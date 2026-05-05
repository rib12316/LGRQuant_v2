import sys
import time
from typing import Dict, Tuple

import argparse
import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
from lgrquant.data import loader as data_loader

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer
)


from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


def tokenization_func(example, tokenizer, max_length):
    return tokenizer(example["text"], truncation=True, max_length=max_length)



def get_deita_10k(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    from datasets import load_dataset
    def format_deita(example):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        roles = {"human": "USER", "gpt": "ASSISTANT"}
        sep = " "
        sep2 = "</s>"
        
        conversations = example["conversations"]
        formatted_text = system_prompt + sep
        
        for sentence in conversations:
            role = roles.get(sentence["from"], sentence["from"])
            content = sentence["value"]
            
            if role == "USER":
                formatted_text += role + ": " + content + sep
            else:
                formatted_text += role + ": " + content + sep2
        
        return {"text": formatted_text}

    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{percent}%]"
    else:
        split = "train"
    
    dataset = load_dataset("hkust-nlp/deita-10k-v0", split=split)
    
    dataset = dataset.map(format_deita, num_proc=os.cpu_count())
    
    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset

def get_alpaca(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    from datasets import load_dataset
    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("tatsu-lab/alpaca", split=split)

    processed_dataset = dataset.map(
        lambda x: tokenization_func(x, tokenizer, max_length), 
        batched=True, 
        batch_size=batch_size, 
        num_proc=os.cpu_count()
    )
    processed_dataset = processed_dataset.shuffle(seed=seed)
    return processed_dataset

def load_quantizers(quantizers_path: str) -> Dict:
    """加载quantizers文件"""
    if not os.path.exists(quantizers_path):
        raise FileNotFoundError(f"Quantizers file not found: {quantizers_path}")

    if quantizers_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        tensor_map = load_file(quantizers_path, device="cpu")
        quantizers: Dict[str, Dict] = {}
        for full_key, tensor in tensor_map.items():
            if full_key.endswith(".weights"):
                q_key = full_key[: -len(".weights")]
                quantizers.setdefault(q_key, {})["weights"] = tensor
            elif full_key.endswith(".scale"):
                q_key = full_key[: -len(".scale")]
                quantizers.setdefault(q_key, {}).setdefault("scales", [None, None])[0] = tensor
            elif full_key.endswith(".zero"):
                q_key = full_key[: -len(".zero")]
                quantizers.setdefault(q_key, {}).setdefault("scales", [None, None])[1] = tensor

        for q_key, q in quantizers.items():
            if "scales" not in q or q["scales"][0] is None:
                raise ValueError(f"Invalid quantizers.safetensors: missing scale for key={q_key}")
            if q["scales"][1] is None:
                q["scales"][1] = torch.zeros_like(q["scales"][0])

        print(f"Loaded quantizers from {quantizers_path}")
        return quantizers

    quantizers = torch.load(quantizers_path, map_location="cpu")
    print(f"Loaded quantizers from {quantizers_path}")
    return quantizers


def parse_true_quant_state_dict(true_quant_state: Dict, prefix: str = "model.layers.") -> Tuple[Dict, Dict]:
    """
    Parse true_quant state dict from unified checkpoint or legacy format.
    Supports both unified checkpoint {"true_quant": {...}} and legacy state dict formats.
    """
    # Handle unified checkpoint format
    if isinstance(true_quant_state, dict) and "true_quant" in true_quant_state:
        print("  [format] unified checkpoint (quantized_model.pth)")
        true_quant_state = true_quant_state["true_quant"]
    elif isinstance(true_quant_state, dict) and "quantizers" in true_quant_state:
        # Unified checkpoint but loaded the whole thing - use true_quant if available
        if "true_quant" in true_quant_state:
            true_quant_state = true_quant_state["true_quant"]
            print("  [format] unified checkpoint (using true_quant field)")

    quantizers = {}
    base_state_dict = {}

    for k, v in true_quant_state.items():
        if not isinstance(k, str):
            continue
        if k.endswith("_qscale") or k.endswith("_qzero"):
            continue

        qscale_key = k + "_qscale"
        if k.startswith(prefix) and qscale_key in true_quant_state:
            scale = true_quant_state[qscale_key]
            qzero_key = k + "_qzero"
            if qzero_key in true_quant_state:
                zero = true_quant_state[qzero_key]
            else:
                zero = torch.zeros_like(scale)

            new_k = k.replace(prefix, "", 1)
            quantizers[new_k] = {"weights": v, "scales": [scale, zero]}
        else:
            base_state_dict[k] = v

    return quantizers, base_state_dict


def get_scheduler(num_training_steps: int):
    def lr_scheduler(optimizer):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
            num_cycles=5,
        )

    return lr_scheduler

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_model_for_training(model):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


class QuantizedLinear(nn.Module):
    """
    包装层，用于在 forward 时动态反量化权重。
    与 HuggingFace Accelerate 兼容，不修改原始模块的 forward 方法。
    """
    def __init__(self, original_module, scale, zero, group_size, asym):
        super().__init__()
        # 保留原 module 仅供读 bias / 识别 Conv1D / 读 nf；forward 不再走它的 linear
        self.original_module = original_module
        self.scale = scale
        self.zero = zero
        self.group_size = group_size
        self.asym = asym

        # quantized_weight 作为 frozen nn.Parameter（而非 buffer），FSDP 才能分片
        qw = original_module.weight.data.detach().clone()
        self.quantized_weight = nn.Parameter(qw, requires_grad=False)
        # 释放原 Linear 的 weight，避免同一份数据占双份显存
        original_module.weight = None

        # 确定维度
        if original_module.__class__.__name__ in ("Conv1D",):
            self.dim = 0
        else:
            self.dim = 1
            
    def get_dequantized_weight(self):
        """动态反量化权重"""
        if self.group_size == -1:
            group_size = self.quantized_weight.shape[self.dim]
        else:
            group_size = self.group_size
            
        # 扩展 scale 和 zero 到与权重相同的形状
        scale_repeated = torch.repeat_interleave(self.scale, repeats=group_size, dim=self.dim)
        if self.asym:
            zero_repeated = torch.repeat_interleave(self.zero, repeats=group_size, dim=self.dim)
            weight = self.quantized_weight * scale_repeated + zero_repeated
        else:
            weight = self.quantized_weight * scale_repeated
            
        return weight
    
    def forward(self, input):
        # 动态计算反量化后的权重
        weight = self.get_dequantized_weight()
        if weight.device != input.device or weight.dtype != input.dtype:
            weight = weight.to(device=input.device, dtype=input.dtype)
        
        # 执行线性变换
        if self.original_module.__class__.__name__ in ("Conv1D",):
            size_out = input.size()[:-1] + (self.original_module.nf,)
            output = torch.mm(input.view(-1, input.size(-1)), weight)
            output = output.view(size_out)
        else:
            output = F.linear(input, weight)
            
        if self.original_module.bias is not None:
            bias = self.original_module.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias
            
        return output


def replace_with_quantized_layers(model, quantizers, args):
    """
    将模型中的 Linear 层替换为 QuantizedLinear 包装层。
    使用 module replacement 而非 forward hook，与 Accelerate 兼容。
    """
    module_to_quantizer = {}
    original_dtype = {}
    
    # 构建模块到量化器的映射
    for q_key in quantizers:
        parts = q_key.split(".")
        if len(parts) < 3:
            continue
        layer_num = parts[0]
        module_path = ".".join(parts[1:-1])
        module_pattern = f"model.layers.{layer_num}.{module_path}"
        module_to_quantizer[module_pattern] = q_key

    replaced_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D",):
            if name in module_to_quantizer:
                q_key = module_to_quantizer[name]
                quantizer = quantizers[q_key]
                weight = quantizer["weights"]
                scale_list = quantizer["scales"]

                original_dtype[name] = module.weight.dtype

                device = module.weight.device

                # 量化权重（frozen）走 bf16
                qw_kwargs = {"dtype": torch.bfloat16, "device": device}
                module.weight.data = weight.to(**qw_kwargs)
                module.weight.requires_grad_(False)

                # scale/zero（trainable）必须 fp32：bf16 mantissa 只 7 bit，
                # adamw lr=2e-5 × grad 量级 1e-5 远小于 bf16 ULP，每步 update 被 round 成 0
                sz_kwargs = {"dtype": torch.float32, "device": device}
                scale = torch.nn.Parameter(scale_list[0].clone().to(**sz_kwargs), requires_grad=True)
                requires_grad = True if args.asym else False
                zero = torch.nn.Parameter(scale_list[1].clone().to(**sz_kwargs), requires_grad=requires_grad)

                # 创建包装层
                quantized_layer = QuantizedLinear(
                    module, scale, zero, args.group_size, args.asym
                )

                # 替换模块
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, quantized_layer)

                replaced_names.append(name)
                
    print(f"Replaced {len(replaced_names)} layers with QuantizedLinear")
    return original_dtype, module_to_quantizer



def recover_original_layers(model, quantizers, module_to_quantizer, original_dtype, args):
    """
    训练完成后，将 QuantizedLinear 恢复为原始 Linear 层，并应用最终的 scale 和 zero。
    """
    recovered_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            # 找到父模块和属性名
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = model.get_submodule(parent_name)
            
            # 获取反量化后的权重
            dequantized_weight = module.get_dequantized_weight()

            # 恢复原始模块的权重
            original_module = module.original_module
            if original_module.weight is None:
                original_module.weight = torch.nn.Parameter(
                    dequantized_weight.to(original_dtype[name])
                )
            else:
                original_module.weight.data = dequantized_weight.to(original_dtype[name])
            
            # 如果训练了 bias，确保 bias 被保留
            if hasattr(args, 'train_bias') and args.train_bias and original_module.bias is not None:
                # bias 已经在原始模块上，不需要额外处理
                pass
            
            # 替换回原始模块
            setattr(parent_module, child_name, original_module)
            
            # 保存 scale 和 zero 到 quantizers
            if name in module_to_quantizer:
                q_key = module_to_quantizer[name]
                quantizers[q_key]["scales"] = [module.scale.detach().cpu(), module.zero.detach().cpu()]
                
            recovered_names.append(name)
            
    print(f"Recovered {len(recovered_names)} layers to original Linear")
    return quantizers



def get_finetune_quantizers(model, quantizers, module_to_quantizer, original_dtype, args):
    """训练完成后，把 QuantizedLinear 的 scale/zero 写回 quantizers dict。

    FSDP 路径下不能直接读 module.scale —— wrap 后 param 是 flat_param 的 1D shard view。
    用 PyTorch 2.4+ 的 distributed.checkpoint API 在所有 rank 上 all-gather 出一份完整
    state_dict（rank0 持有），再按 `<orig_name>.scale` / `<orig_name>.zero` 反查写回。
    """
    is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
        is_fsdp = isinstance(model, _FSDP) or any(
            isinstance(m, _FSDP) for m in model.modules()
        )
    except Exception:
        pass

    if is_fsdp:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_sd = get_model_state_dict(model, options=opts)
        is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if is_rank0:
            saved = 0
            missed = []
            for orig_name, q_key in module_to_quantizer.items():
                scale_key = f"{orig_name}.scale"
                zero_key = f"{orig_name}.zero"
                scale = full_sd.get(scale_key, None)
                zero = full_sd.get(zero_key, None)
                if scale is None:
                    missed.append(orig_name)
                    continue
                quantizers[q_key]["scales"] = [
                    scale.detach().cpu(),
                    zero.detach().cpu() if zero is not None else None,
                ]
                saved += 1
            print(f"saved {saved}/{len(module_to_quantizer)} layers to quantizers (FSDP get_model_state_dict)")
            if missed:
                print(f"  WARN: {len(missed)} keys missing from full_sd, e.g. {missed[:3]}")
        return quantizers

    # 非 FSDP 路径：直接读 module 上的参数
    saved = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) and name in module_to_quantizer:
            q_key = module_to_quantizer[name]
            quantizers[q_key]["scales"] = [
                module.scale.detach().cpu(),
                module.zero.detach().cpu(),
            ]
            saved += 1
    print(f"saved {saved} layers to quantizers")
    return quantizers


def save_quantizers(quantizers: Dict, out_dir: str, save_format: str = "pth", meta: Dict = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if save_format == "pth":
        # 仅保存 unified checkpoint
        unified_ckpt = {
            "quantizers": quantizers,
            "true_quant": None,  # 由 save_quant_model 后续填充
            "meta": meta or {"version": "2.0"}
        }
        out_path = os.path.join(out_dir, "quantized_model.pth")
        torch.save(unified_ckpt, out_path)
        return out_path

    if save_format == "safetensors":
        from safetensors.torch import save_file

        tensor_map: Dict[str, torch.Tensor] = {}
        for q_key, q in quantizers.items():
            if not isinstance(q_key, str) or not isinstance(q, dict):
                continue
            weights = q.get("weights", None)
            scales = q.get("scales", None)
            if torch.is_tensor(weights):
                tensor_map[f"{q_key}.weights"] = weights.detach().cpu()
            if isinstance(scales, (list, tuple)) and len(scales) >= 1 and torch.is_tensor(scales[0]):
                tensor_map[f"{q_key}.scale"] = scales[0].detach().cpu()
            if isinstance(scales, (list, tuple)) and len(scales) >= 2 and torch.is_tensor(scales[1]):
                tensor_map[f"{q_key}.zero"] = scales[1].detach().cpu()

        out_path = os.path.join(out_dir, "quantizers.safetensors")
        save_file(tensor_map, out_path)
        return out_path

    raise ValueError(f"Unknown save_format: {save_format}")


def get_finetune_ln_bias_state_dict(model: nn.Module, args) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    train_ln = bool(getattr(args, "train_LN", False))
    train_bias = bool(getattr(args, "train_bias", False))

    if not (train_ln or train_bias):
        return out

    for name, module in model.named_modules():
        if train_ln and (isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in module.__class__.__name__):
            if hasattr(module, "weight") and torch.is_tensor(module.weight):
                out[f"{name}.weight"] = module.weight.detach().cpu()
            if hasattr(module, "bias") and torch.is_tensor(module.bias):
                out[f"{name}.bias"] = module.bias.detach().cpu()

        if train_bias:
            if isinstance(module, QuantizedLinear):
                bias = getattr(module.original_module, "bias", None)
                if torch.is_tensor(bias):
                    out[f"{name}.bias"] = bias.detach().cpu()
            elif isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D",):
                if name.endswith(".original_module"):
                    continue
                bias = getattr(module, "bias", None)
                if torch.is_tensor(bias):
                    out[f"{name}.bias"] = bias.detach().cpu()

    return out


def save_quant_model(args, model, quantizers, prefix):
    model = model.cpu()
    state_dict = model.state_dict()
    fake_quant, true_quant = state_dict, {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            true_quant[k] = v
        else:
            new_k = k.replace(prefix, "")
            if new_k not in quantizers:
                true_quant[k] = v
            else:
                true_quant[k + '_qscale'] = quantizers[new_k]["scales"][0]
                if args.asym:
                    true_quant[k + '_qzero'] = quantizers[new_k]["scales"][1]
                true_quant[k] = quantizers[new_k]["weights"]

    # 仅保存 unified checkpoint (单个文件)
    unified_ckpt = {
        "quantizers": quantizers,
        "true_quant": true_quant,
        "meta": {
            "bits": getattr(args, "wbits", 2),
            "group_size": getattr(args, "group_size", 64),
            "sym": not getattr(args, "asym", True),
            "asym": getattr(args, "asym", True),
            "version": "2.0"
        }
    }
    unified_path = (args.out_path + "/quantized_model.pth") if args.out_path else "quantized_model.pth"
    torch.save(unified_ckpt, unified_path)
    print(f"Saved unified checkpoint to {unified_path}")
    return unified_ckpt


class DistillTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, distill_loss_type="mse", **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_loss_type = distill_loss_type

    def compute_loss(self, model, inputs, return_outputs=False):
        # 移除 labels 以避免计算原始任务 loss
        if "labels" in inputs:
            inputs.pop("labels")

        # 若 dataset 提供预缓存的 teacher top-K logits/indices，跳过 teacher forward
        cached_top_logits = inputs.pop("teacher_top_logits", None)
        cached_top_indices = inputs.pop("teacher_top_indices", None)
        use_cache = cached_top_logits is not None and cached_top_indices is not None

        # 学生模型前向传播
        outputs = model(**inputs)
        student_logits = outputs.logits

        # 计算 Loss
        if use_cache:
            # 缓存路径只支持 kl_top
            top_student_logits = student_logits.gather(-1, cached_top_indices)
            top_teacher_logits = cached_top_logits.to(top_student_logits.dtype)
            loss = F.kl_div(
                F.log_softmax(top_student_logits, dim=-1).flatten(0, -2),
                F.softmax(top_teacher_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
        else:
            # 在线 teacher forward
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            if self.distill_loss_type == "mse":
                loss = F.mse_loss(student_logits, teacher_logits)
            elif "kl_top" in self.distill_loss_type:
                if self.distill_loss_type == "kl_top":
                    k = 1000
                else:
                    try:
                        k = int(self.distill_loss_type.split("_")[-1])
                    except:
                        k = 1000
                top_teacher_logits, indices = teacher_logits.topk(k, dim=-1, sorted=False)
                top_student_logits = student_logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_student_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_teacher_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
            else:
                loss = F.mse_loss(student_logits, teacher_logits)

        return (loss, outputs) if return_outputs else loss



def main(args):
    # FSDP 通过 torchrun 启动；每个 rank 只看到 1 张卡，本 process 应绑定到 LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    is_main_process = local_rank == 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    if is_main_process:
        print(f'base model {args.model_id}')
    # student 加载到当前 rank 的 GPU；FSDP 启用时由 Trainer 之后做 wrap+shard
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(f"cuda:{local_rank}")
    if getattr(args, "true_quant_path", None) is not None:
        if args.ckpt is not None:
            raise ValueError("--true_quant_path 与 --ckpt 不能同时使用")
        true_quant_state = torch.load(args.true_quant_path, map_location="cpu")
        quantizers, base_state_dict = parse_true_quant_state_dict(true_quant_state, prefix="model.layers.")
        if len(quantizers) == 0:
            raise ValueError(f"checkpoint 中未找到任何 *_qscale 键：{args.true_quant_path}")
        model.load_state_dict(base_state_dict, strict=False)
    else:
        if args.ckpt is not None:
            print(f'ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt), strict=False)

    model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("prepare training data")
    if args.dataset == "alpaca":
        data = get_alpaca(tokenizer, args.data_percent)
    elif args.dataset == "deita":
        data = get_deita_10k(tokenizer, args.data_percent)
    elif args.dataset in ['wikitext2', 'ptb', 'c4', 'pile']:
        model_max_seqlen = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
        if getattr(args, "seqlen", None) is None or args.seqlen <= 0:
            seqlen = model_max_seqlen
        else:
            seqlen = min(model_max_seqlen, args.seqlen)
        print(f"dataset seqlen={seqlen} (model_max={model_max_seqlen})")
        data_list = data_loader.get_loaders(
            args, args.dataset, tokenizer, nsamples=args.nsamples, seqlen=seqlen, eval_mode=False
        )
        
        # Convert list of tuples (input, target) to HF Dataset format
        topk_dir = getattr(args, "teacher_topk_dir", None)

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data_list, topk_dir=None):
                self.data_list = data_list
                self.topk_dir = topk_dir

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, i):
                item = {
                    "input_ids": self.data_list[i][0].squeeze(0),
                    "labels": self.data_list[i][1].squeeze(0),
                    "attention_mask": torch.ones_like(self.data_list[i][0].squeeze(0)),
                }
                if self.topk_dir is not None:
                    cache_path = os.path.join(self.topk_dir, f"sample_{i:05d}.pt")
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    item["teacher_top_logits"] = cached["top_logits"]
                    item["teacher_top_indices"] = cached["top_indices"].long()
                return item

        data = CustomDataset(data_list, topk_dir=topk_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # # data = get_deita_10k(tokenizer, args.data_percent)

    # prepare model for training
    # 加载quantizers
    if getattr(args, "true_quant_path", None) is None:
        if args.quantizers_path is None:
            raise ValueError("请提供 --quantizers_path 或 --true_quant_path")
        quantizers = load_quantizers(args.quantizers_path)
    print("Starting end-to-end finetuning of scale and zero...")
    t1 = time.time()
    
    # Set model to train mode
    model.train()

    # 替换为 QuantizedLinear 层
    original_dtype, module_to_quantizer = replace_with_quantized_layers(model, quantizers, args)
    
    # 收集可训练参数
    params = []
    
    # 收集 QuantizedLinear 中的 scale 和 zero
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            params.append(module.scale)
            if args.asym:
                params.append(module.zero)
    
    # Add LayerNorm parameters if needed
    if hasattr(args, 'train_LN') and args.train_LN:
        for k, m in model.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
    
    # Add bias parameters if needed
    if hasattr(args, 'train_bias') and args.train_bias:
        for k, m in model.named_modules():
            if isinstance(m, QuantizedLinear):
                if hasattr(m.original_module, "bias") and m.original_module.bias is not None:
                    m.original_module.bias.requires_grad_(True)
                    params.append(m.original_module.bias)
            elif isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ("Conv1D",):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)

    # Training
    print_trainable_parameters(model)

    tot_bit=0
    tot_params=0
    

    # Define training arguments
    num_gpus = torch.cuda.device_count()
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 2
    output_dir = args.out_path if args.out_path is not None else "outputs"
    os.makedirs(output_dir, exist_ok=True)

    fsdp_arg = ""
    fsdp_config = None
    if getattr(args, "fsdp", False):
        fsdp_arg = "full_shard auto_wrap"
        fsdp_config = {
            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "use_orig_params": True,
        }

    # FSDP 路径下必须关闭 TrainingArguments.bf16，否则 accelerator 会把
    # FSDP param shard upcast 到 fp32（每卡 14GB 翻倍到 OOM）。model 本身
    # 已在 bf16 加载，纯 bf16 训练即可。
    bf16_training = not getattr(args, "fsdp", False)

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=int(args.train_steps * 0.05),
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        bf16=bf16_training,
        logging_steps=1,
        max_steps=args.train_steps//(per_device_train_batch_size * gradient_accumulation_steps * num_gpus),
        output_dir=output_dir,
        optim="adamw_torch",
        report_to="none",
        save_strategy=args.hf_save_strategy,
        save_steps=args.hf_save_steps if args.hf_save_strategy == "steps" and args.hf_save_steps > 0 else 500,
        fsdp=fsdp_arg,
        fsdp_config=fsdp_config,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Create trainer
    if args.use_distill:
        print("Initializing DistillTrainer...")
        topk_dir = getattr(args, "teacher_topk_dir", None)
        if topk_dir is not None:
            print(f"Using cached teacher top-K logits from {topk_dir}; teacher model skipped")
            teacher_model = None
        else:
            # 原路径：实时 teacher forward（注意：device_map='auto' 在本机会触发 NaN bug）
            print(f'Loading teacher model {args.model_id}')
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.model_id, device_map="auto", torch_dtype=torch.bfloat16
            )
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False

        from transformers import default_data_collator
        trainer = DistillTrainer(
            model=model,
            teacher_model=teacher_model,
            distill_loss_type=args.distill_loss,
            train_dataset=data,
            args=training_args,
            data_collator=default_data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=data,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
    
    model.config.use_cache = False

    # Train the model
    trainer.train()

    # Save model
    model.eval()

    # FSDP 路径下必须传 trainer.model（被 wrap 的对象），才能正确走 FULL_STATE_DICT
    save_target = trainer.model if hasattr(trainer, "model") else model
    quantizers = get_finetune_quantizers(save_target, quantizers, module_to_quantizer, original_dtype, args)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        out_dir = args.out_path if args.out_path is not None else "outputs"
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "bits": getattr(args, "wbits", 2),
            "group_size": getattr(args, "group_size", 64),
            "sym": not getattr(args, "asym", True),
            "asym": getattr(args, "asym", True),
            "version": "2.0"
        }
        saved_path = save_quantizers(quantizers, out_dir=out_dir, save_format=args.quantizers_save_format, meta=meta)
        print(f"quantizers saved to {saved_path}")
        extra_sd = get_finetune_ln_bias_state_dict(model, args)
        if len(extra_sd) > 0:
            extra_path = os.path.join(out_dir, "finetuned_ln_bias.pth")
            torch.save(extra_sd, extra_path)
            print(f"finetuned ln/bias saved to {extra_path}")
    
    # 恢复原始层并保存最终的 unified checkpoint (包含 true_quant)
    quantizers = recover_original_layers(model, quantizers, module_to_quantizer, original_dtype, args)
    save_quant_model(args, model, quantizers, prefix="model.layers.")
    save_dir = args.out_path if args.out_path is not None else "outputs"
    print(f"unified checkpoint saved to {save_dir}/quantized_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="PTQ ckpt",
    )
    parser.add_argument(
        "--dataset", type=str, default="alpaca", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=100, help="Percentage of data to use"
    )
    parser.add_argument(
        "-s", "--train_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Number of training steps"
    )
    parser.add_argument(
        '--quantizers_path', type=str, default=None,
        help='Path to save the quantized model'
    )
    parser.add_argument(
        '--true_quant_path', type=str, default=None,
        help='Load true_quant.pth directly for training (resume scales/zeros and weights)'
    )
    parser.add_argument(
        '--out_path', type=str, default=None,
        help='Path to save the quantized model'
    )
    parser.add_argument(
        "--quantizers_save_format",
        type=str,
        default="pth",
        choices=["pth", "safetensors"],
        help="Format for saving quantizers (pth or safetensors)",
    )
    parser.add_argument(
        "--hf_save_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
        help="Transformers Trainer checkpoint save strategy",
    )
    parser.add_argument(
        "--hf_save_steps",
        type=int,
        default=0,
        help="Save steps when hf_save_strategy=steps (0 uses default 500)",
    )
    parser.add_argument(
        '--asym', action='store_true', default=False,
        help='Use asymmetric quantization (train zero point)'
    )
    parser.add_argument(
        '--group_size', type=int, default=-1,
        help='Group size for quantization (default: -1 means per-channel)'
    )
    parser.add_argument(
        '--train_LN', action='store_true', default=False,
        help='Train LayerNorm parameters'
    )
    parser.add_argument(
        '--train_bias', action='store_true', default=False,
        help='Train bias parameters'
    )
    parser.add_argument(
        '--use_distill', action='store_true', default=False,
        help='Enable distillation training mode (minimize error with original model)'
    )
    parser.add_argument(
        '--distill_loss', type=str, default="mse",
        help='Loss function for distillation: "mse" (default) or "kl_top"'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of samples to use for calibration/finetuning'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='Sequence length for wikitext2/ptb/c4/pile (<=0 uses model max_position_embeddings)'
    )
    parser.add_argument(
        '--teacher_topk_dir', type=str, default=None,
        help='Directory of precomputed teacher top-K logits (sample_*.pt). When set, teacher model is not loaded.'
    )
    parser.add_argument(
        '--fsdp', action='store_true', default=False,
        help='Enable FSDP (full_shard + auto_wrap of Qwen2DecoderLayer). Use torchrun --nproc_per_node=N.'
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
