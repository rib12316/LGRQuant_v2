from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
from torch import nn
import torch
import pdb
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    LlamaTokenizerFast
)

from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModel
)
try:
    import tensorboard
except Exception:
    tensorboard = None
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def _load_quantizers(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        tensor_map = load_file(path, device="cpu")
        if not any(k.endswith(".weights") or k.endswith(".scale") or k.endswith(".zero") for k in tensor_map.keys()):
            return tensor_map
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

        return quantizers

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "w_quantizers" in obj and isinstance(obj["w_quantizers"], dict):
        return obj["w_quantizers"]
    return obj


def _repeat_to_match(weight_2d: torch.Tensor, x: torch.Tensor, dim: int, group_size: int) -> torch.Tensor:
    if x.dim() == 1:
        if dim == 1:
            x = x.view(-1, 1)
        else:
            x = x.view(1, -1)

    if group_size == -1:
        if x.shape[dim] == 1:
            repeats = weight_2d.shape[dim]
        else:
            if weight_2d.shape[dim] % x.shape[dim] != 0:
                raise ValueError(f"Cannot expand: weight[{dim}]={weight_2d.shape[dim]} x[{dim}]={x.shape[dim]}")
            repeats = weight_2d.shape[dim] // x.shape[dim]
    else:
        repeats = group_size

    expanded = torch.repeat_interleave(x, repeats=repeats, dim=dim)
    if expanded.shape[dim] != weight_2d.shape[dim]:
        raise ValueError(f"Expanded mismatch on dim={dim}: got {expanded.shape} expected {weight_2d.shape}")
    return expanded


@torch.no_grad()
def apply_fake_quant_from_quantizers(
    model: nn.Module,
    quantizers: dict,
    w_groupsize: int = -1,
    w_asym: bool = False,
    prefix: str = "model.layers.",
) -> int:
    applied = 0
    for q_key, q in quantizers.items():
        if not isinstance(q_key, str) or not isinstance(q, dict):
            continue
        if "weights" not in q or "scales" not in q:
            continue
        qw = q["weights"]
        scales = q["scales"]
        if not torch.is_tensor(qw) or not isinstance(scales, (list, tuple)) or len(scales) < 1:
            continue
        scale = scales[0]
        zero = scales[1] if len(scales) >= 2 else None
        if not torch.is_tensor(scale):
            continue
        if zero is None or not torch.is_tensor(zero):
            zero = torch.zeros_like(scale)

        parts = q_key.split(".")
        if len(parts) < 2:
            continue
        layer_num = parts[0]
        tail = parts[-1]
        module_path = ".".join(parts[1:-1]) if tail in ("weight", "weights") else ".".join(parts[1:])
        if module_path == "":
            continue
        module_name = f"{prefix}{layer_num}.{module_path}"

        try:
            module = model.get_submodule(module_name)
        except Exception:
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        dim = 0 if module.__class__.__name__ in ("Conv1D",) else 1
        device = module.weight.device
        target_dtype = module.weight.dtype

        qw_f = qw.to(device=device, dtype=torch.float32)
        scale_f = scale.to(device=device, dtype=torch.float32)
        zero_f = zero.to(device=device, dtype=torch.float32)
        if qw_f.dim() != 2:
            continue

        scale_e = _repeat_to_match(qw_f, scale_f, dim=dim, group_size=w_groupsize)
        if w_asym:
            zero_e = _repeat_to_match(qw_f, zero_f, dim=dim, group_size=w_groupsize)
            w = qw_f * scale_e + zero_e
        else:
            w = qw_f * scale_e

        if w.shape != module.weight.data.shape:
            continue
        module.weight.data.copy_(w.to(dtype=target_dtype))
        applied += 1

    return applied


def get_accelerate_model(model_path, args, ckpt_path=None, lora_path=None, output_path=None):
    print(f'loading base model {model_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if ckpt_path:
        print(f'loading ckpt {ckpt_path}...')
        loaded = _load_quantizers(ckpt_path)
        if isinstance(loaded, dict) and any(isinstance(v, dict) and "scales" in v and "weights" in v for v in loaded.values()):
            applied = apply_fake_quant_from_quantizers(
                model,
                loaded,
                w_groupsize=args.w_groupsize,
                w_asym=bool(args.w_asym),
                prefix="model.layers.",
            )
            print(f"applied fake-quant weights from quantizers: {applied}")
        elif isinstance(loaded, dict) and any(hasattr(v, "quantize") for v in loaded.values()):
            applied = 0
            for module_name, quantizer in loaded.items():
                if not isinstance(module_name, str) or not hasattr(quantizer, "quantize"):
                    continue
                try:
                    module = model.get_submodule(module_name)
                except Exception:
                    continue
                if not hasattr(module, "weight") or module.weight is None:
                    continue
                module.weight.data = quantizer.quantize(module.weight.data)
                applied += 1
            print(f"applied fake-quant weights from WeightQuantizer dict: {applied}")
        else:
            model.load_state_dict(loaded, strict=False)
    if lora_path is not None:
        print(f'loading lora adpater {lora_path}...')
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    return model
    
def evaluate(model, tokenizer_path, logger):
    from pp_utils import get_loaders

    results = {}
    seqlen = 2048
    seed = 42
    if True:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            dataloader, testloader = get_loaders(
                dataset,
                seed=seed,
                model=tokenizer_path,
                seqlen=seqlen,
            )
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].cuda()
                # pdb.set_trace()
                # logits = model(batch)['logits']
                outputs = model.model(batch)
                logits = outputs[0]
                logits = model.lm_head(logits)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                    :, 1:
                ].to(model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()

            print('perplexity result:')
            for k,v in results.items():
                print(k, v)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出保存路径")
    parser.add_argument("--ckpt", type=str, default=None, help="检查点路径")
    parser.add_argument("--gptq", action="store_true", help="是否使用 GPTQ 量化")
    parser.add_argument("--merged_path", type=str, default=None, help="合并后的模型路径")
    parser.add_argument("--nsamples", type=int, default=128, help="样本数量")
    parser.add_argument("--w_bits", type=int, default=2, help="权重位宽")
    parser.add_argument("--w_asym", type=bool, default=True, help="是否使用非对称权重量化")
    parser.add_argument("--percdamp", type=float, default=0.01, help="dampening percentage")
    parser.add_argument("--gptq_mse", type=bool, default=False, help="是否使用 MSE 损失")
    parser.add_argument("--act_order", type=bool, default=False, help="是否使用激活顺序")
    parser.add_argument("--w_groupsize", type=int, default=-1, help="权重分组大小")
    args = parser.parse_args()

    model_path = args.model_path
    lora_path = args.lora_path
    if args.merged_path is not None:
        lora_path = None
        model_path = args.merged_path
    output_path = args.output_path
    ckpt = args.ckpt
    model = get_accelerate_model(model_path, args, ckpt, lora_path, output_path)
    model.eval()
    # import pdb
    # pdb.set_trace()s
    # print(model.device)

    # # import datautils
    # # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # # trainloader, testloader = datautils.get_loaders(
    # #     "wikitext2", nsamples=args.nsamples, seed=0, seqlen=2048, model=model_path, cache_dir=""
    # # )

    # # import gptq_utils as gptq_utils
    # # if args.gptq: # GPTQ Weight Quantization
    # #     quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda:0", args)
    # # else: # RTN Weight Quantization
    # #     quantizers = gptq_utils.rtn_fwrd(model, "cuda:0", args)
    # model.to("cuda:0")

    # for n,p in model.named_parameters():
    #     p.requires_grad = False
    # results = evaluate(model ,model_path, logger)
    # print('perplexity result:')
    # for k,v in results.items():
    #     print(k, v)
