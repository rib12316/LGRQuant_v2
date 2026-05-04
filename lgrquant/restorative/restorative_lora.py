# import sys

# # import argparse
# import os
# import torch
# import random
# import numpy as np
# import torch.nn as nn
# import pdb
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     TrainingArguments,
#     Trainer,
#     Seq2SeqTrainer
# )

# # from transformers import LlamaTokenizer, LlamaForCausalLM
# from lgrquant.data.loader import get_qat_dataset


# from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
# from peft import (
#     get_peft_model,
#     LoraConfig,
#     PrefixTuningConfig,
#     PromptEncoderConfig,
#     PromptTuningConfig,
#     TaskType,
# )
# import tensorboard

# def get_scheduler(num_training_steps: int):
#     def lr_scheduler(optimizer):
#         return get_cosine_with_hard_restarts_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=100,
#             num_training_steps=num_training_steps,
#             num_cycles=5,
#         )

#     return lr_scheduler

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )


# def prepare_model_for_training(model):
#     for name, param in model.named_parameters():
#         # freeze base model's layers
#         param.requires_grad = False

#     for param in model.parameters():
#         if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
#             param.data = param.data.to(torch.float32)

#     # For backward compatibility
#     if hasattr(model, "enable_input_require_grads"):
#         model.enable_input_require_grads()
#     else:

#         def make_inputs_require_grad(module, input, output):
#             output.requires_grad_(True)

#         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

#     # enable gradient checkpointing for memory efficiency
#     model.gradient_checkpointing_enable()
#     return model



# def main(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
#     print(f'base model {args.model_id}')
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_id, device_map="auto", torch_dtype=torch.float16
#     )
#     if args.ckpt is not None:
#         print(f'ckpt {args.ckpt}')
#         model.load_state_dict(torch.load(args.ckpt), strict=False)
#     # model=model.to_bettertransformer()

#     model = prepare_model_for_training(model)
#     tokenizer.pad_token = tokenizer.eos_token
#     outputs = 'outputs/' + args.model_id.split('/')[-1] + args.ckpt.split('/')[-1].split('.pt')[0]
#     print('output to ' + outputs)
#     # print("Setup optimizer")
#     # opt = torch.optim.AdamW([
#     #     p
#     #     for p in model.parameters()
#     #     if p.requires_grad
#     # ], lr=training_args.learning_rate)
    
#     # Load dataset
#     print("prepare training data")
#     data = get_qat_dataset(args.dataset, tokenizer, args.data_percent)
#     # pdb.set_trace()

#     print("Setup PEFT")
#     peft_config = LoraConfig(
#         task_type='CAUSAL_LM', inference_mode=False,
#         r=64,
#         lora_alpha=16, lora_dropout=0.1,
#         target_modules=['q_proj','k_proj','v_proj','gate_proj','up_proj','down_proj']
#     )
#     model = get_peft_model(model, peft_config)
#     # Training
#     print_trainable_parameters(model)
#     # replace_with_qlinear(model)
#     # pdb.set_trace()
#     # Print mean bit width
#     tot_bit=0
#     tot_params=0
    
#     # for name, module in model.named_modules():
#     #     if isinstance(module, BinaryInterface):
#     #         module.gen_outlier_mask()
#     #         # print(module.outlier_nbits)
#     #         tot_bit+=(module.outlier_nbits+1)*module.weight.numel()
#     #         tot_params+=module.weight.numel()
#     # print(f"mean_bit: {tot_bit/tot_params} frac: {tot_bit/tot_params/16}")

#     # Define training arguments
#     num_gpus = torch.cuda.device_count()
#     per_device_train_batch_size = 1
#     gradient_accumulation_steps = 2
#     # outputs = 'outputs/llama7b-mix-4-0.1'
#     training_args = TrainingArguments(
#         per_device_train_batch_size=per_device_train_batch_size,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         warmup_steps=args.train_steps * 0.05,
#         learning_rate=1e-4,
#         lr_scheduler_type="cosine",
#         bf16=True,
#         logging_steps=1,
#         max_steps=args.train_steps//(per_device_train_batch_size * gradient_accumulation_steps * num_gpus),
#         # num_train_epochs=5,
#         output_dir=outputs,
#         optim="adamw_torch",
#         report_to="tensorboard",
#     )

#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         train_dataset=data,
#         args=training_args,
#         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     )
    
#     model.config.use_cache = False

#     # Train the model
#     trainer.train()

#     # Save model
#     model.eval()
#     save_dir = outputs + f"/{args.train_steps}"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     # to_regular_linear(model)
#     model.save_pretrained(save_dir)
#     tokenizer.save_pretrained(save_dir)
#     print(f"model saved to {save_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Model Training Script")
#     parser.add_argument(
#         "--model_id",
#         type=str,
#         help="Pretrained model ID",
#     )
#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         help="PTQ ckpt",
#     )
#     parser.add_argument(
#         "--dataset", type=str, default="alpaca", help="Dataset name"
#     )
#     parser.add_argument(
#         "--data_percent", type=float, default=100, help="Percentage of data to use"
#     )
#     parser.add_argument(
#         "-s", "--train_steps", type=int, default=1000, help="Number of training steps"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=42, help="Number of training steps"
#     )


#     args = parser.parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     main(args)


import sys

import os

sys.path.append(os.path.dirname(__file__) or ".")
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import pdb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer
)

# from transformers import LlamaTokenizer, LlamaForCausalLM

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)
try:
    import tensorboard
    _HAS_TENSORBOARD = True
except Exception:
    _HAS_TENSORBOARD = False


def load_quantizers(quantizers_path: str) -> dict:
    if not os.path.exists(quantizers_path):
        raise FileNotFoundError(f"Quantizers file not found: {quantizers_path}")

    if quantizers_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        tensor_map = load_file(quantizers_path, device="cpu")
        quantizers = {}
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

    return torch.load(quantizers_path, map_location="cpu")


def _repeat_to_match(weight_2d: torch.Tensor, x: torch.Tensor, dim: int, group_size: int) -> torch.Tensor:
    if x.dim() == 1:
        if weight_2d.dim() != 2:
            raise ValueError(f"Unsupported weight dim: {weight_2d.dim()}")
        if dim == 1:
            x = x.view(-1, 1)
        else:
            x = x.view(1, -1)

    if x.dim() != weight_2d.dim():
        raise ValueError(f"Scale/zero dim mismatch: x.dim={x.dim()} weight.dim={weight_2d.dim()}")

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
def apply_fake_quant_weights_from_quantizers(
    model: nn.Module,
    quantizers: dict,
    group_size: int = -1,
    asym: bool = False,
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

        scale_e = _repeat_to_match(qw_f, scale_f, dim=dim, group_size=group_size)
        if asym:
            zero_e = _repeat_to_match(qw_f, zero_f, dim=dim, group_size=group_size)
            w = qw_f * scale_e + zero_e
        else:
            w = qw_f * scale_e

        module.weight.data.copy_(w.to(dtype=target_dtype))
        applied += 1

    return applied

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

    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model



def main(args):
    from lgrquant.data.loader import get_qat_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    print(f'base model {args.model_id}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16
    )
    if args.ckpt is not None:
        print(f'ckpt {args.ckpt}')
        model.load_state_dict(torch.load(args.ckpt), strict=False)
    if getattr(args, "quantizers_path", None):
        quantizers = load_quantizers(args.quantizers_path)
        applied = apply_fake_quant_weights_from_quantizers(
            model,
            quantizers,
            group_size=args.quant_group_size,
            asym=args.quant_asym,
            prefix="model.layers.",
        )
        print(f"applied fake-quant weights from quantizers: {applied}")
    # model=model.to_bettertransformer()

    model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token
    ckpt_tag = args.ckpt.split('/')[-1].split('.pt')[0] if args.ckpt is not None else "base"
    outputs = args.out_path + '/' + args.model_id.split('/')[-1] + ckpt_tag
    print('output to ' + outputs)
    # print("Setup optimizer")
    # opt = torch.optim.AdamW([
    #     p
    #     for p in model.parameters()
    #     if p.requires_grad
    # ], lr=training_args.learning_rate)
    
    # Load dataset
    print("prepare training data")
    data = get_qat_dataset(args.dataset, tokenizer, args.data_percent)
    # pdb.set_trace()

    print("Setup PEFT")
    peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16, lora_dropout=0.1,
        target_modules=['q_proj','k_proj','v_proj','gate_proj','up_proj','down_proj']
    )
    model = get_peft_model(model, peft_config)
    # Training
    print_trainable_parameters(model)
    # replace_with_qlinear(model)
    # pdb.set_trace()
    # Print mean bit width
    tot_bit=0
    tot_params=0
    
    # for name, module in model.named_modules():
    #     if isinstance(module, BinaryInterface):
    #         module.gen_outlier_mask()
    #         # print(module.outlier_nbits)
    #         tot_bit+=(module.outlier_nbits+1)*module.weight.numel()
    #         tot_params+=module.weight.numel()
    # print(f"mean_bit: {tot_bit/tot_params} frac: {tot_bit/tot_params/16}")

    # Define training arguments
    num_gpus = torch.cuda.device_count()
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 2
    # outputs = 'outputs/llama7b-mix-4-0.1'
    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.train_steps * 0.05,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        max_steps=args.train_steps//(per_device_train_batch_size * gradient_accumulation_steps * num_gpus),
        # num_train_epochs=5,
        output_dir=outputs,
        optim="adamw_torch",
        report_to="tensorboard" if _HAS_TENSORBOARD else "none",
    )

    # Create trainer
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
    save_dir = outputs + f"/{args.train_steps}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # to_regular_linear(model)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"model saved to {save_dir}")


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
        "-s", "--train_steps", type=int, default=10000, help="Number of training steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Number of training steps"
    )
    parser.add_argument(
        "--quantizers_path",
        type=str,
        default=None,
        help="Quantizers file (.pth or .safetensors) to inject fake-quant weights",
    )
    parser.add_argument(
        "--quant_group_size",
        type=int,
        default=-1,
        help="Group size used by quantizers (-1 for per-channel)",
    )
    parser.add_argument(
        "--quant_asym",
        action="store_true",
        default=False,
        help="Use asymmetric dequantization (apply zero point)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs",
        help="Output path",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank",
    )



    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
