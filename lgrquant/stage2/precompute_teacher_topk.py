"""
Precompute teacher top-K logits for distillation.

Teacher forward in no_grad mode does not trigger the ToCopyBackward0 NaN bug
that affects training (see plan file). Caching top-K logits lets stage2
training skip teacher model entirely, freeing ~28GB/GPU for student FSDP.
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lgrquant.data import loader as data_loader


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[precompute] loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[precompute] loading teacher {args.model_id} (bf16, single GPU)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"[precompute] preparing dataset {args.dataset}, nsamples={args.nsamples}, seqlen={args.seqlen}")
    data_list = data_loader.get_loaders(
        args, args.dataset, tokenizer,
        nsamples=args.nsamples, seqlen=args.seqlen, eval_mode=False,
    )
    assert isinstance(data_list, list), f"expected list, got {type(data_list)}"
    print(f"[precompute] got {len(data_list)} samples")

    t0 = time.time()
    peak_mem = 0
    for i, (inp, _tar) in enumerate(data_list):
        input_ids = inp.to(model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            top_logits, top_indices = logits.topk(args.top_k, dim=-1, sorted=False)

        top_logits = top_logits.squeeze(0).to(torch.float16).cpu()
        top_indices = top_indices.squeeze(0).to(torch.int32).cpu()

        torch.save(
            {"top_logits": top_logits, "top_indices": top_indices},
            os.path.join(args.out_dir, f"sample_{i:05d}.pt"),
        )

        if torch.cuda.is_available():
            cur = torch.cuda.max_memory_allocated() / 1e9
            if cur > peak_mem:
                peak_mem = cur

        if (i + 1) % 20 == 0 or (i + 1) == len(data_list):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(data_list) - i - 1) / rate
            print(
                f"[precompute] {i+1}/{len(data_list)}  "
                f"elapsed={elapsed:.1f}s  rate={rate:.2f}/s  eta={eta:.1f}s  "
                f"peak_mem={peak_mem:.2f}GB",
                flush=True,
            )

    meta = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "top_k": args.top_k,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "peak_gpu_mem_gb": round(peak_mem, 2),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[precompute] done. meta.json written to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "pile"])
    parser.add_argument("--nsamples", type=int, default=1000)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
