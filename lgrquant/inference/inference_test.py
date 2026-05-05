#!/usr/bin/env python3
"""
LGRQuant 真量化推理测试脚本（重写版）
=================================================

支持两种加载方式：
  1. --true_quant_path : 直接加载 true_quant.pth（state_dict 格式：
       model.layers.X.<sub>.weight        int8
       model.layers.X.<sub>.weight_qscale fp32  [out, k/gs]
       model.layers.X.<sub>.weight_qzero  fp32  [out, k/gs]
       其余 embed_tokens / *_layernorm / norm / lm_head / bias 同 HF state_dict
     )
  2. --quantizers_path : 加载训练阶段保存的 quantizers dict，自动转 state_dict

测试指标：
  - 文本生成 sample（直观看输出是否正常）
  - PPL（wikitext2 / c4）
  - 下游 zero-shot 任务（piqa / hellaswag / arc_easy / arc_challenge /
                       winogrande / lambada_openai）
  - 推理速度对比（FP baseline vs W2A16）

W2A16 内核仅支持 __half (fp16)，已在 LinearW2A16.forward 内做 bf16<->fp16
桥接，外部主干可继续保持 bf16。
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from lgrquant.core.linear_w2a16 import LinearW2A16  # noqa: E402
from lgrquant.core.decoupleQ_kernels import (  # noqa: E402
    dQ_preprocess_weights_int2_for_weight_only,
)
from lgrquant.data.loader import get_loaders, get_loaders_legacy  # noqa: E402

# W4 imports are deferred to avoid marlin dependency when using W2 only
def _get_linear_w4a16():
    from lgrquant.core.linear_w4a16 import LinearW4A16, _prewarm_w4a16
    return LinearW4A16, _prewarm_w4a16


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def _disable_init():
    def skip(*a, **kw): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def _get_model(model_path):
    _disable_init()
    name = model_path.lower()
    if "qwen" in name:
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = 2048
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 真量化层装配
# ---------------------------------------------------------------------------
def make_qw2_linear(old_linear: nn.Linear, sd: dict, group_size: int, name: str):
    """
    把一个 nn.Linear 替换成 LinearW2A16，权重/scale/zp/bias 全部就位。

    与 decoupleQ/llama.py 中 make_qw2_linear 的形状惯例保持一致：
      ckpt:  weight  [out, in]   int8
             qscale  [out, k/gs] fp32
             qzero   [out, k/gs] fp32
      LinearW2A16 期望（经过 .t() 之后）：
             weight  [in, out]   int8 -> 内核内做 int2 packing
             scale   [k/gs, out] fp16
             zp      [k/gs, out] fp16
             bias    [out]       fp16
    """
    in_features = old_linear.in_features
    out_features = old_linear.out_features
    has_bias = old_linear.bias is not None

    layer = LinearW2A16(in_features, out_features, has_bias, group_size)

    # ---- weight (int8) ----
    w = sd[f"{name}.weight"]
    assert w.dtype == torch.int8, f"{name}.weight expected int8, got {w.dtype}"
    layer.weight = w.t().contiguous()  # [in, out] int8

    # ---- scale / zero ----
    if f"{name}.weight_qscale" in sd:
        scale = sd[f"{name}.weight_qscale"].cuda().to(torch.float16).t().contiguous()
        layer.scale = scale
    if f"{name}.weight_qzero" in sd:
        zp = sd[f"{name}.weight_qzero"].cuda().to(torch.float16).t().contiguous()
        layer.zp = zp

    # ---- bias ----
    if has_bias:
        if f"{name}.bias" in sd:
            b = sd[f"{name}.bias"]
        else:
            b = old_linear.bias.detach()
        layer.bias = b.cuda().to(torch.float16).contiguous()

    return layer


def make_qw4_linear(old_linear: nn.Linear, sd: dict, group_size: int, name: str):
    """把一个 nn.Linear 替换成 LinearW4A16（sym，无 zp）。"""
    # Lazy import to avoid marlin dependency when using W2 only
    LinearW4A16, _ = _get_linear_w4a16()

    in_features = old_linear.in_features
    out_features = old_linear.out_features
    has_bias = old_linear.bias is not None

    layer = LinearW4A16(in_features, out_features, has_bias, group_size)

    # weight (int8) —— ckpt 里存 [out, in]，转置为 [in, out]
    w = sd[f"{name}.weight"]
    assert w.dtype == torch.int8, f"{name}.weight expected int8, got {w.dtype}"
    layer.weight = w.t().contiguous()  # [in, out] int8

    # scale（sym 路径下只有 qscale，无 qzero）
    if f"{name}.weight_qscale" in sd:
        # ckpt: [out, groups] fp32 -> LinearW4A16 期望 [groups, out] fp16（pack 时会再转）
        scale = sd[f"{name}.weight_qscale"].cuda().to(torch.float16).t().contiguous()
        layer.scales = scale

    # bias
    if has_bias:
        if f"{name}.bias" in sd:
            b = sd[f"{name}.bias"]
        else:
            b = old_linear.bias.detach()
        layer.bias = b.cuda().to(torch.float16).contiguous()

    return layer


def replace_linear_quantized(model, sd: dict, group_size: int, make_fn):
    """通用量化层替换：对 model.model.layers 里的 q/k/v/o/gate/up/down 做替换。"""
    layers = model.model.layers
    targets = [
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ]
    for i, layer in enumerate(layers):
        for parent, child in targets:
            full_name = f"model.layers.{i}.{parent}.{child}"
            if f"{full_name}.weight" not in sd:
                continue
            old = getattr(getattr(layer, parent), child)
            new = make_fn(old, sd, group_size, full_name)
            setattr(getattr(layer, parent), child, new)
    return model


def replace_linear_with_w2(model, sd: dict, group_size: int):
    """W2A16 替换（兼容旧入口）。"""
    return replace_linear_quantized(model, sd, group_size, make_qw2_linear)


def replace_linear_with_w4(model, sd: dict, group_size: int):
    """W4A16 Marlin 替换。"""
    return replace_linear_quantized(model, sd, group_size, make_qw4_linear)


def replace_linear_with_w2_deprecated(model, sd: dict, group_size: int):
    """对 model.model.layers 里的 q/k/v/o/gate/up/down 做替换。"""
    layers = model.model.layers
    targets = [
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ]
    for i, layer in enumerate(layers):
        for parent, child in targets:
            full_name = f"model.layers.{i}.{parent}.{child}"
            if f"{full_name}.weight" not in sd:
                continue
            old = getattr(getattr(layer, parent), child)
            new = make_qw2_linear(old, sd, group_size, full_name)
            setattr(getattr(layer, parent), child, new)
    return model


# ---------------------------------------------------------------------------
# quantizers -> state_dict（兼容老格式）
# ---------------------------------------------------------------------------
def convert_quantizers_to_state_dict(quantizers, asym=True):
    """
    quantizers key 形如:
      "0.self_attn.q_proj.weight"  或  "0.self_attn.q_proj"
    转为:
      model.layers.0.self_attn.q_proj.weight       (int8)
      model.layers.0.self_attn.q_proj.weight_qscale (fp32)
      model.layers.0.self_attn.q_proj.weight_qzero  (fp32)
    """
    sd = {}
    for k, q in quantizers.items():
        if not isinstance(q, dict) or "weights" not in q or "scales" not in q:
            continue
        base = k[:-len(".weight")] if k.endswith(".weight") else k
        full = f"model.layers.{base}.weight"
        sd[full] = q["weights"]
        scales = q["scales"]
        if isinstance(scales, (list, tuple)):
            if len(scales) >= 1:
                sd[f"{full}_qscale"] = scales[0]
            if asym and len(scales) >= 2:
                sd[f"{full}_qzero"] = scales[1]
        else:
            sd[f"{full}_qscale"] = scales
    return sd


# ---------------------------------------------------------------------------
# Unified Checkpoint Loading
# ---------------------------------------------------------------------------
def load_quantized_model(model_path, ckpt_path, group_size, kernel="w2",
                         asym=True, fp16_main=False):
    """
    Load model from unified checkpoint (quantized_model.pth).
    Supports both unified format and legacy formats (true_quant.pth/quantizers.pth).
    """
    print(f"[Load] checkpoint: {ckpt_path}  kernel={kernel}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Format detection: unified > legacy true_quant > legacy quantizers
    if isinstance(ckpt, dict) and "true_quant" in ckpt and ckpt["true_quant"] is not None:
        # Unified checkpoint with complete true_quant (final Stage1 or Stage2 output)
        print("  [format] unified checkpoint (quantized_model.pth)")
        sd = ckpt["true_quant"]
        meta = ckpt.get("meta", {})
        print(f"  [meta] bits={meta.get('bits', 'unknown')}, group_size={meta.get('group_size', group_size)}, sym={meta.get('sym', not asym)}")
    elif isinstance(ckpt, dict) and "quantizers" in ckpt:
        # Unified checkpoint with quantizers but true_quant is None → convert quantizers
        print("  [format] unified checkpoint (true_quant=None, converting from quantizers)")
        sd = convert_quantizers_to_state_dict(ckpt["quantizers"], asym=asym)
    elif isinstance(ckpt, dict) and any(k.endswith('_qscale') for k in ckpt.keys()):
        # Legacy true_quant.pth format (state dict with _qscale/_qzero suffixes)
        print("  [format] legacy true_quant.pth")
        sd = ckpt
    else:
        # Legacy quantizers.pth or raw quantizers dict
        print("  [format] legacy quantizers dict")
        quantizers = ckpt.get("w_quantizers", ckpt) if isinstance(ckpt, dict) else ckpt
        sd = convert_quantizers_to_state_dict(quantizers, asym=asym)

    model = _get_model(model_path)

    # 1) Load non-quantized parameters (embed/norm/lm_head/bias)
    fp_sd = {k: v for k, v in sd.items()
             if not k.endswith(".weight_qscale")
             and not k.endswith(".weight_qzero")
             and v.dtype != torch.int8}
    missing, unexpected = model.load_state_dict(fp_sd, strict=False)
    print(f"  load_state_dict (FP part): missing={len(missing)} unexpected={len(unexpected)}")

    # 2) Replace linear layers with quantized versions
    if kernel == "w2":
        replace_linear_with_w2(model, sd, group_size)
    elif kernel == "w4":
        replace_linear_with_w4(model, sd, group_size)
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    # 3) Optional: convert backbone to fp16
    if fp16_main:
        print("  [opt] convert backbone to fp16")
        model = model.to(torch.float16)

    model = model.cuda()

    # 4) Pre-warm
    if kernel == "w2":
        _prewarm_w2a16(model)
    elif kernel == "w4":
        _, _prewarm_w4a16 = _get_linear_w4a16()
        _prewarm_w4a16(model)

    return model


# ---------------------------------------------------------------------------
# CUDA kernel pre-warm (加载阶段完成 packing，避免第一次 forward 的冷启动开销)
# ---------------------------------------------------------------------------
def _prewarm_w2a16(model):
    n = 0
    for m in model.modules():
        if isinstance(m, LinearW2A16) and not m.weight_processed:
            m.weight = dQ_preprocess_weights_int2_for_weight_only(
                m.weight.to(torch.int8).cpu().contiguous()
            ).cuda()
            if m.scale is not None:
                m.scale = m.scale.to(torch.float16).contiguous()
            if m.zp is not None:
                m.zp = m.zp.to(torch.float16).contiguous()
            if m.bias is not None:
                m.bias = m.bias.to(torch.float16).contiguous()
            m.weight_processed = True
            n += 1
    print(f"  [opt] pre-warmed {n} LinearW2A16 layers")


# Legacy loading functions (kept for backward compatibility)
def load_true_quant_model(model_path, ckpt_path, group_size, kernel="w2",
                          fp16_main=False):
    """Legacy loader for true_quant.pth format."""
    return load_quantized_model(model_path, ckpt_path, group_size, kernel=kernel,
                                asym=True, fp16_main=fp16_main)


def load_quantizers_model(model_path, quantizers_path, group_size, asym=True,
                          fp16_main=False):
    print(f"[Load] quantizers: {quantizers_path}")
    model = _get_model(model_path)
    raw = torch.load(quantizers_path, map_location="cpu")
    quantizers = raw.get("w_quantizers", raw) if isinstance(raw, dict) else raw
    sd = convert_quantizers_to_state_dict(quantizers, asym=asym)
    replace_linear_with_w2(model, sd, group_size)
    if fp16_main:
        print("  [opt] convert backbone to fp16 (remove per-layer dtype cast)")
        model = model.to(torch.float16)
    model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# 文本生成（直观验证）
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_generate(model, tokenizer, prompt, max_new_tokens=128,
                    greedy=False, temperature=0.7, top_p=0.9,
                    repetition_penalty=1.2, autocast_dtype=torch.bfloat16):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature,
                          top_p=top_p, repetition_penalty=repetition_penalty)
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        out = model.generate(**enc, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("---- generated ----")
    print(text)
    print("-------------------")
    return text


# ---------------------------------------------------------------------------
# PPL
# ---------------------------------------------------------------------------
def _load_test_enc(dataset, tokenizer_path, seqlen):
    """只 tokenize 测试集，避免 datautils.get_wikitext2 用 slow tokenizer
    去编码巨大的 train split（会卡死）。"""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if dataset == "wikitext2":
        td = load_dataset("/data1/xx/xxquant/datasets/wikitext",
                          "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(td["text"])
    elif dataset == "ptb":
        td = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n\n".join(td["sentence"])
    elif dataset == "c4":
        td = load_dataset(
            "allenai/c4", "en",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = " ".join(td["text"][:1100])
    else:
        # 兜底走原 get_loaders（有可能慢）
        _, testloader = get_loaders_legacy(dataset, nsamples=128, seed=0,
                                    model=tokenizer_path, seqlen=seqlen)
        return testloader.input_ids if hasattr(testloader, "input_ids") else testloader
    enc = tok(text, return_tensors="pt")
    return enc.input_ids


@torch.no_grad()
def eval_ppl(model, tokenizer_path, datasets, seqlen=2048, nsamples_cal=128,
             autocast_dtype=torch.bfloat16):
    results = {}
    use_cache = model.config.use_cache
    model.config.use_cache = False
    device = next(model.parameters()).device

    for ds in datasets:
        print(f"[PPL] dataset = {ds} (tokenizing test split ...)")
        testenc = _load_test_enc(ds, tokenizer_path, seqlen)

        nsamples = testenc.numel() // seqlen
        nlls = []
        for i in range(nsamples):
            batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            nlls.append(loss * seqlen)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
        print(f"  PPL[{ds}] = {ppl:.4f}")
        results[ds] = ppl

    model.config.use_cache = use_cache
    return results


# ---------------------------------------------------------------------------
# Zero-shot via lm-eval
# ---------------------------------------------------------------------------
def eval_lm_eval(model, tokenizer, tasks, batch_size=16):
    """兼容 lm-eval 0.3.0 与 0.4.x"""
    out = {}
    lm = None

    # ---------- 0.4.x ----------
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        evaluate_fn = lm_eval.simple_evaluate
        for t in tasks:
            print(f"[zero-shot] {t} ...")
            r = evaluate_fn(lm, tasks=[t], batch_size=batch_size)["results"][t]
            acc = round(r.get("acc_norm,none", r.get("acc,none", 0.0)) * 100, 2)
            print(f"  {t} acc = {acc}%")
            out[t] = acc
    except Exception as e0:
        # ---------- 0.3.0 ----------
        try:
            from lm_eval.models.hf_causal import HFCausalLM
            from lm_eval.evaluator import simple_evaluate
            lm = HFCausalLM(
                pretrained=model, tokenizer=tokenizer, batch_size=batch_size
            )
            for t in tasks:
                print(f"[zero-shot] {t} ...")
                r = simple_evaluate(lm, tasks=[t], batch_size=batch_size)["results"][t]
                acc = round(r.get("acc_norm", r.get("acc", 0.0)) * 100, 2)
                print(f"  {t} acc = {acc}%")
                out[t] = acc
        except Exception as e1:
            print(f"WARNING: lm_eval evaluation failed (0.4.x: {e0}, 0.3.0: {e1}), skip.")

    if out:
        out["acc_avg"] = round(sum(out.values()) / len(out), 2)
        print(f"[zero-shot] avg = {out['acc_avg']}%")
    return out


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------
@torch.inference_mode()
def benchmark_speed_legacy(model, tokenizer, prompt, max_length=40,
                           warmup=2, repeats=5, autocast_dtype=torch.bfloat16):
    """复刻旧 inference_test.py 的 benchmark：用 max_length（含 prompt）而非
    max_new_tokens；强制 bf16 autocast；不启用 static cache。便于和历史数字对齐。
    """
    model.eval()
    input_ids = tokenizer([prompt])["input_ids"]
    input_tensor = torch.LongTensor(input_ids).cuda()
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        for _ in range(warmup):
            _ = model.generate(input_tensor, max_length=max_length)
        torch.cuda.synchronize()
        times, out = [], None
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(input_tensor, max_length=max_length)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)
    e2e_ms = float(np.mean(times))
    per_tok = e2e_ms / out.shape[-1]
    print(f"  [legacy] e2e {e2e_ms:.2f} ms, total_len {out.shape[-1]}, "
          f"per-token {per_tok:.2f} ms (avg over {repeats})")
    return {"e2e_ms": e2e_ms, "per_token_ms": per_tok}


@torch.inference_mode()
def benchmark_speed(model, tokenizer, prompt, max_new_tokens=64,
                    warmup=2, repeats=5, autocast_dtype=torch.bfloat16,
                    use_autocast=True):
    model.eval()
    enc = tokenizer([prompt], return_tensors="pt").to("cuda")

    # 启用 static KV-cache
    try:
        model.generation_config.cache_implementation = "static"
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    def _ctx():
        if use_autocast:
            return torch.cuda.amp.autocast(dtype=autocast_dtype)
        import contextlib
        return contextlib.nullcontext()

    # 显存监控
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with _ctx():
        for _ in range(warmup):
            _ = model.generate(**enc, **gen_kwargs)
        torch.cuda.synchronize()
        times, out = [], None
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model.generate(**enc, **gen_kwargs)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)

    e2e_ms = float(np.mean(times))
    n_tokens = out.shape[-1] - enc.input_ids.shape[-1]
    per_tok = e2e_ms / max(n_tokens, 1)

    # 计算峰值显存（GB）
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"  e2e {e2e_ms:.2f} ms, {n_tokens} new tokens, "
          f"per-token {per_tok:.2f} ms, peak_mem {peak_mem:.2f} GB (avg over {repeats})")
    return {"e2e_ms": e2e_ms, "per_token_ms": per_tok, "new_tokens": n_tokens, "peak_mem_gb": peak_mem}


@torch.inference_mode()
def benchmark_prefill(model, tokenizer, seq_lens=(512, 1024, 2048),
                      warmup=2, repeats=5, autocast_dtype=torch.bfloat16,
                      use_autocast=True):
    """长 prompt 一次 forward 的速度——这是 W2A16 mixed-gemm 真正擅长的场景。"""
    model.eval()
    base = "Lorem ipsum dolor sit amet consectetur adipiscing elit " * 4096
    results = {}

    def _ctx():
        if use_autocast:
            return torch.cuda.amp.autocast(dtype=autocast_dtype)
        import contextlib
        return contextlib.nullcontext()

    for L in seq_lens:
        enc = tokenizer(base, return_tensors="pt", truncation=True,
                        max_length=L).to("cuda")
        if enc.input_ids.shape[1] < L:
            print(f"  skip L={L}: tokenizer 裁出来不够长")
            continue
        with _ctx():
            for _ in range(warmup):
                _ = model(**enc)
            torch.cuda.synchronize()
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                t0 = time.time()
                _ = model(**enc)
                torch.cuda.synchronize()
                ts.append((time.time() - t0) * 1000.0)
        ms = float(np.mean(ts))
        print(f"  prefill L={L:>5d}: {ms:.2f} ms ({ms/L:.3f} ms/tok)")
        results[L] = ms
    return results


@torch.inference_mode()
def benchmark_throughput(model, tokenizer, prompt, batch_sizes=(1, 4, 8, 16, 32),
                         seq_len=512, warmup=2, repeats=5,
                         autocast_dtype=torch.bfloat16, use_autocast=True):
    """Batch forward throughput — measures tokens/sec at different batch sizes."""
    model.eval()
    base = "Lorem ipsum dolor sit amet consectetur adipiscing elit " * 4096
    enc_single = tokenizer(base, return_tensors="pt", truncation=True,
                           max_length=seq_len).to("cuda")
    results = {}

    def _ctx():
        if use_autocast:
            return torch.cuda.amp.autocast(dtype=autocast_dtype)
        import contextlib
        return contextlib.nullcontext()

    for bs in batch_sizes:
        ids = enc_single.input_ids.repeat(bs, 1)  # [bs, seq_len]
        try:
            with _ctx():
                for _ in range(warmup):
                    _ = model(ids)
                torch.cuda.synchronize()
                ts = []
                for _ in range(repeats):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _ = model(ids)
                    torch.cuda.synchronize()
                    ts.append((time.time() - t0) * 1000.0)
            ms = float(np.mean(ts))
            total_tokens = bs * seq_len
            tps = total_tokens / (ms / 1000.0)
            print(f"  batch={bs:>3d}, seq={seq_len}: {ms:.2f} ms, "
                  f"{tps:.0f} tok/s")
            results[bs] = {"ms": ms, "tok_per_sec": tps}
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={bs:>3d}: OOM, skipping larger batches")
                torch.cuda.empty_cache()
                break
            raise
    return results


def maybe_compile(model, mode="reduce-overhead"):
    """对 model.forward 做 torch.compile；失败时 fallback。"""
    try:
        print(f"  [opt] torch.compile(mode={mode})")
        model.forward = torch.compile(model.forward, mode=mode, dynamic=False, fullgraph=False)
    except Exception as e:
        print(f"  [warn] torch.compile failed: {e}")
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LGRQuant 真量化推理测试")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--true_quant_path", type=str, default=None)
    parser.add_argument("--quantizers_path", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--asym", action="store_true")
    parser.add_argument("--kernel", choices=["w2", "w4"], default="w2",
                        help="推理内核: w2=LinearW2A16, w4=LinearW4A16(Marlin)")

    # 文本生成
    parser.add_argument("--prompt", type=str, default="who are you?")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--greedy", action="store_true",
                        help="贪心解码（W2 下容易循环，仅用于验证 dtype）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    # 评测开关
    parser.add_argument("--ppl_datasets", type=str, default="wikitext2",
                        help="逗号分隔; 留空则跳过")
    parser.add_argument("--lm_eval_tasks", type=str, default="",
                        help="逗号分隔; 留空则跳过")
    parser.add_argument("--lm_eval_batch_size", type=int, default=16)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_repeats", type=int, default=5)
    parser.add_argument("--benchmark_new_tokens", type=int, default=64)
    parser.add_argument("--benchmark_legacy", action="store_true",
                        help="用旧版口径（max_length=40 + bf16 autocast + "
                             "无 static cache）复现历史数字")
    parser.add_argument("--benchmark_prefill", action="store_true",
                        help="额外测 prefill (长 prompt forward) 加速比")
    parser.add_argument("--prefill_lens", type=str, default="512,1024,2048")
    parser.add_argument("--benchmark_throughput", action="store_true",
                        help="测不同 batch size 下的 forward throughput (tok/s)")
    parser.add_argument("--throughput_batch_sizes", type=str, default="1,4,8,16,32",
                        help="逗号分隔的 batch size 列表")
    parser.add_argument("--throughput_seq_len", type=int, default=512,
                        help="throughput 测试的序列长度")
    parser.add_argument("--compile", action="store_true",
                        help="对 model.forward 做 torch.compile，配合 static "
                             "KV-cache 砍 launch overhead，对 batch=1 解码"
                             "通常有 20%~40% 收益（首次会编译 ~1 min）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16_main", action="store_true",
                        help="把主干（embed/norm/lm_head）也转 fp16，"
                             "并让 autocast 全程跑 fp16，避免每个 LinearW2A16 "
                             "进/出做 bf16<->fp16 cast")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    autocast_dtype = torch.float16 if args.fp16_main else torch.bfloat16
    # 主干已 fp16 时关闭 autocast，避免每 op 入口的 dtype 查询/转换开销
    use_autocast = not args.fp16_main

    # 1) 加载模型
    ckpt_path = args.true_quant_path or args.quantizers_path
    if not ckpt_path:
        raise ValueError("必须指定 --true_quant_path 或 --quantizers_path")

    model = load_quantized_model(
        args.model_path, ckpt_path, args.group_size,
        kernel=args.kernel, asym=args.asym,
        fp16_main=args.fp16_main
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2) 生成 sample
    print("\n======== Generation Sample ========")
    sample_generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        autocast_dtype=autocast_dtype,
    )

    # 3) PPL
    ppl_ds = [d.strip() for d in args.ppl_datasets.split(",") if d.strip()]
    if ppl_ds:
        print("\n======== PPL Evaluation ========")
        eval_ppl(model, args.model_path, ppl_ds,
                 autocast_dtype=autocast_dtype)

    # 4) Zero-shot
    tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
    if tasks:
        print("\n======== Zero-shot Evaluation ========")
        eval_lm_eval(model, tokenizer, tasks,
                     batch_size=args.lm_eval_batch_size)

    # 5) Speed benchmark（W2A16 vs FP）
    if args.benchmark:
        if args.compile:
            maybe_compile(model)
        kernel_label = "W4A16" if args.kernel == "w4" else "W2A16"
        print(f"\n======== Speed: {kernel_label} ========")
        q_speed = benchmark_speed(
            model, tokenizer, args.prompt,
            max_new_tokens=args.benchmark_new_tokens,
            repeats=args.benchmark_repeats,
            autocast_dtype=autocast_dtype,
            use_autocast=use_autocast,
        )
        q_legacy = None
        if args.benchmark_legacy:
            print(f"\n-- {kernel_label} legacy (max_length=40 + bf16 autocast) --")
            q_legacy = benchmark_speed_legacy(
                model, tokenizer, args.prompt, max_length=40,
                repeats=args.benchmark_repeats, autocast_dtype=torch.bfloat16,
            )
        q_prefill = None
        if args.benchmark_prefill:
            print(f"\n-- {kernel_label} prefill --")
            seq_lens = tuple(int(x) for x in args.prefill_lens.split(",") if x.strip())
            q_prefill = benchmark_prefill(
                model, tokenizer, seq_lens=seq_lens,
                repeats=args.benchmark_repeats,
                autocast_dtype=autocast_dtype, use_autocast=use_autocast,
            )
        q_throughput = None
        if args.benchmark_throughput:
            print(f"\n-- {kernel_label} throughput --")
            bs_list = tuple(int(x) for x in args.throughput_batch_sizes.split(",") if x.strip())
            q_throughput = benchmark_throughput(
                model, tokenizer, args.prompt, batch_sizes=bs_list,
                seq_len=args.throughput_seq_len,
                repeats=args.benchmark_repeats,
                autocast_dtype=autocast_dtype, use_autocast=use_autocast,
            )
        del model; gc.collect(); torch.cuda.empty_cache()

        print("\n======== Speed: FP baseline ========")
        fp_model = _get_model(args.model_path)
        if args.fp16_main:
            fp_model = fp_model.to(torch.float16)
        fp_model = fp_model.cuda()
        if args.compile:
            maybe_compile(fp_model)
        fp_speed = benchmark_speed(
            fp_model, tokenizer, args.prompt,
            max_new_tokens=args.benchmark_new_tokens,
            repeats=args.benchmark_repeats,
            autocast_dtype=autocast_dtype,
            use_autocast=use_autocast,
        )
        fp_prefill = None
        if args.benchmark_prefill:
            print("\n-- FP prefill --")
            seq_lens = tuple(int(x) for x in args.prefill_lens.split(",") if x.strip())
            fp_prefill = benchmark_prefill(
                fp_model, tokenizer, seq_lens=seq_lens,
                repeats=args.benchmark_repeats,
                autocast_dtype=autocast_dtype, use_autocast=use_autocast,
            )
        fp_throughput = None
        if args.benchmark_throughput:
            print("\n-- FP throughput --")
            bs_list = tuple(int(x) for x in args.throughput_batch_sizes.split(",") if x.strip())
            fp_throughput = benchmark_throughput(
                fp_model, tokenizer, args.prompt, batch_sizes=bs_list,
                seq_len=args.throughput_seq_len,
                repeats=args.benchmark_repeats,
                autocast_dtype=autocast_dtype, use_autocast=use_autocast,
            )
        fp_legacy = None
        if args.benchmark_legacy:
            print("\n-- FP legacy (max_length=40 + bf16 autocast) --")
            fp_legacy = benchmark_speed_legacy(
                fp_model, tokenizer, args.prompt, max_length=40,
                repeats=args.benchmark_repeats, autocast_dtype=torch.bfloat16,
            )

        print("\n======== Speedup Summary ========")
        print(f"  FP    : e2e {fp_speed['e2e_ms']:.2f} ms, "
              f"per-token {fp_speed['per_token_ms']:.2f} ms, "
              f"peak_mem {fp_speed['peak_mem_gb']:.2f} GB")
        print(f"  {kernel_label} : e2e {q_speed['e2e_ms']:.2f} ms, "
              f"per-token {q_speed['per_token_ms']:.2f} ms, "
              f"peak_mem {q_speed['peak_mem_gb']:.2f} GB")
        print(f"  Speedup (e2e)   = {fp_speed['e2e_ms'] / q_speed['e2e_ms']:.2f}x")
        print(f"  Speedup (token) = "
              f"{fp_speed['per_token_ms'] / q_speed['per_token_ms']:.2f}x")
        mem_ratio = q_speed['peak_mem_gb'] / fp_speed['peak_mem_gb']
        print(f"  Memory ratio    = {mem_ratio:.2f}x ({kernel_label}/FP, lower is better)")

        if q_prefill and fp_prefill:
            print("\n  -- Prefill Speedup --")
            for L in sorted(set(q_prefill) & set(fp_prefill)):
                ratio = fp_prefill[L] / q_prefill[L]
                print(f"    L={L:>5d}: FP {fp_prefill[L]:.2f} ms vs "
                      f"{kernel_label} {q_prefill[L]:.2f} ms  => {ratio:.2f}x")

        if q_throughput and fp_throughput:
            print(f"\n  -- Throughput Speedup (seq_len={args.throughput_seq_len}) --")
            for bs in sorted(set(q_throughput) & set(fp_throughput)):
                q_tps = q_throughput[bs]['tok_per_sec']
                fp_tps = fp_throughput[bs]['tok_per_sec']
                ratio = q_tps / fp_tps
                print(f"    batch={bs:>3d}: FP {fp_tps:.0f} tok/s vs "
                      f"{kernel_label} {q_tps:.0f} tok/s  => {ratio:.2f}x")


if __name__ == "__main__":
    main()
