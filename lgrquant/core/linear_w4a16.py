"""Marlin-backed W4A16 Linear (sym only, group_size in {128, -1}).

与 linear_w2a16.py 1:1 对应：
  - make_qw4_linear() 在 inference_test.py 的加载流程中装配 weight/scale/bias
  - _prewarm_w4a16() 在加载阶段一次性做 int8 -> marlin packed 转换
  - forward() 直接调 marlin.mul，输入/输出统一 fp16
"""
import torch
import torch.nn as nn

# Marlin 需要从项目内的 marlin_ist 目录导入
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'marlin_ist'))
import marlin


class LinearW4A16(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
        super().__init__()
        assert group_size in (128, -1), (
            f"Marlin 只支持 group_size ∈ {{128, -1}}, got {group_size}"
        )
        self.k = in_features
        self.n = out_features
        self.with_bias = bias is not None
        self.group_size = group_size

        # 以下字段在 make_qw4_linear / _prewarm_w4a16 中被 setattr
        self.weight = None          # int8 [in, out] (来自 ckpt，转置后)
        self.qweight = None         # marlin packed int4
        self.scales = None          # fp16 [groups, out]
        self.workspace = None       # marlin GEMM staging buffer
        self.bias = None            # fp16 [out]
        self.weight_processed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_processed, "LinearW4A16: weights not packed yet"
        orig_dtype = x.dtype
        x_fp16 = x if x.dtype == torch.float16 else x.to(torch.float16)

        # marlin.mul 只接受 2D (m, k)，多维 batch 需要 view 展平
        out_shape = x_fp16.shape[:-1] + (self.n,)
        out = torch.empty(out_shape, dtype=torch.float16, device=x_fp16.device)
        marlin.mul(x_fp16.reshape(-1, x_fp16.shape[-1]),
                   self.qweight,
                   out.reshape(-1, self.n),
                   self.scales, self.workspace)

        if self.bias is not None:
            out = out + self.bias

        return out if orig_dtype == torch.float16 else out.to(orig_dtype)


def convert_int8_to_marlin(int8_w: torch.Tensor, fp32_scale: torch.Tensor,
                           group_size: int):
    """把 sym int8 ckpt 权重转换为 marlin packed 格式。

    Args:
        int8_w:      [in_features, out_features] int8 (来自 ckpt weight.t())
        fp32_scale:  [groups, out_features] fp32 (来自 ckpt qscale.t())
        group_size:  128 或 -1

    Returns:
        (qweight, scales, workspace) 可直接赋给 LinearW4A16
    """
    k, n = int8_w.shape
    gs = group_size if group_size > 0 else k

    # 数值稳定性：per-channel (groupsize=-1) 时 Marlin 在 write_result
    # 中先做 fp32→fp16 转换再乘 scale，大 K 维度下未缩放累加值可超
    # fp16 上限 (65504) 产生 Inf。这里将 per-channel scale 复制到每
    # 128 个元素一组，走 grouped kernel 路径（scale 在 matmul 内应用）。
    if group_size <= 0 and k >= 128 and k % 128 == 0:
        gs = 128
        marlin_gs = 128
        # fp32_scale: [1, n] → [k//128, n]
        fp32_scale = fp32_scale.repeat(k // 128, 1)
    else:
        marlin_gs = group_size if group_size > 0 else -1

    assert k % gs == 0, f"k={k} 不能被 group_size={gs} 整除"

    # 确保都在 CUDA 上
    int8_w = int8_w.cuda()
    fp32_scale = fp32_scale.cuda()

    # 1) 反量化为 fp16 dense weight（marlin.pack 需要输入 fp16 Linear）
    # scale: [groups, n], 扩展到 [k, n]
    scale_expanded = fp32_scale.to(torch.float16).repeat_interleave(gs, dim=0)
    w_fp16 = int8_w.to(torch.float16) * scale_expanded  # [k, n]

    # 2) 构造临时 nn.Linear（weight 布局 [out, in]）
    temp_linear = nn.Linear(k, n, bias=False)
    temp_linear.weight.data = w_fp16.t().contiguous()  # [n, k] fp16
    temp_linear = temp_linear.half().cuda()

    # 3) marlin pack
    # pack(linear, scales) 期望 scales 形状 (n, groups) = (n, k//gs)
    # （marlin 内部做 s = scales.t() 变成 (groups, n) 再 broadcast）
    # fp32_scale 是 [groups, n]，需要转置为 [n, groups]
    scales_fp16 = fp32_scale.to(torch.float16).t().contiguous().cuda()  # [n, groups]

    mlayer = marlin.Layer(k, n, groupsize=marlin_gs).cuda()
    mlayer.pack(temp_linear, scales_fp16)

    return mlayer.B, mlayer.s, mlayer.workspace


def _prewarm_w4a16(model):
    """对所有 LinearW4A16 做一次性 marlin packing。"""
    n = 0
    for m in model.modules():
        if isinstance(m, LinearW4A16) and not m.weight_processed:
            assert m.weight is not None and m.scales is not None
            m.qweight, m.scales, m.workspace = convert_int8_to_marlin(
                m.weight, m.scales, m.group_size
            )
            m.weight = None  # 释放原始 int8 权重，节省内存
            if m.bias is not None:
                m.bias = m.bias.to(torch.float16).contiguous()
            m.weight_processed = True
            n += 1
    print(f"  [opt] pre-warmed {n} LinearW4A16 layers")
