"""LinearW4A16 单元测试：对比 marlin GEMM vs 手写 dequant + matmul。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from decoupleQ.linear_w4a16 import LinearW4A16, convert_int8_to_marlin


def test_linear_w4a16_cosine():
    """测试 LinearW4A16 输出与参考实现的 cosine similarity。"""
    k, n = 4096, 11008  # Qwen MLP up_proj 尺寸
    gs = 128
    batch = 4
    DEV = torch.device('cuda:0')

    # 1) 构造随机 sym int8 权重 + fp32 scale
    # 模拟 stage1 产出的 int8 weight [out, in] -> 转置后 [in, out]
    int8_w_out_in = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=DEV)
    int8_w = int8_w_out_in.t().contiguous()  # [k, n] - LinearW4A16 期望的布局

    # scale [groups, out] fp32
    n_groups = k // gs
    scale = torch.rand(n_groups, n, dtype=torch.float32, device=DEV) * 0.01 + 0.001

    # 2) 用 convert_int8_to_marlin 得到 packed 权重
    qw, s, ws = convert_int8_to_marlin(int8_w, scale, gs)
    print(f"packed weight shape: {qw.shape}, scales shape: {s.shape}")

    # 3) 装配 LinearW4A16
    layer = LinearW4A16(k, n, bias=False, group_size=gs)
    layer.weight = int8_w  # 原始 int8，用于参考计算
    layer.qweight = qw
    layer.scales = s
    layer.workspace = ws
    layer.weight_processed = True

    # 4) 输入 fp16
    x = torch.randn(batch, k, dtype=torch.float16, device=DEV)

    # 5) Marlin 输出
    out_marlin = layer(x)

    # 6) 参考输出：dequant + matmul
    # dequant: int8 * scale (sym，无 zp)
    scale_expanded = scale.to(torch.float16).repeat_interleave(gs, dim=0)  # [k, n]
    w_fp16 = int8_w.to(torch.float16) * scale_expanded
    out_ref = torch.matmul(x, w_fp16)

    # 7) 校验
    cosine = torch.nn.functional.cosine_similarity(
        out_marlin.flatten(), out_ref.flatten(), dim=0
    ).item()
    max_diff = (out_marlin - out_ref).abs().max().item()
    rel_err = torch.mean(torch.abs(out_marlin - out_ref)) / torch.mean(torch.abs(out_ref))

    print(f"cosine={cosine:.6f}  max_diff={max_diff:.4f}  rel_err={rel_err.item():.6f}")

    assert cosine > 0.999, f"cosine too low: {cosine}"
    assert max_diff < 0.5, f"max_diff too high: {max_diff}"
    assert rel_err < 0.001, f"rel_err too high: {rel_err.item()}"
    print("PASSED")


if __name__ == "__main__":
    test_linear_w4a16_cosine()
