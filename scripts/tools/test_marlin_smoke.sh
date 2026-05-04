"""验证 marlin 安装 + 探测 API 签名。"""
import torch
import sys
sys.path.insert(0, '/data1/xx/LGRQuant/marlin_ist')
import marlin

def test_marlin_basic():
    print("marlin version:", getattr(marlin, "__version__", "unknown"))

    k, n = 4096, 4096
    gs = 128
    DEV = torch.device('cuda:0')

    # 参考 marlin/test.py 的用法
    tile = 16
    maxq = 2 ** 4 - 1

    # 1. 生成量化权重 (symmetric int4)
    w = torch.randn((k, n), dtype=torch.half, device=DEV)
    # reshape for group-wise quantization
    w_reshaped = w.reshape((-1, gs, n)).permute(1, 0, 2).reshape((gs, -1))
    s = torch.max(torch.abs(w_reshaped), 0, keepdim=True)[0]
    s *= 2 / maxq
    w_int = torch.round(w_reshaped / s).int()
    w_int += (maxq + 1) // 2
    w_int = torch.clamp(w_int, 0, maxq)
    # dequantized reference
    w_dequant = (w_int - (maxq + 1) // 2).half() * s
    # reshape back
    w_dequant = w_dequant.reshape((gs, -1, n)).permute(1, 0, 2).reshape((k, n)).contiguous()
    s = s.reshape((-1, n)).contiguous()

    # 2. 创建 marlin Layer
    layer = marlin.Layer(k, n, groupsize=gs).cuda()
    layer.k = k
    layer.n = n
    layer.groupsize = gs
    layer.B = torch.empty((k // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((k // gs, n), dtype=torch.half, device=DEV)

    # 3. 创建 dummy linear 用于 pack
    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data = w_dequant.t().contiguous()

    # 4. Pack (scales 需要转置)
    layer.pack(linear, s.t())
    print("pack() OK")

    # 5. 测试 mul
    m = 2  # batch size
    A = torch.randn((m, k), dtype=torch.half, device=DEV)
    C = torch.zeros((m, n), dtype=torch.half, device=DEV)

    # workspace size: n // 128 * 16 for thread_k=64, or n // 128 * 16 for thread_k=128
    workspace = torch.zeros(n // 128 * 16, device=DEV)

    # mul(A, B, C, s, workspace, thread_k, thread_n, groupsize)
    marlin.mul(A, layer.B, C, layer.s, workspace, 64, 256, -1)
    torch.cuda.synchronize()
    print("mul() OK, out shape:", C.shape)

    # 6. 数值校验
    C_ref = torch.matmul(A, w_dequant)
    diff = (C - C_ref).abs().max().item()
    print(f"max diff vs fp16 matmul: {diff}")

    # 使用相对误差
    rel_err = torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref))
    print(f"relative error: {rel_err.item():.6f}")

    assert rel_err < 0.001, f"marlin GEMM accuracy too poor: rel_err={rel_err.item()}"
    print("SMOKE PASSED")

if __name__ == "__main__":
    test_marlin_basic()
