"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""
import torch
from torch import nn 

from .decoupleQ_kernels import dQ_preprocess_weights_int2_for_weight_only, dQ_asymm_qw2_gemm


cnt = 0

class LinearW2A16(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool, group_size: int):
        super().__init__()
        self.k = in_features
        self.n = out_features
        self.with_bias = bias is not None
        self.weight = None
        self.bias = torch.zeros((out_features), dtype=torch.float16).cuda()
        self.scale = None
        self.zp = None
        self.group_size = group_size
        self.weight_processed = False        

    def forward(self, input: torch.Tensor):
        if not self.weight_processed:
            assert self.weight != None, "LinearW2A16.forward: need assign weight first"
            if self.with_bias:
                assert self.bias != None, "LinearW2A16.forward: need assign bias if use_bias"
            self.weight = dQ_preprocess_weights_int2_for_weight_only(
                self.weight.to(torch.int8).cpu().contiguous()
            ).cuda()
            assert self.scale != None, "LinearW2A16.forward: need scale"
            self.scale = self.scale.to(torch.float16).contiguous()
            if self.zp is not None:
                self.zp = self.zp.to(torch.float16).contiguous()
            if self.bias is not None:
                self.bias = self.bias.to(torch.float16).contiguous()
            self.weight_processed = True

        # 快路径：input 已 fp16 直接调 kernel；仅 bf16/fp32 时才 cast
        if input.dtype is torch.float16:
            return dQ_asymm_qw2_gemm(
                input, self.weight, self.scale, self.zp, self.bias, self.group_size
            )
        orig_dtype = input.dtype
        out = dQ_asymm_qw2_gemm(
            input.to(torch.float16), self.weight,
            self.scale, self.zp, self.bias, self.group_size,
        )
        return out.to(orig_dtype)

class LinearA16(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool, group_size: int):
        super().__init__()
        self.k = in_features
        self.n = out_features
        self.with_bias = bias is not None
        self.weight = None
        self.bias = torch.zeros((out_features), dtype=torch.bfloat16).cuda()
        self.scale = None
        self.zp = None
        self.group_size = group_size
        self.weight_processed = False        

    def forward(self, input: torch.Tensor):
        if not self.weight_processed:
            w = self.weight.cuda()
            # 若 weight 是量化后的整型 (int8/uint8)，按 (q * scale + zp) 反量化为 fp16
            # 经过 make_qw2_linear 中的 .t() 后，weight 形状为 [k, n]，分组沿 dim=0 (k 轴)
            if w.dtype in (torch.int8, torch.uint8, torch.int32) and self.scale is not None:
                k, n = w.shape
                gs = self.group_size if self.group_size and self.group_size > 0 else k
                assert k % gs == 0, f"k={k} 不能被 group_size={gs} 整除"
                scale = self.scale.to(torch.float16)   # [k/gs, n]
                zp = self.zp.to(torch.float16) if self.zp is not None else None
                w_f = w.to(torch.float16).view(k // gs, gs, n)
                w_f = w_f * scale.unsqueeze(1)
                if zp is not None:
                    w_f = w_f + zp.unsqueeze(1)
                w = w_f.view(k, n).contiguous()
            else:
                w = w.to(torch.float16)
            self.weight = w
            self.weight_processed = True

        output = torch.matmul(input.to(self.weight.dtype), self.weight)
        if self.with_bias and self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


    
