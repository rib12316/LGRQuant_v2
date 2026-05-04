"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
"""
import torch
import torch.nn as nn
from lgrquant.core.moq_quant import Quantizer, repeat_interleave, opt_intW3
import torch.nn.functional as F


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=''):
    if isinstance(module, layers) or module.__class__.__name__ in ("Conv1D",):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        return obj
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = to_device(v, device)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for v in obj:
            new_obj.append(to_device(v, device))
        if isinstance(obj, tuple):
            new_obj = tuple(new_obj)
        return new_obj
    elif isinstance(obj, nn.Module):
        obj = obj.to(device)
        return obj
    return obj


def fp16tofloat(obj):
    if isinstance(obj, torch.Tensor) and obj.dtype in (torch.bfloat16, torch.float16):
        obj = obj.float()
        return obj
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = fp16tofloat(v)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for v in obj:
            new_obj.append(fp16tofloat(v))
        if isinstance(obj, tuple):
            new_obj = tuple(new_obj)
        return new_obj
    elif isinstance(obj, nn.Module):
        for n in list(obj.parameters()) + list(obj.buffers()):
            if n.dtype in (torch.bfloat16, torch.float16):
                n.data = n.data.float()
        return obj
    return obj


class decoupleQ(object):
    def __init__(self, layer, name=''):
        self.layer = layer
        W = layer.weight.data
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif self.layer.__class__.__name__ in ("Conv1D",) and W.ndim == 2:
            W = W.t()
        elif isinstance(self.layer, nn.Linear):
            pass
        else:
            raise NotImplementedError("not support yet")
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # self.H = torch.zeros((self.columns, self.columns), dtype=torch.float, device=W.device)
        self.H = 0
        self.nsamples = 0

    def add_batch(self, inp, out, mask):
        if inp.isnan().any():
            print(f"catch a NAN!!!!!!")
            return
        mask = mask.to(inp.dtype) if mask is not None else None
        if isinstance(self.layer, nn.Linear) or self.layer.__class__.__name__ in ("Conv1D",):
            inp = inp.reshape((-1, inp.shape[-1]))  # [batch, dim]
            if mask is not None:
                mask = mask.reshape((-1, 1))  # [batch, dim]
            inp = inp * mask if mask is not None else inp
            inp = inp.t()  # [dim, batch]

        elif isinstance(self.layer, nn.Conv2d):  # [batch, channel, hight, width]
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            raise NotImplementedError("not support yet")
        # tmp = inp.shape[-1]
        tmp = mask.sum().double() if mask is not None else inp.shape[-1]
        inp = inp.double()
        h = inp.matmul(inp.t())
        self.H = (self.H * self.nsamples + h) / (self.nsamples + tmp)
        self.nsamples += tmp

    def startquant(self, groupsize, symmetric, max_iter_num,
                   inner_iters_for_round, iters_before_round, dev, lr, actorder=True,
                   round_fn="gptq", perdamp=0.01):
        W = self.layer.weight.data.clone().detach()  # [out_channel, in_channel, kernel_size, kernel_size]
        W = W.to(dev).float()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif self.layer.__class__.__name__ in ("Conv1D", ) and W.ndim == 2:
            W = W.t()
        elif isinstance(self.layer, nn.Linear):
            pass
        else:
            raise NotImplementedError

        trt = f"The number of samples for GPTQ must be larger than the dimension of Hession matrix;" \
              f"Otherwise, H must be a singular matrix and cannot be inverted. nsample {self.nsamples}, columns {self.columns}"
        # assert self.nsamples > self.columns, trt
        print(trt)
        H = self.H.to(device=dev, dtype=W.dtype)
        del self.H
        dp = torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += perdamp * dp

        H00 = H
        W00 = W
        # ===================================================================================================
        iters_for_scale = self.quantizer.iters_for_scale
        self.quantizer.iters_for_scale = 0
        scale, zero, scale_out, zero_out, err = self.quantizer.find_params(W, groupsize=groupsize, H=H)
        print(f"get scale via minmax, the init loss is {err.mean().item()}")
        w_int = self.quantizer.get_fake_int_in(W00, scale, zero, groupsize=groupsize)
        del scale, zero
        if max_iter_num >= 1:
            # max_iter_num = 1 is GPTQ
            s0, z0 = repeat_interleave(W, groupsize, scale_out, zero_out)
            h00, w00 = H00.clone(), W00.clone()
            perm, invperm = None, None
            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                invperm = torch.argsort(perm)
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]
            w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, self.quantizer.min_bound,
                                        self.quantizer.max_bound, w00, round_fn="gptq")
            if actorder:
                w_int0 = w_int0[:, invperm]
            tmp = err0 < err
            w_int[tmp] = w_int0[tmp]
            err[tmp] = err0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(f"the success rate of gptq is {rate.item()}, the loss is {err.mean().item()}")

        torch.cuda.empty_cache()
        max_inner_iter = inner_iters_for_round
        iter_num = -1
        for iter_num in range(max_iter_num - 1):
            if iter_num == 0:
                self.quantizer.iters_for_scale = max(iters_for_scale, 2)
                _, _, scale_out0, zero_out0, err0 = self.quantizer.find_params(W, groupsize=groupsize, H=H)
            else:
                scale_out0, zero_out0 = scale_out, zero_out
            s0, z0 = repeat_interleave(W, groupsize, scale_out0, zero_out0)
            h00, w00 = H00.clone(), W00.clone()
            if actorder:
                h00 = h00[perm][:, perm]
                s0, z0 = s0[:, perm], z0[:, perm]
                w00 = w00[:, perm]
            w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, self.quantizer.min_bound, self.quantizer.max_bound, w00,
                                        x_init=None, max_iter=iters_before_round, lr=lr, max_inner_iter=max_inner_iter,
                                        round_fn=round_fn)
            if actorder:
                w_int0 = w_int0[:, invperm]
            if (err0 < 0).any():
                eig = torch.linalg.eigh(H00)
                print("The eigenvalues is", eig.eigenvalues)
                print("The err is ", err0)
                print("The negative err is ", err0[err0 < 0])
                # from IPython import embed; embed(header="negative loss 0")
                # raise ValueError(f"Fatal error, the eigenvalues of hessian is {eig.eigenvalues}")
            tmp = err0 < err
            w_int[tmp] = w_int0[tmp]
            scale_out[tmp] = scale_out0[tmp]
            zero_out[tmp] = zero_out0[tmp]
            err[tmp] = err0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(f"Iter {iter_num}, the success rate of opt_intW is {rate.item()}, the loss is {err.mean().item()}")
            if rate < 1e-4:
                break
            del w_int0, err0, h00, w00, s0, z0, tmp, rate

            try:
                scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                    H=H00, groupsize=groupsize, x0=W00, x_int=w_int)
                if (err0 <= 0).any():
                    eig = torch.linalg.eigh(H00)
                    print(eig.eigenvalues)
                    raise ValueError(f"Fatal error, the eigenvalues os hessian is {eig.eigenvalues}")
            except torch.cuda.OutOfMemoryError:
                print("catch an OutOfMemoryError, the shape of the weight is ", W00.shape,
                      "we will spilt to get scale")
                num_part = 16
                num_channel = W00.shape[0]
                ps = num_channel // num_part
                if num_channel % num_part != 0:
                    break
                try:
                    scale_out0s, zero_out0s, err0s = [], [], []
                    # for oom. Cut along the out_channel dimension,  
                    for k in range(num_part):
                        scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                            H=H00, groupsize=groupsize, x0=W00[k * ps:(k + 1) * ps],
                            x_int=w_int[k * ps:(k + 1) * ps])
                        scale_out0s.append(scale_out0)
                        zero_out0s.append(zero_out0)
                        err0s.append(err0)
                except torch.cuda.OutOfMemoryError:
                    print("catch an OutOfMemoryError again, the shape of the weight is ", W00.shape, "give up")
                    # If it still doesn’t work, give up.
                    torch.cuda.empty_cache()
                    break
                scale_out0, zero_out0, err0 = torch.cat(scale_out0s), torch.cat(zero_out0s), torch.cat(err0s)
            tmp = err0 < err
            err[tmp] = err0[tmp]
            scale_out[tmp] = scale_out0[tmp]
            zero_out[tmp] = zero_out0[tmp]
            rate = torch.sum(tmp) / tmp.shape[0]
            print(
                f"Iter {iter_num}, the success rate of analytic_scale is {rate.item()}, the loss is {err.mean().item()}")
            if rate < 1e-4:
                break
            del scale_out0, zero_out0, err0, tmp, rate

        print(f"after {iter_num} of pure_training, the loss is {err.mean().item()}")
        scale_out0, zero_out0 = repeat_interleave(W, groupsize, scale_out, zero_out)
        if symmetric:
            Q = w_int * scale_out0
        else:
            Q = w_int * scale_out0 + zero_out0
        loss = torch.matmul(torch.matmul((W00 - Q), H), (W00 - Q).t()).diag()
        print("finally the loss is ", loss.mean().item())
        if self.layer.__class__.__name__ in ("Conv1D",) and Q.ndim == 2:
            Q = Q.t()
            w_int = w_int.t()
            scale_out = scale_out.t()
            zero_out = zero_out.t()
        w_int = w_int.reshape(self.layer.weight.shape).to(torch.int8)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        return scale_out, zero_out, w_int, err



    def startquant_second(self, groupsize, symmetric, max_iter_num,
                inner_iters_for_round, iters_before_round, dev, lr, actorder=True,
                round_fn="gptq", perdamp=0.01):
        # ==== 新增：保存原始权重和初始量化结果 ====
        original_weight = self.layer.weight.data.clone()
        W = original_weight.clone().detach()
        W = W.to(dev).float()
        
        # 权重预处理（原有代码）
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif self.layer.__class__.__name__ in ("Conv1D",) and W.ndim == 2:
            W = W.t()
        elif isinstance(self.layer, nn.Linear):
            pass
        else:
            raise NotImplementedError
        
        # 加载Hessian矩阵（原有代码）
        H = self.H.to(device=dev, dtype=W.dtype)
        del self.H
        
        # Hessian正则化（原有代码）
        dp = torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += perdamp * dp
        
        # 保存原始Hessian（原有代码）
        H00 = H
        W00 = W
        
        # 计算初始量化结果（原有代码）
        iters_for_scale = self.quantizer.iters_for_scale
        self.quantizer.iters_for_scale = 0
        scale, zero, scale_out, zero_out, init_err = self.quantizer.find_params(W, groupsize=groupsize, H=H)
        print(f"初始量化损失: {init_err.mean().item():.3e}")
        
        # 获取初始量化权重（原有代码）
        w_int = self.quantizer.get_fake_int_in(W00, scale, zero, groupsize=groupsize)
        
        # ==== 新增：准备优化结果容器 ====
        optimized = False
        best_scale_out = scale_out.clone()
        best_zero_out = zero_out.clone()
        best_w_int = w_int.clone()
        best_err = init_err.clone()
        
        # ==== 新增：尝试优化过程 ====
        try:
            # 优化前准备
            torch.cuda.empty_cache()
            max_inner_iter = inner_iters_for_round
            
            # 第一轮优化：GPTQ（原有代码）
            if max_iter_num >= 1:
                s0, z0 = repeat_interleave(W, groupsize, scale_out, zero_out)
                h00, w00 = H00.clone(), W00.clone()
                perm, invperm = None, None
                
                # 激活重排序（原有代码）
                if actorder:
                    perm = torch.argsort(torch.diag(H), descending=True)
                    invperm = torch.argsort(perm)
                    h00 = h00[perm][:, perm]
                    s0, z0 = s0[:, perm], z0[:, perm]
                    w00 = w00[:, perm]
                
                # 优化整数权重（原有代码）
                w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, self.quantizer.min_bound,
                                            self.quantizer.max_bound, w00, round_fn="gptq")
                
                if actorder:
                    w_int0 = w_int0[:, invperm]
                
                # ==== 新增：更新优化结果 ====
                tmp = err0 < best_err
                best_w_int[tmp] = w_int0[tmp]
                best_err[tmp] = err0[tmp]
                optimized = True
                print(f"GPTQ优化成功，损失降至: {best_err.mean().item():.3e}")
            
            # 后续优化迭代（原有代码）
            for iter_num in range(max_iter_num - 1):
                # 更新量化参数
                try:
                    # ==== 新增：增强Hessian正则化 ====
                    H00_reg = H00.clone()
                    diag_reg = 1e-6 * torch.mean(torch.diag(H00_reg))
                    H00_reg += torch.eye(H00_reg.shape[0], device=H00_reg.device) * diag_reg
                    
                    # ==== 新增：使用双精度计算提高稳定性 ====
                    with torch.no_grad():
                        # 转换为双精度
                        H00_reg_double = H00_reg.double()
                        W00_double = W00.double()
                        best_w_int_double = best_w_int.double()
                        
                        scale_out0, zero_out0, err0 = self.quantizer.get_scale_and_zero_out_group(
                            H=H00_reg_double, 
                            groupsize=groupsize, 
                            x0=W00_double, 
                            x_int=best_w_int_double
                        )
                        
                        # ==== 新增：将结果转换回原始精度 ====
                        scale_out0 = scale_out0.float()
                        zero_out0 = zero_out0.float()
                        err0 = err0.float()
                        
                        # ==== 新增：处理负误差 ====
                        if (err0 < -1e-12).any():
                            print(f"修正负误差: min={err0.min().item():.3e}")
                            err0 = torch.clamp(err0, min=0)
                except Exception as e:
                    print(f"参数更新失败: {e}，跳过本轮优化")
                    continue
                
                # 准备下一轮优化（原有代码）
                s0, z0 = repeat_interleave(W, groupsize, scale_out0, zero_out0)
                h00, w00 = H00.clone(), W00.clone()
                if actorder:
                    h00 = h00[perm][:, perm]
                    s0, z0 = s0[:, perm], z0[:, perm]
                    w00 = w00[:, perm]
                
                # 优化整数权重（原有代码）
                w_int0, _, err0 = opt_intW3(s0, z0, h00, symmetric, 
                                        self.quantizer.min_bound, 
                                        self.quantizer.max_bound, 
                                        w00,
                                        x_init=None, 
                                        max_iter=iters_before_round, 
                                        lr=lr, 
                                        max_inner_iter=max_inner_iter,
                                        round_fn=round_fn)
                
                if actorder:
                    w_int0 = w_int0[:, invperm]
                
                # ==== 新增：更新最佳结果 ====
                tmp = err0 < best_err
                best_w_int[tmp] = w_int0[tmp]
                best_scale_out[tmp] = scale_out0[tmp]
                best_zero_out[tmp] = zero_out0[tmp]
                best_err[tmp] = err0[tmp]
                optimized = True
                print(f"迭代 {iter_num} 优化成功，损失降至: {best_err.mean().item():.3e}")
                
                # ==== 新增：早停检查 ====
                improvement_rate = torch.sum(tmp).item() / tmp.shape[0]
                if improvement_rate < 1e-4:
                    print(f"优化改善率低 ({improvement_rate:.4f})，提前终止")
                    break
        
        # ==== 新增：捕获所有异常并回退 ====
        except Exception as e:
            print(f"优化过程中发生错误: {e}")
            print("将使用初始量化结果")
            # 回退到初始量化结果
            best_scale_out = scale_out
            best_zero_out = zero_out
            best_w_int = w_int
            best_err = init_err
        
        # ==== 新增：最终处理 ====
        if optimized:
            print(f"优化成功，最终损失: {best_err.mean().item():.3e}")
        else:
            print("使用初始量化结果")
        
        scale_out_full, zero_out_full = repeat_interleave(W00, groupsize, best_scale_out, best_zero_out)
        if scale_out_full is None or scale_out_full.shape != best_w_int.shape:
            raise RuntimeError(
                f"scale_out_full shape mismatch: best_w_int={tuple(best_w_int.shape)}, "
                f"scale_out={tuple(best_scale_out.shape)}, scale_out_full={None if scale_out_full is None else tuple(scale_out_full.shape)}, "
                f"groupsize={groupsize}, layer={self.layer.__class__.__name__}"
            )
        if not symmetric:
            if zero_out_full is None or zero_out_full.shape != best_w_int.shape:
                raise RuntimeError(
                    f"zero_out_full shape mismatch: best_w_int={tuple(best_w_int.shape)}, "
                    f"zero_out={tuple(best_zero_out.shape)}, zero_out_full={None if zero_out_full is None else tuple(zero_out_full.shape)}, "
                    f"groupsize={groupsize}, layer={self.layer.__class__.__name__}"
                )

        if symmetric:
            Q = best_w_int * scale_out_full
        else:
            Q = best_w_int * scale_out_full + zero_out_full

        W00_float = W00.float()
        Q_float = Q.float()
        H00_float = H00.float()
        final_loss = torch.matmul(torch.matmul((W00_float - Q_float), H00_float), (W00_float - Q_float).t()).diag().mean()
        print(f"最终量化损失: {final_loss.item():.3e}")

        w_int_apply = best_w_int
        scale_out_apply = best_scale_out
        zero_out_apply = best_zero_out
        Q_apply = Q

        if self.layer.__class__.__name__ in ("Conv1D",) and Q_apply.ndim == 2:
            Q_apply = Q_apply.t()
            w_int_apply = w_int_apply.t()
            scale_out_apply = scale_out_apply.t()
            zero_out_apply = zero_out_apply.t()

        w_int_apply = w_int_apply.reshape(original_weight.shape).to(torch.int8)
        self.layer.weight.data = Q_apply.reshape(original_weight.shape).to(original_weight.dtype)

        return scale_out_apply, zero_out_apply, w_int_apply, best_err

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def replace_forward(layers):
    # modify_forward(layers)
    origin_forward = {}
    for name, module in layers.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D", ):
            # print(f"replace forward for {name}")
            origin_forward[name] = module.forward
            module.forward = linear_forward(module)
        elif isinstance(module, (torch.nn.Conv2d,)):
            # print(f"replace forward for {name}")
            origin_forward[name] = module.forward
            module.forward = conv2d_forward(module)

    return origin_forward


def recover_forward(layers, origin_forward):
    for name, module in layers.named_modules():
        if name in origin_forward:
            # print(f"recover forward for {name}")
            module.forward = origin_forward[name]


def linear_forward(self):
    def tmp(inputs, *args, **kwargs):
        shape = (len(inputs.shape) - 1) * [1] + [inputs.shape[-1]]
        if self.__class__.__name__ in ("Conv1D",):
            size_out = inputs.size()[:-1] + (self.nf,)
            dim = 0
            mdtype = "Conv1D"
        elif isinstance(self, (torch.nn.Linear,)):
            size_out = inputs.size()[:-1] + (self.weight.shape[0],)
            dim = 1
            mdtype = "Linear"
        else:
            raise NotImplementedError("Fatal Error")

        if hasattr(self, "scale"):
            if self.group_size == -1:
                self.group_size = self.weight.shape[dim]
            scale = torch.repeat_interleave(self.scale, repeats=self.group_size, dim=dim)
            zero = torch.repeat_interleave(self.zero, repeats=self.group_size, dim=dim)
            weight = self.weight * scale + zero
        else:
            weight = self.weight
        if mdtype == "Conv1D":
            out = torch.mm(inputs.view(-1, inputs.size(-1)), weight)
        else:
            out = F.linear(inputs, weight)
        if self.bias is not None:
            out = out + self.bias
        out = out.view(size_out)
        return out

    return tmp


def conv2d_forward(self):
    def tmp(inputs, *args, **kwargs):
        shape = [self.weight.shape[0]] + (len(self.weight.shape) - 1) * [1]
        if hasattr(self, "scale"):
            scale = torch.reshape(self.scale, shape=shape)
            zero = torch.reshape(self.zero, shape=shape)
            weight = self.weight * scale + zero
        else:
            weight = self.weight
        out = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    return tmp


@torch.enable_grad()
def minimize_block(args, quantizers, layer, inps, dev, layer_num, masks):
    layer = layer.to(dev)
    full = find_layers(layer)
    params = []
    original_dtype = {}
    for key in full:
        quantizer = quantizers[f"{layer_num}.{key}.weight"]
        weight = quantizer["weights"]
        scale_list = quantizer["scales"]
        original_dtype[key] = full[key].weight.dtype
        dtype = torch.float32
        factory_kwargs = {'device': dev, 'dtype': dtype}
        full[key].weight.data = weight.to(**factory_kwargs)
        full[key].weight.requires_grad_(False)
        scale = torch.nn.Parameter(scale_list[0].clone().to(**factory_kwargs), requires_grad=True)
        requires_grad = True if args.asym else False
        zero = torch.nn.Parameter(scale_list[1].clone().to(**factory_kwargs), requires_grad=requires_grad)
        full[key].register_parameter("scale", scale)
        full[key].register_parameter("zero", zero)
        full[key].group_size = args.group_size
        params.append(scale)
        if args.asym:
            params.append(zero)

    if args.train_LN:
        for k, m in layer.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                    print("add layer norm weight to train")
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    print("add layer norm bias to train")

    if args.train_bias:
        for k, m in layer.named_modules():
            if isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ("Conv1D", ):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    print("add linear bias to train")

    origin_forward = replace_forward(layer)
    lr = args.blockwise_minimize_lr
    opt = torch.optim.Adam(params, lr, eps=2.e-5, betas=(0.9, 0.99), weight_decay=args.blockwise_minimize_wd)
    print("--", opt.param_groups[0]["lr"])
    total_loss = 0.0
    for j in range(args.blockwise_minimize_epoch):
        for idx, b in enumerate(inps):
            b = fp16tofloat(to_device(b, dev))
            if args.out_path is not None:
                label = torch.load(args.out_path+f"/tmp_blockwise/out_{idx}.pth")
            else:
                label = torch.load(f"./tmp_blockwise/out_{idx}.pth")
            mask = masks[idx]
            out = layer(*(b[0]), **(b[1]))

            res = (out[0] - to_device(label["out"][0], dev))
            if mask is not None:
                res = res * mask.float().unsqueeze(-1)
                loss = torch.sum(res * res) / (mask.float().sum() * res.shape[-1])
            else:
                loss = torch.mean(res * res)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
        print(f"the avg loss for training scale zero is {total_loss / len(inps)}")
        total_loss = 0.0

    for key in full:
        scale, zero = full[key].scale, full[key].zero
        quantizers[f"{layer_num}.{key}.weight"]["scales"] = [scale.cpu(), zero.cpu()]
        if full[key].__class__.__name__ in ("Conv1D",):
            dim = 0
        elif isinstance(full[key], torch.nn.Linear):
            dim = 1
        else:
            raise NotImplementedError
        groupsize = args.group_size
        if groupsize == -1:
            groupsize = full[key].weight.data.shape[dim]
        scale = torch.repeat_interleave(scale, repeats=groupsize, dim=dim)
        zero = torch.repeat_interleave(zero, repeats=groupsize, dim=dim)
        full[key].weight.data = full[key].weight.data * scale + zero
        full[key].weight.data = full[key].weight.data.to(original_dtype[key])
        del full[key].scale, full[key].zero, full[key].group_size

    recover_forward(layer, origin_forward)
    print()

@torch.enable_grad()
def finetune_scale(args, model, quantizers, dev):
    import time
    import transformers
    from transformers import Seq2SeqTrainer, LlamaTokenizer, AdamW
    from datautils_e2e import make_data_module
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    print("Starting end-to-end finetuning of scale and zero...")
    t1 = time.time()
    
    # Set model to train mode
    model.train()

    params = []
    original_dtype = {}
    module_to_quantizer = {}
    for q_key in quantizers:
        # Extract layer number and module name from quantizer key
        # Example: "0.self_attn.q_proj.weight" -> layer_num=0, module_path="self_attn.q_proj"
        parts = q_key.split(".")
        if len(parts) < 3:
            continue
        layer_num = parts[0]
        module_path = ".".join(parts[1:-1])  # Remove layer_num and "weight"
        # Construct module name pattern: "model.layers.{layer_num}.{module_path}"
        module_pattern = f"model.layers.{layer_num}.{module_path}"
        module_to_quantizer[module_pattern] = q_key

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D",):
            # Check if this module has quantized weights
            if name in module_to_quantizer:
                q_key = module_to_quantizer[name]
                quantizer = quantizers[q_key]
                weight = quantizer["weights"]
                scale_list = quantizer["scales"]
                original_dtype[name] = module.weight.dtype
                dtype = torch.float32
                factory_kwargs = {'dtype': dtype}
                module.weight.data = weight.to(**factory_kwargs)
                module.weight.requires_grad_(False)
                
                # Register scale and zero as parameters
                scale = torch.nn.Parameter(scale_list[0].clone().to(**factory_kwargs), requires_grad=True)
                requires_grad = True if args.asym else False
                zero = torch.nn.Parameter(scale_list[1].clone().to(**factory_kwargs), requires_grad=requires_grad)
                
                module.register_parameter("scale", scale)
                module.register_parameter("zero", zero)
                module.group_size = args.group_size
                
                params.append(scale)
                if args.asym:
                    params.append(zero)
    
    # Add LayerNorm parameters if needed
    if hasattr(args, 'train_LN') and args.train_LN:
        for k, m in model.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)) or "Norm" in m.__class__.__name__:
                if hasattr(m, "weight"):
                    m.weight.requires_grad_(True)
                    params.append(m.weight)
                    # print(f"add layer norm weight to train: {k}")
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    # print(f"add layer norm bias to train: {k}")
    
    # Add bias parameters if needed
    if hasattr(args, 'train_bias') and args.train_bias:
        for k, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or m.__class__.__name__ in ("Conv1D",):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad_(True)
                    params.append(m.bias)
                    # print(f"add linear bias to train: {k}")
    


    # Replace forward method to use scale and zero
    origin_forward = replace_forward(model)
    
    from memory_utils import distribute_model
    if args.train_distribute:                     # 如果需要分布式
        distribute_model(model)
    
    
    # Set up optimizer
    lr = args.e2e_lr if hasattr(args, 'e2e_lr') else args.blockwise_minimize_lr
    opt = AdamW(params, lr, eps=2.e-5, betas=(0.9, 0.99), weight_decay=0)
    print(f"Optimizer learning rate: {opt.param_groups[0]['lr']}")
    
    # Load tokenizer and dataset
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    if tokenizer._pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data module with alpaca dataset
    data_args = type('DataArgs', (), {
        'dataset': 'alpaca',
        'source_max_len': 384,
        'target_max_len': 128,
        'max_train_samples': None,
        'max_eval_samples': 16,
        'eval_dataset_size': 16,
        'conv_temp': 'llama-2',
        'mask_use': True,
        'dataset_format': 'alpaca',
        'overwrite_cache': False,
        'preprocessing_num_workers': 32,
        'do_predict': False,
        'do_eval': False,
        'do_mmlu_eval': True,
        'do_train':True,
        'group_by_length':True,
        'predict_with_generate':False,
    })()
    
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir='./output_e2e',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        weight_decay=0,
        max_steps=args.max_steps,#10000
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy='steps',
        # evaluation_strategy='steps',
        # eval_steps=200,#2000
        save_total_limit=5,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=0.3,
        group_by_length=True,
        report_to='none',
        do_train=True,
        ddp_find_unused_parameters=False,  # 避免DDP错误
        dataloader_num_workers=4,
    )
    import argparse
    data_module = make_data_module(tokenizer=tokenizer, args=data_args)

    
    # Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(opt, None),
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'}
    )
    
    # Start training
    print("Starting training...")
    train_result = trainer.train()
    print(f"Training completed. Metrics: {train_result.metrics}")

    # Save trained scale and zero back to quantizers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ in ("Conv1D",):
            if hasattr(module, "scale") and hasattr(module, "zero"):
                # Find the corresponding key in quantizers
                if name in module_to_quantizer:
                    q_key = module_to_quantizer[name]
                    scale, zero = module.scale, module.zero
                    quantizers[q_key]["scales"] = [scale.cpu(), zero.cpu()]
                    #Apply scale and zero to weights
                if module.__class__.__name__ in ("Conv1D",):
                    dim = 0
                elif isinstance(module, torch.nn.Linear):
                    dim = 1
                else:
                    raise NotImplementedError
                
                groupsize = args.group_size
                if groupsize == -1:
                    groupsize = module.weight.data.shape[dim]
                scale_repeated = torch.repeat_interleave(scale, repeats=groupsize, dim=dim)
                zero_repeated = torch.repeat_interleave(zero, repeats=groupsize, dim=dim)
                module.weight.data = module.weight.data * scale_repeated + zero_repeated
                module.weight.data = module.weight.data.to(original_dtype[name])
                # Clean up
                del module.scale, module.zero, module.group_size

    recover_forward(model, origin_forward)

    
    print(f"End-to-end finetuning completed. Time cost: {time.time() - t1:.2f} seconds")
    return quantizers
