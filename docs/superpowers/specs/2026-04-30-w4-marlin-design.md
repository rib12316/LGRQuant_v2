# LGRQuant — 4-bit Marlin 推理内核 + 推理评估扩展（Spec）

- 日期：2026-04-30
- 项目根：`/data1/xx/LGRQuant`
- 模型：`Qwen2.5-3B`，路径 `/data2/llms/Qwen2.5-3b`
- 主环境：conda `dq_xx`（Python 3.9.23 / torch 2.1.0+cu121 / L40S sm_89）
- 状态：Spec — 待用户复核

---

## 1. 目标 (Goal)

在 LGRQuant 项目中新增 **4-bit 推理内核（原版 Marlin）** 与一条**可选的 4-bit 推理评估子流程**，并新增一份**只跑 Stage 1 + Stage 2 的快速量化流水线**。优先里程碑分级如下：

- **M1（关键、当前唯一焦点）**：Marlin 内核适配正确 + 权重打包正确 + 推理产出非乱码。
- **M2（次级）**：显存 ↓ ≥ 50%（vs FP16 baseline）。per-token 推理速度 ↑ ≥ 10%（vs FP16 baseline）。
- **M3（再次级）**：lm-eval 4 任务（piqa / hellaswag / winogrande / boolq）每项 acc 与 FP16 差距 < 10 pts。

PPL（wikitext2）仅作为观察项，不作为通过条件。

## 2. 范围 (Scope)

**包含**：
- 新增 `decoupleQ/linear_w4a16.py`（Marlin 适配的 W4A16 Linear）。
- 新增 `pipeline_fast.sh`（仅 Stage 1 + Stage 2，支持 `WBITS=2|4` 切换）。
- 修改 `inference_test.py`：新增 `--kernel {w2,w4}` 参数；抽出 `replace_linear_generic`；W4 路径走 `LinearW4A16`。
- 修改 `quant_finetune.py`：新增 `save_true_quant_from_finetune`，让 Stage 2 在保存 `quantizers.pth` 之外，**也产出与 Stage 1 完全同结构的 `true_quant.pth`**（包含 embed/norm/lm_head/bias/layernorm 等 FP 主干 + int8 weight + scale [+ zp]）。
- 修改 `run_inference_test.sh`：新增 `KERNEL=w2|w4` 环境变量切换；默认 `w2` 完全保留旧行为。
- 新增 `tools/sanity_true_quant.py`、`tools/check_w4_targets.sh` 辅助脚本。
- 文档：本 spec、后续 implementation plan。

**不包含**：
- 不改动 `pipeline.sh`（保留 4 阶段全流程）。
- 不改动 Stage 3 / Stage 4。
- 不引入 vllm / auto-gptq / autoawq / bitsandbytes（仅 fallback 时再考虑）。
- 不在 `dq_xx` 内尝试运行 `quant_finetune.py`（Stage 2 仍跑 `ptq1.61`）。

## 3. 约束 (Constraints)

- C1 **环境策略**（D1）：Stage 1 + 推理评估全跑 `dq_xx`；Stage 2 跑 `ptq1.61`。
- C2 **环境变更须用户审批**：任何 pip / conda 包变更必须先报告版本号，由用户下载后反馈再继续（这是用户硬性要求）。
- C3 **硬件**：单卡 L40S（46 GB）足够；sm_89 满足 Marlin sm_80+ 要求。
- C4 **量化对称性**：W4 路径强制 sym（无 zero-point），W2 路径保留现状 asym。
- C5 **group_size**：W4 强制 128（Marlin 原版限制），W2 保留 64。
- C6 仅复用 `pipeline.sh` 的 Stage 1（`llama.py`）+ Stage 2（`quant_finetune.py`），跳过 Stage 3 / 4。

## 4. 关键决策记录

| ID | 决策 |
|----|------|
| D1 | Stage 1 + 推理评估 = `dq_xx`；Stage 2 = `ptq1.61` |
| D2 | 4-bit 内核首选 **原版 Marlin**（IST-DASLab/marlin），fallback GPTQ-Marlin |
| D3 | 推理评估口径 = 与现有 W2A16 `inference_test.py` 完全一致 |
| D4 | 推理入口 = 同一 `run_inference_test.sh` + `KERNEL=w2|w4` |
| D5 | 新增 `pipeline_fast.sh`（仅 Stage1+2，`WBITS=2|4`） |
| D6 | Stage 2 必须额外产出 `true_quant.pth`（修补 `quant_finetune.py`） |
| D7 | `pipeline_fast.sh` Stage 2 `--train_steps=2000` |
| D8 | W4 量化路径强制 sym；W2 保留 asym |
| D9 | 实现路线 = Option A（原版 Marlin + 自包装 `LinearW4A16`） |
| D10 | 里程碑：M1 = 内核+打包正确 → M2 = 性能达标 → M3 = lm-eval acc 差距 < 10pts |

## 5. 文件改动地图

```
LGRQuant/
├── pipeline.sh                        (不动)
├── pipeline_fast.sh                   (新增)
├── run_inference_test.sh              (改：KERNEL=w2|w4)
├── inference_test.py                  (改：--kernel + W4 装配分支)
├── quant_finetune.py                  (改：训练后多写一份 true_quant.pth)
├── llama.py                           (不动)
├── decoupleQ/
│   ├── linear_w2a16.py                (不动)
│   └── linear_w4a16.py                (新增)
├── tools/
│   ├── sanity_true_quant.py           (新增)
│   └── check_w4_targets.sh            (新增)
└── docs/superpowers/specs/
    └── 2026-04-30-w4-marlin-design.md (本文件)
```

## 6. 数据流

```
[FP16 Qwen2.5-3B]
    │
    │ pipeline_fast.sh  WBITS=4  (Stage 1, dq_xx)
    │   llama.py --wbits 4 --group-size 128 (no --asym)
    ▼
[w4/stage1/true_quant.pth]   sym:  weight(int8) + qscale(fp32)  + FP backbone
    │
    │ pipeline_fast.sh  WBITS=4  (Stage 2, ptq1.61)
    │   quant_finetune.py --group_size 128 --train_steps 2000 (no --asym)
    │   + save_true_quant_from_finetune()   ← D6
    ▼
[w4/stage2/true_quant.pth]   结构与 stage1 完全一致
    │
    │ run_inference_test.sh  KERNEL=w4  (dq_xx)
    │   inference_test.py --kernel w4 --true_quant_path ... --group_size 128
    ▼
LinearW4A16 装配 → marlin.mul GEMM → Generation / PPL / lm-eval / Speed Benchmark vs FP16
```

## 7. 组件设计

### 7.1 `decoupleQ/linear_w4a16.py`

`LinearW4A16(nn.Module)`，与 `LinearW2A16` 1:1 对应：

- 字段：`qweight`（marlin packed int4，存为 int32 view） / `scales`（fp16 [k/gs, n] 或 [1, n]） / `workspace`（per-layer staging buffer） / `bias`（fp16 [n]）。
- `forward(x)`：将 x 视情况 cast 到 fp16，分配 fp16 输出 buffer，调 `marlin.mul(x, qweight, out, scales, workspace)`，加 bias，恢复 dtype。
- 约束：`group_size ∈ {128, -1}`；`x.dtype` ∈ {fp16, bf16}；输出 fp16。

辅助函数：

- `convert_qweight_to_marlin(int8_w, fp32_scale, group_size)`：从 ckpt 的 sym int8 + fp32 scale 构造 marlin packed weight + scales + workspace。实现先把 int8 还原成 dense fp16，再走 `marlin.Layer.pack(...)`（具体调用与 marlin 实际版本对齐）。
- `replace_linear_with_marlin(model, sd, group_size)`：与 `replace_linear_with_w2` 同模板，循环替换 q/k/v/o/gate/up/down 7 类 Linear。

> **API 对齐说明**：marlin 0.1 / 0.1.1 的 `Layer.pack` 入参略有差异。**待用户下载 marlin 后告知确切版本号，再把 packing 代码收敛到该版本的 API**。

### 7.2 `inference_test.py` 修改

- 新增 `--kernel {w2,w4}`，默认 `w2`。
- 把 `replace_linear_with_w2` / `_prewarm_w2a16` 抽象成 `replace_linear_generic(model, sd, group_size, layer_ctor, prewarm_fn)`，W2 / W4 共用模板。
- W4 路径：使用 `LinearW4A16` + `convert_qweight_to_marlin`，加载阶段一次性完成 packing（不在 forward 内 lazy 转换，避免首次 inference 抖动）。
- 其它流程（generation / PPL / lm-eval / benchmark / FP baseline 对比）一行不改。

### 7.3 `quant_finetune.py` 修改

新增函数（设计参考 `llama.py:save_quant_model`）：

```python
def save_true_quant_from_finetune(args, model, quantizers,
                                  prefix="model.layers.",
                                  out_name="true_quant.pth"):
    """合成与 Stage 1 完全一致结构的 true_quant.pth：
       FP 主干（embed/norm/lm_head/bias/layernorm 等）来自当前 model.state_dict()；
       量化层 weight 来自 quantizers[k]['weights']（int8）；
       weight_qscale / weight_qzero 来自 quantizers[k]['scales']。
       sym 路径下若 scales[1] 为 None，则不写 _qzero 字段。
    """
```

调用点：训练主流程末尾、`save_quantizers(...)` 之后。

### 7.4 `pipeline_fast.sh`（新增）

- 入口环境变量：`BASE_MODEL` / `GPU_ID` / `WBITS={2,4}` / `OUT_ROOT` / `S2_TRAIN_STEPS=2000`。
- `WBITS=2`：`GS=64 ASYM_FLAG="--asym"`。
- `WBITS=4`：`GS=128 ASYM_FLAG=""`（sym）。
- Stage 1 在 `dq_xx` 跑；Stage 2 在 `ptq1.61` 跑。
- 简洁监控（`monitor_simple` 一份 GPU 显存采样），写 `fast_summary.txt`。
- 任一 Stage 退出非 0 时立即终止并报错。

### 7.5 `run_inference_test.sh` 修改

- 新增 `KERNEL=w2|w4` 默认 `w2`。
- W2 默认参数与现状一致；W4 默认 `QUANT_PT=/data2/xx_llms/LGRQuant_fast/w4/stage2/true_quant.pth`、`GROUP_SIZE=128`、不传 `--asym`、传 `--kernel w4`。
- 进入脚本时统一 `conda activate dq_xx`。

## 8. 错误处理（3-Strike 协议）

| 故障点 | Strike-1 | Strike-2 | Strike-3 |
|--------|----------|----------|----------|
| `pip install marlin` 失败 | 检查 nvcc / torch 兼容 | `pip install marlin@<git tag>` 或源码编译 | fallback 至 GPTQ-Marlin (auto-gptq) |
| Marlin pack API 不匹配 | 对齐版本号 API | 走替代入口 `Layer.from_fp16` 等 | 询问用户 |
| W4 推理输出乱码（M1 失败） | 校验 packing 路径（int8→packed→反算 fp16 矩阵 cosine） | 临时禁用 fp16_main、上 fp32 reference | 询问用户回退 GPTQ-Marlin |
| 显存 ratio > 0.5（M2 失败） | 检查 KV cache fp16 / 关无关缓存 | 改 `--max_new_tokens` 排查 weight 占比 | 报告用户 |
| 速度 < 1.10×（M2 失败） | 加 `--compile` + static cache | 测 prefill / batch>1 加速 | 报告用户 |

所有错误强制写入 `progress.md` 的 Errors 表。

## 9. 测试计划

- **T1** Marlin 装包 + smoke `marlin.mul` 通过。
- **T2** `LinearW4A16` 单元测试：随机 [K=4096, N=4096] 权重 cosine ≥ 0.99 vs reference dequant + matmul。
- **T3** `quant_finetune.py` 修补 sanity：跑 10 步 Stage 2，确认 `true_quant.pth` key 集合包含 FP 主干。
- **T4** `pipeline_fast.sh WBITS=2` 与现 `pipeline.sh` Stage1+2 等价烟测。
- **T5** `pipeline_fast.sh WBITS=4` 全跑出 `w4/stage2/true_quant.pth`。
- **T6** `KERNEL=w4 bash run_inference_test.sh` 全套评估，先确认 generation sample 非乱码（M1）。
- **T7** `tools/check_w4_targets.sh` 解析 Speedup Summary，PASS=M2 达成。
- **T8** lm-eval 4 任务 acc 与 FP16 baseline 比较，M3 检查。

## 10. 环境变更清单（待用户审批）

按 C2，下列包必须由用户下载后反馈：

1. **marlin**
   - 来源：`https://github.com/IST-DASLab/marlin`
   - 推荐：`pip install marlin==0.1.1`（或最新 stable tag）
   - 备注：先确认是否有 prebuilt wheel；若无则需要 nvcc 在 dq_xx 内可用。
   - 用户反馈实际安装到的版本号后，spec 中的 packing API 调用会按该版本固化。

> 现阶段**不需要**安装 auto-gptq / vllm / bnb 等任何其它包；仅当 marlin 适配失败时才考虑 fallback。

## 11. 里程碑顺序与开发节奏

1. M1（最高优）：装好 marlin → 写 `LinearW4A16` → 写 packing → 改 `inference_test.py` → 用一份小 W4 ckpt 验证非乱码。
2. M1 完成后再连接整条 `pipeline_fast.sh` + `quant_finetune.py` 的 Stage 2 修补 + `run_inference_test.sh` 切换。
3. M2：测速度 / 显存指标。
4. M3：跑 lm-eval 完整 4 任务对比。

## 12. Open Questions（已知遗留）

- Marlin 实际版本号 → 决定 `LinearW4A16.pack` 调用形式（待用户下载后反馈）。
- Stage 2 sym 训练在 `quant_finetune.py` 内是否需要任何额外开关（现 `--asym` 是 store_true，不传即 sym，但需 verify 内部 `quantizer.configure(perchannel=True, sym=...)` 路径在 sym 时的行为）。
