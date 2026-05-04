# LGRQuant_v2 端到端测试指南

> **假设前提**: dq_xx 和 abq-llm 环境已创建，CUDA 内核已编译

## 测试前准备

### 1. 激活环境并验证

```bash
# 验证 dq_xx 环境 (Stage1 + Inference)
conda activate dq_xx
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证 W2/W4 内核
python -c "from lgrquant.core.linear_w2a16 import LinearW2A16; print('W2 kernel OK')"
python -c "from lgrquant.core.linear_w4a16 import LinearW4A16; print('W4 kernel OK')"
```

### 2. 配置路径

```bash
cd /data1/xx/LGRQuant_v2

# 复制配置模板
cp configs/paths_template.yaml configs/paths.yaml

# 编辑 configs/paths.yaml，填入你的实际路径
vim configs/paths.yaml
```

**configs/paths.yaml 示例**:
```yaml
# 模型路径 - 使用你本地已有的模型
model_path: "/data2/llms/Qwen2.5-7B-Instruct"

# 输出根目录
output_root: "/data2/xx_llms/LGRQuant_v2_test"

# Conda 环境路径 (用于 pipeline 脚本自动切换)
dq_xx_python: "/data0/miniconda3/envs/dq_xx/bin/python"
abq_llm_python: "/data0/miniconda3/envs/abq-llm/bin/python"

# Teacher 缓存目录
teacher_cache_dir: "/data2/xx_llms/LGRQuant_v2_test/teacher_topk_cache"
```

### 3. 选择测试配置

```bash
# 使用 W2 配置
cp configs/w2.yaml configs/test_config.yaml

# 或使用 W4 配置  
cp configs/w4.yaml configs/test_config.yaml
```

**建议**: 先用小模型测试 (如 Qwen2.5-3B 或 7B)，而不是 14B

---

## 完整测试流程

### Stage 1: 量化 (dq_xx 环境)

```bash
conda activate dq_xx
cd /data1/xx/LGRQuant_v2

# 方法1: 使用统一脚本
bash scripts/stage1/run.sh configs/test_config.yaml

# 方法2: 直接运行 Python (推荐用于调试)
python -m lgrquant.stage1.quantize \
    --model /data2/llms/Qwen2.5-7B-Instruct \
    --dataset wikitext2 \
    --wbits 2 \
    --groupsize 64 \
    --save \
    --out_path /data2/xx_llms/LGRQuant_v2_test/stage1
```

**预期输出**:
- `quantized_model.pth` - 统一格式检查点
- `true_quant.pth` - 兼容格式
- `fake_quant.pth` - 兼容格式
- `quantizers.pth` - 原始格式

**检查**:
```bash
ls -lh /data2/xx_llms/LGRQuant_v2_test/stage1/
```

---

### Precompute: Teacher Logits 缓存 (abq-llm 环境)

```bash
conda activate abq-llm
cd /data1/xx/LGRQuant_v2

# 使用脚本
bash scripts/stage2/precompute.sh

# 或手动运行
python -m lgrquant.stage2.precompute_teacher_topk \
    --model_id /data2/llms/Qwen2.5-7B-Instruct \
    --dataset wikitext2 \
    --nsamples 1000 \
    --seqlen 2048 \
    --top_k 1000 \
    --out_dir /data2/xx_llms/LGRQuant_v2_test/teacher_topk_cache
```

**预期输出**:
- `sample_00000.pt` ~ `sample_00999.pt` (每个约 12MB)
- `meta.json`

**检查**:
```bash
ls /data2/xx_llms/LGRQuant_v2_test/teacher_topk_cache/ | wc -l
# 应显示 1001 个文件 (1000 samples + meta.json)
```

---

### Stage 2: 端到端蒸馏微调 (abq-llm 环境)

```bash
conda activate abq-llm
cd /data1/xx/LGRQuant_v2

# 使用脚本
bash scripts/stage2/run.sh

# 或手动运行
python -m lgrquant.stage2.quant_finetune \
    --model_id /data2/llms/Qwen2.5-7B-Instruct \
    --true_quant_path /data2/xx_llms/LGRQuant_v2_test/stage1/true_quant.pth \
    --teacher_topk_dir /data2/xx_llms/LGRQuant_v2_test/teacher_topk_cache \
    --out_path /data2/xx_llms/LGRQuant_v2_test/stage2 \
    --train_steps 200 \
    --group_size 64 \
    --asym \
    --use_distill \
    --distill_loss kl_top \
    --dataset wikitext2 \
    --nsamples 1000
```

**预期输出**:
- `quantizers.pth` - 微调后的量化器
- `quantized_model.pth` - 统一格式 (包含微调后的参数)
- `finetuned_ln_bias.pth` - 微调的 LayerNorm 和 bias

**检查**:
```bash
ls -lh /data2/xx_llms/LGRQuant_v2_test/stage2/
```

---

### Inference: 推理测试 (dq_xx 环境)

```bash
conda activate dq_xx
cd /data1/xx/LGRQuant_v2

# W2 推理测试
bash scripts/inference/run_w2.sh

# 或手动运行
python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path /data2/llms/Qwen2.5-7B-Instruct \
    --true_quant_path /data2/xx_llms/LGRQuant_v2_test/stage2/true_quant.pth \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    --prompt "Artificial intelligence is"
```

**预期输出**:
```
======== Generation Sample ========
Artificial intelligence is ...

======== PPL Evaluation ========
[PPL] dataset = wikitext2 ...
  PPL[wikitext2] = X.XXXX
```

**Zero-shot 评测** (可选，需要更多时间):
```bash
python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path /data2/llms/Qwen2.5-7B-Instruct \
    --true_quant_path /data2/xx_llms/LGRQuant_v2_test/stage2/true_quant.pth \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    --lm_eval_tasks "piqa,hellaswag,arc_easy"
```

---

## 一键完整流程

如果你不想分步执行，使用 pipeline 脚本：

```bash
cd /data1/xx/LGRQuant_v2

# 完整流程: Stage1 -> Precompute -> Stage2 -> Inference
bash scripts/pipeline/run_full.sh configs/test_config.yaml
```

**注意**: pipeline 脚本会自动切换 conda 环境，但你需要先确保 configs/paths.yaml 中配置了正确的 Python 路径。

---

## 快速验证测试 (推荐)

为了快速验证流程是否通畅，使用减少的样本数：

```bash
# Stage 1 (少量样本)
python -m lgrquant.stage1.quantize \
    --model /data2/llms/Qwen2.5-3B \
    --dataset wikitext2 \
    --wbits 2 \
    --groupsize 64 \
    --save \
    --nsamples 32 \
    --out_path /tmp/lgrquant_test/stage1

# Precompute (少量样本)
python -m lgrquant.stage2.precompute_teacher_topk \
    --model_id /data2/llms/Qwen2.5-3B \
    --dataset wikitext2 \
    --nsamples 32 \
    --out_dir /tmp/lgrquant_test/teacher_cache

# Stage 2 (少量 steps)
python -m lgrquant.stage2.quant_finetune \
    --model_id /data2/llms/Qwen2.5-3B \
    --true_quant_path /tmp/lgrquant_test/stage1/true_quant.pth \
    --teacher_topk_dir /tmp/lgrquant_test/teacher_cache \
    --out_path /tmp/lgrquant_test/stage2 \
    --train_steps 10 \
    --nsamples 32

# Inference
python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path /data2/llms/Qwen2.5-3B \
    --true_quant_path /tmp/lgrquant_test/stage2/true_quant.pth \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2
```

---

## 故障排查

### 1. "No module named 'lgrquant'"

**解决**: 确保在 `/data1/xx/LGRQuant_v2` 目录下运行，或使用 `python -m`

```bash
cd /data1/xx/LGRQuant_v2
export PYTHONPATH=/data1/xx/LGRQuant_v2:$PYTHONPATH
```

### 2. "undefined symbol" 加载 W2 内核失败

**解决**: 确保在 `dq_xx` 环境中，且内核是用该环境的 PyTorch 编译的

```bash
conda activate dq_xx
python -c "import torch; print(torch.__version__)"  # 应为 2.1.x

# 如果仍失败，重新编译
bash scripts/build_kernels.sh --w2-only
```

### 3. Stage 2 OOM (显存不足)

**解决**: 减少 batch size 或 nsamples

```bash
python -m lgrquant.stage2.quant_finetune \
    ... \
    --nsamples 512  # 减少到 512
```

### 4. "teacher_topk_cache" 找不到

**解决**: 确保先运行了 precompute

```bash
ls /data2/xx_llms/LGRQuant_v2_test/teacher_topk_cache/
# 应该有 sample_00000.pt 等文件
```

### 5. 模型加载失败

**解决**: 检查模型路径

```bash
ls /data2/llms/Qwen2.5-7B-Instruct/config.json
```

---

## 成功标志

如果看到以下输出，说明全流程成功：

1. **Stage 1**: "saved X layers to quantizers"
2. **Precompute**: "[precompute] done. meta.json written"
3. **Stage 2**: "quantizers saved to ..." 和 "model saved"
4. **Inference**: "PPL[wikitext2] = X.XXXX" 和生成的文本

---

## 性能参考 (Qwen2.5-7B-Instruct, W2)

| Stage | 时间 | 显存 |
|-------|------|------|
| Stage 1 | ~30 min | ~30 GB |
| Precompute | ~7 min | ~28 GB |
| Stage 2 (200 steps) | ~25 min | ~39 GB |
| Inference | ~2 min | ~15 GB |

**最终指标**:
- PPL (wikitext2): ~8-9 (vs FP16 ~7-8)
- PIQA: ~77 (vs FP16 ~80)

---

## 日志位置

所有日志统一保存在 `logs/` 目录：

```bash
ls -t logs/
# stage1_20260505_120000.log
# precompute_20260505_123000.log
# stage2_20260505_124000.log
# inference_w2_20260505_130000.log
```

查看日志：
```bash
tail -f logs/stage1_*.log
```
