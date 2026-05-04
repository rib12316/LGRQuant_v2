#!/bin/bash
# LGRQuant_v2 快速测试脚本 (全流程)
# 使用小模型和少量样本，用于验证流程通畅

set -e

# 配置
MODEL_PATH="${MODEL_PATH:-/data2/llms/Qwen2.5-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/lgrquant_quick_test}"
NSAMPLES="${NSAMPLES:-32}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"

echo "============================================"
echo "LGRQuant_v2 Quick Test"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_ROOT"
echo "Samples: $NSAMPLES"
echo "Train steps: $TRAIN_STEPS"
echo ""

mkdir -p "$OUTPUT_ROOT"

# Stage 1
echo "[1/4] Stage 1: Quantization"
echo "--------------------------------------------"
conda activate dq_xx
cd /data1/xx/LGRQuant_v2

python -m lgrquant.stage1.quantize \
    --model "$MODEL_PATH" \
    --dataset wikitext2 \
    --wbits 2 \
    --groupsize 64 \
    --save \
    --nsamples $NSAMPLES \
    --out_path "$OUTPUT_ROOT/stage1" \
    2>&1 | tee "$OUTPUT_ROOT/stage1.log"

echo "✓ Stage 1 completed"
echo ""

# Precompute
echo "[2/4] Precompute: Teacher Logits Caching"
echo "--------------------------------------------"
conda activate abq-llm

python -m lgrquant.stage2.precompute_teacher_topk \
    --model_id "$MODEL_PATH" \
    --dataset wikitext2 \
    --nsamples $NSAMPLES \
    --seqlen 2048 \
    --top_k 1000 \
    --out_dir "$OUTPUT_ROOT/teacher_cache" \
    2>&1 | tee "$OUTPUT_ROOT/precompute.log"

echo "✓ Precompute completed"
echo ""

# Stage 2
echo "[3/4] Stage 2: E2E Distillation"
echo "--------------------------------------------"
python -m lgrquant.stage2.quant_finetune \
    --model_id "$MODEL_PATH" \
    --true_quant_path "$OUTPUT_ROOT/stage1/true_quant.pth" \
    --teacher_topk_dir "$OUTPUT_ROOT/teacher_cache" \
    --out_path "$OUTPUT_ROOT/stage2" \
    --train_steps $TRAIN_STEPS \
    --group_size 64 \
    --asym \
    --use_distill \
    --distill_loss kl_top \
    --dataset wikitext2 \
    --nsamples $NSAMPLES \
    2>&1 | tee "$OUTPUT_ROOT/stage2.log"

echo "✓ Stage 2 completed"
echo ""

# Inference
echo "[4/4] Inference: Evaluation"
echo "--------------------------------------------"
conda activate dq_xx

python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path "$MODEL_PATH" \
    --true_quant_path "$OUTPUT_ROOT/stage2/true_quant.pth" \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    --prompt "Artificial intelligence is" \
    2>&1 | tee "$OUTPUT_ROOT/inference.log"

echo "✓ Inference completed"
echo ""

# Summary
echo "============================================"
echo "Quick Test Completed Successfully!"
echo "============================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_ROOT/"
echo ""
echo "Logs:"
ls -lh "$OUTPUT_ROOT/"*.log
echo ""
echo "To clean up: rm -rf $OUTPUT_ROOT"
