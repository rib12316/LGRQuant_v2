#!/bin/bash
# LGRQuant_v2 快速测试脚本 (全流程)
# 使用小模型和少量样本，用于验证流程通畅

set -e

export CUDA_VISIBLE_DEVICES=0
# CUDA 内存优化：防止 PyTorch 缓存分配器碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 初始化 conda (脚本中激活环境必需)
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || {
    echo "ERROR: Cannot initialize conda. Please run: conda init bash"
    exit 1
}

# 配置 — 根据你的实际路径修改
MODEL_PATH="${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/xx_llms/quick_test_output}"
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

mkdir -p "$OUTPUT_ROOT/logs"

# -------------------------------------------------------------------------
# Stage 1: Quantization (dq_xx 环境)
# -------------------------------------------------------------------------
echo "[1/4] Stage 1: Quantization"
echo "--------------------------------------------"
conda activate dq_xx
cd /data1/xx/LGRQuant_v2

python -m lgrquant.stage1.quantize \
    --model "$MODEL_PATH" \
    --dataset wikitext2 \
    --true-sequential \
    --act-order \
    --new-eval \
    --wbits 2 \
    --group-size 64 \
    --nsamples $NSAMPLES \
    --max-iter-num 4 \
    --iters-before-round 200 \
    --inner-iters-for-round 5 \
    --round-fn gptq \
    --blockwise-minimize-epoch 4 \
    --blockwise-minimize-lr 1.0e-5 \
    --finetune-scale-mode layerwise \
    --save \
    --save_true_only \
    --out_path "$OUTPUT_ROOT/stage1" \
    2>&1 | tee "$OUTPUT_ROOT/logs/stage1.log"

echo "✓ Stage 1 completed"
echo ""

# -------------------------------------------------------------------------
# Precompute: Teacher Logits Caching (abq-llm 环境)
# -------------------------------------------------------------------------
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
    2>&1 | tee "$OUTPUT_ROOT/logs/precompute.log"

echo "✓ Precompute completed"
echo ""

# 清理 CUDA 缓存（Teacher 模型释放产生的碎片）
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || true

# -------------------------------------------------------------------------
# Stage 2: E2E Distillation (abq-llm 环境)
# -------------------------------------------------------------------------
echo "[3/4] Stage 2: E2E Distillation"
echo "--------------------------------------------"
python -m lgrquant.stage2.quant_finetune \
    --model_id "$MODEL_PATH" \
    --true_quant_path "$OUTPUT_ROOT/stage1/quantized_model.pth" \
    --teacher_topk_dir "$OUTPUT_ROOT/teacher_cache" \
    --out_path "$OUTPUT_ROOT/stage2" \
    --train_steps $TRAIN_STEPS \
    --group_size 64 \
    --asym \
    --use_distill \
    --distill_loss kl_top \
    --dataset wikitext2 \
    --nsamples $NSAMPLES \
    2>&1 | tee "$OUTPUT_ROOT/logs/stage2.log"

echo "✓ Stage 2 completed"
echo ""

# -------------------------------------------------------------------------
# Inference: Evaluation (dq_xx 环境)
# -------------------------------------------------------------------------
echo "[4/4] Inference: Evaluation"
echo "--------------------------------------------"
conda activate dq_xx

python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path "$MODEL_PATH" \
    --true_quant_path "$OUTPUT_ROOT/stage2/quantized_model.pth" \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    --prompt "Artificial intelligence is" \
    2>&1 | tee "$OUTPUT_ROOT/logs/inference.log"

echo "✓ Inference completed"
echo ""

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo "============================================"
echo "Quick Test Completed Successfully!"
echo "============================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_ROOT/"
echo ""
echo "Logs:"
ls -lh "$OUTPUT_ROOT/logs/"
echo ""
echo "To clean up: rm -rf $OUTPUT_ROOT"
