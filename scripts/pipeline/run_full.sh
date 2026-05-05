#!/bin/bash
# LGRQuant Full Pipeline: Stage1 -> Precompute -> Stage2 -> Inference
# 用法: MODEL_PATH=/data2/llms/Qwen2.5-7B-Instruct OUTPUT_ROOT=/data2/xx_llms/output bash scripts/pipeline/run_full.sh
# 所有配置通过环境变量传入，各子脚本自带 conda 初始化和默认值

set -e

# CUDA 内存优化
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 环境变量默认值（可由调用者覆盖）
export MODEL_PATH="${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/data2/xx_llms/pipeline_output}"
export NSAMPLES="${NSAMPLES:-128}"
export TRAIN_STEPS="${TRAIN_STEPS:-200}"
export WBITS="${WBITS:-2}"
export GROUP_SIZE="${GROUP_SIZE:-64}"

# 各阶段输出路径
STAGE1_OUT="${OUTPUT_ROOT}/stage1"
STAGE2_OUT="${OUTPUT_ROOT}/stage2"
CACHE_DIR="${OUTPUT_ROOT}/teacher_cache"

mkdir -p "$OUTPUT_ROOT"

echo "=========================================="
echo "LGRQuant Full Pipeline"
echo "=========================================="
echo "Model:    $MODEL_PATH"
echo "Output:   $OUTPUT_ROOT"
echo "Wbits:    $WBITS  GroupSize: $GROUP_SIZE"
echo "Samples:  $NSAMPLES  Steps: $TRAIN_STEPS"
echo "Start:    $(date)"
echo ""

# ---- Stage 1: Quantization (dq_xx) ----
echo "[Pipeline] Stage 1: ARQ Quantization"
echo "----------------------------------------"
STAGE1_OUTPUT="$STAGE1_OUT" \
    bash "$SCRIPT_DIR/../stage1/run.sh"

# ---- Precompute: Teacher Logits Cache (abq-llm) ----
echo ""
echo "[Pipeline] Precompute: Teacher Top-K Logits Caching"
echo "----------------------------------------"
PRECOMPUTE_OUT="$CACHE_DIR" \
    bash "$SCRIPT_DIR/../stage2/precompute.sh"

# 清理 CUDA 缓存（Teacher 模型释放产生的碎片）
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')" 2>/dev/null || true

# ---- Stage 2: E2E Distillation (abq-llm) ----
echo ""
echo "[Pipeline] Stage 2: E2E Distillation"
echo "----------------------------------------"
STAGE1_OUTPUT="$STAGE1_OUT" PRECOMPUTE_OUT="$CACHE_DIR" STAGE2_OUTPUT="$STAGE2_OUT" \
    bash "$SCRIPT_DIR/../stage2/run.sh"

# 清理 CUDA 缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ---- Inference: Evaluation (dq_xx) ----
echo ""
echo "[Pipeline] Inference: W2 Evaluation"
echo "----------------------------------------"
QUANT_PATH="$STAGE2_OUT/quantized_model.pth" \
    bash "$SCRIPT_DIR/../inference/run_w2.sh"

echo ""
echo "=========================================="
echo "Full Pipeline Completed!"
echo "=========================================="
echo "Output: $OUTPUT_ROOT"
echo "End:    $(date)"
echo ""
ls -lh "$STAGE1_OUT/" "$STAGE2_OUT/" 2>/dev/null || true