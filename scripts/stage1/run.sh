#!/bin/bash
# LGRQuant Stage 1: ARQ Quantization
# Environment: dq_xx (PyTorch 2.1)
# 可单独运行: bash scripts/stage1/run.sh
# 也可被 pipeline 调用: MODEL_PATH=... OUTPUT_ROOT=... bash scripts/stage1/run.sh

set -e

# conda 初始化
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Setup logging
LOG_FILE="logs/stage1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 1: ARQ Quantization"
echo "=========================================="
echo "Model:    ${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
echo "Output:   ${STAGE1_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage1}"
echo "Start:    $(date)"
echo "Log:      $LOG_FILE"
echo ""

conda activate dq_xx
cd "$PROJECT_ROOT"

python -m lgrquant.stage1.quantize \
    --model "${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}" \
    --dataset "${DATASET:-c4}" \
    --true-sequential \
    --act-order \
    --new-eval \
    --wbits "${WBITS:-2}" \
    --group-size "${GROUP_SIZE:-64}" \
    --nsamples "${NSAMPLES:-128}" \
    --max-iter-num 4 \
    --iters-before-round 200 \
    --inner-iters-for-round 5 \
    --round-fn gptq \
    --blockwise-minimize-epoch 4 \
    --blockwise-minimize-lr 1.0e-5 \
    --finetune-scale-mode layerwise \
    --save \
    --save_true_only \
    --out_path "${STAGE1_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage1}"

echo ""
echo "Stage 1 completed!"
echo "Output: ${STAGE1_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage1}"
echo "End: $(date)"
