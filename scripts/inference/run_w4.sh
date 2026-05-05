#!/bin/bash
# LGRQuant Inference: W4 (4-bit) Model
# Environment: dq_xx (PyTorch 2.1)

set -e

# conda 初始化
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/inference_w4_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Inference: W4 (4-bit)"
echo "=========================================="
echo "Model:    ${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
echo "Checkpoint: ${QUANT_PATH:-./outputs/stage2/quantized_model.pth}"
echo "Start:    $(date)"
echo "Log:      $LOG_FILE"
echo ""

conda activate dq_xx
cd "$PROJECT_ROOT"

python -m lgrquant.inference.inference_test \
    --kernel w4 \
    --model_path "${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}" \
    --true_quant_path "${QUANT_PATH:-./outputs/stage2/quantized_model.pth}" \
    --ppl_datasets wikitext2 \
    --prompt "Artificial intelligence is"

echo ""
echo "Inference completed!"
echo "End: $(date)"
