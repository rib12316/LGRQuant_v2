#!/bin/bash
# LGRQuant Inference: W4 (4-bit) Model
# Environment: dq_xx (PyTorch 2.1)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/inference_w4_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Inference: W4 (4-bit)"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

python -m lgrquant.inference.inference_test \
    --kernel w4 \
    --model_path "${MODEL_PATH:-/data2/llms/Qwen2.5-7B-Instruct}" \
    --quantizers_path "${QUANT_PATH:-./outputs/stage2/quantizers.pth}" \
    --ppl_datasets wikitext2 \
    "$@"

echo ""
echo "Inference completed!"
echo "End time: $(date)"
