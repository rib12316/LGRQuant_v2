#!/bin/bash
# LGRQuant Inference: Llama Model
# Environment: dq_xx (PyTorch 2.1)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/inference_llama_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Inference: Llama"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Run inference with Llama model
python lgrquant/inference/inference_test.py \
    --model_path "${MODEL_PATH:-/data2/llms/llama-2-7b}" \
    --quantizers_path "${QUANT_PATH:-./outputs/stage2/quantizers.pth}" \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    "$@"

echo ""
echo "Inference completed!"
echo "End time: $(date)"
