#!/bin/bash
# LGRQuant Stage 1: ARQ Quantization
# Environment: dq_xx (PyTorch 2.1)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse arguments
CONFIG="${1:-configs/w2.yaml}"

# Setup logging
LOG_FILE="logs/stage1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 1: ARQ Quantization"
echo "=========================================="
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Load config values (requires yq or python)
# For now, use default values
python -m lgrquant.stage1.quantize \
    --model "${MODEL_PATH:-/data2/llms/Qwen2.5-7B-Instruct}" \
    --dataset wikitext2 \
    --wbits 2 \
    --groupsize 64 \
    --save \
    --out_path "${OUTPUT_ROOT:-./outputs}/stage1" \
    "$@"

echo ""
echo "Stage 1 completed!"
echo "Output: ${OUTPUT_ROOT:-./outputs}/stage1"
echo "End time: $(date)"
