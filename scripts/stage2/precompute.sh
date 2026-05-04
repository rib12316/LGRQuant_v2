#!/bin/bash
# LGRQuant Stage 2 Precompute: Teacher Top-K Logits Caching
# Environment: abq-llm (PyTorch 2.4+)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/precompute_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 2 Precompute"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Run precompute
python -m lgrquant.stage2.precompute_teacher_topk \
    --model_id "${MODEL_ID:-/data2/llms/Qwen2.5-7B-Instruct}" \
    --dataset wikitext2 \
    --nsamples 1000 \
    --seqlen 2048 \
    --top_k 1000 \
    --out_dir "${OUTPUT_ROOT:-./outputs}/teacher_topk_cache" \
    "$@"

echo ""
echo "Precompute completed!"
echo "End time: $(date)"
