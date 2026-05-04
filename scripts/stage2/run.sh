#!/bin/bash
# LGRQuant Stage 2: End-to-End Distillation Fine-tuning
# Environment: abq-llm (PyTorch 2.4+)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/stage2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 2: E2E Distillation"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

python -m lgrquant.stage2.quant_finetune \
    --model_id "${MODEL_ID:-/data2/llms/Qwen2.5-7B-Instruct}" \
    --true_quant_path "${STAGE1_OUTPUT:-./outputs/stage1}/true_quant.pth" \
    --teacher_topk_dir "${PRECOMPUTE_OUTPUT:-./outputs/teacher_topk_cache}" \
    --out_path "${OUTPUT_ROOT:-./outputs}/stage2" \
    --train_steps 200 \
    --group_size 64 \
    --asym \
    --use_distill \
    --distill_loss kl_top \
    --dataset wikitext2 \
    --nsamples 1000 \
    "$@"

echo ""
echo "Stage 2 completed!"
echo "Output: ${OUTPUT_ROOT:-./outputs}/stage2"
echo "End time: $(date)"
