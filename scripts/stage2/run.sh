#!/bin/bash
# LGRQuant Stage 2: End-to-End Distillation Fine-tuning
# Environment: abq-llm (PyTorch 2.4+)

set -e

# conda 初始化
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/stage2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 2: E2E Distillation"
echo "=========================================="
echo "Model:    ${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
echo "Stage1:   ${STAGE1_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage1}"
echo "Cache:    ${PRECOMPUTE_OUT:-${OUTPUT_ROOT:-./outputs}/teacher_topk_cache}"
echo "Output:   ${STAGE2_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage2}"
echo "Start:    $(date)"
echo "Log:      $LOG_FILE"
echo ""

conda activate abq-llm
cd "$PROJECT_ROOT"

python -m lgrquant.stage2.quant_finetune \
    --model_id "${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}" \
    --true_quant_path "${STAGE1_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage1}/quantized_model.pth" \
    --teacher_topk_dir "${PRECOMPUTE_OUT:-${OUTPUT_ROOT:-./outputs}/teacher_topk_cache}" \
    --out_path "${STAGE2_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage2}" \
    --train_steps "${TRAIN_STEPS:-200}" \
    --group_size "${GROUP_SIZE:-64}" \
    --asym \
    --use_distill \
    --distill_loss kl_top \
    --dataset wikitext2 \
    --nsamples "${NSAMPLES:-128}"

echo ""
echo "Stage 2 completed!"
echo "Output: ${STAGE2_OUTPUT:-${OUTPUT_ROOT:-./outputs}/stage2}"
echo "End:    $(date)"
