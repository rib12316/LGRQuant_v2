#!/bin/bash
# LGRQuant Stage 2 Precompute: Teacher Top-K Logits Caching
# Environment: abq-llm (PyTorch 2.4+)
# 必须先于 Stage 2 运行！

set -e

# conda 初始化
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

LOG_FILE="logs/precompute_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Stage 2 Precompute"
echo "=========================================="
echo "Model:    ${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}"
echo "Cache:    ${PRECOMPUTE_OUT:-${OUTPUT_ROOT:-./outputs}/teacher_topk_cache}"
echo "Start:    $(date)"
echo "Log:      $LOG_FILE"
echo ""

conda activate abq-llm
cd "$PROJECT_ROOT"

python -m lgrquant.stage2.precompute_teacher_topk \
    --model_id "${MODEL_PATH:-/data2/llms/Qwen2.5-3B-Instruct}" \
    --dataset wikitext2 \
    --nsamples "${S2_NSAMPLES:-1000}" \
    --seqlen 2048 \
    --top_k 1000 \
    --out_dir "${PRECOMPUTE_OUT:-${OUTPUT_ROOT:-./outputs}/teacher_topk_cache}"

echo ""
echo "Precompute completed!"
echo "Cache: ${PRECOMPUTE_OUT:-${OUTPUT_ROOT:-./outputs}/teacher_topk_cache}"
echo "End:    $(date)"
