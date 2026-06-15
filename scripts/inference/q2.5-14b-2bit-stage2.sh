#!/usr/bin/env bash
# Qwen2.5-14B 2-bit (stage2 quantizers) 真量化推理
set -e

eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

LOG_FILE="${SCRIPT_DIR}/../../logs/q2.5-14b-2bit-stage2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "LGRQuant Inference: Qwen2.5-14B W2A16 (stage2)"
echo "=========================================="
echo "Model:      /data2/llms/qwen-2.5-14b"
echo "Checkpoint: /data2/xx_llms/LGRQuant_fast/qwen-2.5-14b/w2/stage2/quantizers.pth"
echo "Start:      $(date)"
echo "Log:        $LOG_FILE"
echo ""

conda activate dq_xx
cd "$PROJECT_ROOT"

python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path /data2/llms/qwen-2.5-14b \
    --quantizers_path /data2/xx_llms/LGRQuant_fast/qwen-2.5-14b/w2/stage2/quantizers.pth \
    --group_size 64 \
    --asym \
    --prompt "who are you?" \
    --max_new_tokens 128 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --ppl_datasets wikitext2 \
    --lm_eval_tasks "piqa,hellaswag,winogrande,boolq" \
    --lm_eval_batch_size 16 \
    --benchmark \
    --benchmark_repeats 10 \
    --benchmark_new_tokens 512 \
    --benchmark_prefill \
    --prefill_lens "512,768,1024" \
    --benchmark_throughput \
    --throughput_batch_sizes "1,2,4,8" \
    --throughput_seq_len 1024

echo ""
echo "Inference completed!"
echo "End: $(date)"

# ── Generate Results Summary Table ──
python "$SCRIPT_DIR/generate_summary.py" "$LOG_FILE" "Qwen2.5-14B" "W2A16"
