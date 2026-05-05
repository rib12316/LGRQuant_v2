#!/bin/bash
# Test Inference only (requires Stage 1 or 2 output)

set -e

MODEL="${1:-/data2/llms/Qwen2.5-3B}"
CKPT="${2:-/tmp/lgrquant_test/stage1/quantized_model.pth}"

echo "Testing Inference..."
echo "Model: $MODEL"
echo "Checkpoint: $CKPT"

conda activate dq_xx
cd /data1/xx/LGRQuant_v2

python -m lgrquant.inference.inference_test \
    --kernel w2 \
    --model_path "$MODEL" \
    --true_quant_path "$CKPT" \
    --group_size 64 \
    --asym \
    --ppl_datasets wikitext2 \
    --prompt "Artificial intelligence is"

echo "✓ Inference test passed!"
