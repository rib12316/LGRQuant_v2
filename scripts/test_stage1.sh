#!/bin/bash
# Test Stage 1 only

set -e

MODEL="${1:-/data2/llms/Qwen2.5-3B}"
NSAMPLES="${2:-32}"
OUT="${3:-/tmp/lgrquant_test/stage1}"

echo "Testing Stage 1..."
echo "Model: $MODEL"
echo "Samples: $NSAMPLES"
echo "Output: $OUT"

conda activate dq_xx
cd /data1/xx/LGRQuant_v2

python -m lgrquant.stage1.quantize \
    --model "$MODEL" \
    --dataset wikitext2 \
    --wbits 2 \
    --group-size 64 \
    --save \
    --nsamples $NSAMPLES \
    --out_path "$OUT"

echo "✓ Stage 1 test passed!"
echo "Output: $OUT/"
ls -lh "$OUT/"
