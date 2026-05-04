#!/bin/bash
# LGRQuant Full Pipeline: Stage1 -> Precompute -> Stage2 -> Inference

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

CONFIG="${1:-configs/w2.yaml}"

echo "=========================================="
echo "LGRQuant Full Pipeline"
echo "=========================================="
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo ""

# Stage 1
echo "[Pipeline] Stage 1: ARQ Quantization"
bash "$SCRIPT_DIR/../stage1/run.sh" "$CONFIG"

# Precompute (Stage 2 preparation)
echo ""
echo "[Pipeline] Precompute: Teacher Top-K Logits Caching"
bash "$SCRIPT_DIR/../stage2/precompute.sh" "$CONFIG"

# Stage 2
echo ""
echo "[Pipeline] Stage 2: E2E Distillation"
bash "$SCRIPT_DIR/../stage2/run.sh" "$CONFIG"

# Inference
echo ""
echo "[Pipeline] Inference: W2 Evaluation"
bash "$SCRIPT_DIR/../inference/run_w2.sh" "$CONFIG"

echo ""
echo "=========================================="
echo "Full Pipeline Completed!"
echo "End time: $(date)"
