# W2 vs W4 Configuration Guide

## Overview

| Feature | W2 (2-bit) | W4 (4-bit) |
|---------|-----------|-----------|
| Bits per weight | 2 | 4 |
| Group size | 64 (typical) | -1 (per-channel) |
| Symmetric | No (asymmetric) | Yes |
| Kernel | decoupleQ (custom) | Marlin |
| Accuracy | Lower | Higher |
| Speed | Slower | Faster (Marlin optimized) |

## Configuration

### W2 (configs/w2.yaml)

```yaml
bits: 2
group_size: 64
sym: false      # asymmetric: uses zero-point
asym: true
```

**When to use W2:**
- Maximum compression needed
- Can tolerate some accuracy loss
- Model will be deployed on memory-constrained devices

### W4 (configs/w4.yaml)

```yaml
bits: 4
group_size: -1  # per-channel quantization
sym: true       # symmetric: no zero-point
asym: false
```

**When to use W4:**
- Balanced compression and accuracy
- Production deployment where accuracy matters
- Using Marlin for fast inference

## Key Differences

### Group Size

- **W2**: `group_size=64` means weights are grouped in chunks of 64 for quantization
- **W4**: `group_size=-1` means per-channel quantization (one scale per output channel)

### Symmetry

- **W2 (asymmetric)**: Uses both scale and zero-point
  - `quantized = round((original - zero) / scale)`
- **W4 (symmetric)**: Uses scale only, zero-point is 0
  - `quantized = round(original / scale)`

### CUDA Kernels

- **W2**: Custom kernel from ByteDance-Seed/decoupleQ
- **W4**: Marlin kernel from IST-DASLab/marlin

## Pipeline Differences

The pipeline commands are the same, just change the config:

```bash
# W2 quantization
bash scripts/pipeline/run_full.sh configs/w2.yaml

# W4 quantization
bash scripts/pipeline/run_full.sh configs/w4.yaml

# W2 inference
bash scripts/inference/run_w2.sh

# W4 inference
bash scripts/inference/run_w4.sh
```

## Accuracy vs Compression

| Model | FP16 | W4 | W2 |
|-------|------|----|----|
| Qwen2.5-7B | 100% | ~98% | ~95% |
| Qwen2.5-14B | 100% | ~98% | ~94% |

*Note: Exact values depend on calibration data and training steps.*
