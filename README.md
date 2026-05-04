# LGRQuant_v2

Low-bit (2-bit / 4-bit) Large Language Model Quantization Framework

## Overview

LGRQuant_v2 is a clean, modular quantization framework for LLMs, supporting 2-bit (W2A16) and 4-bit (W4A16) quantization. It implements a three-stage pipeline:

1. **Stage 1** (`dq_xx` env): ARQ (Alternating Rounding & Quantization) + Block-wise distillation
2. **Precompute** (`abq-llm` env): Teacher model top-K logits caching for efficient Stage 2
3. **Stage 2** (`abq-llm` env): End-to-end distillation fine-tuning
4. **Inference** (`dq_xx` env): Quantized model evaluation (PPL, Zero-shot)

## Hardware Requirements

- GPU: NVIDIA A100 / H100 (recommended) or RTX 3090/4090
- VRAM:
  - Stage 1 (W2, 7B): ~30GB
  - Stage 2 (W2, 7B): ~39GB
  - Stage 2 (W2, 14B): ~60GB (single GPU) or use FSDP for multi-GPU
- CUDA: 11.8+ (for kernel compilation)

## Quick Start

### 1. Clone and Setup

```bash
cd /data1/xx/LGRQuant_v2

# Create and activate conda environments
# (See env_setup.md for detailed instructions)
```

### 2. Compile CUDA Kernels

**IMPORTANT**: You MUST recompile the CUDA kernels for your specific PyTorch/CUDA version.

```bash
# One-command build for both W2 and W4 kernels
bash scripts/build_kernels.sh

# Verify installation
bash scripts/verify_setup.sh

# Or manually verify:
python -c "from lgrquant.core.linear_w2a16 import LinearW2A16; print('W2 OK')"
python -c "from lgrquant.core.linear_w4a16 import LinearW4A16; print('W4 OK')"
```

**Note**: The `decoupleQ_kernels.so` file in the repository is compiled for PyTorch 2.1 + CUDA 11.8 (dq_xx environment). If you get "undefined symbol" errors, you need to recompile with `bash scripts/build_kernels.sh`.

### 3. Configure Paths

```bash
cp configs/paths_template.yaml configs/paths.yaml
# Edit configs/paths.yaml with your actual paths
```

### 4. Run Pipeline

```bash
# Full pipeline (Stage1 -> Precompute -> Stage2 -> Inference)
bash scripts/pipeline/run_full.sh configs/w2.yaml

# Or run stages separately
bash scripts/stage1/run.sh configs/w2.yaml
bash scripts/stage2/precompute.sh
bash scripts/stage2/run.sh
bash scripts/inference/run_w2.sh
```

## Directory Structure

```
LGRQuant_v2/
├── lgrquant/              # Core Python package
│   ├── core/              # Quantization algorithms (decoupleQ)
│   ├── stage1/            # Stage 1: ARQ quantization
│   ├── stage2/            # Stage 2: E2E distillation
│   ├── inference/         # Inference and evaluation
│   ├── restorative/       # LoRA recovery (optional)
│   ├── data/              # Unified data loader
│   └── utils/             # Utilities (PPL calculation)
├── scripts/               # Shell scripts
│   ├── build_kernels.sh   # One-command kernel build
│   ├── stage1/
│   ├── stage2/
│   ├── inference/
│   └── pipeline/
├── configs/               # Configuration files
├── third_party/           # External dependencies
│   └── marlin_ist/        # Marlin W4 kernel (git submodule)
├── logs/                  # Log files (flat structure)
└── docs/                  # Documentation
```

## Environment Setup

See `docs/env_setup.md` for detailed environment setup instructions.

## Documentation

- `docs/architecture.md` - System architecture and design
- `docs/env_setup.md` - Environment setup guide
- `docs/w2_vs_w4.md` - W2 vs W4 configuration differences

## License

This project is built on top of OPTQ/GPTQ and other open-source projects. See individual files for SPDX license identifiers.

## Acknowledgments

- ByteDance-Seed/decoupleQ for the W2 quantization algorithm
- IST-DASLab/Marlin for the W4 CUDA kernels
- EleutherAI/lm-evaluation-harness for evaluation tools
