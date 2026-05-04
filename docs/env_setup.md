# Environment Setup Guide

LGRQuant_v2 requires two separate conda environments due to PyTorch ABI compatibility.

## Environment 1: dq_xx (Stage 1 & Inference)

Used for:
- Stage 1 quantization (ARQ)
- Inference with W2/W4 kernels

```bash
conda create -n dq_xx python=3.9
conda activate dq_xx
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.37.0
pip install lm-eval==0.4.4
pip install datasets
pip install accelerate
pip install sentencepiece
pip install protobuf
```

## Environment 2: abq_llm (Stage 2)

Used for:
- Stage 2 end-to-end distillation
- Precompute teacher logits

```bash
conda create -n abq-llm python=3.10
conda activate abq-llm
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.40.0
pip install peft==0.11.0
pip install datasets
pip install accelerate
pip install sentencepiece
pip install protobuf
```

## Verify Installation

### dq_xx

```bash
conda activate dq_xx
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA available: True
```

### abq_llm

```bash
conda activate abq-llm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.4.0+cu121
CUDA available: True
```

## Why Two Environments?

The CUDA kernels (`decoupleQ_kernels.so` for W2, Marlin for W4) are compiled against specific PyTorch ABI versions:

- **W2 kernel**: Compiled with PyTorch 2.1, requires `dq_xx`
- **Marlin W4 kernel**: Can be compiled with either version, but must match the runtime PyTorch version

The `abq-llm` environment uses PyTorch 2.4+ which supports FSDP (Fully Sharded Data Parallel) for efficient Stage 2 training.

## Switching Between Environments

The pipeline scripts automatically handle environment switching:

```bash
# In pipeline scripts
conda activate dq_xx
python -m lgrquant.stage1.llama ...  # Stage 1 uses dq_xx

conda activate abq-llm
python -m lgrquant.stage2.quant_finetune ...  # Stage 2 uses abq_llm
```
