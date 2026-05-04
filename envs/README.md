# Environment Setup

## dq_xx (Stage 1 & Inference)

```bash
conda create -n dq_xx python=3.9
conda activate dq_xx
pip install -r dq_xx/requirements.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## abq_llm (Stage 2)

```bash
conda create -n abq_llm python=3.10
conda activate abq_llm
pip install -r abq_llm/requirements.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- `dq_xx` uses PyTorch 2.1 with CUDA 11.8 (compatible with decoupleQ W2 kernel)
- `abq_llm` uses PyTorch 2.4+ with CUDA 12.1 (for FSDP support in Stage 2)
- CUDA kernels must be compiled with matching PyTorch/CUDA versions
