# LGRQuant_v2 Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LGRQuant_v2                             │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1 (dq_xx)  │  Stage 2 (abq_llm)  │  Inference (dq_xx)   │
│  ───────────────  │  ─────────────────  │  ─────────────────   │
│  ARQ Quantization │  E2E Distillation   │  Quantized Model     │
│  Block-wise       │  Teacher Top-K      │  PPL Evaluation      │
│  Calibration      │  Logits Cache       │  Zero-shot Tasks     │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Unified Checkpoint   │
         │   quantized_model.pth  │
         └──────────────────────┘
```

## Module Structure

### lgrquant.core

Core quantization algorithms and CUDA kernels.

```
lgrquant/core/
├── quant.py          # ARQ (Alternating Rounding & Quantization)
├── moq_quant.py      # Quantizer class, scale/zero calculation
├── linear_w2a16.py   # W2 linear layer (int2 weights, fp16 activation)
├── linear_w4a16.py   # W4 linear layer (int4 weights, fp16 activation)
└── decoupleQ_kernels.so  # Compiled W2 CUDA kernel
```

### lgrquant.stage1

Stage 1: Initial quantization using ARQ + block-wise distillation.

**Input**: FP16/BF16 pretrained model
**Output**: `quantized_model.pth` with quantized weights

Key steps:
1. Load pretrained model
2. Run ARQ quantization layer by layer
3. Block-wise distillation to refine weights
4. Save quantized checkpoint

### lgrquant.stage2

Stage 2: End-to-end distillation fine-tuning.

**Input**: Quantized model from Stage 1
**Output**: Refined quantized model

Key files:
- `quant_finetune.py`: Main training script
- `precompute_teacher_topk.py`: Teacher logits caching

### lgrquant.inference

Inference and evaluation.

**Input**: Quantized model
**Output**: PPL, generation samples, zero-shot task results

### lgrquant.data

Unified data loader supporting multiple datasets:
- wikitext2
- c4 / c4_new
- ptb / ptb_new
- pile
- red_pajama (for Stage 2)
- alpaca (for Stage 2)

## Data Flow

```
Stage 1:
  Raw Model ──► ARQ Quant ──► Block Distill ──► quantized_model.pth

Stage 2 Precompute:
  Raw Model ──► Teacher Forward ──► Top-K Logits ──► teacher_cache/

Stage 2:
  quantized_model.pth + teacher_cache/
      │
      ▼
  E2E Distillation ──► quantized_model.pth (refined)

Inference:
  quantized_model.pth ──► Load ──► Evaluate
```

## Key Design Decisions

### 1. Unified Data Loader

Merged `data_utils.py` and `datautils.py` into `lgrquant/data/loader.py`:
- Single source of truth for data loading
- Two function variants for backward compatibility:
  - `get_loaders()`: Modern API (takes tokenizer)
  - `get_loaders_legacy()`: Legacy API (creates tokenizer from model path)

### 2. Checkpoint Format

Uses unified checkpoint format:
```python
{
    "quantizers": {...},    # Quantizer objects
    "true_quant": {...},    # Quantized weights (state_dict)
    "meta": {...}           # Metadata
}
```

Benefits:
- Single file for portability
- Metadata tracking
- Backward compatible with separate files

### 3. Environment Separation

Two conda environments for ABI compatibility:
- `dq_xx`: PyTorch 2.1 for Stage 1 & Inference
- `abq_llm`: PyTorch 2.4+ for Stage 2

### 4. Flat Log Structure

All logs in `logs/*.log` with timestamp naming:
- `stage1_20260504_143022.log`
- `stage2_20260504_144200.log`
- `inference_20260504_145000.log`

## Extension Points

### Adding New Datasets

Edit `lgrquant/data/loader.py`:

```python
def get_loaders(...):
    if 'my_dataset' in name:
        return get_my_dataset(...)

def get_my_dataset(nsamples, seqlen, tokenizer, eval_mode=False):
    # Implementation
    pass
```

### Adding New Quantization Bits

1. Implement new linear layer in `lgrquant/core/linear_wXa16.py`
2. Add config in `configs/wX.yaml`
3. Update kernel compilation in `scripts/build_kernels.sh`

## References

- DecoupleQ: [ByteDance-Seed/decoupleQ](https://github.com/ByteDance-Seed/decoupleQ)
- Marlin: [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)
- GPTQ: [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
