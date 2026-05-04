# Configuration Files

This directory contains configuration files for LGRQuant_v2.

## Files

- `w2.yaml` - 2-bit quantization configuration
- `w4.yaml` - 4-bit quantization configuration
- `paths_template.yaml` - Template for system-specific paths

## Usage

1. Copy `paths_template.yaml` to `paths.yaml`:
   ```bash
   cp configs/paths_template.yaml configs/paths.yaml
   ```

2. Edit `paths.yaml` with your actual paths:
   - `model_path`: Path to your Hugging Face model
   - `output_root`: Where to save quantized models
   - `dq_xx_python`: Python executable for dq_xx environment
   - `abq_llm_python`: Python executable for abq-llm environment

3. Run scripts with the configuration:
   ```bash
   bash scripts/pipeline/run_full.sh --config configs/w2.yaml
   ```

## Customization

You can create custom configs by copying and modifying `w2.yaml` or `w4.yaml`:

```bash
cp configs/w2.yaml configs/w2_custom.yaml
# Edit w2_custom.yaml
```
