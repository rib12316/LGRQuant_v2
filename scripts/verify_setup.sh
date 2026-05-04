#!/bin/bash
# LGRQuant_v2 Setup Verification Script
# Run this after environment setup and kernel compilation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "LGRQuant_v2 Setup Verification"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

cd "$PROJECT_ROOT"

# 1. Check Python version
echo "1. Python Environment"
echo "   Version: $(python --version 2>&1)"
echo "   Path: $(which python)"

# 2. Check PyTorch
echo ""
echo "2. PyTorch"
python -c "import torch; print(f'   Version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null && check_pass "PyTorch OK" || check_fail "PyTorch import failed"

# 3. Check core imports
echo ""
echo "3. Core Module Imports"
python -c "from lgrquant.core import quant; print('   quant module OK')" 2>/dev/null && check_pass "quant module" || check_fail "quant module"
python -c "from lgrquant.core import moq_quant; print('   moq_quant module OK')" 2>/dev/null && check_pass "moq_quant module" || check_fail "moq_quant module"

# 4. Check CUDA kernels (only in dq_xx env)
echo ""
echo "4. CUDA Kernels"
python -c "from lgrquant.core.linear_w2a16 import LinearW2A16; print('   W2 kernel OK')" 2>/dev/null && check_pass "W2 kernel (LinearW2A16)" || check_warn "W2 kernel - may need recompilation with 'bash scripts/build_kernels.sh --w2-only'"

python -c "from lgrquant.core.linear_w4a16 import LinearW4A16; print('   W4 kernel OK')" 2>/dev/null && check_pass "W4 kernel (LinearW4A16)" || check_warn "W4 kernel - may need recompilation with 'bash scripts/build_kernels.sh --w4-only'"

# 5. Check data loader
echo ""
echo "5. Data Loader"
python -c "from lgrquant.data import loader; print('   loader module OK')" 2>/dev/null && check_pass "data loader" || check_fail "data loader"

# 6. Check configs
echo ""
echo "6. Configuration Files"
[ -f "configs/w2.yaml" ] && check_pass "configs/w2.yaml" || check_fail "configs/w2.yaml missing"
[ -f "configs/w4.yaml" ] && check_pass "configs/w4.yaml" || check_fail "configs/w4.yaml missing"
[ -f "configs/paths_template.yaml" ] && check_pass "configs/paths_template.yaml" || check_fail "configs/paths_template.yaml missing"

if [ ! -f "configs/paths.yaml" ]; then
    check_warn "configs/paths.yaml not created (copy from paths_template.yaml and edit)"
fi

# 7. Check scripts
echo ""
echo "7. Executable Scripts"
[ -x "scripts/build_kernels.sh" ] && check_pass "scripts/build_kernels.sh" || check_fail "scripts/build_kernels.sh not executable"
[ -x "scripts/stage1/run.sh" ] && check_pass "scripts/stage1/run.sh" || check_fail "scripts/stage1/run.sh not executable"
[ -x "scripts/stage2/run.sh" ] && check_pass "scripts/stage2/run.sh" || check_fail "scripts/stage2/run.sh not executable"
[ -x "scripts/inference/run_w2.sh" ] && check_pass "scripts/inference/run_w2.sh" || check_fail "scripts/inference/run_w2.sh not executable"

echo ""
echo "================================"
echo "Verification Complete"
echo "================================"
echo ""
echo "Next steps:"
echo "1. If W2/W4 kernel checks failed, run: bash scripts/build_kernels.sh"
echo "2. Create configs/paths.yaml: cp configs/paths_template.yaml configs/paths.yaml"
echo "3. Edit configs/paths.yaml with your actual paths"
echo "4. Run full pipeline: bash scripts/pipeline/run_full.sh configs/w2.yaml"
