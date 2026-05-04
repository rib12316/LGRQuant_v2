#!/bin/bash
# =====================================================================
# LGRQuant_v2 CUDA 内核一键编译脚本
# 编译 W2 (decoupleQ) 和 W4 (marlin) 内核
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="/tmp/lgrquant_build_$$"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ---------------------------------------------------------------------------
# Build W2 kernel (decoupleQ)
# ---------------------------------------------------------------------------
build_w2_kernel() {
    log_info "Building W2 CUDA kernel (decoupleQ)..."
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Clone ByteDance-Seed/decoupleQ if not exists
    if [ ! -d "decoupleQ" ]; then
        log_info "Cloning ByteDance-Seed/decoupleQ..."
        git clone https://github.com/ByteDance-Seed/decoupleQ.git
    fi
    
    cd decoupleQ
    
    # Pull submodules (cutlass + TensorRT-LLM dependencies)
    log_info "Initializing submodules..."
    git submodule update --init
    
    # Build
    log_info "Building W2 kernel..."
    # Note: This requires proper CMake and CUDA setup
    # The actual build steps may vary based on the decoupleQ repository structure
    
    if [ -f "build.sh" ]; then
        # Edit build.sh to match environment (if needed)
        # export TORCH_CUDA_ARCH_LIST="8.0+PTX"
        bash build.sh
    else
        log_warn "build.sh not found, trying cmake directly..."
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc)
    fi
    
    # Copy compiled kernel to project
    if [ -f "decoupleQ_kernels.so" ]; then
        cp decoupleQ_kernels.so "$PROJECT_ROOT/lgrquant/core/"
        log_info "W2 kernel installed to lgrquant/core/decoupleQ_kernels.so"
    else
        log_error "W2 kernel build failed - decoupleQ_kernels.so not found"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
}

# ---------------------------------------------------------------------------
# Build W4 kernel (marlin)
# ---------------------------------------------------------------------------
build_w4_kernel() {
    log_info "Building W4 CUDA kernel (marlin)..."
    
    cd "$PROJECT_ROOT/third_party/marlin_ist"
    
    if [ ! -f "setup.py" ]; then
        log_error "marlin_ist/setup.py not found"
        exit 1
    fi
    
    # Build and install
    pip install -e . -v
    
    log_info "W4 kernel (marlin) installed successfully"
    
    cd "$PROJECT_ROOT"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_info "LGRQuant_v2 CUDA Kernel Build Script"
    log_info "====================================="
    log_info "Project root: $PROJECT_ROOT"
    
    # Check Python and PyTorch
    if ! python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_error "PyTorch not found. Please activate your conda environment first."
        exit 1
    fi
    
    # Check CUDA
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_warn "CUDA not available. Kernels will be built for CPU only (if supported)."
    fi
    
    # Parse arguments
    BUILD_W2=true
    BUILD_W4=true
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --w2-only)
                BUILD_W4=false
                shift
                ;;
            --w4-only)
                BUILD_W2=false
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --w2-only    Build only W2 (decoupleQ) kernel"
                echo "  --w4-only    Build only W4 (marlin) kernel"
                echo "  --help, -h   Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Build kernels
    if [ "$BUILD_W2" = true ]; then
        build_w2_kernel
    fi
    
    if [ "$BUILD_W4" = true ]; then
        build_w4_kernel
    fi
    
    # Cleanup
    rm -rf "$BUILD_DIR"
    
    log_info "====================================="
    log_info "Build completed successfully!"
    log_info ""
    log_info "To verify the installation, run:"
    log_info "  python -c 'from lgrquant.core.linear_w2a16 import LinearW2A16; print(\"W2 OK\")'"
    log_info "  python -c 'from lgrquant.core.linear_w4a16 import LinearW4A16; print(\"W4 OK\")'"
}

main "$@"
