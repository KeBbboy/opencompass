#!/bin/bash
# Install KIVI CUDA extension for KV cache quantization

set -e  # Exit on error

echo "========================================="
echo "Installing KIVI CUDA Extension"
echo "========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_DIR="${SCRIPT_DIR}/quant"

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA version:"
nvcc --version

echo ""
echo "PyTorch CUDA version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" || {
    echo "ERROR: PyTorch not found or CUDA not available"
    exit 1
}

echo ""
echo "Building CUDA extension in: ${QUANT_DIR}"
cd "${QUANT_DIR}"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.egg-info

# Build and install
echo "Compiling CUDA kernels..."
python setup.py install

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "You can now use KIVI quantization by setting:"
echo "  method='full_KIVI'"
echo ""
