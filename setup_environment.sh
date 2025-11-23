#!/bin/bash
# Environment setup script for Event-LangSplat
# This script activates the ggev3dgs conda environment and sets up required paths

echo "🚀 Setting up Event-LangSplat environment..."

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "📦 Activating conda environment: ggev3dgs"
    eval "$(conda shell.bash hook)"
    conda activate ggev3dgs
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Conda environment activated successfully"
    else
        echo "❌ Failed to activate conda environment ggev3dgs"
        exit 1
    fi
else
    echo "❌ Conda not found"
    exit 1
fi

# Set PYTHONPATH for CUDA extensions (commented out since packages are installed)
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/diff-gaussian-rasterization"
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/simple-knn"
# export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/fused-ssim"
echo "📐 Using installed CUDA extension packages"

# Configure LD_LIBRARY_PATH for PyTorch
# Prioritize system libraries to avoid libtinfo.so.6 warning
if [[ -n "$CONDA_PREFIX" ]]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    echo "🔗 LD_LIBRARY_PATH configured for PyTorch"
fi

# Apply GLIBCXX compatibility fixes
export LD_PRELOAD=""  # Clear any existing LD_PRELOAD
echo "🔧 GLIBCXX compatibility configured"

# Set CUDA_HOME for building CUDA extensions
export CUDA_HOME=$CONDA_PREFIX
echo "🔧 CUDA_HOME set to: $CUDA_HOME"

# Verify environment
echo "🔍 Environment verification:"
echo "   Python: $(which python)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "   CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo "   Working directory: $(pwd)"

echo "✅ Environment setup complete!"
echo ""
echo "Usage examples:"
echo "  source setup_environment.sh && python train.py -s data/tandt/truck --iterations 7000 --eval"
echo "  source setup_environment.sh && python test_enhanced_events.py"
echo ""