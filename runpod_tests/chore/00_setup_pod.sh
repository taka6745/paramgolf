#!/bin/bash
# 00_setup_pod.sh — Install deps and clone repo
# Runs on: any GPU pod (3060 or H100)
# Time: ~2 min

set -e

echo "=== POD SETUP ==="
echo

# Update system + install everything we need
# git/curl/wget = repo clone, downloads
# htop = monitoring
# bc = floating-point math in shell scripts (used by validate/v0*.sh and unknown/u0*.sh)
# build-essential = sometimes needed for pip wheels
apt-get update -qq && apt-get install -y -qq \
    git curl wget htop bc \
    build-essential \
    ca-certificates

# Clone the repo
cd /workspace
if [ ! -d "paramgolf" ]; then
    git clone https://github.com/taka6745/paramgolf.git
fi
cd paramgolf

# CRITICAL: do NOT install torch via pip if the system already has a working one.
# RunPod base images come with torch + matching CUDA driver pre-installed.
# Installing a newer torch from pip will overwrite it with a version that
# expects a newer driver than the pod has, breaking CUDA initialization.

# If a previous run created a broken venv that overwrites system torch, nuke it
if [ -d ".venv" ]; then
    echo "Found existing venv — checking if it works..."
    if ! .venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  ✗ existing venv has broken torch (CUDA init fails)"
        echo "  removing .venv to fall back to system torch..."
        rm -rf .venv
    else
        echo "  ✓ existing venv works"
    fi
fi

echo "Checking pre-installed torch + CUDA..."
TORCH_OK=0
if python3 -c "import torch; assert torch.cuda.is_available(); torch.cuda.get_device_name(0)" 2>/dev/null; then
    echo "  ✓ system torch works with CUDA — using pre-installed torch"
    TORCH_OK=1
    PIP="pip install --progress-bar off"
else
    echo "  system torch missing or broken — installing fresh"
    # Detect CUDA version to pick the right torch wheel
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "12.1")
    echo "  detected CUDA version: $CUDA_VER"
    # Map to PyTorch wheel index
    case "$CUDA_VER" in
        12.4*|12.5*|12.6*|12.7*|12.8*) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        12.1*|12.2*|12.3*)             TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        11.8*|11.9*|12.0*)             TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)                             TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
    echo "  using torch index: $TORCH_INDEX"
    PIP="pip install --progress-bar off"

    if [ ! -d ".venv" ]; then
        python3 -m venv .venv --system-site-packages  # inherit system torch as fallback
    fi
    source .venv/bin/activate
    $PIP --upgrade pip
    echo "Installing torch (matching CUDA $CUDA_VER, ~2 GB)..."
    $PIP torch --index-url "$TORCH_INDEX"
fi

# Whether or not torch was installed, install everything else with the same pip
echo
echo "Installing additional Python deps (sentencepiece, numpy, datasets, etc.)..."
$PIP \
    sentencepiece \
    numpy \
    huggingface_hub \
    datasets \
    tqdm \
    zstandard \
    brotli \
    scikit-learn

# Verify GPU
echo
echo "Verifying GPU..."
python3 -c "import torch; print(f'  torch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

echo
echo "✓ Pod setup complete"
