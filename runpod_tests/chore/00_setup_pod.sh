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

# Create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate and install Python deps
source .venv/bin/activate
pip install -q --upgrade pip

# Core ML stack
pip install -q \
    'torch>=2.4.0' \
    sentencepiece \
    numpy

# Data + tokenizer
pip install -q \
    huggingface_hub \
    datasets \
    tqdm

# Compression (for artifact packing)
pip install -q \
    zstandard \
    brotli

# DC500 categories (KMeans clustering)
pip install -q scikit-learn

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

echo
echo "✓ Pod setup complete"
