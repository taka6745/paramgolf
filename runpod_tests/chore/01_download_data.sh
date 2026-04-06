#!/bin/bash
# 01_download_data.sh — Download FineWeb sp1024 shards (and tokenizer)
# Runs on: any GPU pod
# Time: ~8 min on first run, instant on subsequent (idempotent)

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== DOWNLOAD FINEWEB DATA ==="
echo

# Skip if already have enough shards
if [ -d "data/datasets/fineweb10B_sp1024" ]; then
    SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
    if [ "$SHARD_COUNT" -ge 10 ]; then
        echo "✓ Already have $SHARD_COUNT sp1024 train shards, skipping download"
        ls -la data/datasets/fineweb10B_sp1024/ | head -15
        exit 0
    fi
    echo "Found $SHARD_COUNT sp1024 shards (want ≥10), downloading more..."
fi

# Download SP-1024 baseline data (default tokenizer + pre-tokenized shards)
echo "Downloading 10 sp1024 train shards from HuggingFace..."
python3 data/cached_challenge_fineweb.py \
    --variant sp1024 \
    --train-shards 10

echo
echo "✓ Data downloaded:"
ls -la data/datasets/fineweb10B_sp1024/ | head -15
echo
echo "Tokenizer:"
ls -la data/tokenizers/ 2>/dev/null || echo "  (no tokenizers dir yet — bundled inside shards)"
