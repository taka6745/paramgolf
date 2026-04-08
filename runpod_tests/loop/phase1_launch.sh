#!/bin/bash
# Phase 1 launch script — runs on the H100 pod.
# Assumes tokenize has finished:
#   data/datasets/datasets/fineweb10B_sp8192/*.bin exist
#   data/datasets/tokenizers/fineweb_8192_bpe.model exists
# Fixes the nested path layout and kicks off a 600-second seed=42 run.

set -e
cd /workspace/paramgolf

echo "=== PHASE 1 LAUNCH START $(date -u +%H:%M:%SZ) ==="

# --- Fix path layout from --output-root=data/datasets ---
# The download script created:
#   data/datasets/datasets/fineweb10B_sp8192/*.bin
#   data/datasets/tokenizers/fineweb_8192_bpe.model
# But train_gpt_phase1.py (DATA_DIR=./data/) expects:
#   data/datasets/fineweb10B_sp8192/*.bin
#   data/tokenizers/fineweb_8192_bpe.model
#
# Use symlinks to bridge.

if [ ! -L data/datasets/fineweb10B_sp8192_link ] && [ -d data/datasets/datasets/fineweb10B_sp8192 ]; then
    # Remove any empty dir placeholder
    rmdir data/datasets/fineweb10B_sp8192 2>/dev/null || true
    ln -sfn datasets/fineweb10B_sp8192 data/datasets/fineweb10B_sp8192
fi

mkdir -p data/tokenizers
if [ ! -f data/tokenizers/fineweb_8192_bpe.model ] && [ -f data/datasets/tokenizers/fineweb_8192_bpe.model ]; then
    ln -sfn ../datasets/tokenizers/fineweb_8192_bpe.model data/tokenizers/fineweb_8192_bpe.model
fi

# --- Verify the paths exist and are readable ---
echo "=== VERIFY PATHS ==="
ls -la data/datasets/fineweb10B_sp8192/ | head -5
ls -la data/tokenizers/fineweb_8192_bpe.model
python3 -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='data/tokenizers/fineweb_8192_bpe.model')
print(f'tokenizer vocab_size: {sp.vocab_size()}')
assert sp.vocab_size() == 8192, f'expected 8192, got {sp.vocab_size()}'
print('OK')
"

# --- Count shards ---
TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "train_shards=$TRAIN_SHARDS val_shards=$VAL_SHARDS"

if [ "$TRAIN_SHARDS" -lt 1 ] || [ "$VAL_SHARDS" -lt 1 ]; then
    echo "ERROR: no shards found, tokenize didn't finish?"
    exit 1
fi

# --- SMOKE TEST (30 seconds) ---
echo "=== SMOKE TEST (30s wallclock) ==="
mkdir -p logs
SEED=42 \
MAX_WALLCLOCK_SECONDS=30 \
TTT_ENABLED=0 \
DATA_DIR=./data/ \
VOCAB_SIZE=8192 \
timeout 180 python3 -u train_gpt_phase1.py 2>&1 | tee logs/phase1_smoke_test.log | tail -40

if ! grep -q "val_loss" logs/phase1_smoke_test.log; then
    echo "SMOKE TEST FAILED: no val_loss in log"
    tail -50 logs/phase1_smoke_test.log
    exit 2
fi

echo "=== SMOKE TEST OK, launching full Shot 1 ==="

# --- FULL SHOT 1: 600s run with TTT enabled ---
echo "=== SHOT 1 START $(date -u +%H:%M:%SZ) ==="
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=1 \
DATA_DIR=./data/ \
VOCAB_SIZE=8192 \
python3 -u train_gpt_phase1.py 2>&1 | tee logs/phase1_shot1_seed42.log

echo "=== SHOT 1 COMPLETE $(date -u +%H:%M:%SZ) ==="
echo "=== RESULT ==="
grep "val_bpb" logs/phase1_shot1_seed42.log | tail -5
