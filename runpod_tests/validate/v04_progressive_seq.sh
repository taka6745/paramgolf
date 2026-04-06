#!/bin/bash
# v04_progressive_seq.sh — Verify progressive seq scheduler triggers correctly
# Hardware: any GPU with ≥10GB
# Time: ~30 sec on 3080 Ti
#
# This test EXPECTS train_gpt.py to be patched with progressive seq support.
# The patch script (chore/08) doesn't add progressive seq yet, so this test
# will report SKIPPED instead of FAIL until that patch lands.
#
# When the patch is applied, this test runs train_gpt.py with PROGRESSIVE_SEQ=1
# and a SHORT wallclock and verifies the phase transition log message appears.

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== V04: PROGRESSIVE SEQ PHASE TRANSITION ==="
echo

# Skip cleanly if train_gpt.py doesn't support PROGRESSIVE_SEQ env var yet
if ! grep -q 'PROGRESSIVE_SEQ' train_gpt.py 2>/dev/null; then
    echo "△ SKIP: train_gpt.py doesn't have progressive seq patches yet"
    echo "  This test will pass once chore/08 (or a follow-up) adds progressive_seq support."
    echo "  No-op for now."
    exit 0
fi

echo "Running 30s with progressive seq (25s @ seq=128, 5s @ seq=1024)..."

PROGRESSIVE_SEQ=1 \
PHASE1_SEQ_LEN=128 \
PHASE1_LR_MULT=25.0 \
PHASE1_FRACTION=0.85 \
PHASE2_SEQ_LEN=1024 \
PHASE1_NGRAM_WEIGHT=0.40 \
PHASE2_NGRAM_WEIGHT=0.05 \
\
ITERATIONS=10000 \
MAX_WALLCLOCK_SECONDS=30 \
TRAIN_LOG_EVERY=50 \
TRAIN_SEQ_LEN=128 \
TRAIN_BATCH_TOKENS=65536 \
GRAD_ACCUM_STEPS=1 \
VAL_BATCH_SIZE=131072 \
VAL_LOSS_EVERY=0 SKIP_FINAL_EVAL=1 \
WARMUP_STEPS=10 \
python3 train_gpt.py 2>&1 | tee runpod_tests/logs/v04_progressive.log

echo
echo "=== VALIDATION ==="

if grep -q "PHASE TRANSITION" runpod_tests/logs/v04_progressive.log; then
    echo "✓ Phase transition triggered"
else
    echo "✗ Phase transition NOT triggered (check Patch C is applied)"
    exit 1
fi

if grep -q 'nan\|NaN\|inf\|Inf' runpod_tests/logs/v04_progressive.log; then
    echo "✗ FAIL: NaN/Inf detected"
    exit 1
fi

INITIAL_LOSS=$(grep 'step:1\b' runpod_tests/logs/v04_progressive.log | grep -oE 'train_loss:[0-9.]+' | head -1 | cut -d: -f2)
FINAL_LOSS=$(grep 'step:' runpod_tests/logs/v04_progressive.log | grep -oE 'train_loss:[0-9.]+' | tail -1 | cut -d: -f2)
echo "  Loss: $INITIAL_LOSS → $FINAL_LOSS"

if [ -n "$FINAL_LOSS" ] && (( $(echo "$FINAL_LOSS < $INITIAL_LOSS" | bc -l) )); then
    echo "✓ PASS: progressive seq runs without errors, loss decreases"
    exit 0
else
    echo "✗ FAIL: loss did not decrease"
    exit 1
fi
