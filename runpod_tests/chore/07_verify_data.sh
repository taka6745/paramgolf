#!/bin/bash
# 07_verify_data.sh — Sanity check chore outputs (SP-1024 path)
# Time: ~1 min

set -e
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

echo "=== VERIFY DATA ==="
echo

PASS=0
FAIL=0

check() {
    local name=$1
    local cmd=$2
    if eval "$cmd" > /dev/null 2>&1; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL+1))
    fi
}

echo "Data:"
check "SP-1024 train shards (>=1)" "[ \$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) -ge 1 ]"
check "SP-1024 val shard exists" "ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null"

echo
echo "N-gram tables (SP-1024):"
check "Bigram table" "[ -f data/bigram_tab_1024v.npy ]"
check "Bigram shape (2048, 1024)" "python3 -c 'import numpy; t=numpy.load(\"data/bigram_tab_1024v.npy\"); assert t.shape==(2048,1024), t.shape'"
check "Trigram table" "[ -f data/trigram_logprobs_1024v.npy ]"
check "4-gram table" "[ -f data/fourgram_logprobs_1024v.npy ]"

echo
echo "DC categories:"
check "DC500" "[ -f data/dist_cats_500_1024.npz ]"

echo
echo "Quantization:"
check "Lloyd-Max codebook" "[ -f data/lloyd_max_codebook_64.npy ]"
check "Codebook has 64 levels" "python3 -c 'import numpy; t=numpy.load(\"data/lloyd_max_codebook_64.npy\"); assert len(t)==64'"

echo
echo "GPU:"
check "CUDA available" "python3 -c 'import torch; assert torch.cuda.is_available()'"

echo
echo "==============="
echo "PASS: $PASS / $((PASS+FAIL))"
[ $FAIL -eq 0 ] && echo "✓ ALL CHORES COMPLETE" || echo "✗ $FAIL CHECKS FAILED — fix before proceeding"
exit $FAIL
