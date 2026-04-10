#!/bin/bash
# submission/dry_run.sh — H100 SUBMISSION DRY RUN
#
# Single-command launcher for the submission-grade config we want to test on H100.
# This is the FULL stack we'd ship as a real comp record. Targets PR #1493
# (merged leaderboard #1, val_bpb 1.0810 on 8×H100 SXM) plus our deltas:
#   PR #1493 stack (verified from openai/parameter-golf merged PR description):
#   - SP8192 vocab
#   - 3-Layer Depth Recurrence (NUM_LOOPS=2, LOOP_START=3, LOOP_END=5)
#   - Parallel Residuals (PR #1493 applies L7+; we apply all-layers — see notes)
#   - QK-Gain 5.25
#   - Legal Score-First TTT (TTT_ENABLED=1, TTT_LR=0.005, TTT_EPOCHS=3)
#   - EMA_DECAY=0.9965, WARMDOWN_FRAC=0.72, ENABLE_LOOPING_AT=0.35
#   - NO pre-quant TTT, NO SLOT, NO n-gram cache, NO ETLB (PR #1493 explicit)
#   Our deltas on top:
#   - NUM_LAYERS=6 + MLP_MULT=2 (CHAMP_D validated, val_bpb 1.39943 on 3090)
#   - MATRIX_BITS=8 (our int8 quant breakthrough — eliminates the int6 quant gap)
#   - USE_PARALLEL_MUON=1 (our batched Newton-Schulz speedup)
#   - max-autotune-no-cudagraphs compile mode + cudnn.benchmark
#   - 600s training cap, full 16 MB artifact target
#
# Usage on a fresh pod:
#   bash submission/bootstrap.sh   # FIRST do data setup (slow, one-time)
#   bash submission/dry_run.sh     # then this
#
# Or as a one-liner via curl on a fresh pod (after bootstrap completes):
#   curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/submission/dry_run.sh | bash
#
# Expected on 1×H100 PCIe: val_bpb ~1.10-1.20 (validates the projection)
# Expected on 8×H100 SXM:  val_bpb ~1.00-1.08 (potentially beats PR #1493 = 1.0810)

set -eu
REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
cd "$REPO_DIR"

echo "============================================================"
echo "[dry_run] SUBMISSION DRY RUN starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "  Stack: PR #1493 leaderboard #1 (1.0810) + our int8 quant + parallel muon"
echo "         + NUM_LAYERS=6 compute-efficient + max-autotune compile"
echo "============================================================"

# === Submission-grade config ===
export SEED=42
export MAX_WALLCLOCK_SECONDS=600

# Model architecture — CHAMP_D validated config (val_bpb 1.39943 on 3090, 600s)
export NUM_LAYERS=6                         # was 8 — reverted to CHAMP_D validated
export MLP_MULT=2

# PR #1493 (leaderboard #1, val_bpb 1.0810) architecture techniques
export NUM_LOOPS=2                          # 3-layer depth recurrence
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35               # PR #1493 (we default to 0.5)
export QK_GAIN_INIT=5.25                    # PR #1493
export USE_PARALLEL_RESIDUALS=1             # PR #1493 (note: PR uses L7+, our impl is all-layers)
export EMA_DECAY=0.9965                     # PR #1493 (we default to 0.997)
export WARMDOWN_FRAC=0.72                   # PR #1493 (we default to 0.667)

# Our discovered speed wins
export USE_PARALLEL_MUON=1                  # batched Newton-Schulz
export TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
export USE_CUDNN_BENCHMARK=1

# Our discovered quant fix (int8 instead of int6 — preserves converged quality)
export MATRIX_BITS=8
export USE_CMP_QUANT_VALUE_DEDUP=0          # disable alphabet snap (less compression but cleaner quant)

# TTT — legal score-first only (matches PR #1493 exactly)
# PR #1493 PR description: "no pre-quant TTT, no ETLB, no n-gram cache, no SLOT — fully compliant"
export PREQUANT_TTT_ENABLED=0               # disabled — PR #1493 explicitly does not use pre-quant TTT
export PREQUANT_TTT_EPOCHS=0
export TTT_ENABLED=1                        # legal score-first TTT (eval chunk, then train chunk)
export TTT_EPOCHS=3                         # PR #1493
export TTT_LR=0.005                         # PR #1493
export TTT_FREEZE_BLOCKS=0
export SLIDING_WINDOW_ENABLED=1             # full sliding eval

# Standard Phase 1 wins (already defaulted in run.sh, declared here for clarity)
export USE_GATED_ATTENTION=1
export USE_NORMUON=1
export USE_NORM_PCT_DROPOUT=1
export USE_NGRAM_BIAS=1
export USE_NGRAM_BACKOFF=1
export USE_NGR_LOG_FREQ_INV=1               # uses train data for sample (rule-fix in commit 4b16703)
export USE_CTX_PARTITIONED_TAB=1
export USE_PREFETCH_LOADER=1

bash submission/run.sh

echo
echo "============================================================"
echo "[dry_run] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Submission val_bpb candidates (use the LAST one for the PR):"
echo "  - quantized val_bpb         (no TTT)"
echo "  - quantized_sliding_window  (sliding eval, no TTT)"
echo "  - legal_ttt_exact val_bpb   (legal score-first TTT — THIS IS THE SUBMISSION NUMBER)"
echo
grep -E "^(quantized|legal_ttt_exact|quantized_sliding_window)" /tmp/paramgolf_bootstrap.log 2>/dev/null || \
  grep -E "val_bpb" /tmp/paramgolf_bootstrap.log 2>/dev/null | tail -15
