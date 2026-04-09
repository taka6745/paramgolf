#!/bin/bash
# phase2/bootstrap.sh — one-command setup for Phase 2 speed work on a fresh pod.
#
# Usage: curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/phase2/bootstrap.sh | bash
#
# Difference from submission/bootstrap.sh:
#   - Uses submission/setup.sh for base pod environment
#   - Adds phase2/setup_phase2.sh for Phase 2-specific deps (FA3, etc)
#   - Uses submission/get_data.sh for data (tokenize + n-grams) unchanged
#   - Runs phase2/warm_compile_cache.py to populate torch.compile cache (S1)
#   - Uses phase2/run.sh (compile enabled) instead of submission/run.sh
#
# Phase 1 bootstrap is still the single command for reproducing Phase 1:
#   curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/submission/bootstrap.sh | bash
#
# Both bootstraps share data (tokenize + n-gram tables are identical, built by
# submission/get_data.sh into /workspace/paramgolf/data/), so running Phase 2
# after Phase 1 on the same pod skips tokenize and goes straight to train.

set -eu
exec > >(tee -a /tmp/paramgolf_phase2_bootstrap.log) 2>&1

REPO_URL="${REPO_URL:-https://github.com/taka6745/paramgolf.git}"
REPO_DIR="${REPO_DIR:-/workspace/paramgolf}"
BRANCH="${BRANCH:-main}"

echo "============================================================"
echo "= paramgolf PHASE 2 bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ="
echo "============================================================"
echo "REPO_URL=$REPO_URL"
echo "REPO_DIR=$REPO_DIR"
echo "BRANCH=$BRANCH"

# Step 1: git (re)clone the repo
if ! command -v git >/dev/null 2>&1; then
    echo "[p2-bootstrap] installing git..."
    apt-get update -qq && apt-get install -y -qq git
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[p2-bootstrap] cloning $REPO_URL into $REPO_DIR..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "[p2-bootstrap] repo exists, pulling latest $BRANCH..."
    cd "$REPO_DIR"
    git fetch origin "$BRANCH"
    git reset --hard "origin/$BRANCH"
fi

cd "$REPO_DIR"

# Step 2: base pod env (same as Phase 1)
echo
echo "============================================================"
echo "[p2-bootstrap] STEP 1/5: submission/setup.sh (base pod env)"
echo "============================================================"
bash submission/setup.sh

# Step 3: Phase 2-specific deps (FA3, extra tools)
echo
echo "============================================================"
echo "[p2-bootstrap] STEP 2/5: phase2/setup_phase2.sh (Phase 2 deps)"
echo "============================================================"
if [ -f phase2/setup_phase2.sh ]; then
    bash phase2/setup_phase2.sh
else
    echo "[p2-bootstrap] phase2/setup_phase2.sh not present yet — skipping (S2 not landed)"
fi

# Step 4: data (tokenize + n-grams, shared with Phase 1)
echo
echo "============================================================"
echo "[p2-bootstrap] STEP 3/5: submission/get_data.sh (tokenize + n-grams)"
echo "============================================================"
bash submission/get_data.sh

# Step 5: warm the torch.compile cache (Shot 1)
echo
echo "============================================================"
echo "[p2-bootstrap] STEP 4/5: phase2/warm_compile_cache.py (populate inductor cache)"
echo "============================================================"
if [ -f phase2/warm_compile_cache.py ]; then
    python3 phase2/warm_compile_cache.py
else
    echo "[p2-bootstrap] phase2/warm_compile_cache.py not present yet — skipping (S1 not landed)"
fi

# Step 6: train + eval + serialize via the Phase 2 run wrapper
echo
echo "============================================================"
echo "[p2-bootstrap] STEP 5/5: phase2/run.sh (train with compile enabled)"
echo "============================================================"
if [ -f phase2/run.sh ]; then
    bash phase2/run.sh
else
    echo "[p2-bootstrap] phase2/run.sh not present yet — falling back to submission/run.sh"
    bash submission/run.sh
fi

echo
echo "============================================================"
echo "[p2-bootstrap] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo "Submission artifact: $REPO_DIR/final_model.int6.ptz"
echo "Log: /tmp/paramgolf_phase2_bootstrap.log + $REPO_DIR/submission/logs/"
