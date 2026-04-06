#!/bin/bash
# bootstrap.sh — One-line setup for a fresh RunPod box
#
# Usage (paste into SSH terminal on the RunPod box):
#   curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/runpod_tests/bootstrap.sh | bash
#
# This script:
#   1. Installs git if missing
#   2. Clones (or pulls) the paramgolf repo to /workspace/paramgolf
#   3. Makes all runpod_tests/ scripts executable
#   4. Prints the next commands to run

set -e

echo "================================================================================"
echo "PARAMGOLF — RUNPOD BOOTSTRAP"
echo "================================================================================"
echo

# 1. Install git if missing (some minimal images don't have it)
echo "[1/4] Checking git..."
if ! command -v git &>/dev/null; then
    echo "  git not found, installing..."
    apt-get update -qq
    apt-get install -y -qq git ca-certificates
fi
echo "  $(git --version)"

# 2. Clone or update repo
echo
echo "[2/4] Cloning paramgolf to /workspace/paramgolf..."
mkdir -p /workspace
cd /workspace
if [ ! -d "paramgolf" ]; then
    git clone https://github.com/taka6745/paramgolf.git
    echo "  ✓ cloned"
else
    cd paramgolf
    git fetch origin
    git pull origin main || true
    cd ..
    echo "  ✓ updated existing clone"
fi

# 3. Verify runpod_tests exists
echo
echo "[3/4] Verifying runpod_tests/..."
cd /workspace/paramgolf
if [ ! -d "runpod_tests" ]; then
    echo "  ✗ FAIL: runpod_tests/ not found in repo"
    echo "  The repo may not have been pushed with the test suite yet."
    exit 1
fi
echo "  ✓ runpod_tests/ exists"

# 4. Make all scripts executable
echo
echo "[4/4] Making scripts executable..."
chmod +x runpod_tests/*.sh 2>/dev/null || true
chmod +x runpod_tests/chore/*.sh runpod_tests/chore/*.py 2>/dev/null || true
chmod +x runpod_tests/validate/*.sh runpod_tests/validate/*.py 2>/dev/null || true
chmod +x runpod_tests/unknown/*.sh 2>/dev/null || true
echo "  ✓ all scripts executable"

# Detect GPU
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

# Done — print next steps
echo
echo "================================================================================"
echo "✓ BOOTSTRAP COMPLETE"
echo "================================================================================"
echo
echo "  Repo:  /workspace/paramgolf"
echo "  GPU:   $GPU"
echo
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo
echo "Run tests one at a time (recommended for first time):"
echo "  cd /workspace/paramgolf/runpod_tests"
echo "  ./setup.sh        # ~30 min, ~\$0.10  — deps + data + n-gram tables"
echo "  ./validate.sh     # ~1 hr,  ~\$0.20  — confirm code runs on this GPU"
echo "  ./unknown.sh      # ~3-4 hr,~\$0.70  — the actual research"
echo "  ./export_logs.sh  # uploads logs and prints download URLs"
echo
echo "Or chain everything in one command (~5 hrs, ~\$1):"
echo "  cd /workspace/paramgolf/runpod_tests && \\"
echo "    ./setup.sh && ./validate.sh && ./unknown.sh && ./export_logs.sh"
echo
echo "Run a specific unknown test only:"
echo "  cd /workspace/paramgolf/runpod_tests"
echo "  ./unknown.sh u02         # progressive seq only"
echo "  ./unknown.sh u02 u03 u04 # progressive + cache + full stack"
echo
echo "Tail a log from another SSH session:"
echo "  tail -f /workspace/paramgolf/runpod_tests/logs/unknown.log"
echo
echo "================================================================================"
