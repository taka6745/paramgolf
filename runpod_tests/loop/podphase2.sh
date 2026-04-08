#!/bin/bash
# podphase2.sh — phase 2 of pod bootstrap: n-gram tables + DC500 + verify.
#
# Prerequisites: phase 1 (podbootstrap3.sh) must have completed: repo cloned,
# venv set up, FineWeb shards downloaded, patcher applied, run_forever running.
#
# This phase:
#   - Builds bigram/trigram/4gram tables via chore/04_build_ngrams.py (~30 s on 100M tokens)
#   - Builds DC500 categories via chore/05_build_dc500.py
#   - Runs chore/07_verify_data.sh to confirm all required files are present
#   - Restarts run_forever.sh so it picks up the new n-gram tables
#
# Usage:
#   POD_HOST=vwkkjkevpvyrfs-6441169a@ssh.runpod.io POD_ID=B \
#       bash /tmp/podphase2.sh > /tmp/phase2_B.log 2>&1

set -u

if [ -z "${POD_HOST:-}" ] || [ -z "${POD_ID:-}" ]; then
    echo "FATAL: POD_HOST and POD_ID env vars are required" >&2
    exit 1
fi

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "=== podphase2.sh: $POD_ID via $POD_HOST starting at $(date -u) ==="

INNER_SCRIPT=$(cat <<'INNER_EOF'
#!/bin/bash
set -u
cd /workspace/paramgolf
echo "PHASE2_START at $(date -u)"

# Verify phase 1 prerequisites
if [ ! -d data/datasets/fineweb10B_sp1024 ]; then
    echo "PHASE2_FATAL: missing FineWeb shards (phase 1 incomplete)"
    exit 2
fi

# 04: n-gram tables (bigram/trigram/4gram for SP-1024, ~30 sec on 100M tokens)
if [ ! -f data/bigram_tab_1024v.npy ] || [ ! -f data/trigram_logprobs_1024v.npy ] || [ ! -f data/fourgram_logprobs_1024v.npy ]; then
    echo "PHASE2_RUN: 04_build_ngrams.py"
    python3 runpod_tests/chore/04_build_ngrams.py 2>&1 | tail -15 || echo "PHASE2_WARN: 04_build_ngrams.py failed"
else
    echo "PHASE2: n-gram tables already built"
fi

# 05: DC500 categories
if [ ! -f data/dist_cats_500_1024.npz ]; then
    if [ -f runpod_tests/chore/05_build_dc500.py ]; then
        echo "PHASE2_RUN: 05_build_dc500.py"
        python3 runpod_tests/chore/05_build_dc500.py 2>&1 | tail -10 || echo "PHASE2_WARN: 05_build_dc500.py failed"
    fi
else
    echo "PHASE2: DC500 already built"
fi

# 07: verify all required data is present
if [ -f runpod_tests/chore/07_verify_data.sh ]; then
    echo "PHASE2_RUN: 07_verify_data.sh"
    bash runpod_tests/chore/07_verify_data.sh 2>&1 | tail -25 || true
fi

# Restart run_forever to pick up new n-gram tables
echo "PHASE2: restarting run_forever"
pkill -f run_forever.sh 2>/dev/null || true
pkill -f experiment_runner.py 2>/dev/null || true
sleep 2

nohup bash runpod_tests/loop/run_forever.sh > runpod_tests/loop/run_forever.out 2>&1 &
disown

sleep 4
RF_PID=$(pgrep -f run_forever.sh | head -1)
if [ -n "$RF_PID" ]; then
    echo "PHASE2_RUN_FOREVER_ALIVE pid=$RF_PID"
    head -10 runpod_tests/loop/run_forever.out 2>/dev/null
else
    echo "PHASE2_RUN_FOREVER_DEAD"
    tail -20 runpod_tests/loop/run_forever.out 2>/dev/null
fi

echo "PHASE2_DONE at $(date -u)"
INNER_EOF
)

B64=$(echo "$INNER_SCRIPT" | base64 | tr -d '\n')

unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 \
    "$POD_HOST" <<SSH_STDIN
echo $B64 | base64 -d > /tmp/phase2_inner.sh && bash /tmp/phase2_inner.sh
echo PHASE2_EXITING
exit
SSH_STDIN

SSH_RC=$?
echo "=== podphase2.sh: $POD_ID exit_code=$SSH_RC at $(date -u) ==="
exit $SSH_RC
