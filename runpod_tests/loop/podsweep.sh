#!/bin/bash
# podsweep.sh — status-check + fix all 7 pods.
#
# For each pod:
#   - Show current run_forever process state
#   - Show last few lines of run_forever.out
#   - Show nvidia-smi util/memory
#   - Show count of in-flight Python processes
#   - Set pod_id.txt to the assigned letter
#   - Kill any zombie train_gpt.py processes
#   - If run_forever is dead, restart it
#
# Usage:
#   POD_HOST=<host> POD_ID=<letter> bash /tmp/podsweep.sh

set -u
if [ -z "${POD_HOST:-}" ] || [ -z "${POD_ID:-}" ]; then
    echo "FATAL: POD_HOST and POD_ID required" >&2
    exit 1
fi
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

INNER=$(cat <<INNER_EOF
#!/bin/bash
set -u
echo "SWEEP_START POD=$POD_ID at \$(date -u)"
cd /workspace/paramgolf 2>/dev/null || { echo "SWEEP_FATAL: no repo"; exit 2; }

# State snapshot
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1
echo "--- ps ---"
ps -ef | grep -E 'run_forever|experiment_runner|train_gpt|01_download' | grep -v grep | head -10
echo "--- branch ---"
git branch --show-current
git rev-parse --short HEAD
echo "--- pod_id.txt ---"
cat runpod_tests/loop/pod_id.txt 2>/dev/null || echo "(missing)"
echo "--- last results.jsonl rows ---"
tail -3 runpod_tests/loop/results.jsonl 2>/dev/null || echo "(no results)"
echo "--- last 10 lines run_forever.out ---"
tail -10 runpod_tests/loop/run_forever.out 2>/dev/null || echo "(no run_forever.out)"

# FIX: write pod_id.txt
echo "$POD_ID" > runpod_tests/loop/pod_id.txt
echo "SWEEP_FIX: pod_id.txt = \$(cat runpod_tests/loop/pod_id.txt)"

# FIX: pull latest experiments queue
git pull --rebase --autostash origin main 2>&1 | tail -3 || true
echo "SWEEP: now at \$(git rev-parse --short HEAD)"

# DO NOT kill train_gpt processes — that destroys in-flight experiments.
# run_forever's own crash detection handles real zombies. Earlier version of this
# sweep killed train_gpt unconditionally and was wiping experiments on every C5 fire.
TRAIN_GPT_PIDS=\$(pgrep -f 'python3.*train_gpt.py' 2>/dev/null)
if [ -n "\$TRAIN_GPT_PIDS" ]; then
    echo "SWEEP: train_gpt running (pids: \$TRAIN_GPT_PIDS) — leaving alone"
fi

# Check run_forever liveness
RF_PID=\$(pgrep -f run_forever.sh | head -1)
if [ -n "\$RF_PID" ]; then
    echo "SWEEP: run_forever ALIVE pid=\$RF_PID"
else
    # Only restart if 01_download_data isn't running (i.e., bootstrap is done)
    DL=\$(pgrep -f 01_download_data 2>/dev/null)
    if [ -n "\$DL" ]; then
        echo "SWEEP: bootstrap still running (01_download pid=\$DL), not restarting run_forever"
    else
        echo "SWEEP_FIX: run_forever DEAD, restarting"
        nohup bash runpod_tests/loop/run_forever.sh > runpod_tests/loop/run_forever.out 2>&1 &
        disown
        sleep 3
        echo "SWEEP: new run_forever pid=\$(pgrep -f run_forever.sh | head -1)"
    fi
fi

echo "SWEEP_DONE POD=$POD_ID at \$(date -u)"
INNER_EOF
)

B64=$(echo "$INNER" | base64 | tr -d '\n')

unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 \
    "$POD_HOST" <<SSH_STDIN
echo $B64 | base64 -d > /tmp/sweep_inner.sh && bash /tmp/sweep_inner.sh
exit
SSH_STDIN
