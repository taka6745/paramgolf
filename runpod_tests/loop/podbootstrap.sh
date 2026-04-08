#!/bin/bash
# podbootstrap3.sh — STDIN dispatch (matches /tmp/podrun.sh's working pattern).
#
# Sends the bootstrap commands via ssh STDIN with -tt PTY mode. The remote
# shell reads commands as if typed; an explicit `exit` at the end terminates
# the session so ssh returns instead of hanging at the prompt.
#
# Usage:
#   POD_HOST=vwkkjkevpvyrfs-6441169a@ssh.runpod.io POD_ID=B \
#       bash /tmp/podbootstrap3.sh > /tmp/bootstrap_B.log 2>&1

set -u

if [ -z "${POD_HOST:-}" ] || [ -z "${POD_ID:-}" ]; then
    echo "FATAL: POD_HOST and POD_ID env vars are required" >&2
    exit 1
fi

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "=== podbootstrap3.sh: $POD_ID via $POD_HOST starting at $(date -u) ==="

# Build the inner script as a single base64 blob, decode + run on the pod via stdin.
# Single base64 blob is robust against PTY echo: even if the shell echoes each char
# back, the only line that gets executed is the base64 command itself.
INNER_SCRIPT=$(cat <<INNER_EOF
#!/bin/bash
set -u
export POD_ID="$POD_ID"
echo "REMOTE_BOOTSTRAP_START POD_ID=\$POD_ID at \$(date -u)"

cd /workspace
if [ ! -d paramgolf ]; then
    git clone https://github.com/taka6745/paramgolf.git
fi
cd paramgolf
git fetch origin 2>&1 | tail -3
git checkout main 2>&1 | tail -3
git pull --rebase --autostash origin main 2>&1 | tail -3 || true
echo "REMOTE_AT_COMMIT \$(git rev-parse --short HEAD)"

mkdir -p data/datasets data/tokenizers runpod_tests/loop/logs

# Block 1
if [ -f runpod_tests/chore/00_setup_pod.sh ]; then
    echo "REMOTE_RUN: 00_setup_pod.sh"
    bash runpod_tests/chore/00_setup_pod.sh 2>&1 | tail -8 || true
fi

# Block 2: data download
if [ ! -d data/datasets/fineweb10B_sp1024 ] || [ -z "\$(ls -A data/datasets/fineweb10B_sp1024 2>/dev/null)" ]; then
    if [ -f runpod_tests/chore/01_download_data.sh ]; then
        echo "REMOTE_RUN: 01_download_data.sh"
        bash runpod_tests/chore/01_download_data.sh 2>&1 | tail -15 || true
    fi
else
    echo "REMOTE: shards already present"
fi

# Block 3: patcher with G4 integrity
if [ ! -f train_gpt.py.bak ]; then
    cp train_gpt.py train_gpt.py.bak
fi
cp train_gpt.py.bak train_gpt.py
echo "REMOTE_RUN: 08_patch_train_gpt.sh"
bash runpod_tests/chore/08_patch_train_gpt.sh 2>&1 | tail -10
PATCH_EXIT=\$?
if [ "\$PATCH_EXIT" -ne 0 ]; then
    echo "REMOTE_FATAL patcher_exit=\$PATCH_EXIT"
    exit 4
fi

# Block 4: pod_id + run_forever launch
echo "\$POD_ID" > runpod_tests/loop/pod_id.txt
echo "REMOTE_POD_ID=\$(cat runpod_tests/loop/pod_id.txt)"

pkill -f run_forever.sh 2>/dev/null || true
pkill -f experiment_runner.py 2>/dev/null || true
sleep 2

nohup bash runpod_tests/loop/run_forever.sh > runpod_tests/loop/run_forever.out 2>&1 &
disown

sleep 4
RF_PID=\$(pgrep -f run_forever.sh | head -1)
if [ -n "\$RF_PID" ]; then
    echo "REMOTE_RUN_FOREVER_ALIVE pid=\$RF_PID"
else
    echo "REMOTE_RUN_FOREVER_DEAD"
    tail -10 runpod_tests/loop/run_forever.out 2>/dev/null
fi

echo "REMOTE_BOOTSTRAP_DONE POD_ID=\$POD_ID at \$(date -u)"
INNER_EOF
)

B64=$(echo "$INNER_SCRIPT" | base64 | tr -d '\n')

# Send the base64 decode + run command via SSH stdin, followed by exit.
# CRITICAL: see runpod_tests/loop/SSH_TROUBLESHOOTING.md for why we disable
# the agent and use IdentityAgent=none. RunPod's proxy rejects -tt PTY
# connections when the macOS SSH agent is involved.
unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 \
    "$POD_HOST" <<SSH_STDIN
echo $B64 | base64 -d > /tmp/bootstrap_inner.sh && bash /tmp/bootstrap_inner.sh
echo REMOTE_DONE_EXITING
exit
SSH_STDIN

SSH_RC=$?
echo "=== podbootstrap3.sh: $POD_ID exit_code=$SSH_RC at $(date -u) ==="
exit $SSH_RC
