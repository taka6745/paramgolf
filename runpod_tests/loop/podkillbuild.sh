#!/bin/bash
# podkillbuild.sh — kill any stuck 05_build_dc500.py process on a pod
set -u
if [ -z "${POD_HOST:-}" ] || [ -z "${POD_ID:-}" ]; then
    echo "FATAL: POD_HOST POD_ID required" >&2
    exit 1
fi
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
INNER='echo POD='"$POD_ID"'; ps -ef | grep build_dc500 | grep -v grep; pkill -9 -f 05_build_dc500.py 2>/dev/null && echo killed || echo nothing_to_kill; sleep 1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
B64=$(echo "$INNER" | base64 | tr -d '\n')
unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$SSH_KEY" -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 "$POD_HOST" <<SSH
echo $B64 | base64 -d | bash
exit
SSH
