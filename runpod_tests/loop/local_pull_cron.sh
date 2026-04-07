#!/bin/bash
# local_pull_cron.sh — runs on the user's Mac to pull results from the pod every N min.
#
# Install:
#   crontab -e
#   */10 * * * * /Users/takodamundy/Documents/personal_repos/paramgolf/runpod_tests/loop/local_pull_cron.sh >> /tmp/podpull_cron.log 2>&1
#
# Each invocation:
#   1. Pulls results.jsonl + leaderboard.txt from the pod via SSH heredoc
#   2. Runs the local analyze.py to print a summary into the cron log
#   3. Notifies via macOS notification if a new top-3 result has appeared
#
# Self-contained — relies only on ssh + python3 + base64.
set -u
LOG=/tmp/podpull_cron.log
DEST=/tmp/podpull_results
mkdir -p "$DEST"

POD_HOST="${POD_HOST:-tyf0q5l1kgefgx-64410a6f@ssh.runpod.io}"
SSH_KEY="${SSH_KEY:-/Users/takodamundy/.ssh/id_ed25519}"

cmd_file=$(mktemp)
out_file=$(mktemp)
trap 'rm -f "$cmd_file" "$out_file"' EXIT

cat > "$cmd_file" <<'PODEOF'
cd /workspace/paramgolf
echo "===CRON_START==="
echo "TIME: $(date -u)"
echo "===RESULTS_BEGIN==="
test -f runpod_tests/loop/results.jsonl && cat runpod_tests/loop/results.jsonl | base64
echo "===RESULTS_END==="
echo "===LEADERBOARD_BEGIN==="
test -f runpod_tests/loop/leaderboard.txt && cat runpod_tests/loop/leaderboard.txt | base64
echo "===LEADERBOARD_END==="
echo "PROC:"
ps -ef | grep -E 'experiment_runner|train_gpt' | grep -v grep | head -3
echo "===CRON_END==="
exit
PODEOF

ssh -tt -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$POD_HOST" < "$cmd_file" 2>&1 \
    | tr -d '\r' \
    | LANG=C perl -pe 's/\e\[[0-9;?]*[a-zA-Z]//g; s/\e\][^\a]*\a//g' \
    > "$out_file"

# Extract base64 between markers and decode
awk '/===RESULTS_BEGIN===/{p=1; next} /===RESULTS_END===/{p=0} p{print}' "$out_file" \
    | grep -E '^[A-Za-z0-9+/]+=*$' \
    | tr -d '\n' \
    | base64 -d > "$DEST/results.jsonl" 2>/dev/null

awk '/===LEADERBOARD_BEGIN===/{p=1; next} /===LEADERBOARD_END===/{p=0} p{print}' "$out_file" \
    | grep -E '^[A-Za-z0-9+/]+=*$' \
    | tr -d '\n' \
    | base64 -d > "$DEST/leaderboard.txt" 2>/dev/null

# Count results
N=$(wc -l < "$DEST/results.jsonl" 2>/dev/null | tr -d ' ')
TOP=$(head -1 "$DEST/leaderboard.txt" 2>/dev/null)
echo "$(date -u) — pulled $N results — $TOP"

# Macos notification if best changed
BEST=$(grep -E '^[A-Z][0-9]?_' "$DEST/leaderboard.txt" 2>/dev/null | head -1 | awk '{print $1, $2}')
LAST_BEST_FILE="$DEST/last_best.txt"
if [ -n "$BEST" ]; then
    if [ ! -f "$LAST_BEST_FILE" ] || ! diff -q "$LAST_BEST_FILE" <(echo "$BEST") >/dev/null 2>&1; then
        echo "$BEST" > "$LAST_BEST_FILE"
        osascript -e "display notification \"$BEST\" with title \"paramgolf new top result\"" 2>/dev/null || true
    fi
fi
