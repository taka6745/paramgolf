#!/bin/bash
# watchdog.sh — restart the experiment loop if it died.
# Designed to be invoked from cron every minute.
#
# Crontab entry (set up by install_cron.sh):
#   * * * * * /workspace/paramgolf/runpod_tests/loop/watchdog.sh >> /workspace/paramgolf/runpod_tests/loop/watchdog.log 2>&1
set -u
cd /workspace/paramgolf

if pgrep -f experiment_runner.py >/dev/null 2>&1; then
    exit 0
fi

# Loop is dead — restart it
echo "$(date -u) — restarting experiment_runner (was dead)"
pkill -f 'python3 train_gpt.py' 2>/dev/null || true
sleep 2
mkdir -p runpod_tests/loop/logs
nohup bash runpod_tests/loop/run_forever.sh >> runpod_tests/loop/run_forever.out 2>&1 &
disown $! 2>/dev/null || true
echo "$(date -u) — restarted, new PID $!"
