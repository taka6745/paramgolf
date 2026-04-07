#!/bin/bash
# install_cron.sh — install the watchdog cron entry on the pod.
# Adds a `* * * * *` job that calls watchdog.sh every minute.
# Idempotent — won't add a duplicate entry.
set -u
chmod +x /workspace/paramgolf/runpod_tests/loop/watchdog.sh
LINE='* * * * * /workspace/paramgolf/runpod_tests/loop/watchdog.sh >> /workspace/paramgolf/runpod_tests/loop/watchdog.log 2>&1'
( crontab -l 2>/dev/null | grep -v 'runpod_tests/loop/watchdog.sh' ; echo "$LINE" ) | crontab -
echo "✓ cron installed:"
crontab -l | grep watchdog
