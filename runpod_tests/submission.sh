#!/bin/bash
# submission.sh — Run the actual submission build (NOT a test)
# Usage: ./submission.sh
#
# This is separate from validate.sh / unknown.sh because submissions
# AREN'T tests. The unknown/ folder is for finding new things and A/B
# comparisons; submission/ is for producing the actual artifact you ship.
#
# Hardware: 8xH100 (will refuse to run on smaller GPUs)
# Time: ~45 min (3 seeds × ~15 min each)
# Cost: ~$15
#
# Outputs:
#   logs/submission.log         — runner summary
#   logs/submission/seed_42.log — full training + eval for each seed
#   logs/submission/seed_314.log
#   logs/submission/seed_999.log
#   logs/submission/results.txt — mean ± std for the submission JSON

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/submission.log"
mkdir -p "$LOG_DIR/submission"

{
    echo "================================================================================"
    echo "SUBMISSION RUN — $(date)"
    echo "Host:       $(hostname)"
    echo "Repo:       $REPO_ROOT"
    echo "GPU:        $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "GPU count:  $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
    echo "================================================================================"
    echo
} > "$LOG_FILE"

# Build absolute test list
TESTS=()
for pattern in "$SCRIPT_DIR/submission/"*.sh; do
    for script in $pattern; do
        if [ -f "$script" ]; then
            TESTS+=("$script")
        fi
    done
done

PASS=0
FAIL=0
FAILED_TESTS=()

run_submission() {
    local script=$1
    local name
    name=$(basename "$script" | sed 's/\.[^.]*$//')

    echo
    echo "================================================================================"
    echo ">>> START: $name"
    echo ">>> TIME: $(date '+%H:%M:%S')"
    echo "================================================================================"

    local start
    start=$(date +%s)
    local exit_code=0

    cd "$REPO_ROOT"
    bash "$script" 2>&1
    exit_code=$?

    local end
    end=$(date +%s)
    local duration=$((end - start))

    echo
    if [ $exit_code -eq 0 ]; then
        echo "<<< RESULT: $name PASS (${duration}s)"
        PASS=$((PASS + 1))
    else
        echo "<<< RESULT: $name FAIL (exit=$exit_code, ${duration}s)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$name")
    fi
    echo "================================================================================"
}

{
    if [ ${#TESTS[@]} -eq 0 ]; then
        echo "✗ NO SUBMISSION SCRIPTS FOUND in $SCRIPT_DIR/submission/"
        ls "$SCRIPT_DIR/submission/" 2>&1 || echo "  (directory does not exist)"
    fi

    for script in "${TESTS[@]}"; do
        run_submission "$script"
    done

    echo
    echo "================================================================================"
    echo "SUBMISSION SUMMARY"
    echo "================================================================================"
    echo "TIME:    $(date)"
    echo "TESTS:   ${#TESTS[@]}"
    echo "PASSED:  $PASS"
    echo "FAILED:  $FAIL"
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "FAILED:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
