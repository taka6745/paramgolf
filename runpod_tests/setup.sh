#!/bin/bash
# setup.sh — Run all chore tests, output to logs/setup.log
# Usage: ./setup.sh

set -u  # don't exit on test failure (let later tests still run)
cd "$(dirname "$0")"

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR"

# Header
{
    echo "================================================================================"
    echo "SETUP RUN — $(date)"
    echo "Host: $(hostname)"
    echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
} > "$LOG_FILE"

# Test runner
PASS=0
FAIL=0
FAILED_TESTS=()

run_test() {
    local script=$1
    local name=$(basename "$script" | sed 's/\.[^.]*$//')

    echo
    echo "================================================================================"
    echo ">>> START: $name"
    echo ">>> TIME: $(date '+%H:%M:%S')"
    echo "================================================================================"

    local start=$(date +%s)
    local exit_code=0

    if [[ "$script" == *.py ]]; then
        .venv/bin/python3 "$script" 2>&1
        exit_code=$?
    else
        bash "$script" 2>&1
        exit_code=$?
    fi

    local end=$(date +%s)
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

# Run all tests in chore/, in order
{
    for script in chore/00_*.sh chore/01_*.sh chore/02_*.sh chore/03_*.sh chore/04_*.py chore/05_*.py chore/06_*.py chore/07_*.sh; do
        if [ -f "$script" ]; then
            run_test "$script"
        fi
    done

    # Summary
    echo
    echo "================================================================================"
    echo "SETUP SUMMARY"
    echo "================================================================================"
    echo "TIME:    $(date)"
    echo "PASSED:  $PASS"
    echo "FAILED:  $FAIL"
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "FAILED TESTS:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo
    if [ $FAIL -eq 0 ]; then
        echo "STATUS: ALL CHORES COMPLETE — proceed to ./validate.sh"
    else
        echo "STATUS: $FAIL CHORE(S) FAILED — fix before proceeding"
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
