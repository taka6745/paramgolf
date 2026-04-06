#!/bin/bash
# validate.sh — Run all validate tests, output to logs/validate.log
# Usage: ./validate.sh

set -u

# Always run from repo root so relative paths work
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Source venv if it exists
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/validate.log"
mkdir -p "$LOG_DIR"

{
    echo "================================================================================"
    echo "VALIDATE RUN — $(date)"
    echo "Host: $(hostname)"
    echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
} > "$LOG_FILE"

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

    cd "$REPO_ROOT"
    if [[ "$script" == *.py ]]; then
        python3 "$SCRIPT_DIR/$script" 2>&1
        exit_code=$?
    else
        bash "$SCRIPT_DIR/$script" 2>&1
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

{
    # Run all v01-v10 in order
    for script in validate/v01*.sh validate/v02*.py validate/v03*.py validate/v04*.sh \
                  validate/v05*.py validate/v06*.py validate/v07*.py validate/v08*.py \
                  validate/v09*.py validate/v10*.sh; do
        if [ -f "$SCRIPT_DIR/$script" ]; then
            run_test "$script"
        fi
    done

    echo
    echo "================================================================================"
    echo "VALIDATE SUMMARY"
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
        echo "STATUS: ALL VALIDATIONS PASSED — proceed to ./unknown.sh"
    else
        echo "STATUS: $FAIL VALIDATION(S) FAILED — debug before unknown/"
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
