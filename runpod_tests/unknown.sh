#!/bin/bash
# unknown.sh — Run all unknown tests, output to logs/unknown.log
# Usage: ./unknown.sh [u01 u02 ... uNN]
# If no args: runs all unknown tests in order
# If args: runs only specified tests

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
LOG_FILE="$LOG_DIR/unknown.log"
mkdir -p "$LOG_DIR"

{
    echo "================================================================================"
    echo "UNKNOWN RUN — $(date)"
    echo "Host: $(hostname)"
    echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "VRAM: $(python3 -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB") if torch.cuda.is_available() else print("none")' 2>/dev/null || echo 'unknown')"
    echo "================================================================================"
    echo
    echo "WARNING: Running unknown/ tests on a non-H100 GPU will give different"
    echo "absolute numbers than the eventual H100 submission. Use these results"
    echo "for RELATIVE comparisons (config A vs B), not absolute BPP estimates."
    echo
} > "$LOG_FILE"

PASS=0
FAIL=0
FAILED_TESTS=()
RESULTS=()

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
    bash "$SCRIPT_DIR/$script" 2>&1
    exit_code=$?

    local end=$(date +%s)
    local duration=$((end - start))

    echo
    if [ $exit_code -eq 0 ]; then
        echo "<<< RESULT: $name PASS (${duration}s)"
        PASS=$((PASS + 1))
        RESULTS+=("$name PASS ${duration}s")
    else
        echo "<<< RESULT: $name FAIL (exit=$exit_code, ${duration}s)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$name")
        RESULTS+=("$name FAIL ${duration}s")
    fi
    echo "================================================================================"
}

# If args provided, run only those. Otherwise run all in order.
if [ $# -gt 0 ]; then
    {
        echo "Running specific tests: $@"
        echo
        for arg in "$@"; do
            for script in unknown/${arg}*.sh; do
                if [ -f "$SCRIPT_DIR/$script" ]; then
                    run_test "$script"
                fi
            done
        done
    } 2>&1 | tee -a "$LOG_FILE"
else
    {
        echo "Running all unknown tests in order"
        echo
        # Order: u01 (arch), u06 (speed), u07/u08 (GLA chain), u02 (progressive), u03 (cache), u04 (full), u05 (3-seed)
        for script in unknown/u01*.sh unknown/u06*.sh unknown/u07*.sh unknown/u08*.sh \
                      unknown/u02*.sh unknown/u03*.sh unknown/u04*.sh unknown/u05*.sh; do
            if [ -f "$SCRIPT_DIR/$script" ]; then
                run_test "$script"
            fi
        done
    } 2>&1 | tee -a "$LOG_FILE"
fi

# Summary
{
    echo
    echo "================================================================================"
    echo "UNKNOWN SUMMARY"
    echo "================================================================================"
    echo "TIME:    $(date)"
    echo "PASSED:  $PASS"
    echo "FAILED:  $FAIL"
    echo
    echo "RESULTS:"
    for r in "${RESULTS[@]}"; do
        echo "  $r"
    done

    # Extract any val_bpb scores found in logs
    echo
    echo "VAL_BPB SCORES (if any):"
    if [ -d "logs" ]; then
        find logs -name "*.log" -exec grep -H 'final_int8_zlib_roundtrip val_loss' {} \; 2>/dev/null | \
            grep -oE 'logs/[^:]*|val_bpb:[0-9.]+' | paste - - | head -20 || echo "  (none yet)"
    fi

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo
        echo "FAILED TESTS:"
        for t in "${FAILED_TESTS[@]}"; do
            echo "  - $t"
        done
    fi
    echo "================================================================================"
} 2>&1 | tee -a "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
exit $FAIL
