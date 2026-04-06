#!/bin/bash
# 03_retokenize.sh — SKIPPED for now (using sp1024 baseline directly)
#
# We're using sp1024 data straight from the cached_challenge download.
# No re-tokenization needed for the SP-1024 path.

set -e
echo "=== RE-TOKENIZE (SKIPPED — using sp1024 baseline directly) ==="
echo
ls data/datasets/ 2>/dev/null || echo "  (no datasets/ yet)"
echo
echo "✓ Skipped (sp1024 data is already tokenized)"
exit 0
