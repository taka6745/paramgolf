#!/bin/bash
# 02_build_tokenizer.sh — SKIPPED for now (using sp1024 baseline)
#
# We're using the default sp1024 tokenizer that train_gpt.py uses by default.
# BPE-8192 needs download_hf_docs_and_tokenize.py with a custom spec, which
# is a 30+ min job and needs the docs_selected.jsonl download. We'll do that
# as an upgrade after the SP-1024 path validates end-to-end.

set -e
echo "=== BUILD TOKENIZER (SKIPPED — using sp1024 baseline) ==="
echo
echo "  Default sp1024 tokenizer comes with the cached_challenge_fineweb data."
echo "  To upgrade to BPE-8192 later:"
echo "    1. Edit data/tokenizer_specs.json to add bpe_8192 entry"
echo "    2. python3 data/download_hf_docs_and_tokenize.py --output-root data/"
echo
echo "✓ Skipped (sp1024 is bundled with downloaded data)"
exit 0
