#!/usr/bin/env python3
"""
04_build_ngrams.py — Build bigram, trigram, 4-gram tables for SP-1024

Runs on: any pod (CPU work)
Time: ~4 min
Outputs:
    data/bigram_tab_1024v.npy        (small, 2048 hash buckets × 1024 vocab)
    data/trigram_logprobs_1024v.npy
    data/fourgram_logprobs_1024v.npy

These get loaded by train_gpt.py and added as logit bias during training.
Using SP-1024 baseline. To rebuild for BPE-8192 later, change VOCAB and DATA_DIR.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

VOCAB = 1024
HASH_BUCKETS = 2048  # 2x vocab for low collision
DATA_DIR = Path("data/datasets/fineweb10B_sp1024")

# Hash functions (signed polynomial — confirmed +0.003 BPP on Mac)
def hash_bigram(prev):
    return (prev * 36313) % HASH_BUCKETS

def hash_trigram(prev2, prev1):
    return ((prev2 * 36313 + prev1 * 27191) % HASH_BUCKETS)

def hash_fourgram(prev3, prev2, prev1):
    return ((prev3 * 36313 + prev2 * 27191 + prev1 * 51497) % HASH_BUCKETS)


def load_tokens():
    """Load all training tokens from SP-1024 shards."""
    print(f"Loading tokens from {DATA_DIR}...")
    if not DATA_DIR.exists():
        print(f"  ✗ {DATA_DIR} does not exist")
        print(f"  Did 01_download_data.sh run successfully?")
        print(f"  Available datasets: {list(Path('data/datasets').glob('*')) if Path('data/datasets').exists() else 'none'}")
        sys.exit(1)

    all_tokens = []
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    if not shard_paths:
        print(f"  ✗ No fineweb_train_*.bin shards in {DATA_DIR}")
        print(f"  Files in dir: {list(DATA_DIR.iterdir())}")
        sys.exit(1)

    for shard_path in shard_paths:
        # Each shard is a header + uint16 tokens
        with open(shard_path, "rb") as f:
            header = np.frombuffer(f.read(1024), dtype=np.int32)
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
        all_tokens.append(tokens)
        print(f"  {shard_path.name}: {len(tokens):,} tokens")
    return np.concatenate(all_tokens)


def build_bigram(tokens):
    print(f"\nBuilding bigram table ({HASH_BUCKETS} buckets × {VOCAB} vocab)...")
    counts = np.zeros((HASH_BUCKETS, VOCAB), dtype=np.int32)
    for i in range(len(tokens) - 1):
        h = hash_bigram(int(tokens[i]))
        counts[h, int(tokens[i+1])] += 1
        if i % 1_000_000 == 0:
            print(f"  {i:,}/{len(tokens):,}")

    # Convert to log probabilities (Laplace smoothing)
    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    log_probs = np.log(probs).astype(np.float32)
    return log_probs


def build_trigram(tokens):
    print(f"\nBuilding trigram table...")
    counts = np.zeros((HASH_BUCKETS, VOCAB), dtype=np.int32)
    for i in range(2, len(tokens)):
        h = hash_trigram(int(tokens[i-2]), int(tokens[i-1]))
        counts[h, int(tokens[i])] += 1
        if i % 1_000_000 == 0:
            print(f"  {i:,}/{len(tokens):,}")

    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return np.log(probs).astype(np.float32)


def build_fourgram(tokens):
    print(f"\nBuilding 4-gram table...")
    counts = np.zeros((HASH_BUCKETS, VOCAB), dtype=np.int32)
    for i in range(3, len(tokens)):
        h = hash_fourgram(int(tokens[i-3]), int(tokens[i-2]), int(tokens[i-1]))
        counts[h, int(tokens[i])] += 1
        if i % 1_000_000 == 0:
            print(f"  {i:,}/{len(tokens):,}")

    counts = counts.astype(np.float64) + 0.1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return np.log(probs).astype(np.float32)


def main():
    bigram_path = f"data/bigram_tab_{VOCAB}v.npy"
    trigram_path = f"data/trigram_logprobs_{VOCAB}v.npy"
    fourgram_path = f"data/fourgram_logprobs_{VOCAB}v.npy"

    if all(os.path.exists(f) for f in [bigram_path, trigram_path, fourgram_path]):
        print("✓ All n-gram tables already exist, skipping")
        return

    tokens = load_tokens()
    print(f"\nTotal tokens: {len(tokens):,}")

    bigram = build_bigram(tokens)
    np.save(bigram_path, bigram)
    print(f"✓ Saved bigram: {bigram.shape}, {bigram.nbytes/1024/1024:.2f} MB")

    trigram = build_trigram(tokens)
    np.save(trigram_path, trigram)
    print(f"✓ Saved trigram: {trigram.shape}, {trigram.nbytes/1024/1024:.2f} MB")

    fourgram = build_fourgram(tokens)
    np.save(fourgram_path, fourgram)
    print(f"✓ Saved 4-gram: {fourgram.shape}, {fourgram.nbytes/1024/1024:.2f} MB")

    print("\n✓ All n-gram tables built")


if __name__ == "__main__":
    main()
