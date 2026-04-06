#!/usr/bin/env python3
"""
04_build_ngrams.py — Build bigram, trigram, 4-gram tables for BPE-8192

Runs on: any pod (CPU work)
Time: ~4 min
Outputs:
    data/bigram_tab_8192v.npy        (2.94 MB, 16384 hash buckets)
    data/trigram_logprobs_8192v.npy  (0.64 MB)
    data/fourgram_logprobs_8192v.npy (0.32 MB)

These get loaded by train_gpt.py and added as logit bias during training.
"""

import os
import numpy as np
from collections import defaultdict
from pathlib import Path

VOCAB = 8192
HASH_BUCKETS = 16384  # 2x vocab for low collision
DATA_DIR = Path("data/datasets/fineweb10B_bpe8192")

# Hash functions (signed polynomial — confirmed +0.003 BPP on Mac)
def hash_bigram(prev):
    return (prev * 36313) % HASH_BUCKETS

def hash_trigram(prev2, prev1):
    return ((prev2 * 36313 + prev1 * 27191) % HASH_BUCKETS)

def hash_fourgram(prev3, prev2, prev1):
    return ((prev3 * 36313 + prev2 * 27191 + prev1 * 51497) % HASH_BUCKETS)


def load_tokens():
    """Load all training tokens from BPE-8192 shards."""
    print("Loading tokens from shards...")
    all_tokens = []
    for shard_path in sorted(DATA_DIR.glob("fineweb_train_*.bin")):
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
    if all(os.path.exists(f) for f in [
        "data/bigram_tab_8192v.npy",
        "data/trigram_logprobs_8192v.npy",
        "data/fourgram_logprobs_8192v.npy",
    ]):
        print("✓ All n-gram tables already exist, skipping")
        return

    tokens = load_tokens()
    print(f"\nTotal tokens: {len(tokens):,}")

    bigram = build_bigram(tokens)
    np.save("data/bigram_tab_8192v.npy", bigram)
    print(f"✓ Saved bigram: {bigram.shape}, {bigram.nbytes/1024/1024:.2f} MB")

    trigram = build_trigram(tokens)
    np.save("data/trigram_logprobs_8192v.npy", trigram)
    print(f"✓ Saved trigram: {trigram.shape}, {trigram.nbytes/1024/1024:.2f} MB")

    fourgram = build_fourgram(tokens)
    np.save("data/fourgram_logprobs_8192v.npy", fourgram)
    print(f"✓ Saved 4-gram: {fourgram.shape}, {fourgram.nbytes/1024/1024:.2f} MB")

    print("\n✓ All n-gram tables built")


if __name__ == "__main__":
    main()
