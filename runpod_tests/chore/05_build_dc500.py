#!/usr/bin/env python3
"""
05_build_dc500.py — Build Distributional Categories (DC500)

Runs on: any pod (CPU work)
Time: ~2 min

DC500 = automatically discovered token clusters based on co-occurrence.
Validated -0.010 BPP on Mac. Used as additional logit bias signal.
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

VOCAB = 8192
N_CATEGORIES = 500
DATA_DIR = Path("data/datasets/fineweb10B_bpe8192")
OUTPUT = "data/dist_cats_500_8192.npz"


def main():
    if os.path.exists(OUTPUT):
        print(f"✓ {OUTPUT} already exists, skipping")
        return

    # Load bigram table (already built)
    bigram_log = np.load("data/bigram_tab_8192v.npy")
    print(f"Loaded bigram table: {bigram_log.shape}")

    # Convert to probability distribution per token
    # Each token's row = its distributional fingerprint
    # We cluster these fingerprints into 500 categories
    probs = np.exp(bigram_log)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Take only the rows that correspond to vocab tokens
    # (the bigram table is hashed, but we want per-token clustering)
    # For DC, we need each TOKEN's distribution, not each hash bucket
    # Approximate: use hash buckets directly (may merge some tokens)

    print(f"Clustering {probs.shape[0]} distributions into {N_CATEGORIES} categories...")
    km = KMeans(n_clusters=N_CATEGORIES, n_init=3, max_iter=100, random_state=42)
    labels = km.fit_predict(probs)

    # Build the lookup: token_id -> category, category -> next_token_dist
    cat_centers = km.cluster_centers_  # (500, vocab)
    cat_log_centers = np.log(cat_centers + 1e-10).astype(np.float32)

    np.savez(OUTPUT,
             token_to_cat=labels.astype(np.int16),
             cat_log_probs=cat_log_centers)

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"✓ Saved {OUTPUT}: {size_mb:.2f} MB")
    print(f"  token_to_cat: {labels.shape}")
    print(f"  cat_log_probs: {cat_log_centers.shape}")


if __name__ == "__main__":
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        os.system("pip install -q scikit-learn")
        from sklearn.cluster import KMeans
    main()
