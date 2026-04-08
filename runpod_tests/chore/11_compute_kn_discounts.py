#!/usr/bin/env python3
"""11_compute_kn_discounts.py — compute per-bucket KN-style discounts.

For each n-gram bucket B, computes a confidence discount in [0, 1]:
  discount[B] = 1 - H_normalized[B]
where H_normalized is the entropy of softmax(log_probs[B,:]) normalized by log(VOCAB).

Intuition (Modified Kneser-Ney inspired):
- Buckets with concentrated distributions (low entropy) are confident → keep bias.
- Buckets with flat distributions (high entropy) are uncertain → discount bias.

Outputs (each shape (HASH_BUCKETS,) float32):
  data/bigram_kn_discount.npy
  data/trigram_kn_discount.npy
  data/fourgram_kn_discount.npy

Run-once. Idempotent (skips if outputs exist + newer than inputs).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

DATA = Path("data")

INPUTS = [
    ("bigram_logprobs.npy", "bigram_kn_discount.npy"),
    ("trigram_logprobs.npy", "trigram_kn_discount.npy"),
    ("fourgram_logprobs.npy", "fourgram_kn_discount.npy"),
]


def compute_discount(logp_path: Path, out_path: Path):
    if not logp_path.exists():
        print(f"  ⚠ {logp_path} missing — skipping {out_path.name}")
        return
    if out_path.exists() and out_path.stat().st_mtime >= logp_path.stat().st_mtime:
        print(f"  ✓ {out_path.name} up-to-date — skipping")
        return
    t0 = time.time()
    print(f"  loading {logp_path.name}...")
    logp = np.load(logp_path).astype(np.float32)  # (HASH_BUCKETS, VOCAB)
    HASH_BUCKETS, VOCAB = logp.shape
    print(f"    shape={logp.shape}")
    # Softmax along vocab axis (numerically stable)
    logp_max = logp.max(axis=-1, keepdims=True)
    exp_logp = np.exp(logp - logp_max)
    Z = exp_logp.sum(axis=-1, keepdims=True)
    P = exp_logp / np.maximum(Z, 1e-12)
    # Entropy per bucket
    eps = 1e-12
    H = -(P * np.log(np.maximum(P, eps))).sum(axis=-1)  # (HASH_BUCKETS,)
    H_norm = H / np.log(float(VOCAB))  # in [0, 1]
    # Discount: confident bucket (low entropy) → discount near 1 (keep bias)
    #           uncertain bucket (high entropy) → discount near 0 (drop bias)
    discount = (1.0 - H_norm).clip(0.0, 1.0).astype(np.float32)
    np.save(out_path, discount)
    print(f"    → {out_path.name}: mean={discount.mean():.4f} std={discount.std():.4f} "
          f"min={discount.min():.4f} max={discount.max():.4f} ({time.time()-t0:.1f}s)")


def main():
    if not DATA.exists():
        print(f"FATAL: {DATA} not found")
        return 1
    print(f"11_compute_kn_discounts: building per-bucket KN-style discounts in {DATA}/")
    for inp, out in INPUTS:
        compute_discount(DATA / inp, DATA / out)
    print("11_compute_kn_discounts: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
