#!/usr/bin/env python3
"""
v09_lloyd_max_quant.py — Verify Lloyd-Max codebook gives lower MSE than linear int6

Loads the precomputed codebook, quantizes random log-normal weights,
compares MSE against uniform int6.

Expected: Lloyd-Max gives ~70-90% less MSE than linear.
"""

import sys
import numpy as np


def linear_int6_quantize(weights):
    w_min, w_max = weights.min(), weights.max()
    step = (w_max - w_min) / 64
    grid = np.linspace(w_min + step/2, w_max - step/2, 64)
    idx = np.argmin(np.abs(weights[:, None] - grid[None, :]), axis=1)
    return grid[idx]


def lloyd_max_quantize(weights, codebook):
    idx = np.argmin(np.abs(weights[:, None] - codebook[None, :]), axis=1)
    return codebook[idx]


def main():
    print("=== V09: LLOYD-MAX QUANTIZATION ===\n")

    # Load codebook
    codebook = np.load("data/lloyd_max_codebook_64.npy")
    print(f"Codebook: {len(codebook)} levels, range [{codebook.min():.4f}, {codebook.max():.4f}]")

    # Generate log-normal weights (matches our finding σ=1.6)
    np.random.seed(42)
    n = 100_000
    weights = np.random.lognormal(mean=-5.0, sigma=1.6, size=n)
    weights = weights * np.random.choice([-1, 1], n)  # symmetric

    # Clip to codebook range (real weights would be similar)
    weights = np.clip(weights, codebook.min(), codebook.max())

    # Quantize both ways
    print(f"\nQuantizing {n:,} weights...")
    q_lin = linear_int6_quantize(weights)
    q_llm = lloyd_max_quantize(weights, codebook)

    mse_lin = np.mean((weights - q_lin) ** 2)
    mse_llm = np.mean((weights - q_llm) ** 2)

    improvement = (1 - mse_llm / mse_lin) * 100

    print(f"\nLinear int6 MSE:    {mse_lin:.8f}")
    print(f"Lloyd-Max int6 MSE: {mse_llm:.8f}")
    print(f"Improvement:        {improvement:.1f}%")

    # Validation
    if improvement < 50:
        print(f"\n✗ FAIL: improvement {improvement:.0f}% is less than expected (>50%)")
        print(f"  This might mean the codebook isn't tuned for log-normal weights.")
        print(f"  Re-run chore/06_lloyd_max.py with actual model weights.")
        return 1

    print(f"\n✓ PASS: Lloyd-Max saves {improvement:.0f}% quantization error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
