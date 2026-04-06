#!/usr/bin/env python3
"""
v08_hedge_mixer.py — Verify HedgeMixer adapts weights toward better expert

Two synthetic experts (one good, one bad). After many updates, the
hedge mixer should put almost all weight on the good expert.

Expected: PASS in <30 sec.
"""

import sys
import numpy as np

sys.path.insert(0, "/workspace/paramgolf")
from eval_ngram_cache import HedgeMixer


def main():
    print("=== V08: HEDGE MIXER ===\n")

    np.random.seed(42)
    V = 256

    # Two experts:
    # Expert A: always assigns 80% to the correct token
    # Expert B: random uniform
    mixer = HedgeMixer(n_experts=2, eta=0.1)

    print(f"Expert A: 80% accurate")
    print(f"Expert B: random")
    print(f"Initial weights: {mixer.weights}\n")

    n_steps = 200
    for step in range(n_steps):
        actual = np.random.randint(0, V)

        # Expert A: 80% on correct, rest uniform
        a_probs = np.full(V, 0.20 / V)
        a_probs[actual] = 0.80

        # Expert B: uniform
        b_probs = np.full(V, 1.0 / V)

        # Mix
        mixed = mixer.mix([a_probs, b_probs])

        # Update based on expert performance
        mixer.update([a_probs[actual], b_probs[actual]])

    print(f"Final weights: {mixer.weights}")
    print(f"  Expert A: {mixer.weights[0]:.3f}")
    print(f"  Expert B: {mixer.weights[1]:.3f}")

    # Validation: Expert A should have most of the weight
    if mixer.weights[0] > 0.90:
        print(f"\n✓ PASS: hedge correctly identified Expert A ({mixer.weights[0]:.1%})")
        return 0
    else:
        print(f"\n✗ FAIL: hedge didn't converge to good expert")
        return 1


if __name__ == "__main__":
    sys.exit(main())
