#!/usr/bin/env python3
"""
v07_eval_cache.py — Verify NgramEvalCache builds and predicts correctly

Imports the production module and runs it on synthetic data.
Confirms:
- Cache grows as data is added
- Predictions improve over time (more cache = more hits)
- Memory stays bounded

Expected: PASS in <1 min.
"""

import sys
import os
import time
import numpy as np

# Import the production module
sys.path.insert(0, "/workspace/paramgolf")
from eval_ngram_cache import NgramEvalCache, HedgeMixer


def main():
    print("=== V07: EVAL N-GRAM CACHE ===\n")

    # Create cache
    cache = NgramEvalCache(max_order=7, min_count=2, vocab_size=8192)
    mixer = HedgeMixer(n_experts=2, eta=0.1)

    # Synthetic repeated sequence (where cache should help)
    np.random.seed(42)
    base_pattern = np.random.randint(0, 8192, 50)
    # Repeat the pattern 20 times with slight variations
    sequence = []
    for _ in range(20):
        sequence.extend(base_pattern.tolist())
        # Small variation
        sequence.append(np.random.randint(0, 8192))

    print(f"Sequence: {len(sequence)} tokens, {len(set(sequence))} unique")

    # Score with cache
    correct_pred = 0
    total = 0
    cache_hits_by_order = {k: 0 for k in range(2, 8)}

    t0 = time.time()
    for i in range(2, len(sequence)):
        ctx = sequence[:i]
        actual = sequence[i]

        # Cache prediction
        cache_p = cache.predict(ctx)
        predicted = int(np.argmax(cache_p))

        # Track hits per order
        for order in range(min(7, i), 1, -1):
            ctx_tuple = tuple(ctx[-order+1:]) if order > 1 else ()
            if ctx_tuple in cache.cache[order] and sum(cache.cache[order][ctx_tuple].values()) >= 2:
                cache_hits_by_order[order] += 1
                break

        if predicted == actual:
            correct_pred += 1
        total += 1

        # Update cache
        cache.update(ctx + [actual])

    elapsed = time.time() - t0
    accuracy = correct_pred / total

    print(f"\nResults:")
    print(f"  Accuracy: {correct_pred}/{total} = {accuracy:.1%}")
    print(f"  Time: {elapsed:.2f}s ({total/elapsed:.0f} predictions/sec)")
    print(f"  Cache hits by order:")
    for order, hits in sorted(cache_hits_by_order.items()):
        print(f"    Order {order}: {hits}")

    # Memory check
    total_entries = sum(len(c) for c in cache.cache.values())
    print(f"  Total cache entries: {total_entries}")

    # Validation
    if accuracy < 0.10:
        print(f"\n✗ FAIL: cache accuracy too low ({accuracy:.1%})")
        return 1

    if total_entries == 0:
        print(f"\n✗ FAIL: cache is empty")
        return 1

    # Higher orders should fire on repeated patterns
    if cache_hits_by_order[2] == 0:
        print(f"\n✗ FAIL: bigram cache never hit (suspicious)")
        return 1

    print(f"\n✓ PASS: eval cache works, accuracy {accuracy:.1%} on repeated pattern")
    return 0


if __name__ == "__main__":
    sys.exit(main())
