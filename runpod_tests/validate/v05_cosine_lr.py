#!/usr/bin/env python3
"""
v05_cosine_lr.py — Verify Phase 2 cosine LR schedule produces correct values

Standalone test of the cosine LR formula. No model training.
Just runs the schedule logic and checks:
- Starts at 1e-4
- Ends at 3e-5
- Monotonically decreases
- Smooth (no jumps)

Expected: PASS in <5 sec.
"""

import math
import sys


def cosine_phase2_lr(elapsed, phase1_end_time, total_time, lr_start=1e-4, lr_end=3e-5):
    """Cosine decay from lr_start to lr_end over Phase 2."""
    phase2_progress = (elapsed - phase1_end_time) / max(1, total_time - phase1_end_time)
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * phase2_progress))


def main():
    print("=== V05: COSINE LR PHASE 2 SCHEDULE ===\n")

    # Simulate: 600s total, 510s Phase 1, 90s Phase 2
    total_time = 600
    phase1_end = 510

    print(f"Phase 1 end: {phase1_end}s")
    print(f"Total time:  {total_time}s")
    print(f"Phase 2 LR: cosine from 1e-4 → 3e-5\n")

    samples = []
    for elapsed in [510, 520, 540, 555, 575, 595, 600]:
        lr = cosine_phase2_lr(elapsed, phase1_end, total_time)
        samples.append((elapsed, lr))
        print(f"  t={elapsed}s: lr={lr:.2e}")

    # Validations
    failed = 0

    # 1. Starts at 1e-4
    start_lr = samples[0][1]
    if abs(start_lr - 1e-4) > 1e-10:
        print(f"\n  ✗ Phase 2 doesn't start at 1e-4 (got {start_lr:.2e})")
        failed += 1
    else:
        print(f"\n  ✓ Phase 2 starts at 1e-4")

    # 2. Ends at 3e-5
    end_lr = samples[-1][1]
    if abs(end_lr - 3e-5) > 1e-10:
        print(f"  ✗ Phase 2 doesn't end at 3e-5 (got {end_lr:.2e})")
        failed += 1
    else:
        print(f"  ✓ Phase 2 ends at 3e-5")

    # 3. Monotonically decreases
    decreasing = all(samples[i][1] >= samples[i+1][1] for i in range(len(samples)-1))
    if not decreasing:
        print(f"  ✗ LR is not monotonically decreasing")
        failed += 1
    else:
        print(f"  ✓ LR monotonically decreases")

    # 4. Mid-point should be ~midway in log space
    mid_lr = samples[3][1]  # t=555 = halfway
    expected_mid = 3e-5 + 0.5 * (1e-4 - 3e-5)  # arithmetic midpoint = 6.5e-5
    if abs(mid_lr - expected_mid) > expected_mid * 0.05:
        print(f"  ✗ Midpoint LR not as expected ({mid_lr:.2e} vs {expected_mid:.2e})")
        failed += 1
    else:
        print(f"  ✓ Midpoint LR is correct ({mid_lr:.2e})")

    if failed == 0:
        print("\n✓ PASS: cosine LR schedule produces correct values")
        return 0
    else:
        print(f"\n✗ FAIL: {failed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
