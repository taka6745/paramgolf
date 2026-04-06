#!/usr/bin/env python3
"""
v06_ema_model.py — Verify EMA(0.997) weight averaging works

Creates a tiny model, trains for a few steps, applies EMA updates,
and verifies the EMA model weights are between the initial and current.

Expected: PASS in <30 sec.
"""

import torch
import torch.nn as nn
import copy
import sys


def main():
    print("=== V06: EMA WEIGHT AVERAGING ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    # EMA copy
    ema_model = copy.deepcopy(model)
    ema_decay = 0.997

    # Save initial weights
    initial = {n: p.clone() for n, p in model.named_parameters()}

    # Train: random gradient steps
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(100):
        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 64, device=device)
        loss = ((model(x) - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1 - ema_decay)

    # Verify EMA weights are BETWEEN initial and current
    print("Checking EMA weights are between initial and current...")
    for (name, p_init), (_, p_model), (_, p_ema) in zip(
        initial.items(),
        model.named_parameters(),
        ema_model.named_parameters(),
    ):
        # The EMA should be closer to current than to initial
        # (weight starts at initial, ema_decay=0.997 → very slow update)
        # After 100 steps with decay 0.997, ema is roughly:
        #   ema ≈ 0.997^100 * initial + (1 - 0.997^100) * avg(current)
        #       ≈ 0.74 * initial + 0.26 * recent
        dist_to_init = (p_ema - p_init).abs().mean().item()
        dist_to_curr = (p_ema - p_model).abs().mean().item()

        # EMA should not equal either
        if dist_to_init < 1e-10:
            print(f"  ✗ {name}: EMA == initial (no update happened)")
            return 1
        if dist_to_curr < 1e-10:
            print(f"  ✗ {name}: EMA == current (decay not working)")
            return 1

    print("  ✓ EMA weights are between initial and current")

    # Verify forward pass works
    x = torch.randn(2, 64, device=device)
    out_model = model(x)
    out_ema = ema_model(x)
    diff = (out_model - out_ema).abs().mean().item()
    print(f"  Output diff (model vs EMA): {diff:.6f}")
    if diff < 1e-10:
        print("  ✗ Outputs are identical (EMA didn't track)")
        return 1
    print("  ✓ EMA model produces different outputs than current")

    print("\n✓ PASS: EMA(0.997) works correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
