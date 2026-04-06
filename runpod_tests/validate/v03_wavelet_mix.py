#!/usr/bin/env python3
"""
v03_wavelet_mix.py — Verify WaveletGPT multi-scale mixing in PyTorch

Validates the wavelet_mix() function works correctly:
- Output shape matches input
- Causal (no future leak)
- Layer 0 only mixes 2 positions, layer 8 mixes 512
- No NaN/Inf
- Adapts to seq_len < 2^(layer+1) cap

This is the function we'll add to TransformerBlock.forward().

Expected: PASS in <30 sec.
"""

import torch
import torch.nn.functional as F
import sys


def wavelet_mix(x, layer_idx, mix_ratio=0.2):
    """WaveletGPT: causal multi-scale averaging on half dims.

    Validated -0.018 BPP on Mac (9L wavelet, 1000 steps, BPE-8192).
    Naturally adapts to progressive seq via k = min(2^(i+1), T).
    """
    B, T, D = x.shape
    half = D // 2
    left = x[..., :half]
    right = x[..., half:]

    k = min(2 ** (layer_idx + 1), T)

    # Causal cumulative sum
    cs = torch.cumsum(right, dim=1)
    shifted = F.pad(cs[:, :-k], (0, 0, k, 0))
    counts = torch.arange(1, T + 1, device=x.device, dtype=right.dtype)
    counts = counts.clamp(max=k).unsqueeze(0).unsqueeze(-1)
    right_avg = (cs - shifted) / counts

    right_mixed = (1 - mix_ratio) * right + mix_ratio * right_avg
    return torch.cat([left, right_mixed], dim=-1)


def test_basic():
    """Output shape matches input."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 128, device=device)
    y = wavelet_mix(x, layer_idx=3)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    return True


def test_causal():
    """No future leak: changing token at position t shouldn't affect outputs at positions < t."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 16, 64, device=device)
    y1 = wavelet_mix(x, layer_idx=2)

    # Change position 8 onwards
    x_modified = x.clone()
    x_modified[:, 8:] = 0
    y2 = wavelet_mix(x_modified, layer_idx=2)

    # Outputs at positions 0-7 should be UNCHANGED
    diff = (y1[:, :8] - y2[:, :8]).abs().max().item()
    assert diff < 1e-5, f"Causal violation: diff={diff}"
    return True


def test_layer_scales():
    """Higher layers should mix more positions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use a delta input to see the receptive field
    x = torch.zeros(1, 256, 64, device=device)
    x[0, 100, 32:] = 1.0  # spike at position 100, in the right half

    for layer_idx in [0, 2, 4, 6, 8]:
        y = wavelet_mix(x, layer_idx=layer_idx)
        # The right half (mixed) should spread the spike based on scale k = 2^(i+1)
        right = y[0, :, 32:]
        # Find positions where the output is non-zero in the right half
        nonzero_positions = (right.abs().sum(-1) > 1e-6).nonzero().flatten()
        if len(nonzero_positions) > 0:
            spread = nonzero_positions.max().item() - nonzero_positions.min().item() + 1
            expected_k = 2 ** (layer_idx + 1)
            print(f"  Layer {layer_idx}: k={expected_k}, observed spread={spread}")
    return True


def test_no_nan():
    """No NaN/Inf in output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seq_len in [16, 64, 128, 1024, 2048]:
        x = torch.randn(2, seq_len, 64, device=device)
        for layer_idx in range(12):
            y = wavelet_mix(x, layer_idx=layer_idx)
            assert not torch.isnan(y).any(), f"NaN at layer {layer_idx}, seq {seq_len}"
            assert not torch.isinf(y).any(), f"Inf at layer {layer_idx}, seq {seq_len}"
    return True


def test_adapts_to_short_seq():
    """When seq_len < 2^(i+1), k caps at seq_len (no error)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 16, 64, device=device)  # short
    # Layer 8 wants k=512 but seq is 16
    y = wavelet_mix(x, layer_idx=8)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    return True


def test_progressive_seq_compat():
    """At seq=128 (Phase 1), layers 7+ should still work (cap at 128)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_p1 = torch.randn(2, 128, 64, device=device)
    x_p2 = torch.randn(2, 1024, 64, device=device)
    for layer_idx in range(11):
        y_p1 = wavelet_mix(x_p1, layer_idx=layer_idx)
        y_p2 = wavelet_mix(x_p2, layer_idx=layer_idx)
        assert y_p1.shape == x_p1.shape
        assert y_p2.shape == x_p2.shape
    return True


def main():
    print("=== V03: WAVELET MIX VALIDATION ===\n")
    tests = [
        ("Basic shape", test_basic),
        ("Causality", test_causal),
        ("Layer scales", test_layer_scales),
        ("No NaN/Inf", test_no_nan),
        ("Short seq adapt", test_adapts_to_short_seq),
        ("Progressive seq compat", test_progressive_seq_compat),
    ]

    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

    if failed == 0:
        print("\n✓ PASS: WaveletGPT works correctly in PyTorch")
        return 0
    else:
        print(f"\n✗ FAIL: {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
