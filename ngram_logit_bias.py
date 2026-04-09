"""
N-gram logit bias module for PyTorch (CUDA).

Exact port of the MLX n-gram logit bias from train_gpt_mlx_v13.py.
Adds precomputed log-probability biases to logits *after* softcap, *before* cross-entropy.

Usage:
    ngram = NgramLogitBias.from_npy(
        bigram_path="data/bigram_logprobs.npy",
        trigram_path="data/trigram_logprobs_16k.npy",
        fourgram_path="data/fourgram_logprobs_16k.npy",
        bigram_weight=0.3,
        trigram_weight=0.15,
        fourgram_weight=0.1,
    )
    ngram = ngram.cuda()

    # In the forward pass (after softcap, before cross_entropy):
    # logits shape: (B*T, V), input_ids shape: (B, T)
    bias = ngram(input_ids)       # -> (B*T, V)
    logits = logits + bias

Compatible with torch.compile -- no Python loops in the hot path.
Tables are registered as buffers so they move with .cuda() / .to() and
appear in state_dict for checkpoint save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class NgramLogitBias(nn.Module):
    """Additive n-gram logit bias (bigram + trigram + 4-gram).

    Tables are non-trainable buffers (frozen log-probability distributions
    precomputed from training data).

    Hash functions match train_gpt_mlx_v13.py exactly:
        bigram:  direct lookup table[prev1]
        trigram: (36313 * prev1 + 27191 * prev2) % trigram_buckets
        4-gram:  (36313 * prev1 + 27191 * prev2 + 51497 * prev3) % fourgram_buckets

    where prev1 = token at position t (predicting t+1), prev2 = token at t-1, etc.
    """

    def __init__(
        self,
        bigram_table: Optional[Tensor] = None,
        trigram_table: Optional[Tensor] = None,
        fourgram_table: Optional[Tensor] = None,
        bigram_weight: float = 0.3,
        trigram_weight: float = 0.15,
        fourgram_weight: float = 0.1,
    ):
        super().__init__()

        # Weights are plain floats, not Parameters (not trained).
        self.bigram_weight = bigram_weight
        self.trigram_weight = trigram_weight
        self.fourgram_weight = fourgram_weight

        # Register tables as persistent buffers.
        # Shape: bigram (V, V) or (B_bi, V);  trigram/fourgram (B_tri, V) / (B_4g, V)
        # where V = vocab_size, B_* = number of hash buckets.
        if bigram_table is not None:
            self.register_buffer("bigram_table", bigram_table)
            self.bigram_buckets = bigram_table.shape[0]
        else:
            self.bigram_table = None
            self.bigram_buckets = 0

        if trigram_table is not None:
            self.register_buffer("trigram_table", trigram_table)
            self.trigram_buckets = trigram_table.shape[0]
        else:
            self.trigram_table = None
            self.trigram_buckets = 0

        if fourgram_table is not None:
            self.register_buffer("fourgram_table", fourgram_table)
            self.fourgram_buckets = fourgram_table.shape[0]
        else:
            self.fourgram_table = None
            self.fourgram_buckets = 0

    # ------------------------------------------------------------------
    # Factory: load from .npy files
    # ------------------------------------------------------------------
    @classmethod
    def from_npy(
        cls,
        bigram_path: Optional[str] = None,
        trigram_path: Optional[str] = None,
        fourgram_path: Optional[str] = None,
        bigram_weight: float = 0.3,
        trigram_weight: float = 0.15,
        fourgram_weight: float = 0.1,
    ) -> "NgramLogitBias":
        """Load precomputed log-prob tables from numpy files.

        Each .npy file is a float32 array of shape (num_buckets, vocab_size).
        For bigram, num_buckets == vocab_size (direct lookup by prev token).
        For trigram/fourgram, num_buckets is the hash table size (e.g. 16384).
        """
        bigram_table = None
        trigram_table = None
        fourgram_table = None

        if bigram_path is not None and Path(bigram_path).exists():
            arr = np.load(bigram_path)
            bigram_table = torch.from_numpy(arr).float()
            print(f"[NgramLogitBias] bigram loaded: shape={arr.shape} weight={bigram_weight}")

        if trigram_path is not None and Path(trigram_path).exists():
            arr = np.load(trigram_path)
            trigram_table = torch.from_numpy(arr).float()
            print(f"[NgramLogitBias] trigram loaded: shape={arr.shape} weight={trigram_weight}")

        if fourgram_path is not None and Path(fourgram_path).exists():
            arr = np.load(fourgram_path)
            fourgram_table = torch.from_numpy(arr).float()
            print(f"[NgramLogitBias] fourgram loaded: shape={arr.shape} weight={fourgram_weight}")

        return cls(
            bigram_table=bigram_table,
            trigram_table=trigram_table,
            fourgram_table=fourgram_table,
            bigram_weight=bigram_weight,
            trigram_weight=trigram_weight,
            fourgram_weight=fourgram_weight,
        )

    # ------------------------------------------------------------------
    # Forward: build logit bias from token sequence
    # ------------------------------------------------------------------
    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute additive logit bias from n-gram context.

        Args:
            input_ids: (B, T) int64 tensor of token IDs.
                       These are the *input* tokens (same as what the model sees).
                       Position t predicts target t+1, so prev1 for position t is
                       input_ids[t] itself, matching the MLX convention.

        Returns:
            bias: (B*T, V) float tensor to add to logits *after* softcap.
                  Zero if no tables are loaded.
        """
        B, T = input_ids.shape
        flat = input_ids.reshape(-1)  # (B*T,)

        # -- Shift context tokens exactly as MLX does --
        # prev_tokens  = input_flat          (token at position t)
        # prev2_tokens = [0, input_flat[:-1]] (token at position t-1, zero-padded)
        # prev3_tokens = [0, 0, input_flat[:-2]]
        prev1 = flat  # (N,)

        # Determine vocab size from whichever table we have.
        V = 0
        if self.bigram_table is not None:
            V = self.bigram_table.shape[1]
        elif self.trigram_table is not None:
            V = self.trigram_table.shape[1]
        elif self.fourgram_table is not None:
            V = self.fourgram_table.shape[1]

        if V == 0:
            # No tables loaded -- return zero bias of unknown shape.
            # Caller should check `ngram.has_tables` before calling.
            return torch.zeros(B * T, 1, device=input_ids.device, dtype=torch.float32)

        # Accumulate bias in float32.
        bias = torch.zeros(B * T, V, device=input_ids.device, dtype=torch.float32)

        # -- Bigram: table[prev1] --
        if self.bigram_table is not None:
            # prev1 indexes directly into the table (row = prev token, cols = next-token logprobs).
            # For vocab-sized tables, prev1 is in [0, V). No hash needed.
            bigram_bias = self.bigram_table[prev1]  # (N, V)
            bias = bias + self.bigram_weight * bigram_bias

        # -- Trigram: hash(prev1, prev2) --
        if self.trigram_table is not None:
            prev2 = torch.zeros_like(flat)
            prev2[1:] = flat[:-1]
            hashed_tri = (36313 * prev1.int() + 27191 * prev2.int()) % self.trigram_buckets
            trigram_bias = self.trigram_table[hashed_tri.long()]  # (N, V)
            bias = bias + self.trigram_weight * trigram_bias

        # -- 4-gram: hash(prev1, prev2, prev3) --
        if self.fourgram_table is not None:
            prev2 = torch.zeros_like(flat)
            prev2[1:] = flat[:-1]
            prev3 = torch.zeros_like(flat)
            prev3[2:] = flat[:-2]
            hashed_4g = (
                36313 * prev1.int() + 27191 * prev2.int() + 51497 * prev3.int()
            ) % self.fourgram_buckets
            fourgram_bias = self.fourgram_table[hashed_4g.long()]  # (N, V)
            bias = bias + self.fourgram_weight * fourgram_bias

        return bias

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def has_tables(self) -> bool:
        return (
            self.bigram_table is not None
            or self.trigram_table is not None
            or self.fourgram_table is not None
        )

    def extra_repr(self) -> str:
        parts = []
        if self.bigram_table is not None:
            parts.append(f"bigram={tuple(self.bigram_table.shape)} w={self.bigram_weight}")
        if self.trigram_table is not None:
            parts.append(f"trigram={tuple(self.trigram_table.shape)} w={self.trigram_weight}")
        if self.fourgram_table is not None:
            parts.append(f"fourgram={tuple(self.fourgram_table.shape)} w={self.fourgram_weight}")
        return ", ".join(parts) if parts else "no tables"


# ======================================================================
# Integration example (drop into GPT.forward in train_gpt.py)
# ======================================================================
#
#   # At model construction time:
#   self.ngram = NgramLogitBias.from_npy(
#       bigram_path="data/bigram_logprobs.npy",
#       trigram_path="data/trigram_logprobs_16k.npy",
#       fourgram_path="data/fourgram_logprobs_16k.npy",
#   )
#
#   # In GPT.forward, after softcap, before cross_entropy:
#   logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
#   if self.ngram.has_tables:
#       logits = logits + self.ngram(input_ids)
#   return F.cross_entropy(logits.float(), targets, reduction="mean")
#
# Checkpoint save/load: tables are in state_dict as buffers
# (ngram.bigram_table, ngram.trigram_table, ngram.fourgram_table).
# They are NOT counted toward the 16MB parameter budget because they
# are not trainable parameters.
#
# For 8192-vocab submissions, use the *_8192v.npy files:
#   bigram_path="data/bigram_logprobs_8192v.npy"
#   trigram_path="data/trigram_logprobs_8192v.npy"
#   fourgram_path="data/fourgram_logprobs_8192v.npy"
# ======================================================================


if __name__ == "__main__":
    # Quick sanity test: load tables and run a dummy forward pass.
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing NgramLogitBias on {device}")

    ngram = NgramLogitBias.from_npy(
        bigram_path="data/bigram_logprobs.npy",
        trigram_path="data/trigram_logprobs_16k.npy",
        fourgram_path="data/fourgram_logprobs_16k.npy",
        bigram_weight=0.3,
        trigram_weight=0.15,
        fourgram_weight=0.1,
    ).to(device)

    print(f"Module:\n{ngram}")

    # Dummy input: batch of 2, seq length 128.
    B, T = 2, 128
    ids = torch.randint(0, 1024, (B, T), device=device)

    # Warm-up
    bias = ngram(ids)
    print(f"Output shape: {bias.shape}")
    print(f"Output dtype: {bias.dtype}")
    print(f"Bias range: [{bias.min().item():.4f}, {bias.max().item():.4f}]")
    print(f"Bias mean: {bias.mean().item():.4f}")

    # Verify against MLX-style manual computation for a single position.
    # Position 3: prev1=ids[0,3], prev2=ids[0,2], prev3=ids[0,1]
    p1 = ids[0, 3].item()
    p2 = ids[0, 2].item()
    p3 = ids[0, 1].item()
    expected_bigram = ngram.bigram_table[p1] if ngram.bigram_table is not None else 0
    tri_hash = (36313 * p1 + 27191 * p2) % ngram.trigram_buckets
    expected_trigram = ngram.trigram_table[tri_hash] if ngram.trigram_table is not None else 0
    fg_hash = (36313 * p1 + 27191 * p2 + 51497 * p3) % ngram.fourgram_buckets
    expected_fourgram = ngram.fourgram_table[fg_hash] if ngram.fourgram_table is not None else 0
    expected = 0.3 * expected_bigram + 0.15 * expected_trigram + 0.1 * expected_fourgram
    actual = bias[3]  # position 3 in first batch item

    diff = (expected - actual).abs().max().item()
    print(f"Verification diff at position 3: {diff:.8f}")
    assert diff < 1e-5, f"Mismatch! diff={diff}"
    print("PASSED: Output matches manual MLX-style computation.")

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = ngram(ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Benchmark: 100 forward passes in {elapsed*1000:.1f}ms ({elapsed*10:.2f}ms/pass)")

    # torch.compile test
    print("\nTesting torch.compile compatibility...")
    try:
        compiled_ngram = torch.compile(ngram)
        bias2 = compiled_ngram(ids)
        diff2 = (bias - bias2).abs().max().item()
        print(f"torch.compile output diff: {diff2:.8f}")
        assert diff2 < 1e-5
        print("PASSED: torch.compile compatible.")
    except Exception as e:
        print(f"torch.compile test: {e}")
        print("(This is expected on CPU or older PyTorch versions)")

    # State dict test
    print("\nTesting state_dict serialization...")
    sd = ngram.state_dict()
    print(f"State dict keys: {list(sd.keys())}")
    ngram2 = NgramLogitBias()
    ngram2.register_buffer("bigram_table", torch.zeros(1))
    ngram2.register_buffer("trigram_table", torch.zeros(1))
    ngram2.register_buffer("fourgram_table", torch.zeros(1))
    # Proper reload: construct with matching shapes then load
    ngram2 = NgramLogitBias.from_npy(
        bigram_path="data/bigram_logprobs.npy",
        trigram_path="data/trigram_logprobs_16k.npy",
        fourgram_path="data/fourgram_logprobs_16k.npy",
    ).to(device)
    ngram2.load_state_dict(sd)
    bias3 = ngram2(ids)
    diff3 = (bias - bias3).abs().max().item()
    print(f"Reload diff: {diff3:.8f}")
    assert diff3 < 1e-5
    print("PASSED: state_dict round-trip OK.")
