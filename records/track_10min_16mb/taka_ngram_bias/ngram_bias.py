"""
N-gram logit bias module for Parameter Golf.

Usage:
    1. Call precompute_ngram_tables() before training to build tables from training data
    2. Call add_ngram_bias() in the model's forward pass to add biases to output logits
    3. Tables are stored as numpy arrays, can be int8+zstd compressed for the artifact

This module works with both MLX (Apple Silicon) and PyTorch (CUDA).
"""
import numpy as np
from pathlib import Path


def precompute_ngram_tables(
    train_shard_path: str,
    vocab_size: int = 1024,
    bigram_buckets: int = 0,  # 0 = direct table (vocab x vocab)
    trigram_buckets: int = 8192,
    fourgram_buckets: int = 8192,
    smoothing: float = 1.0,
    output_dir: str = "data",
) -> dict:
    """Precompute n-gram log-probability tables from a training shard.

    Returns dict with 'bigram', 'trigram', 'fourgram' numpy arrays.
    Also saves to disk as .npy files.
    """
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(train_shard_path, dtype="<u2", offset=header_bytes).astype(np.int64)
    n = len(tokens)
    print(f"ngram_bias: loaded {n} tokens from {train_shard_path}")

    tables = {}

    # Bigram: direct P(next | prev)
    bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(bigram_counts, (tokens[:-1], tokens[1:]), 1)
    bigram_probs = (bigram_counts + smoothing) / (bigram_counts.sum(axis=1, keepdims=True) + smoothing * vocab_size)
    tables["bigram"] = np.log(bigram_probs).astype(np.float32)
    np.save(f"{output_dir}/bigram_logprobs.npy", tables["bigram"])
    print(f"ngram_bias: bigram table {tables['bigram'].shape}, {tables['bigram'].nbytes / 1e6:.1f}MB")

    # Trigram: hash(prev2, prev1) -> bucket
    h3 = (36313 * tokens[1:-1] + 27191 * tokens[:-2]) % trigram_buckets
    tri_counts = np.zeros((trigram_buckets, vocab_size), dtype=np.float64)
    np.add.at(tri_counts, (h3, tokens[2:]), 1)
    tri_probs = (tri_counts + smoothing) / (tri_counts.sum(axis=1, keepdims=True) + smoothing * vocab_size)
    tables["trigram"] = np.log(tri_probs).astype(np.float32)
    np.save(f"{output_dir}/trigram_logprobs.npy", tables["trigram"])
    print(f"ngram_bias: trigram table {tables['trigram'].shape}, {tables['trigram'].nbytes / 1e6:.1f}MB")

    # 4-gram: hash(prev3, prev2, prev1) -> bucket
    h4 = (36313 * tokens[2:-1] + 27191 * tokens[1:-2] + 51497 * tokens[:-3]) % fourgram_buckets
    four_counts = np.zeros((fourgram_buckets, vocab_size), dtype=np.float64)
    np.add.at(four_counts, (h4, tokens[3:]), 1)
    four_probs = (four_counts + smoothing) / (four_counts.sum(axis=1, keepdims=True) + smoothing * vocab_size)
    tables["fourgram"] = np.log(four_probs).astype(np.float32)
    np.save(f"{output_dir}/fourgram_logprobs.npy", tables["fourgram"])
    print(f"ngram_bias: fourgram table {tables['fourgram'].shape}, {tables['fourgram'].nbytes / 1e6:.1f}MB")

    return tables


# ---- PyTorch version (for CUDA train_gpt.py) ----

def add_ngram_bias_torch(
    logits,  # [N, vocab_size] tensor
    input_ids_flat,  # [N] tensor (previous tokens = input to model)
    bigram_table,  # [vocab, vocab] tensor or None
    trigram_table=None,  # [buckets, vocab] tensor or None
    fourgram_table=None,  # [buckets, vocab] tensor or None
    bigram_weight: float = 0.2,
    trigram_weight: float = 0.15,
    fourgram_weight: float = 0.1,
    trigram_buckets: int = 8192,
    fourgram_buckets: int = 8192,
):
    """Add n-gram logit biases to model output logits. PyTorch version."""
    import torch

    prev1 = input_ids_flat  # shape [N]

    if bigram_table is not None:
        logits = logits + bigram_weight * bigram_table[prev1]

    if trigram_table is not None:
        prev2 = torch.cat([torch.zeros(1, dtype=torch.long, device=prev1.device), prev1[:-1]])
        h3 = (36313 * prev1.long() + 27191 * prev2.long()) % trigram_buckets
        logits = logits + trigram_weight * trigram_table[h3]

    if fourgram_table is not None:
        prev2 = torch.cat([torch.zeros(1, dtype=torch.long, device=prev1.device), prev1[:-1]])
        prev3 = torch.cat([torch.zeros(2, dtype=torch.long, device=prev1.device), prev1[:-2]])
        h4 = (36313 * prev1.long() + 27191 * prev2.long() + 51497 * prev3.long()) % fourgram_buckets
        logits = logits + fourgram_weight * fourgram_table[h4]

    return logits


# ---- MLX version (for Apple Silicon train_gpt_mlx.py) ----

def add_ngram_bias_mlx(
    logits,  # [N, vocab_size] mx.array
    input_ids_flat,  # [N] mx.array
    bigram_table,  # [vocab, vocab] mx.array or None
    trigram_table=None,
    fourgram_table=None,
    bigram_weight: float = 0.2,
    trigram_weight: float = 0.15,
    fourgram_weight: float = 0.1,
    trigram_buckets: int = 8192,
    fourgram_buckets: int = 8192,
):
    """Add n-gram logit biases to model output logits. MLX version."""
    import mlx.core as mx

    prev1 = input_ids_flat

    if bigram_table is not None:
        logits = logits + bigram_weight * bigram_table[prev1].astype(logits.dtype)

    if trigram_table is not None:
        prev2 = mx.concatenate([mx.zeros((1,), dtype=mx.int32), prev1[:-1]])
        h3 = (36313 * prev1.astype(mx.int32) + 27191 * prev2.astype(mx.int32)) % trigram_buckets
        logits = logits + trigram_weight * trigram_table[h3].astype(logits.dtype)

    if fourgram_table is not None:
        prev2 = mx.concatenate([mx.zeros((1,), dtype=mx.int32), prev1[:-1]])
        prev3 = mx.concatenate([mx.zeros((2,), dtype=mx.int32), prev1[:-2]])
        h4 = (36313 * prev1.astype(mx.int32) + 27191 * prev2.astype(mx.int32) + 51497 * prev3.astype(mx.int32)) % fourgram_buckets
        logits = logits + fourgram_weight * fourgram_table[h4].astype(logits.dtype)

    return logits


def compress_tables_for_artifact(tables: dict, output_path: str):
    """Compress n-gram tables to int8+zstd for the 16MB artifact."""
    import zstandard as zstd
    import pickle

    compressed = {}
    for name, table in tables.items():
        vmin, vmax = float(table.min()), float(table.max())
        scale = (vmax - vmin) / 255.0
        quantized = np.round((table - vmin) / scale).astype(np.uint8)
        compressed[name] = {
            "quantized": quantized,
            "vmin": vmin,
            "scale": scale,
            "shape": table.shape,
        }

    raw = pickle.dumps(compressed, protocol=pickle.HIGHEST_PROTOCOL)
    blob = zstd.ZstdCompressor(level=22).compress(raw)
    with open(output_path, "wb") as f:
        f.write(blob)
    print(f"ngram_bias: compressed {len(tables)} tables to {len(blob) / 1e6:.1f}MB at {output_path}")
    return len(blob)


def load_compressed_tables(path: str) -> dict:
    """Load int8+zstd compressed n-gram tables."""
    import zstandard as zstd
    import pickle

    with open(path, "rb") as f:
        blob = f.read()
    compressed = pickle.loads(zstd.ZstdDecompressor().decompress(blob))

    tables = {}
    for name, data in compressed.items():
        dequant = data["quantized"].astype(np.float32) * data["scale"] + data["vmin"]
        tables[name] = dequant.reshape(data["shape"])
    return tables
