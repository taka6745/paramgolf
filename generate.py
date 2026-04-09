#!/usr/bin/env python3
"""
Generate text from a saved paramgolf model.

Usage:
  python3 generate.py                                    # defaults: baseline 200-step model
  python3 generate.py --model logs/v3_leakyrelu_50_mlx_model.npz --prompt "The meaning of life is"
  python3 generate.py --model logs/mlx_baseline_mlx_model.npz --prompt "Once upon a time" --tokens 200
  python3 generate.py --model logs/v2_50step_mlx_model.npz --temperature 0.5
"""
from __future__ import annotations

import argparse
import math
import sys
import os

import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# ---------------------------------------------------------------------------
# Model architecture (must match train_gpt_mlx_v3.py / train_gpt_mlx.py)
# ---------------------------------------------------------------------------

COMPUTE_DTYPE = mx.bfloat16

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc(x)
        h = mx.where(h >= 0, h, 0.5 * h)  # LeakyReLU(0.5)^2
        return self.proj(h * h)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, logit_softcap: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def logits(self, input_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        raw = x @ self.tok_emb.weight.astype(x.dtype).T
        c = self.logit_softcap
        return c * mx.tanh(raw / c)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model: GPT, sp: spm.SentencePieceProcessor, prompt: str,
             max_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
    token_ids = sp.encode(prompt)
    if not token_ids:
        token_ids = [0]

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    # Print the prompt text
    sys.stdout.write(prompt)
    sys.stdout.flush()

    for _ in range(max_tokens):
        # Only feed the last 1024 tokens (context window)
        ctx = token_ids[-1024:]
        x = mx.array([ctx], dtype=mx.int32)
        logit_out = model.logits(x)
        # Take logits for the last position
        next_logits = logit_out[-1].astype(mx.float32)
        mx.eval(next_logits)

        if temperature < 1e-6:
            # Greedy
            next_token = int(mx.argmax(next_logits).item())
        else:
            # Temperature + top-k sampling
            next_logits = next_logits / temperature
            if top_k > 0 and top_k < next_logits.shape[0]:
                # Zero out everything below top-k
                logits_np = np.array(next_logits)
                threshold = np.partition(logits_np, -top_k)[-top_k]
                logits_np[logits_np < threshold] = -1e9
                next_logits = mx.array(logits_np)
            probs = mx.softmax(next_logits, axis=-1)
            next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

        token_ids.append(next_token)
        # Decode just the new token
        new_text = sp.decode([next_token])
        sys.stdout.write(new_text)
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Generated {max_tokens} tokens")


def main():
    parser = argparse.ArgumentParser(description="Generate text from a paramgolf model")
    parser.add_argument("--model", default="logs/mlx_baseline_mlx_model.npz",
                        help="Path to .npz model file")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece .model file")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Text prompt to continue from")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy, higher = more random)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    # Model architecture (must match what was used during training)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    args = parser.parse_args()

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab_size={sp.vocab_size()})")

    # Build model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    # Load weights
    print(f"Loading model: {args.model}")
    weights = dict(mx.load(args.model))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model: {n_params:,} parameters, {args.num_layers}L {args.dim}d")

    # Generate
    generate(model, sp, args.prompt, max_tokens=args.tokens,
             temperature=args.temperature, top_k=args.top_k)


if __name__ == "__main__":
    main()
