# Validate — Confirm Mac Wins Port to CUDA

These tests run on **3060 ($0.18/hr)**. Goal: confirm that things validated on Mac actually compile and run on CUDA without surprises. **No new findings here — just sanity checks.**

## Hardware

3060 (12 GB). Total time: ~1 hr. Cost: ~$0.20.

## Test List

| # | Script | What | Time |
|---|---|---|---|
| 01 | `v01_smoke_test.sh` | Base port: 50 steps, loss decreases | 3 min |
| 02 | `v02_ngram_bias.py` | N-gram bias forward pass | 30 sec |
| 03 | `v03_wavelet_mix.py` | WaveletGPT in PyTorch | 30 sec |
| 04 | `v04_progressive_seq.sh` | Phase transition triggers correctly | 3 min |
| 05 | `v05_cosine_lr.py` | Cosine LR schedule outputs | 5 sec |
| 06 | `v06_ema_model.py` | EMA averaging works | 30 sec |
| 07 | `v07_eval_cache.py` | Cache module on real model | 1 min |
| 08 | `v08_hedge_mixer.py` | Hedge mixer adapts weights | 30 sec |
| 09 | `v09_lloyd_max_quant.py` | Lloyd-Max post-training quant | 1 min |
| 10 | `v10_full_stack_smoke.sh` | Everything combined, 100 steps | 5 min |

## Pass Criteria

- Each test exits with code 0
- No NaN/Inf in any output
- Loss decreases over training steps
- Quantized models are within ~5% of unquantized (validation that quant works)

## What This Catches

- PyTorch version incompatibilities (`enable_gqa` etc.)
- CUDA-specific bugs (device ops, dtype mismatches)
- Numerical precision issues (fp16/bf16)
- API differences from MLX (Mac) version

## What This DOES NOT Test

- Actual quality at scale — that's for `unknown/`
- Performance / throughput — different hardware = different bottlenecks
- Distributed training (single GPU only)

## Run Order

Tests are independent. You can run them in any order, but they're numbered for logical flow. If `v01` fails, fix the port before running anything else.
