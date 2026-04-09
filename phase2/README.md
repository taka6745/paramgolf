# phase2/ — Speed work on the locked-in Phase 1 model

This directory holds the Phase 2 speed optimizations. Everything here is an
**overlay** on top of `submission/` — the Phase 1 locked-in stack. Files in
this directory either:

1. Replace a `submission/` counterpart with a modified version (e.g., `run.sh`
   with `TORCH_COMPILE_DISABLE=0`), OR
2. Are new files that don't exist in `submission/` (e.g., `kernels/` Triton
   implementations, `warm_compile_cache.py`).

**`submission/train.py` stays in `submission/` — it is NOT duplicated here.**
Any Phase 2 modifications to `train.py` are in-place edits to the existing
file gated behind new env vars (e.g., `USE_FUSED_NGRAM_KERNEL=1`), so a Phase 1
run with those env vars unset behaves identically to the Phase 1 dry-run.

## Why overlay instead of fork

- **Single source of truth** for `train.py`: 731 lines in `submission/train.py`, no drift between two copies
- **Env var gates** turn Phase 2 optimizations on/off cleanly
- **`git blame` works**: Phase 2 changes show up on the same file as Phase 1, easier to track provenance
- **Phase 1 is reproducible** without modification: `curl -sL .../submission/bootstrap.sh | bash` still works as it did at the Phase 1 snapshot

## How to run Phase 2 on a fresh pod

```bash
curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/phase2/bootstrap.sh | bash
```

That script:
1. Clones the repo
2. Runs `submission/setup.sh` (torch, brotli, sentencepiece, hf_hub)
3. Runs `phase2/setup_phase2.sh` (extra Phase 2 deps: FA3 wheel / source build, any fused kernel deps)
4. Runs `submission/get_data.sh` (tokenize + build n-grams — unchanged from Phase 1)
5. Runs `phase2/warm_compile_cache.py` (~5 min one-time torch.compile warmup that populates the inductor cache)
6. Runs `phase2/run.sh` (the Phase 2 version with `TORCH_COMPILE_DISABLE=0` and any other speed flags flipped on)

ETA on a cheap 3090: ~40-60 min total (tokenize is still the bottleneck; compile warmup adds 5 min; train is ~8-15 min once compile is hot).

## Expected speedup vs Phase 1 baseline

Per `PHASE2_PLAN.md`:

| Cumulative shots | ms/step | vs P1 | steps in 600s |
|---|---|---|---|
| P1 baseline (1× H100 SXM, eager, SDPA) | ~3300 | 1.0× | 180 |
| +S1 torch.compile | ~1100 | 3× | 540 |
| +S2 FA3 (or FA2) | ~770 | 4.3× | 780 |
| +S3 CUDAGraph | ~440 | 7.5× | 1350 |
| +S4 fused n-gram (optional) | ~400 | 8.3× | 1500 |
| Stretch full Phase 2 | ~220 | **15×** | 2700 |

The actual val_bpb improvement follows directly from the step count: more
training steps → lower train_loss → lower val_bpb → closer to the comp records.

## Files shipped so far

| file | status | purpose |
|---|---|---|
| `README.md` | ✅ | this file |
| `bootstrap.sh` | ✅ | Phase 2 single-command launcher (chains submission/ + phase2/) |
| `metrics.py` | ✅ | Structured JSONL telemetry helper for per-step timing + GPU/CPU/RAM |
| `warm_compile_cache.py` | ✅ | Shot 1: runs short training pass to populate inductor cache |
| `run.sh` | ✅ | Shot 1: Phase 2 run with torch.compile enabled |
| `kernels/.gitkeep` | ✅ | placeholder dir for future Triton kernels |
| `setup_phase2.sh` | ⏳ not yet | S2: FA3 wheel install or source build |
| `kernels/ngram_bias_kernel.py` | ⏳ not yet | S4: fused n-gram bias gather + bias-add |
| `kernels/gptq_dequant_matmul.py` | ⏳ not yet | S5: fused int6 dequant + matmul for eval |

## Tier 0 speed levers shipped in submission/ (in-place edits, env-var gated)

These edits land in `submission/` directly because they're small enough to be
gated by env vars without duplicating code:

| Shot | File | Env var | Status |
|---|---|---|---|
| Free Inductor patch | `submission/run.sh` | `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` | ✅ default on |
| CUDA allocator expandable segments | `submission/run.sh` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ✅ default on |
| CPU data prefetch thread + pinned RAM | `submission/train.py` ShuffledSequenceLoader | `USE_PREFETCH_LOADER=1 PREFETCH_DEPTH=8` | ✅ default on |
| Prefetch prefill during pretime | `submission/train.py` ShuffledSequenceLoader.prefill() | `PREFETCH_PREFILL_BATCHES=8` | ✅ default on |

These are active on **both** the Phase 1 `submission/bootstrap.sh` path AND
the Phase 2 `phase2/bootstrap.sh` path. Phase 2 additionally enables torch.compile
(which Phase 1 has disabled to avoid the 5+ min cold-start penalty).

## Shots NOT YET shipped (future work)

See `PHASE2_PLAN.md` and `PHASE2_RESEARCH.md` for the full roadmap. High-priority
items still pending:

- **Shot 2 (FA3 sourcing)** — not on PyPI; build from source or find a private wheel
- **Shot 9 (FA3 varlen + window attention + mixed seq_len)** — PR #1212 pattern, the comp's fastest record at 69.6 ms/step
- **Shot 10 (Parameter Banking + Parallel Muon)** — PR #399's 15× optimizer speedup
- **Shot 14 (Training megakernel)** — world-first opportunity, 5-7 days dev
- **Shot 0b (Batched + streaming KV sliding eval)** — world-novel, 5-15× eval speedup
- **Shot 17 (Fuzzy LR bandit per microbatch)** — user's "dial-in" hint
- **Shot 19 (GPU-resident successive halving)** — user's "GPU tests" hint

## Decisions that diverged from the research agent's suggestions

1. **grad_accum 8 → 1 SKIPPED**: research agent claimed 30-50% free win but didn't check activation memory. Our 56 GB peak at microbatch=48 seqs would become 448 GB at microbatch=384 seqs — blows H100 80GB 8×. Keeping grad_accum=8.

2. **CPU n-gram precompute thread SKIPPED**: research agent claimed ~5-10% speedup from moving bigram/trigram/fourgram hash+gather to CPU. Math shows it's actually 48× SLOWER because GPU HBM bandwidth (3 TB/s) makes gather ops near-free, while CPU hits both a 100 GB/s memory wall AND a 50 GB/s PCIe Gen5 transfer wall. Pivoted to prefetch prefill during pretime instead.

Both skips documented in commit messages and in the relevant code files so
future-me doesn't re-fall for the same bad advice.

## Dependencies

- Phase 1 `submission/` must be present (we `source` / chain to its scripts)
- Phase 2 adds torch.compile cache dir (`~/.cache/torch/inductor`), optionally
  a FA3 wheel (not on PyPI; see `PHASE2_PLAN.md` Shot 2 for sourcing options),
  optionally Triton (already a torch dep)

## What Phase 2 explicitly does NOT do

- ❌ Change the model architecture or hyperparams (model locked from Phase 1)
- ❌ Add new patches (that's a separate "Phase X" if we decide to)
- ❌ 8×H100 distributed training (separate phase after Phase 2 lands)
- ❌ Re-train the SentencePiece tokenizer or rebuild n-grams

See `PHASE2_PLAN.md` for the full shot list and stop conditions.
