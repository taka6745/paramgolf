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

## Files (live when the respective shot lands)

| file | purpose | shot |
|---|---|---|
| `README.md` | this file | meta |
| `bootstrap.sh` | Phase 2 single-command launcher | always |
| `setup_phase2.sh` | extra Phase 2 pod setup on top of `submission/setup.sh` (FA3 wheel/source, cache dir creation) | S2 |
| `run.sh` | Phase 2 version of `submission/run.sh` with compile enabled + warm-cache + new speed flags | S1+ |
| `warm_compile_cache.py` | small helper that runs a short training pass to populate the inductor compile cache | S1 |
| `kernels/.gitkeep` | placeholder for S4+ Triton kernels | S4 |
| `kernels/ngram_bias_kernel.py` | fused n-gram bias gather + bias-add (S4) | S4 |
| `kernels/gptq_dequant_matmul.py` | fused int6 dequant + matmul for eval path (S5) | S5 |

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
