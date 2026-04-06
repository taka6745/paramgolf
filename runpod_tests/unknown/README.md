# Unknown — The Actual Research

This is where we **find new stuff**. Tests we don't know the answer to.

## ⚠️ 3060 vs H100 — what transfers and what doesn't

**Transfers from 3060 → H100:**
- Code correctness, no NaN, loss decreases
- Architecture relative comparisons (u01)
- Patch validation

**Does NOT transfer:**
- Absolute speed (H100 is 5-10x faster, different memory bandwidth)
- Progressive seq's REAL value (3060 can't fit batch=524K → different regime)
- Step counts in 10 min wallclock
- Eval cache benefit (needs well-trained model = needs H100 scale)

**Conclusion:** Run u01 on 3060. Run u02-u08 on 1xH100. Only u05 needs 8xH100.

## Hardware Strategy

**3060 only for u01. Everything else needs H100.**

| Test | GPU | Time | Cost | Why this hardware |
|---|---|---|---|---|
| u01 | 3060 | 30 min | $0.10 | Architecture relative comparison transfers to H100 |
| u06 | 1xH100 | 5 min | $0.30 | Pure point: real H100 ms/step |
| u07 | 1xH100 | 30 min | $1.50 | Optional. GLA install + 50-step smoke |
| u08 | 1xH100 | 30 min | $1.50 | Optional. Only if u07 shows GLA is faster |
| u02 | 1xH100 | 30 min | $1.50 | Progressive seq's value depends on real H100 timing + batch size |
| u03 | 1xH100 | 30 min | $1.50 | Needs trained model from u02 |
| u09 | 1xH100 | 30 min | $1.50 | Two-stage continual low-LR (Mac validated -0.026) |
| u10 | 1xH100 | 30 min | $1.50 | Eval-time α + T (Mac validated -0.012) |
| u04 | 1xH100 | 30 min | $1.50 | Full stack single seed |
| u05 | 8xH100 | 45 min | $15 | 3-seed final submission only |

**Total: ~$26 (all 10 tests), ~$23 (without u07+u08), ~$20 (skip u09+u10 too).** Add $15 buffer = ~$38-41.

## Test Order & Decision Tree

```
[3060]
u01 (architecture sweep on 3060)
  ↓ pick best (NUM_LAYERS, MLP_EXPANSION)

[switch to 1xH100]
u06 (speed baseline)
  ↓ confirm ms/step matches projections
u07 (GLA shootout — optional)
  ↓ if GLA ≥30% faster, run u08
u08 (GLA + progressive — optional)
  ↓ if GLA + progressive wins, use GLA for u02-u05
u02 (standard or GLA + progressive seq)
  ↓ if -0.05 BPP gain, keep; else fall back
u03 (eval cache, reuses u02 model)
  ↓ if -0.05 BPP gain, keep; else debug
u09 (continual low-LR stage 2)
  ↓ if -0.01 BPP, add stage 2 to u04
u10 (eval-time α + temperature)
  ↓ if -0.005 BPP, add to u04 eval pipeline
u04 (full stack — combines all winners from above)
  ↓ best single-seed result

[switch to 8xH100]
u05 (3-seed)
  ↓ submission
```

## Test List

### Quality (the main path)

| # | Script | What we don't know | Hardware | Expected outcome |
|---|---|---|---|---|
| u01 | `u01_arch_sweep.sh` | Best layer count + MLP expansion at scale | 3060 | 11L+3xMLP wins, but verify |
| u02 | `u02_progressive_seq.sh` | Does -0.15 BPP transfer to CUDA? | 3060 → 1xH100 | -0.05 to -0.20 BPP improvement |
| u03 | `u03_eval_cache.sh` | Does eval cache give -0.10 BPP? | 1xH100 | -0.05 to -0.15 BPP |
| u04 | `u04_full_stack.sh` | Do all techniques stack? | 1xH100 | 0.85-0.95 BPP final |
| u05 | `u05_3seed_final.sh` | What's our submission variance? | 8xH100 | std < 0.01 BPP |

### Speed + alternative architectures

| # | Script | What we don't know | Hardware | Expected outcome |
|---|---|---|---|---|
| u06 | `u06_speed_baseline.sh` | Real H100 ms/step for our configs | 1xH100 | seq=128 ~3-5ms, seq=1024 ~25-40ms |
| u07 | `u07_gla_shootout.sh` | Is GLA 30%+ faster than standard at 50 steps? | 1xH100 | maybe 1.5-2x speedup at long seq |
| u08 | `u08_gla_progressive.sh` | Does GLA + progressive seq COMPOUND? | 1xH100 | More steps in Phase 2 → fixes the 8x under-resource? |

### Newer Mac findings worth porting

| # | Script | What we don't know | Hardware | Expected outcome |
|---|---|---|---|---|
| u09 | `u09_continual_lowlr.sh` | Does two-stage training (full LR → matrix_lr/4) help on CUDA? | 1xH100 | Mac validated -0.026 BPP |
| u10 | `u10_eval_temp_alpha.sh` | Does eval-time α=0.06 + T=0.93 stack on CUDA? | 1xH100 | Mac validated -0.012 BPP |

**Decision rules:**
- u07: if GLA ≥30% faster at similar 50-step loss → run u08
- u08: if GLA + progressive beats standard + progressive (u02) → use GLA for u04+
- If GLA fails or hurts → stick with standard attention

## Key Principles

1. **One variable at a time.** Each test isolates ONE change so we can attribute the gain.
2. **3060 first, H100 second.** Save money on the cheap GPU. Only escalate when the result needs scale to be interpretable.
3. **Reuse trained models.** u03 should reuse the model from u02. Don't retrain when you don't have to.
4. **Document failures.** If a test fails, write down WHY. The decision tree depends on knowing what works.

## Pre-flight Checklist

Before running unknown/:
- [ ] All chore/ scripts passed
- [ ] All validate/ scripts passed
- [ ] You have $30+ in RunPod credits
- [ ] You know how to SSH out and stop the pod manually if something hangs

## Cost Budget

Estimated total: **$20-25** for u01-u05.

Add $10-15 buffer for retries / debugging = **$30-40 total budget**.

## Stop Conditions

- A test crashes with an error you can't fix → STOP, debug locally
- Loss diverges (NaN/Inf) → STOP, check the patches
- Cost exceeds $50 → STOP, reassess what's worth the spend
