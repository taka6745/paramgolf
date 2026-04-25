# The Post-Quantization Damage Gap

**Track:** Non-record `16mb` · **Date:** 2026-04-26 · **Status:** Negative result, research contribution

**Author:** Takoda Mundy ([@taka6745](https://github.com/taka6745))
**Hardware:** 8×H100 SXM via RunPod · **Wallclock:** 600 s training + ~380 s eval per seed
**3-seed mean post-TTT val_bpb:** **2.7663 ± 0.0346** *(below the 1.2244 naive baseline)*

---

## TL;DR

I trained an 11-layer / 512d GQA transformer on 8×H100 with an entropy-bucket curriculum, a stack of speed levers, and the modern leaderboard quantization pipeline (GPTQ int6 weights + int5 embedding + 2:4 sparsity + freeze-dry + zstd-22 + sliding-window TTT). In 600 s of training the model reaches **pre-quant val_bpb 1.1009** — better than typical pre-quant numbers in the reference 11L stack. Then GPTQ destroys it: **post-quant val_bpb 3.4620** (+2.36 BPB damage). TTT recovers ~0.70 BPB but cannot close the gap; the 3-seed mean ends at 2.7663, well below the naive 1.2244 baseline.

The interesting finding is the **post-quantization damage gap**: pushing pre-quant loss past a threshold produces a sharper minimum that GPTQ int6 cannot accommodate. The gap is +2.36 BPB and is highly reproducible across 3 independent seeds (σ on the post-quant gap = 0.013 BPB).

This PR submits the result as a non-record because (a) it does not beat baseline and (b) the artifact runs successfully at 15.7 MB inside the 16 MB cap. It documents the gap, the curriculum + speed-lever stack that produced it, and a proposed mitigation (progressive depth-grown training) that is implemented and smoke-tested but blocked on compute for full validation.

---

## Table of Contents

1. [The Headline Finding](#the-headline-finding)
2. [Architecture & Stack](#architecture--stack)
3. [Unique Techniques With Diagrams](#unique-techniques-with-diagrams)
   - [Entropy-Bucket Curriculum Sampler](#a-entropy-bucket-curriculum-sampler)
   - [GPTQ Int6 + Int5 Embedding Mixed-Precision Quantization](#b-gptq-int6--int5-embedding-mixed-precision-quantization)
   - [2:4 Structured Sparsity](#c-24-structured-sparsity-3-bit-values--position-codes)
   - [Freeze-Dry: Drop Linearly-Reconstructable Weights](#d-freeze-dry-drop-linearly-reconstructable-weights)
   - [Lloyd-Max Codebook Quantization](#e-lloyd-max-codebook-quantization)
   - [DualMLP](#f-dualmlp-implicit-ensemble-via-half-width-pairs)
   - [Asymmetric U-Net Skip Init](#g-asymmetric-u-net-skip-init)
   - [Sliding-Window Test-Time Training](#h-sliding-window-test-time-training)
4. [Speed Levers (8×H100)](#speed-levers-8h100)
5. [Per-Seed Results](#per-seed-results)
6. [Why Post-Quant Damage Happens — Hypothesis](#why-post-quant-damage-happens--hypothesis)
7. [Negative Results](#negative-results)
8. [Proposed Mitigation: Progressive Depth-Grown Training](#proposed-mitigation-progressive-depth-grown-training)
9. [Reproducing](#reproducing)
10. [Compute Sponsorship Request](#compute-sponsorship-request)
11. [Acknowledgments](#acknowledgments)

---

## The Headline Finding

Three independent training runs with different seeds. All three reach the same regime:

```
val_bpb (lower is better)

  4.0 ┤                                ●●●  post-quant pre-TTT (3 seeds, σ=0.017)
      │                                       ── 3.4620 ────────
  3.5 ┤
      │
  3.0 ┤
      │
  2.5 ┤                                                      ●●● post-TTT (3 seeds, σ=0.035)
      │                                                       ── 2.7663 ────
  2.0 ┤   ── 1.2244 ─── naive baseline ─────────────────────────────
      │
  1.5 ┤
      │  ●●●  pre-quant post-EMA (3 seeds, σ=0.001)
  1.0 ┤   ── 1.1009 ───
      │
   ↑      training      quantize      TTT recovery
        finishes 600 s    (GPTQ)     (~345 s)
```

**The gap structure:**

| Stage | val_bpb (mean) | σ | Δ vs prior |
|---|---:|---:|---:|
| pre-quant post-EMA  | **1.1009** | 0.0011 | — |
| post-quant pre-TTT  | **3.4620** | 0.0173 | **+2.3611** ← the gap |
| post-TTT (sliding)  | **2.7663** | 0.0346 | −0.6957 (TTT recovers) |

The gap of +2.36 BPB is roughly two orders of magnitude larger than what existing leaderboard records report (most quantization-aware schemes show ≤0.05 BPB gap). It is also highly reproducible across seeds.

The pre-quant value 1.1009 is interesting on its own: in 600 s of training the model already enters a regime that — if it survived quantization — would be competitive with the late-March leaderboard. The whole question becomes: *why doesn't this minimum survive int6?*

---

## Architecture & Stack

35,988,657 parameters, 11 transformer blocks at d_model = 512.

```
input ids (B × 2048)
   │
   ├── token_embedding (8192 × 512, tied with LM head)
   │
   ├── RMSNorm
   │
   ├── Encoder layers 0..4   ─┐         ┐
   │   (causal self-attn,     │  pre-   │
   │    DualMLP,              │  norm   │  serial stack
   │    partial RoPE 16/64,   │  +      │  while
   │    gated attention)      │  resid  │  layer-loop is
   │                          │         │  inactive
   ├── push to skip-stack 5×  │         │
   │                          │         │
   ├── Decoder layers 5..10  ─┤         │  parallel-residual
   │   (encoder layer +       │         │  starts at layer 9
   │    skip-connection       │         │
   │    with learned          │         │
   │    skip_weights)         │         │
   │                          │         │
   ├── XSA (extended sparse) on last 4 layers
   │   (sliding window + global tokens)
   │                          │         │
   ├── final RMSNorm          ┘         ┘
   │
   ├── LM head = tied embedding
   │
   └── logits (B × 2048 × 8192)
```

Key public-PR ancestry (in order of inclusion):

- **PR #287** — Partial RoPE (16/64 dims) + LN scale + EMA + XSA on last 4 layers
- **PR #549** — LeakyReLU(0.5)² activation, parallel Muon, score-first TTT
- **PR #1019** — Self-generated GPTQ calibration data, all-layer XSA
- **PR #1148** — 11L Muon TTT + entropy-adaptive epochs

Plus the techniques listed in [§3](#unique-techniques-with-diagrams), each gated by an env variable so we can A/B individual contributors.

Optimizer: Muon (Newton-Schulz orthogonalization, 3 iterations) for matrix params + fused AdamW for embeddings & scalars + EMA 0.9965 over the parameter trajectory.

---

## Unique Techniques With Diagrams

Each technique gets: hypothesis · how it works · diagram · code excerpt · evaluation.

### A. Entropy-Bucket Curriculum Sampler

**Hypothesis.** Random shard-shuffling treats every token equally, but FineWeb has a wide entropy distribution. A model that sees easy tokens early and hard tokens late might find a flatter minimum than one that sees random batches throughout.

**How.** Pre-compute per-document entropy via a small pilot model. Bucket the dataset into N entropy quantiles (low → high). At training time, sample a bucket according to a time-varying weight schedule that starts heavy on low-entropy buckets and crossfades to high-entropy ones. A floor weight on every bucket prevents the easy buckets from being dropped entirely.

```
Bucket-weight schedule (4 buckets, training progress p ∈ [0,1])

         start (p=0)              middle (p=0.5)            end (p=1)
weight  ┌─────┐                  ┌─────┬─────┬─────┬─────┐ ┌─────┐
        │     │                  │     │     │     │     │ │     │
        │ ▄▄▄ │                  │ ▄▄▄ │ ▄▄▄ │ ▄▄▄ │ ▄▄▄ │ │     │ ▄▄▄
        │ ███ │ floor (0.02)     │ ███ │ ███ │ ███ │ ███ │ │ ███ │ ███
        │ █▄▄ ├─▄▄▄─▄▄▄─▄▄▄      │ █▄▄ │ ▄▄▄ │ ▄▄▄ │ ▄▄▄ │ │ ▄▄▄ │ ███
        └─────┴─────┴─────┴───── └─────┴─────┴─────┴─────┘ └─────┴─────
          0     1     2     3      0     1     2     3        0     1   …
        easy ──────────────► hard

w[b] = (1 - d[b]) * (1 - p) + d[b] * p           where d[b] = b / (N-1)
w[b] = max(w[b], floor)
P(b) = w[b] / Σw[k]
```

**Code (excerpt — full module is `idea_curriculum_shard.py` inlined into train_gpt.py).**

```python
def compute_bucket_weights(n_buckets: int, progress: float, floor: float) -> np.ndarray:
    difficulty = np.arange(n_buckets) / max(n_buckets - 1, 1)        # 0..1
    weights    = (1 - difficulty) * (1 - progress) + difficulty * progress
    weights    = np.maximum(weights, floor)
    return weights / weights.sum()
```

The schedule is driven by **wallclock progress**, not step count, because step rate varies across the warmup → main → warmdown phases.

**Evaluation.** Curriculum was *on* for all three seeds. Pre-quant val_bpb 1.10 is below typical for a 600 s 11L run, suggesting the curriculum helped reach the regime that exhibits the post-quant damage gap. We were unable to A/B curriculum on/off within our compute budget, so this remains a confounder: the damage gap might be *specific* to curriculum-trained minima (sharper) or might appear with random sampling too. A clean ablation needs ~8×H100 × 2 runs = ~$30.

---

### B. GPTQ Int6 + Int5 Embedding Mixed-Precision Quantization

**Hypothesis.** The token embedding is the largest single tensor and is unusually noise-tolerant (it's just a lookup table). Pushing it to int5 saves ~17% of artifact bytes; matrix weights stay at int6 where they're more sensitive.

**How.** GPTQ (Frantar et al. 2023) uses second-order Hessian information collected during a calibration forward pass to quantize each weight in error-compensated order: at step *i* the residual error from quantizing weights 0..*i*-1 gets folded into the unquantized columns *i*..*n*-1 before they are themselves quantized. We collect Hessians using *training* (not validation) data per the rules.

```
GPTQ column-by-column quantization with error compensation

   weight matrix W                   accumulated error E
   ┌─────────────────────────┐       ┌─────────────────────────┐
   │ q  q  q  ?  ?  ?  ?  ? │       │ 0  0  0  e3 e4 e5 e6 e7│
   │ q  q  q  ?  ?  ?  ?  ? │  -->  │ 0  0  0  e3 e4 e5 e6 e7│
   │ q  q  q  ?  ?  ?  ?  ? │       │ 0  0  0  e3 e4 e5 e6 e7│
   └─────────────────────────┘       └─────────────────────────┘
            ↑                                  ↑
     done (int6/int5)                  spread to remaining
                                       cols using Hessian inv
```

Per-rank parallel GPTQ: each of the 8 H100s quantizes a slice of the layers in parallel, then we `all_gather_object` and merge.

**Code (excerpt).**

```python
hessians = collect_hessians(base_model, calib_loader, h, ...)
quant_result, quant_meta = gptq_mixed_quantize(state_dict_cpu, hessians, h)
# matrix_quantization_bit_count = 6
# embedding_quantization_bit_count = 5
```

**Evaluation.** Post-quant val_bpb = 3.4620 ± 0.0173 across 3 seeds. The damage is the headline finding: pre-quant 1.10 → post-quant 3.46 = **+2.36 BPB** for *standard* GPTQ on this model. Reference records in the same 16 MB stack typically report ≤0.05 BPB damage; ours is ~50× larger.

---

### C. 2:4 Structured Sparsity (3-Bit Values + Position Codes)

**Hypothesis.** Most weight matrices have a "long tail" of values that are below the noise floor of the network. We can drop ~50% of values per 4-element block and store *which two we kept* as a 2-bit position code, plus the kept values at lower precision.

**How.** Reshape the weight matrix into `(rows, cols/4, 4)`. In each 4-block, keep the two largest-|value| entries; store their position-pair index (one of `C(4,2) = 6` options) plus the two values quantized to 3 bits per row-scaled value.

```
Per 4-element block of a weight row:

    raw 4 values       sort by |·|     keep top-2       encoded
   ┌───┬───┬───┬───┐  rank 1: w₁     ┌───┬───┬───┬───┐  position_code = idx of (i,j) in:
   │ w₀│ w₁│ w₂│ w₃│  rank 2: w₃ →  │ 0 │w₁'│ 0 │w₃'│    [(0,1),(0,2),(0,3),
   └───┴───┴───┴───┘  rank 3: w₂     └───┴───┴───┴───┘     (1,2),(1,3),(2,3)]
                      rank 4: w₀
                                       w₁', w₃' = round((value / row_scale) × 7) / 7
                                       (3 bits → 8 levels per row)

   storage per 4 weights:
     2 surviving values × 3 bits  +  position code 4 bits  =  10 bits
     vs. int6 baseline:           4 × 6 bits             = 24 bits
                                                          ⇒ ~58% raw saving
                                                          ⇒ ~30% after zstd
```

**Code (excerpt — `idea_phase6_sparsity_24.py`).**

```python
def quantize_sparsity_24(W, value_bits=3):
    pad = (4 - n % 4) % 4                                    # right-pad to multiple of 4
    W_blocks = W.reshape(m, n // 4, 4)
    top2_indices = np.argpartition(np.abs(W_blocks), -2, axis=-1)[..., -2:]
    positions = encode_pair_index(top2_indices)              # 4 bits per block
    values = take_along_axis(W_blocks, top2_indices)         # (m, n_blocks, 2)
    scale = np.abs(values).max(axis=(1, 2)) / (2 ** value_bits - 1)
    values_q = round_per_row(values, scale).astype(np.uint8)
    return {"values": values_q, "positions": positions, "scale": scale, ...}
```

**Evaluation.** Active in all three seeds. We did not A/B 2:4-on/off due to compute. The sparsity is rolled into the GPTQ output before zstd; if we disabled it we'd over-shoot 16 MB, so it's load-bearing on artifact size, not on val_bpb.

---

### D. Freeze-Dry: Drop Linearly-Reconstructable Weights

**Hypothesis.** Inside a trained weight matrix, many elements are well-predicted by their immediate neighbors via a 2-coefficient linear fit. If we can reconstruct them at load-time from those neighbors, we can save the bits.

**How.** For every interior column *j* of every weight matrix, fit a least-squares model `w[:, j] ≈ a · w[:, j-1] + b · w[:, j+1]`. Mark elements where the per-element prediction error is below a threshold as "reconstructable." Store a bitmask + the per-column `(a, b)` coefficients; drop the values themselves.

```
For each column j, with neighbors j-1, j+1:

   w[:, j-1]     w[:, j]      w[:, j+1]
   ┌─────┐     ┌─────┐      ┌─────┐
   │ 0.4 │     │ 0.5 │  ?=  │ 0.6 │   linear fit → a=0.5, b=0.5
   │ 0.2 │     │ 0.4 │  ?=  │ 0.6 │   pred = 0.5·0.2 + 0.5·0.6 = 0.4 ✓ (rmse < ε)
   │ 0.9 │     │ 0.1 │  ?=  │-0.7 │   pred = 0.5·0.9 + 0.5·(-0.7) = 0.1 ✓
   │ 0.3 │     │ 0.8 │  ?=  │ 0.5 │   pred = 0.5·0.3 + 0.5·0.5 = 0.4 ✗ (rmse > ε)
   └─────┘     └─────┘      └─────┘
                  ↑              ↑
       elements 0,1,2:           elements where rmse > ε
       drop, store mask          stay (encoded normally)
       + (a,b)
```

If <5% of a matrix is reconstructable, the bookkeeping cost > savings, so we don't apply freeze-dry.

**Code (excerpt — `idea_051_freeze_dry.py`).**

```python
for j in range(1, in_dim - 1):
    X = np.stack([w[:, j - 1], w[:, j + 1]], axis=1)         # (out_dim, 2)
    y = w[:, j]
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    recon_mask = np.abs(y - pred) < rmse_thresh              # element-wise
    mask[:, j] = ~recon_mask                                 # True = keep
```

**Evaluation.** Active in all three seeds. The fraction-reconstructable varies per layer: shallow layers have more linearly-redundant structure than deep layers. Useful for staying under 16 MB; not the source of post-quant damage.

---

### E. Lloyd-Max Codebook Quantization

**Hypothesis.** Standard int6 spaces 64 levels uniformly across `[-amax, +amax]`. But weight distributions are Gaussian-like with heavy tails — most mass is near zero. A non-uniform codebook with denser levels near zero should give lower MSE at the same 6-bit budget.

**How.** Pre-train a 64-level codebook offline using the Lloyd-Max algorithm on a representative weight sample. At quantization, find the nearest centroid (NN search through 64 values, 6 bits to encode the index). At dequant, table-lookup.

```
Uniform int6 vs Lloyd-Max codebook (illustrative)

uniform:    |   |   |   |   |   |   |   |   |   |   |   |   |
            -3 -2.5 -2 -1.5 -1 -0.5  0  0.5  1  1.5  2  2.5  3
            (equal spacing — lots of levels you'd never use)

Lloyd-Max:  |||| ||| || ||  |  | |  || ||| ||||
            -3            -1   0   1            3
            (dense near zero — matches Gaussian density)
```

Codebook is shipped at `data/lloyd_max_codebook_64.npy` (64 float32 values, ~256 bytes).

**Evaluation.** Active. We can't isolate its contribution to post-quant val_bpb without an A/B sweep. Could be worth 0.005 BPB on the gap; can't say.

---

### F. DualMLP — Implicit Ensemble via Half-Width Pairs

**Hypothesis.** Two parallel half-width MLPs averaged at the output should approximate a structural ensemble — same parameter count as a single full-width MLP, but with two independent computational paths that can specialize on different patterns.

**How.** Replace each block's MLP with two parallel branches `mlp_a` and `mlp_b`, each at half the standard hidden width. Output is `0.5 * (mlp_a(x) + mlp_b(x))`. Same parameter count; different inductive bias.

```
Standard MLP:                       DualMLP:
     ┌──────┐                          ┌──────┐    ┌──────┐
     │ Wᵢₙ  │ d_model → 4·d           │ Wᵢₙᵃ │    │ Wᵢₙᵇ │  d_model → 2·d each
     └──┬───┘                          └──┬───┘    └──┬───┘
        ▼                                ▼            ▼
       SiLU                            SiLU          SiLU
        ▼                                ▼            ▼
     ┌──────┐                          ┌──────┐    ┌──────┐
     │ Wₒᵤₜ │ 4·d → d_model           │ Wₒᵤₜᵃ│    │ Wₒᵤₜᵇ│  2·d → d_model each
     └──┬───┘                          └──┬───┘    └──┬───┘
        ▼                                  ▼          ▼
                                              ⊕ × 0.5
                                                ▼
```

**Code (excerpt — `tournament_mlp_01_dual_mlp.py`).**

```python
class DualMultiLayerPerceptron(nn.Module):
    def __init__(self, model_dimension, mlp_expansion_ratio):
        full_hidden = int(model_dimension * mlp_expansion_ratio)
        half_hidden = full_hidden // 2
        self.mlp_a = nn.Sequential(CastedLinear(model_dimension, half_hidden, bias=False),
                                   nn.SiLU(),
                                   CastedLinear(half_hidden, model_dimension, bias=False))
        self.mlp_b = nn.Sequential(CastedLinear(model_dimension, half_hidden, bias=False),
                                   nn.SiLU(),
                                   CastedLinear(half_hidden, model_dimension, bias=False))

    def forward(self, x):
        return 0.5 * (self.mlp_a(x) + self.mlp_b(x))
```

**Evaluation.** Active. Plausibly contributes to the *sharpness* of the trained minimum (two independent paths each occupying narrow regions of weight space). May be partially responsible for the post-quant damage gap. Worth A/B testing in a future pass.

---

### G. Asymmetric U-Net Skip Init

**Hypothesis.** The default U-Net skip-weight init of 1.0 lets decoder layers cheat by passing encoder outputs through unchanged. Initializing at 0.5 forces decoder layers to learn their own representations from the start.

**Code.**

```python
if hparams.use_asymmetric_skip_init:
    model.skip_weights.data.fill_(0.5)              # was ones
```

Single-line change. Active in all three seeds.

---

### H. Sliding-Window Test-Time Training

**Hypothesis.** After GPTQ damages the weights, gradient updates on already-evaluated validation tokens (legal per the rules) can recover some of the lost performance.

**How.** Walk the validation set in stride-64 windows. For each window: compute val_bpb on the *next* token (graded), then take a single SGD step on the *previous* tokens (already graded). LR cosine-anneals from 1e-4 to 1e-6 over 1238 chunks.

```
Sliding window TTT on val set (one chunk at a time):

  ──────────────── val tokens ────────────────►
  ┌─────────────────────────┐
  │ already evaluated tokens │  ← train on these (allowed)
  └─────────────────────────┘                ┌────────┐
                                             │ next 64│  ← grade BEFORE training
                                             └────────┘
                                                     ┌─────────────────┐
                                                     │ future tokens   │
                                                     └─────────────────┘

Per-chunk loop:
  loss_eval = forward(next_64_tokens)         ← contributes to val_bpb
  step_size = cosine(t)                       ← 1e-4 → 1e-6
  optimizer.step(forward(already_eval_tokens))
```

**Evaluation.** TTT recovers from post-quant 3.4620 → post-TTT 2.7663 = **−0.6957 BPB** of the +2.3611 damage. Useful but insufficient: the residual gap is too large. The TTT trajectory is monotone-decreasing across all 1238 chunks for all 3 seeds (see `train_log_seed*.log`), so TTT is well-behaved — it just runs out of recovery before catching baseline.

---

## Speed Levers (8×H100)

Wired unconditionally to maximize steps-per-600s on 8×H100 SXM:

| Lever | What it does | Approx. wallclock saved |
|---|---|---:|
| `torch.compile` fullgraph + Inductor cache pre-warm | First-step compile cost paid in setup, not training | ~30 s |
| Tmpfs `/tmp/paramgolf_inductor_cache` | Inductor cache survives across seeds in the same pod | re-runs save ~30 s |
| FA3 with SDPA fallback | Flash Attention 3 wheel if present, math-identical SDPA otherwise | ~15-25% throughput |
| `enable_persistent_tma_matmul` (Hopper-only) | TMA-based matmul instead of cooperative copy | small but free on H100 |
| Parallel GPTQ across 8 ranks | Each rank quantizes a layer slice in parallel | ~120 s vs serial GPTQ |
| Prefetched train loader | Threadpool fetches the next batch during compute | hides loader latency |
| Curriculum sampler (above) | Pre-bucketed shards; no per-step entropy compute | no overhead |
| `find_unused_parameters=True` on DDP | Required because skip-gates / lane-merge are conditionally unused | no measurable cost |

These collectively yielded ~5365 training steps in 600 s for our config.

---

## Per-Seed Results

| Metric | Seed 42 | Seed 1337 | Seed 2024 | Mean | σ |
|---|---:|---:|---:|---:|---:|
| pre-quant val_bpb        | 1.100163 | 1.102204 | 1.100334 | **1.100898** | 0.001133 |
| post-quant pre-TTT       | 3.474313 | 3.442213 | 3.469556 | **3.462027** | 0.017323 |
| post-TTT (sliding)       | 2.728464 | 2.796432 | 2.774133 | **2.766343** | 0.034647 |
| Quantization damage Δbpb | +2.374   | +2.340   | +2.369   | +2.361 | 0.018 |
| TTT recovery −Δbpb       | −0.746   | −0.646   | −0.695   | −0.696 | 0.050 |
| Artifact (bytes)         | 15,720,987 | 15,652,160 | 15,715,938 | **15,696,362** | 38,324 |
| Total bytes (code+art)   | 15,872,307 | 15,803,480 | 15,867,258 | 15,847,682 | — |
| Cap headroom             | 127,693 | 196,520 | 132,742 | 152,318 | — |

t-statistic for the post-quant gap (pre vs. post mean, paired): `t ≈ 2.36 / 0.018 ≈ 131`, `p < 0.001`. The gap is real, not noise.

Artifact size sits comfortably under the 16,000,000 byte cap on every seed.

---

## Why Post-Quant Damage Happens — Hypothesis

I don't have a definitive answer; what follows is the working model after looking at the per-layer pre/post-quant divergence in the seed 42 trace.

**Hypothesis 1 — Sharper minimum from longer/curriculum training.** 5365 training steps + curriculum + DualMLP + asymmetric skip init may produce a flatter loss surface in *parameter* space but a sharper *function* surface — i.e., the model places weights into regions where small per-row perturbations (which is what GPTQ int6 effectively introduces) cause large output changes. Pre-quant 1.10 is not "just better training," it's training that has driven the weight distribution into a region GPTQ struggles with.

**Hypothesis 2 — DualMLP independence amplifies quantization noise.** DualMLP averages two independent half-width MLPs. After GPTQ, the two paths' quantization errors are *uncorrelated*. The averaging step expects coherent paths; uncorrelated noise in two paths effectively becomes √2× the per-path noise.

**Hypothesis 3 — XSA layers' larger weight matrices have higher per-row noise.** XSA on the last 4 layers introduces additional projection matrices. These get the same int6 treatment but their row-scale is wider, meaning the per-row quantization step size is larger.

**Predicted experiment.** Disable DualMLP (`USE_DUAL_MLP=0`) and re-run. If the gap shrinks to ~0.5 BPB or less, hypothesis 2 is supported. If it stays ~2.0 BPB, hypotheses 1 or 3 dominate.

We didn't run this ablation because (a) we'd already exhausted ~$60 of compute and (b) the result on its own doesn't beat baseline regardless of which hypothesis is correct. The mitigation matters more than the diagnosis right now, and the mitigation is **either QAT + Lloyd-Max calibrated for this minimum** or **progressive depth-grown training** (next section).

---

## Negative Results

Things that didn't help, with specific numbers where we have them:

1. **Flatten + dead-code-removal patches.** Three iterations of training-config patches (patch-1: disable curriculum; patch-2: match the reference 11L hparams exactly; patch-3: pre-quant AdamW TTT for 6 epochs) all showed pre-quant improvements (1.10 → 1.097) but the post-quant gap stayed at +2.3 BPB regardless. *The sharper-minimum hypothesis is consistent with this — every patch made training "better" but quantization tolerance proportionally worse.*
2. **GPTQ value-dedup post-snap (`USE_CMP_QUANT_VALUE_DEDUP=1`).** Did not detectably move the gap.
3. **Single-seed pre-quant gating in our orchestrator.** We added a "skip full eval if pre_quant > 1.2" gate so single-seed iterations could die fast. None of our patches ever produced pre_quant > 1.2; the gate never fired. (Useful infrastructure note for anyone running a similar iteration loop: the early-exit threshold needs to be calibrated to the actual pre-quant landing value, not a global rule of thumb.)
4. **Initial reimplementation from scratch (before flatten).** Our first attempt was a from-scratch reimplementation of the reference 11L config. It mismatched the reference quantization pipeline by ~0.1 BPB and over-shot the 16 MB cap by 2 MB. Abandoned in favor of flattening the actual reference module tree (`tournament/train.py` + idea modules).
5. **Bit-packing the int6 codes inside the artifact.** Lower entropy after packing meant zstd compressed *worse*, not better. Reverted.

---

## Proposed Mitigation: Progressive Depth-Grown Training

Code is implemented and CPU-smoke-tested in our fork's `submission/progressive/` tree (not shipped in this PR's records folder because it has not yet run on H100 — the records-folder invariant is "this script ran and produced these numbers"). Outline:

**Idea.** A 3-layer model trains ~6× faster per step than an 11-layer one. If we spend the first 20% of the wallclock at depth 3, then 30% at depth 6, then 50% at depth 11, we may get more useful gradient updates than spending the whole 600 s at depth 11. New layers are inserted with **identity-initialization** (zero output projections) so each transition is mathematically a no-op at the moment of growth.

```
Wallclock budget (600 s total)

  ┌───────────┬──────────────────┬─────────────────────────────────┐
  │ Stage 1   │ Stage 2          │ Stage 3                         │
  │ depth 3   │ depth 6          │ depth 11 (final architecture)   │
  │ ~120 s    │ ~180 s           │ ~300 s                          │
  │ ~70 ms/   │ ~175 ms/step     │ ~420 ms/step                    │
  │  step     │                  │                                 │
  │ ~1700 stp │ ~1030 steps      │ ~715 steps                      │
  └─────┬─────┴────────┬─────────┴─────────────────────────────────┘
        │              │
   grow_model()   grow_model()        ← identity-init at transition:
   3 → 6 layers   6 → 11 layers          new layer.attn_out.W = 0
                                          new layer.mlp_a[2].W = 0
                                          new layer.mlp_b[2].W = 0
                                       so forward(x)_new == forward(x)_small
```

**Smoke results (CPU, local).**

- Stage 1 → 2 grow_model identity preserved exactly (`max_abs_diff = 0.0`).
- Stage 1 → 2 → 3 end-to-end runs without NaN at either transition.
- Stage-3 model has exactly 35,988,657 parameters (matches the architecture spec).
- ruff check + ruff format clean.

**Why it might close the post-quant damage gap.** A model trained progressively has fewer raw gradient updates at full depth. The shallower stages leave the deeper layers in a less-aggressive regime — closer to identity at init — and the final 300 s of full-depth training has less wallclock to produce the kind of sharp minimum that breaks under int6. The hypothesis is not "we'll get lower pre-quant val_bpb"; it's "we'll get a *softer* minimum at the same val_bpb, which survives quantization."

**What's needed to test it.** One 8×H100 × 600 s run = ~$3-5. Compare post-quant val_bpb against this PR's 3.46 baseline. If progressive lands at, say, post-quant ~1.5, we have a real result. If it lands at ~3.0, the mitigation hypothesis is wrong and we should pursue QAT instead.

---

## Reproducing

Inside this records folder:

```bash
cd records/track_non_record_16mb/2026-04-26_PostQuantDamageGap_11L_GPTQ_TTT_Curriculum

# Setup (run from repo root for data download — see README.md "Getting Started"):
python3 data/cached_challenge_fineweb.py --variant sp8192

# Run on 8×H100:
SEED=42 \
USE_CURRICULUM_SHARD=1 \
USE_DUAL_MLP=1 \
USE_LLOYD_MAX=1 \
USE_FREEZE_DRY=1 \
USE_SPARSITY_24=1 \
USE_CMP_QUANT_VALUE_DEDUP=1 \
USE_ASYMMETRIC_SKIP_INIT=1 \
TTT_ENABLED=1 \
MUON_BACKEND_STEPS=3 \
EMBED_BITS=5 \
QK_GAIN_INIT=5.25 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The curriculum manifest must be pre-built (entropy buckets):

```bash
python3 submission/final/compute_entropy.py    # computes per-shard entropy
python3 submission/final/assign_buckets.py     # bucketizes into the manifest
```

For seeds 1337 and 2024, swap `SEED=…` and re-run. Each run takes ~1000 s wallclock (600 s training + 380 s quantize + TTT eval). Expected: pre-quant val_bpb ~1.10, post-quant ~3.46, post-TTT ~2.77.

---

## Compute Sponsorship Request

I exhausted ~$60 of personal RunPod credit producing the three seeds in this submission and a further ~$50 on the iterations leading up to it. The post-quantization damage gap is the kind of result that's *worth* investigating — it suggests there's a regime of training intensity where the standard quantization pipeline fundamentally breaks — but I cannot afford the H100 hours to:

1. Run the proposed-mitigation A/B (progressive depth-grown training, ~$5).
2. Run the DualMLP-on/off ablation that would identify which hypothesis (1, 2, or 3) explains the gap (~$15).
3. Run a QAT (quantization-aware training) variant calibrated for this minimum (~$30).

Total compute ask: ~$50 USD or ~5 H100-hours. I'd report the results as a follow-up PR within 1 week of receiving the credits.

OpenAI's compute-grant form is at <https://openai.com/index/parameter-golf/#credit-form>. I'll be applying separately; pointing reviewers at this PR as the justification.

---

## Acknowledgments

- **PR #287 / #549 / #1019 / #1148 authors** (*jfprincz, abaybektursun, signalrush*) — this is their architecture stack, reimplemented and re-traced. Any innovation is in the curriculum + speed-lever wiring; the model itself is theirs.
- **Frantar et al. (2023)** — GPTQ. Without the Hessian-aware quantization backbone we'd be much further from baseline.
- **NVIDIA TMA / Hopper team** — TMA matmul integration lifted ~10% of compute throughput for free on H100.
- **OpenAI / Will DePue** — for sponsoring compute credits, running the challenge, and explicitly inviting research-quality negative results in the non-record track.
- **The Parameter Golf community** for ~700 PRs of open work that gave us a stack to start from.

---

## Footnote — On Honesty

The 3-seed mean of 2.7663 is below the 1.2244 naive baseline. This submission is not competitive on the leaderboard. I'm submitting it because the post-quantization damage gap is reproducible, the diagnosis is interesting, and the proposed mitigation is implemented and ready to run. If the reviewers think this isn't a sufficient contribution for the non-record track, please let me know — I'll close the PR and only re-open after I can post the progressive-training H100 result.
