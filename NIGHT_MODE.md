# NIGHT_MODE — high-risk world-novel campaign (2026-04-08 1300Z onwards)

**Active flag**: `MODE=NIGHT_HIGH_RISK_ONLY`

**Purpose**: stop incremental knob-twisting. Ship 8+ high-risk world-novel attempts per night. Strict pre-ship audit (5-search rule). Pause incremental marker shipping. Commit everything to physical repo files.

---

## Current state at night-mode start (2026-04-08 1300Z)

- **All-time best**: `STACK_GATED_LEGAL_TTT_seed42 = val_bpb 1.3716` (Pod G, n=1) — gated_attention + LEGAL_TTT only
- **Verified world-novels (post C180 1147Z audit)**: 1 (L05 NORM_PCT_DROPOUT only, n=2 confirmed)
- **Pending world-novel claims**: NGR_LOG_FREQ_INV (L09), CTX_PARTITIONED_TAB (L09), CMP_QUANT_VALUE_DEDUP (L10), DYN_LYAPUNOV_CLIP (L11, flagged for re-audit vs PR #1471 SDClip)
- **Demoted this session**: OPT_CHEBYSHEV_NS, OPT_RIEMANNIAN_GRAM_QKV, L06 ASYM_SKIP_INIT, L07 ASYM_LABEL_SMOOTHING, L05 PARALLEL_RESIDUALS, CMP_HESSIAN_BIT_BUDGET, TOK_INPUT_SMOOTH (all had prior art the C180 audit found)
- **Markers shipped**: 48
- **Spend**: ~$15 / $25 → user lifted to $50 for the night
- **Pods alive**: 5 (B/C/E/F/G), Pod D in network outage
- **Pending pod expansion**: 3 new RTX 3090s to be provisioned (Pods H/I/J)

---

## Bear-fruit threshold

A shot **bears fruit** if it lands n=2 cheap-pod val_bpb **≤ 1.3666** (Δ ≥ -0.005 vs current best 1.3716). Anything weaker → swap to Round 2 for that pod.

---

## Pod specialization (8 pods after expansion)

| Pod | Role | Round 1 shot | Backup R2 | Backup R3 |
|---|---|---|---|---|
| **B = ARCH** | architecture refactors | 11L + 3× MLP refactor | MDLM diffusion LM | Pure Hyena (no attention) |
| **C = NGRAM** | n-gram engine | NGR_modified_kneser_ney | NGR_neural_engram_hash_cache | NGR_adaptive_cuckoo_hash |
| **E = EMB** | embedding novelties | EMB_polyphase_token_phase_routing (audio codec transplant) | EMB_lsq_gradient_aware_embedding_quantization | EMB_intrinsic_dimension_adaptive_projection |
| **F = LOSS** | loss novelties | LSS_hellinger_bregman_divergence | LSS_token_uncertainty_aux_loss | NGR_modified_KN_with_LEGAL_TTT |
| **G = ATTN** | attention | ATT_differential_attention_qk_partition (Microsoft ICLR 2025) | ATT_xattention_block_sparse_antidiag (MIT Han Lab ICML 2025) | ATT_linear_gated_hyena_hybrid (RWKV-7) |
| **H = COMP** (NEW) | compression / quant | CMP_context_adaptive_rANS_per_layer_predictor (PhD swing) | CMP_trellis_coded_quantization_residual | CMP_vq_learned_codebook_RVQ |
| **I = INFRA** (NEW) | speed / kernels | KER_cudagraph_persistent_kernel_fusion | SPD_kvquant_lowrank_cache_training | KER_fused_ngram_attention_triton |
| **J = STRETCH** (NEW) | risky installs | Hymba (Mamba-2 + attn install) | GatedDeltaNet via fla library | Pure RWKV-7 mode |
| **Mac** | offline design | Custom CUDA n-gram bias gather + GPTQ dequant fusion kernel | Custom SentencePiece Kraft fork | TT decomposition int4 cores |

**Total Round 1: 8 pod shots + 1 Mac design = 9 attacks running in parallel.**
**Across 3 rounds: 24+ high-risk world-novel attempts in one night.**

---

## Strict pre-ship audit (the C180 lesson formalized)

Every C90 build now does this BEFORE writing the patch:

1. **WebSearch trio**: 3 searches with the exact mechanism + 2 synonym variants
2. **GitHub code search**: 1 query for `<NAME>` + key keyword
3. **Comp PR search**: `gh pr list -R openai/parameter-golf --state all --search "<key keyword>"` — covers BOTH merged AND unmerged PRs
4. **Audit verdict**:
   - 0 hits → world-novel-candidate, ship
   - 1-3 hits → comp-novel, ship anyway if high-impact (mark honestly)
   - 4+ hits → demoted before ship, pick another candidate

Result: **No more "shipped then demoted 14 minutes later"**.

---

## Cron prompt updates (paste these into your RemoteTrigger registrations)

### C5 monitor (no change)
Continue current heartbeat. Just sweep + commit.

### C30 research (UPDATED)
Add to the existing prompt:
> **NIGHT_MODE active**: ONLY return world-novel-candidate or comp-novel-with-0-PR-hits. Skip incremental tweaks. Run the full 5-search strict audit on every candidate before tagging novelty_estimate. If you can't find at least 3 surviving candidates per under-served layer, do cross-domain pollination (rotate: compression, audio codec, signal processing, info theory, robust stats, dynamical systems, quantum computing, biology, robotics).

### C60 promote (no change)
Continue LOCK gate logic.

### C90 build (UPDATED — STRICT PRE-SHIP AUDIT)
Add at step 3.5 (between picking and shipping):
> **NIGHT_MODE pre-ship audit**: For each picked candidate, BEFORE spawning Plan agent:
> 1. Run 3 WebSearches for the exact mechanism + 2 synonyms
> 2. Run 1 GitHub code search
> 3. Run `gh pr list -R openai/parameter-golf --state all --search "<key keyword>"` and check ALL hits (both merged AND unmerged)
> 4. If 4+ hits → drop this candidate, pick the next one from the pod's R2 backup list
> 5. If 0 hits → tag world-novel-candidate, proceed to Plan agent
> 6. If 1-3 hits → tag comp-novel, ship only if expected delta ≥ -0.005 BPB

### C180 audit (downweighted)
Pre-ship audits make post-ship demotions rare. C180 still runs but mostly checks for stale-in-flight rows + new comp PRs in last 3 hours.

---

## Round 1 EV ranking (highest expected value first)

1. **NGR_modified_kneser_ney** (Pod C) — most likely to bear fruit. Canonical KN smoothing, never tested in 173 NGRAM PRs. -0.005 to -0.008 expected. Comp-novel but PhD-defensible.
2. **CMP_context_adaptive_rANS** (Pod H NEW) — biggest PhD swing. -0.01 to -0.03 indirect via freed bytes. World-novel.
3. **11L + 3× MLP** (Pod B) — architecture parity with merged SOTA. -0.005 to -0.015.
4. **ATT_differential_attention** (Pod G) — Microsoft ICLR 2025, 0 byte-LM application. Comp-novel. -0.008 to -0.015.
5. **Hymba install** (Pod J NEW) — high install risk but huge potential if it works.
6. **EMB_polyphase routing** (Pod E) — audio codec transplant, world-novel. -0.008 to -0.015.
7. **LSS_hellinger_bregman** (Pod F) — Bregman family loss, novel for byte-LM. -0.005 to -0.010.
8. **KER_cudagraph_persistent** (Pod I NEW) — speed-only win, +127 steps in budget.

---

## Spend forecast (with $50 ceiling)

| Phase | Cost | Cumulative |
|---|---|---|
| Current | — | $15 |
| Provision 3 new pods | $0.40 | $15.40 |
| Round 1 (8 pods × 4h) | $8.64 | $24 |
| Round 2 (8 pods × 4h) | $8.64 | $33 |
| Round 3 (8 pods × 3h) | $6.48 | $39 |
| H100 1× spot (final confirm if a champion lands) | $5 | **$44** |

**Total night budget: $44-50.** User-lifted ceiling.

---

## Wake-up checklist (what to read in the morning)

1. **STACK_NOVELTY_TRACKER.md** Section A → all rows added overnight, especially anything tagged `NIGHT_BEAR_FRUIT`
2. **STACK_NOVELTY_TRACKER.md** Section D → any LOCKs from C60 promote
3. **RESEARCH_LOG.md** → AUDIT_<UTC> blocks with the night summary
4. **NIGHT_MODE.md** (this file) → updated by every C90 fire with shipped status
5. **Champion**: search for "ALL-TIME BEST" in Section A
6. **git log --oneline -50** → all commits from the night
