---
id: IDEA-020
slug: suffix-array-online-cache
created: 2026-04-16
updated: 2026-04-16
status: audited
layer: L09
novelty_class: WN
expected_bpb: [-0.020, -0.005]
cost_hours: 5.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l09-suffix-array-bwt-based-lookup
prior_art_checked: 2026-04-16
next_step: prior-art-audit-then-prototype
---

# IDEA-020: Suffix-array-backed online n-gram cache for variable-length context matching

> **Hypothesis**: Replacing the fixed-length hash table in the moonshot's online cache (IDEA-012) with a suffix array of the already-scored val stream enables **variable-length longest-match** retrieval — each query finds the longest suffix of the current context that has ever been seen, rather than being capped at k=5 or k=7. This reduces val_bpb by 0.005-0.020 on long-range repetitive structure (code, URLs, repeated phrases) that fixed-k misses.

## Method

After each scored val byte, append to a growing byte stream `val_so_far[]`. Maintain an online suffix array `SA` over `val_so_far[]` with **segment doubling** (amortized O(log n) insertion using Burrows-Wheeler + dynamic suffix tree tricks, or simpler: rebuild SA every K=1024 bytes with O(n log n) amortized).

Query at eval time:
```python
# b is next byte to score; need P(b | val_so_far_context)
ctx = val_so_far[-MAX_MATCH_LEN:]
# Find longest l in [1..MAX_MATCH_LEN] such that ctx[-l:] appears in val_so_far[:-l]
l_star = suffix_array_longest_match(val_so_far, ctx)
# For each occurrence of ctx[-l_star:], look at the byte that followed → counts[next]
counts = collect_next_bytes(val_so_far, ctx[-l_star:])
# Distribution: smoothed counts with Dirichlet alpha = 1
p_sa = (counts + 1) / (counts.sum() + 256)
# Blend with LM
mixed_p = alpha * lm_p + (1 - alpha) * p_sa
bpb_contrib = -log2(mixed_p[b])
# Update: append b to val_so_far, schedule SA rebuild
val_so_far.append(b)
```

Key difference from IDEA-012 (fixed hash k): **l_star adapts per query**. For highly-repetitive val regions (URLs, code blocks), l_star can reach 20+; for prose, 3-5.

**Integration point**: eval-time, plugged into the same hedge-mixer scaffold as IDEA-012/019. Requires pybind to a C++ suffix-array library OR a pure-Python rolling-hash approximation. ~400 LOC in C++, ~150 LOC pure Python (slower but simpler).

## Expected BPB

- **Range**: [-0.020, -0.005]
- **Mechanism**: long-match retrieval captures repetitive structure (URL prefixes, code keywords, repeated entity names) that fixed-k hash misses. FineWeb val has real internal repetition — URLs, `<br>` tags, boilerplate — that this should exploit.
- **Lower bound**: -0.005 (if LM already captures most repetition via attention)
- **Upper bound**: -0.020 (if longest-match coverage is significantly wider than fixed-k cache)

## Testable prediction

- **Metric**: val_bpb at seed 42, stock config + SA cache
- **Threshold**: ≤ 1.072 (−0.010 from 1.082)
- **Secondary diagnostic**: histogram of l_star distribution across val — should show bimodal (short-match common, long-match tail)

## Falsification criterion

Kill if val_bpb ≥ 1.080 at 2 seeds (less than 0.002 improvement — SA overhead not worth it).

## Stacking plan

- **Composes with**: LM (orthogonal axis), IDEA-019 CTW (different structural prior)
- **Composes with but partially-redundant with**: IDEA-012 moonshot (both provide "cache" functionality; IDEA-012's fixed-k vs IDEA-020's variable-length capture similar info but with different bias/efficiency tradeoff)
- **Conflicts with**: nothing
- **Blocks**: nothing
- **Budget footprint**: 0 bytes (eval-time RAM; SA and byte stream live in memory)

## Prior-art audit

Audited 2026-04-16 by Loop A fire 15 (Explore subagent).

- **Arxiv**:
  - **"Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens"** (Liu et al. Jan 2024, COLM 2024, arxiv 2401.17377) — **close match**. Builds ∞-gram LM using suffix array indices on 5T tokens, longest-match retrieval at ms latency, interpolation with neural LM. **But: Infini-gram is OFFLINE** (static suffix array on fixed corpus); IDEA-020 is ONLINE (grows SA on val stream, causal per comp rules).
  - "SuffixDecoding" (NeurIPS 2025, arxiv 2411.04975) — suffix trees for speculative decoding; not for prediction/scoring.
- **Comp PRs** (openai/parameter-golf): none mention `suffix`, `BWT`, `longest-match` in titles. parameter-golf already ships IDEA-012 (hash), IDEA-019 (CTW tree), n-gram bias tables. Suffix-array cache is new.
- **Verdict**: **partial-overlap-with-Infini-gram + novel online-causal twist**. Infini-gram validates the long-match-from-suffix-array idea as useful; the novel step is making it **online + causal + transformer-hedged at eval time**, which nobody has shipped.
- **Checked by**: claude 2026-04-16

## Lineage

- `STACK_NOVELTY_TRACKER_v2.md §L09` open-novelty "Suffix array / BWT-based lookup"
- RESEARCH_PROTOCOL.md §1 grid cell (L09 × suffix arrays / BWT)
- Complements IDEA-012 (cross-layer moonshot) and IDEA-019 (CTW standalone) — three different structural priors for the eval-time cache

## Risks

- Suffix array construction can be expensive at 100MB+ of val. Mitigation: use divsufsort (O(n log n), very fast in practice) or segment-based lazy SA.
- Long-match queries may over-predict on very rare contexts (few-occurrence matches). Dirichlet-1 smoothing mitigates but may not be enough; consider Kneser-Ney on top.

## Notes

Three-way cache comparison is the most valuable outcome: (IDEA-012 hash cache) vs (IDEA-019 CTW) vs (IDEA-020 SA). Best of the three becomes the moonshot's cache layer.
