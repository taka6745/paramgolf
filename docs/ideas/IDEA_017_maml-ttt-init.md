---
id: IDEA-017
slug: maml-ttt-init
created: 2026-04-16
updated: 2026-04-16
status: draft
layer: L10
novelty_class: WN
expected_bpb: [-0.015, -0.005]
cost_hours: 6.0
depends_on: []
blocks: []
supersedes: []
stack_row: STACK_NOVELTY_TRACKER_v2.md#l10-maml-style-meta-learning-for-ttt
prior_art_checked: null
next_step: prior-art-audit-then-prototype
---

# IDEA-017: MAML-pretrained TTT init — meta-train for 1-step test-time adaptation

> **Hypothesis**: Meta-training the model such that a single SGD step on a val chunk produces the same adaptation quality as our current 3-epoch Score-First TTT lets us either (a) get the same BPB with 3× less eval wallclock (freeing budget for larger models / more inference stacking) or (b) get MORE BPB at the current eval budget by running more adaptation steps. Expected delta: 0.005–0.015 BPB.

## Method

Classical MAML: during training, simulate the TTT protocol and backprop through it.

```python
# In the inner training loop, every K regular steps:
# 1. Split a train batch into "support" and "query" halves
# 2. Apply K_inner SGD steps on the support half using current weights θ → θ'
# 3. Compute loss on the query half using θ'
# 4. THIS is the outer-loop loss; backprop through the inner steps (via torch.func / functorch)
# 5. Update θ using the outer-loop gradient

outer_loss = []
for task in range(N_meta_tasks):
    support, query = split(batch)
    theta_prime = theta.clone()
    for inner_step in range(K_inner):
        g = grad(forward_ce(theta_prime, support), theta_prime)
        theta_prime = theta_prime - lr_inner * g  # differentiable
    outer_loss.append(forward_ce(theta_prime, query))
total_outer_loss = mean(outer_loss)
# backprop total_outer_loss through theta_prime → theta
# (torch.func.grad + functorch primitives handle this)
```

With `K_inner=1` (one-step adaptation), we meta-train so that the model's weights are in a region where a single gradient step dramatically improves val NLL. At eval time, our Score-First TTT becomes cheaper by a factor of `TTT_EPOCHS / 1` (currently 3×).

**Integration point**: wrap `submission/train.py` training loop with an outer MAML step every K=64 regular steps. Uses `torch.func.grad` for second-order gradients. Memory cost is 2× model-state; fine on H100 80 GB for our 30M model.

## Expected BPB

- **Range**: [-0.015, -0.005]
- **Mechanism 1** (same budget, more adaptation): same 600s eval time → 9 TTT epochs instead of 3 → more adaptation → lower BPB. Comp PR #1242 reported −0.014 for Score-First TTT at 3 epochs; extrapolating to 9 epochs is ~−0.005 additional.
- **Mechanism 2** (lower budget, larger model): shave ~5 min off the 10 min eval budget, re-spend on a 12-layer model instead of 11, re-amortize over training.
- **Lower bound**: -0.005 (if MAML doesn't significantly improve one-step adaptation)
- **Upper bound**: -0.015 (if meta-learning gives us 2-3× efficiency in TTT)

## Testable prediction

- **Metric**: val_bpb at 1-epoch TTT ≤ current 3-epoch TTT val_bpb
  - Specifically: val_bpb @ 1ep-MAML ≤ 1.082 (our 3-epoch baseline). If equal, then 9-epoch MAML at same wallclock ≤ 1.072.
- **After**: 600s training with MAML outer step, 3-seed confirmation

## Falsification criterion

Kill if:
- val_bpb @ 1ep-MAML ≥ 1.090 at 2 seeds (MAML broke the base model)
- OR val_bpb @ 9ep-MAML-at-same-wallclock ≥ 1.080 (no gain from cheaper adaptation)

## Stacking plan

- **Composes with**: Score-First TTT (we keep this, just make it more efficient)
- **Composes with**: IDEA-004 (Pre-Quant AdamW TTT) — orthogonal (training-time vs eval-time adaptation)
- **Conflicts with**: nothing structural
- **Blocks**: future evolution toward "online MAML" during eval itself
- **Budget footprint**: 0 bytes (training-time only; eval uses standard TTT)

## Prior-art audit

_To be filled by next Loop A fire with Explore subagent._

- **Arxiv (2023-2026)**: search "MAML language model", "meta-learning test-time adaptation LM", "MAML TTT transformer"
- **Comp PRs**: grep for `maml`, `meta-learn`, `meta_learn`, `reptile` in comp PR titles
- **Verdict**: TBD; expect MAML itself is well-known, but combining with Score-First TTT and byte-LM comp scale is likely novel
- **Checked by**: _pending_

## Lineage

- RESEARCH_PROTOCOL.md §1 grid cell (L10 × MAML / meta-learning)
- `STACK_NOVELTY_TRACKER_v2.md §L10` row "MAML pretraining for TTT-ready init"
- OpenAI "requests for PRs" wishlist includes "E2E TTT" (end-to-end test-time training) — this is a concrete implementation route

## Risks

- Second-order gradients are expensive and unstable. `torch.func.grad` may not scale cleanly to our model shape without careful memory planning.
- MAML can destabilize if inner-LR is wrong. Mitigation: small inner-LR (0.001), ramp up.
- Outer MAML step is 3-5× more expensive than a regular training step (forward + backward through K_inner inner gradient steps). Budget accordingly — may need to run MAML only every 32-64 regular steps, not every step.
- Eval-time: MAML-trained model may not behave identically to Score-First TTT expectations. Check with micro-ablation first.

## Notes

Strongly OpenAI-wishlist-aligned (E2E TTT explicitly listed). If this works, it's a paper-worthy finding AND unlocks the "more adaptation budget" argument for stacking other ideas.
