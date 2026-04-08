# PHASE1_RESULTS.md — append-only shot ledger

**Comp**: openai/parameter-golf
**Pod**: `9lfji49c6ngy9a` (paramgolf-phase1-h100, NVIDIA H100 PCIe 80GB, RunPod-rented)
**Plan**: PHASE1_PLAN.md
**Trainer**: `train_gpt_phase1.py` (clean decoded PR #1477, no patcher hunks)
**Cost cap**: $15

Each row: shot id, env diff, wallclock, val_bpb, artifact_bytes, ms/step, status,
timestamp. Append only.

| shot | env | wallclock | val_bpb | artifact_bytes | ms/step | status | utc |
|---|---|---|---|---|---|---|---|
| (none yet) | | | | | | pre-tokenize | |

---

## Comp anchors (for comparison — not our runs)

| PR | stack | val_bpb | hardware |
|---|---|---|---|
| #1477 | SP8192 + Parallel Residuals + Score-First TTT | **1.0822** | 8×H100 |
| #1476 | SP8192 + QK5 + Legal TTT | 1.0842 | 8×H100 |
| #1471 | SP8192 + SDClip + 3-Layer Depth Recurrence + EMA | 1.0866 | 8×H100 |
| #1019 | AR Self-Gen GPTQ + XSA-all + BigramHash3072 | 1.1147 | 8×H100 |

## Our pre-Phase-1 baseline (overnight cheap-pod 4070 Ti S2)

| stack | val_bpb | n_seeds | hardware |
|---|---|---|---|
| STACK_GATED_LEGAL_TTT (2-component minimal) | **1.3711** | 2 | 4070 Ti |

Phase 1 success criterion: get on H100 + SP8192 stack and land within 0.02-0.05 BPB
of the comp anchor (i.e. ~1.10-1.15 expected on 1×H100 with our smaller batch).
