# Submission — The Actual Build

**This is NOT a test folder.** Tests live in `validate/` (sanity checks)
and `unknown/` (research/exploration).

This folder is for producing the actual artifact you ship to OpenAI.

## Contents

| Script | Hardware | Cost | What |
|---|---|---|---|
| `3seed_final.sh` | 8xH100 | ~$15 | Run 3 seeds with the winning config, compute mean ± std |

## How to use

```bash
# Spin up an 8xH100 pod (NOT a 3060)
# SSH in, then:
cd /workspace/paramgolf/runpod_tests
./submission.sh
```

`submission.sh` (at the top level of `runpod_tests/`) is the runner that
executes everything in this folder. It's separate from `unknown.sh` because
this isn't an exploration loop — it's a one-shot build.

## When to run

Only AFTER you have:
1. ✅ Validated the code on a 3060 (`./validate.sh` passes)
2. ✅ Picked the winning architecture from `unknown/u01_arch_sweep`
3. ✅ Confirmed progressive seq is worth it from `unknown/u02_progressive_seq`
4. ✅ Confirmed eval cache is worth it from `unknown/u03_eval_cache`
5. ✅ A single-seed dry run (`unknown/u04_full_stack`) gave a competitive number

If any of those is missing, run them first. Don't burn $15 on an 8xH100 to
discover that something basic is broken.

## Outputs

```
logs/submission/
├── seed_42.log     full training + eval log for seed 42
├── seed_314.log    seed 314
├── seed_999.log    seed 999
└── results.txt     mean ± std + the JSON snippet for submission.json
```

After running, take the mean from `results.txt` and copy it into
`records/track_10min_16mb/2026-04-XX_OURNAME/submission.json`.
