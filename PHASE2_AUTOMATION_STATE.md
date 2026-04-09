# PHASE2_AUTOMATION_STATE.md — unified experiment driver state

**Pod**: M = `4jfptzwhy9exy9` (RTX 3090 24 GB, $0.46/h, eu-cz-1)
**Cron**: `137c5635` at `9,26,43 * * * *`
**Per-fire budget**: 10 min
**Total budget cap**: $5 / ~11 h wallclock on Pod M

## Experiments

| exp | description | status | val_bpb (unquant / quant) | ms/step | log | notes |
|---|---|---|---|---|---|---|
| **E1** | Shot 0e validation: Phase 1 stack + fix, submission/bootstrap.sh, no compile | **running** | — / — | — | `/tmp/paramgolf_bootstrap.log` | launched 05:41Z, in get_data.sh HF download |
| E2 | Phase 2 Shot 1 (torch.compile on) via phase2/bootstrap.sh | pending | | | | waits on E1 done |
| E3 | Code + test Shot 17 (fuzzy LR bandit, ~80 LOC) | pending | | | | needs coding |
| E4 | Code + test Shot 0b (streaming KV eval, ~250 LOC) | pending | | | | needs coding |
| E5 | Code + test Shot 10 (Parameter Banking + Parallel Muon, ~200 LOC) | pending | | | | needs coding |

## Fire log

| fire | utc | action | next |
|---|---|---|---|
| 1 | 20260409T0544Z | E1 running: get_data.sh downloading docs_selected.jsonl from HF. GPU 0%, container disk 60/80 GB (75%), 0 shards yet. Process alive (PID 307). | wait for tokenize + train + eval; next fire ~17 min |
| 2 | 20260409T0609Z | E1 still running: tokenize at 6.4M/15.4M docs (42%), 77 train shards written, hard-link fix confirmed (container disk 60→53 GB after cache drop), GPU 0% (tokenize CPU-only). No errors. | wait for tokenize to complete + shards + n-grams + train; next fire ~17 min |

## Running tally

- Pod M uptime: ~28 min
- Pod M spend: ~$0.22
- Total commits by driver: 1 (fire 1)
- E1 phase: tokenize ~42% done, expecting another ~30-40 min to tokenize+ngram+train
