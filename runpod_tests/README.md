# RunPod Test Suite

Three folders, three runner scripts. Run on a 3060 first, ship winners to H100.

## TL;DR — One-line bootstrap

SSH into the RunPod box, then paste this single command:

```bash
curl -sL https://raw.githubusercontent.com/taka6745/paramgolf/main/runpod_tests/bootstrap.sh | bash
```

This installs git, clones the repo to `/workspace/paramgolf`, makes scripts executable, and prints the next commands.

Then run the tests:

```bash
cd /workspace/paramgolf/runpod_tests
./setup.sh          # data prep, tokenizer, n-grams         → logs/setup.log
./validate.sh       # confirm code runs, no NaN            → logs/validate.log
./unknown.sh        # the actual research                  → logs/unknown.log
./export_logs.sh    # uploads logs and prints download URLs
```

Or chain everything in one command (~5 hrs, ~$1 on a 3060):
```bash
cd /workspace/paramgolf/runpod_tests && ./setup.sh && ./validate.sh && ./unknown.sh && ./export_logs.sh
```

## Structure

```
runpod_tests/
├── README.md            this file
├── setup.sh             ← run all chore tests
├── validate.sh          ← run all validate tests
├── unknown.sh           ← run all unknown tests (or pass: ./unknown.sh u01 u02)
├── export_logs.sh       ← upload all logs off the pod
│
├── logs/                created by the runners
│   ├── setup.log
│   ├── validate.log
│   └── unknown.log
│
├── chore/               one-time setup, individual scripts
│   ├── 00_setup_pod.sh
│   ├── 01_download_data.sh
│   ├── 02_build_tokenizer.sh
│   ├── 03_retokenize.sh
│   ├── 04_build_ngrams.py
│   ├── 05_build_dc500.py
│   ├── 06_lloyd_max.py
│   └── 07_verify_data.sh
│
├── validate/            mac-validated wins, just confirm CUDA works
│   ├── v01_smoke_test.sh
│   ├── v02_ngram_bias.py
│   ├── v03_wavelet_mix.py
│   ├── v04_progressive_seq.sh
│   ├── v05_cosine_lr.py
│   ├── v06_ema_model.py
│   ├── v07_eval_cache.py
│   ├── v08_hedge_mixer.py
│   ├── v09_lloyd_max_quant.py
│   └── v10_full_stack_smoke.sh
│
└── unknown/             the actual research
    ├── u01_arch_sweep.sh
    ├── u02_progressive_seq.sh
    ├── u03_eval_cache.sh
    ├── u04_full_stack.sh
    ├── u05_3seed_final.sh    (8xH100 only)
    ├── u06_speed_baseline.sh
    ├── u07_gla_shootout.sh
    └── u08_gla_progressive.sh
```

## Strategy: 3060 first, H100 for winners

The user plan is **everything on 3060, only winners to H100**. That's a valid strategy provided you treat 3060 results as **relative comparisons**, not absolute predictions:

| What 3060 tells you | What it doesn't |
|---|---|
| Code compiles, no NaN | Real H100 BPP |
| Loss decreases | Real H100 ms/step |
| Config A vs Config B (relative) | Whether progressive seq's full value transfers |
| Patches apply correctly | Real eval cache impact at H100 batch sizes |

**Caveat for u02 / u03 / u04 on 3060:** The progressive seq finding came from a 3080 Ti with batch=65K-200K. On a 3060 with 12GB you'll need batch ~8K-16K. The RELATIVE improvement (progressive vs baseline) should still hold, but the absolute numbers won't match what you'll see on H100.

**Recommendation:** run all unknown/ on 3060, identify the 1-2 winning configs, then re-run ONLY those on a single 1xH100 to get real numbers, then 3-seed final on 8xH100.

## Cost (3060-only path)

| Step | GPU | Time | Cost |
|---|---|---|---|
| `./setup.sh` | 3060 | ~30 min | $0.10 |
| `./validate.sh` | 3060 | ~1 hr | $0.20 |
| `./unknown.sh` (all u01-u04, u06-u08) | 3060 | ~3-4 hrs | $0.70 |
| **3060 phase total** | | ~5 hrs | **~$1.00** |
| Re-run winners on 1xH100 | H100 | 1-2 hrs | $3-6 |
| 3-seed final | 8xH100 | 45 min | $15 |
| **GRAND TOTAL** | | ~7 hrs | **~$22** |

Buffer for debugging: +$10 = **~$32 total**.

## Decision flow

```
./setup.sh → setup.log
  ↓ if PASS
./validate.sh → validate.log
  ↓ if PASS
./unknown.sh → unknown.log
  ↓ identify winners (best val_bpb for each config option)
./export_logs.sh
  ↓ download logs to laptop
analyze on laptop, decide final config
  ↓
spin up 1xH100, re-run only winning config
  ↓ if real H100 number is good
spin up 8xH100, run u05_3seed_final.sh
  ↓
submit
```

## How the runners work

Each runner (`setup.sh`, `validate.sh`, `unknown.sh`) does:

1. Detects GPU and writes a header to its log
2. Runs each individual test script in order
3. Wraps each test in a clear START/END block with timing
4. Records PASS/FAIL per test
5. Writes a summary at the end with total pass/fail count
6. Exits with the number of failed tests (so you can chain: `./setup.sh && ./validate.sh && ./unknown.sh`)

The logs are pure text — easy to grep, easy to read, easy to upload.

## How `export_logs.sh` works

1. Bundles `logs/` into a `tar.gz` file
2. Tries 3 free upload services in order: transfer.sh → 0x0.st → file.io
3. Also uploads each individual log separately (in case the bundle fails)
4. Prints all URLs

You then copy the URLs to your local machine and `curl -O` them.

**No SCP, no SSH key setup, no GitHub auth required.** Just URLs.

## Selective runs

If you only want to run specific unknown tests (e.g., re-run u02 after a fix):

```bash
./unknown.sh u02            # only u02
./unknown.sh u02 u03        # u02 and u03
./unknown.sh u06 u07 u08    # speed-related tests only
```

## What if a test fails?

- Check the log for the failed test name (look for `<<< RESULT: ... FAIL`)
- Read the lines BEFORE the FAIL marker for the actual error
- Fix and re-run just that test directly: `bash unknown/u02_progressive_seq.sh`
- Or re-run the whole runner: `./unknown.sh`

## Things to know

- **Don't run setup.sh twice on the same volume** — most chore steps short-circuit if outputs exist
- **Validate is idempotent** — safe to re-run
- **Unknown is NOT idempotent** — each run will append to existing logs and produce new model checkpoints
- **export_logs.sh tries 3 services** — if one fails, the bundle will still upload via another
- **transfer.sh files expire in 14 days**; **0x0.st in 30 days**; **file.io after first download** — save the URLs!
