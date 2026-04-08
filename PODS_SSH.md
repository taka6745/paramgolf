# PODS_SSH.md — RunPod test fleet (2026-04-08)

7 pods spun up for parallel TEST_PLAN_TODAY.md execution. All use community cloud (direct TCP SSH preferred over the `ssh.runpod.io` proxy because community pods can have proxy quirks).

**Key**: SSH key is `~/.ssh/id_ed25519`. Workspace path on every pod: `/workspace`.

## Pod 1 — paramgolf-v2 (ORIGINAL pod, RTX 3080 Ti 12 GB)

- **ID**: `tyf0q5l1kgefgx`
- **User hash**: `64410a6f`
- **SSH proxy**: `ssh tyf0q5l1kgefgx-64410a6f@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@136.61.20.181 -p 4129 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `136.61.20.181:4129 → :22`
- **Role**: anchor pod, has the existing loop, results.jsonl history, and `data/` artifacts

## Pod 2 — paramgolf-test-1 (RTX 3090 24 GB)

- **ID**: `vwkkjkevpvyrfs`
- **User hash**: `6441169a`
- **SSH proxy**: `ssh vwkkjkevpvyrfs-6441169a@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.156 -p 19650 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.156:19650 → :22`

## Pod 3 — paramgolf-test-2 (RTX 3090 24 GB)

- **ID**: `1yo8wu8n77nbv8`
- **User hash**: `64411ad5`
- **SSH proxy**: `ssh 1yo8wu8n77nbv8-64411ad5@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: TBD (image only showed proxy line — request from user if proxy fails)
- **Direct TCP**: TBD

## Pod 4 — paramgolf-test-3 (RTX 3090 24 GB)

- **ID**: `1nqdd6aajwqofk`
- **User hash**: `64411378`
- **SSH proxy**: `ssh 1nqdd6aajwqofk-64411378@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@99.69.17.69 -p 10323 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `99.69.17.69:10323 → :22`

## Pod 5 — paramgolf-test-4 (RTX 3090 24 GB)

- **ID**: `9g10r6i4rst296`
- **User hash**: `644114cb`
- **SSH proxy**: `ssh 9g10r6i4rst296-644114cb@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@99.69.17.69 -p 11168 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `99.69.17.69:11168 → :22`

## Pod 6 — paramgolf-test-5 (RTX 3090 24 GB)

- **ID**: `373y5iemxa5s9o`
- **User hash**: `64411631`
- **SSH proxy**: `ssh 373y5iemxa5s9o-64411631@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@194.26.196.165 -p 18620 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `194.26.196.165:18620 → :22`

## Pod 7 — paramgolf-test-6 (RTX 4070 Ti 12 GB)

- **ID**: `7yp2f7j6rm9unm`
- **User hash**: `64410d97`
- **SSH proxy**: `ssh 7yp2f7j6rm9unm-64410d97@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **SSH over exposed TCP**: `ssh root@87.197.146.56 -p 40261 -i ~/.ssh/id_ed25519`
- **Direct TCP**: `87.197.146.56:40261 → :22`

---

## Quick reference table

| # | name | GPU | id | direct ssh |
|---|---|---|---|---|
| 1 | paramgolf-v2 | RTX 3080 Ti 12 GB | `tyf0q5l1kgefgx` | `ssh root@136.61.20.181 -p 4129 -i ~/.ssh/id_ed25519` |
| 2 | paramgolf-test-1 | RTX 3090 24 GB | `vwkkjkevpvyrfs` | `ssh root@194.26.196.156 -p 19650 -i ~/.ssh/id_ed25519` |
| 3 | paramgolf-test-2 | RTX 3090 24 GB | `1yo8wu8n77nbv8` | TBD (proxy: `1yo8wu8n77nbv8-64411ad5@ssh.runpod.io`) |
| 4 | paramgolf-test-3 | RTX 3090 24 GB | `1nqdd6aajwqofk` | `ssh root@99.69.17.69 -p 10323 -i ~/.ssh/id_ed25519` |
| 5 | paramgolf-test-4 | RTX 3090 24 GB | `9g10r6i4rst296` | `ssh root@99.69.17.69 -p 11168 -i ~/.ssh/id_ed25519` |
| 6 | paramgolf-test-5 | RTX 3090 24 GB | `373y5iemxa5s9o` | `ssh root@194.26.196.165 -p 18620 -i ~/.ssh/id_ed25519` |
| 7 | paramgolf-test-6 | RTX 4070 Ti 12 GB | `7yp2f7j6rm9unm` | `ssh root@87.197.146.56 -p 40261 -i ~/.ssh/id_ed25519` |

---

## Important notes

1. **Per-pod user hash** — every pod has its OWN `<podid>-<hash>` SSH proxy address. Do NOT assume the hash is the same across pods (this burned us earlier).
2. **Community cloud quirks** — community pods can have flaky proxy access. Prefer the direct-TCP `ssh root@<ip> -p <port>` form when possible.
3. **Workspace path** — on every pod, the persistent volume is `/workspace`. Always `cd /workspace` before cloning.
4. **Bootstrap procedure** (run once per pod after first login):
   ```bash
   cd /workspace
   git clone https://github.com/taka6745/paramgolf.git
   cd paramgolf
   bash runpod_tests/chore/08_patch_train_gpt.sh
   python3 data/download_hf_docs_and_tokenize.py  # 10-15 min FineWeb shards
   # n-gram tables — copy from anchor pod (`tyf0q5l1kgefgx`) via /tmp/podpush.sh
   ```
5. **Loop runner** — start with `cd /workspace/paramgolf && bash runpod_tests/loop/run_forever.sh &` after bootstrapping.
6. **Killing pods** — when a test phase finishes: `runpodctl remove pod <id>` for that pod ONLY. Keep the anchor (`tyf0q5l1kgefgx`) alive for results aggregation.

---

## Spend tracking

- 7 pods @ ~$0.22-0.30/h = ~$1.50-2.10/h fleet cost
- $36 budget, $6.70 already spent → $29.30 headroom
- Soft cap $25 → ~9 hours of full fleet runtime
- **Plan**: aim to finish TEST_PLAN_TODAY.md phases A-F in 6 hours, kill the test pods, keep anchor + 1-2 spares for H100 escalation

Last updated: 2026-04-08 ~01:30 UTC (after the user shared 6 web-UI screenshots).
