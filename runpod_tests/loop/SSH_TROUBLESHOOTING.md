# SSH Troubleshooting — RunPod community pod auth via ssh.runpod.io

This file documents SSH issues we've hit with RunPod's community pod SSH proxy and the fixes that worked. **Read this before debugging SSH again.**

---

## The bug we kept hitting

**Symptom**: `ssh -tt -i ~/.ssh/id_ed25519 -o ... <pod>-<hash>@ssh.runpod.io` returns:
```
<pod>-<hash>@ssh.runpod.io: Permission denied (publickey).
```

We hit this multiple times. Sometimes after manual recovery, sometimes after running too many sweeps in a row.

## What it ISN'T

- Not a wrong key. `~/.ssh/id_ed25519.pub` IS the key authorized on the pods (the user's `takoda.mundy@biarri.com` ed25519, fingerprint `SHA256:xO+TnqNSuf95AsZzKRIGhFZw3smTdqiMFe/H8egwCTc`).
- Not a stale `known_hosts` entry. Clearing `known_hosts` doesn't fix it (and clearing makes it worse temporarily).
- Not a server-side key revocation. The verbose `ssh -vvv` output shows `Server accepts key` in some configurations.
- Not Pod A being gone. All other pods (B-G) hit it too.
- Not `MaxAuthTries` exhausted. Even with a single `-i` and `IdentitiesOnly=yes`, it fails.

## What it IS — root cause

**The SSH agent corrupts the auth handshake on RunPod's `-tt` PTY-allocating sessions.** When `SSH_AUTH_SOCK` is set (i.e., the local user has the macOS SSH agent running with keys loaded — which Sourcetree does by default), `ssh` consults the agent during key exchange even when `-i` is passed. RunPod's Go-based SSH proxy fork has a bug or quirk where:
- **PTY mode (`-tt`)**: rejects auth when the agent is involved
- **Non-PTY mode**: accepts auth, but then refuses with "Your SSH client doesn't support PTY" because RunPod's proxy requires PTY allocation for community pods

The combination is a catch-22 unless you disable the agent entirely.

## The fix (works every time)

**In any shell script that ssh's to RunPod community pods**, set this BEFORE the `ssh` invocation:

```bash
unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o IdentityAgent=none \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 \
    "$POD_HOST" <<SSH_STDIN
... your commands here ...
SSH_STDIN
```

Critical flags explained:
- **`unset SSH_AUTH_SOCK`** — disables the agent at the env var level
- **`-F /dev/null`** — bypasses `~/.ssh/config` entirely (we have a Sourcetree-Generated GitLab section that confused things)
- **`-i "$SSH_KEY"`** — explicit identity file (not via agent)
- **`-o IdentitiesOnly=yes`** — only try the explicit `-i` key, don't try other keys
- **`-o IdentityAgent=none`** — second belt-and-braces to disable the agent at the SSH option level
- **`-o StrictHostKeyChecking=accept-new`** — auto-accept new host keys (the `ssh.runpod.io` host key)
- **`-o UserKnownHostsFile=/dev/null`** — don't write to (or read from) the global known_hosts
- **`-o ConnectTimeout=20`** — fail fast on dead pods
- **`-tt`** — RunPod's proxy requires a PTY; without it you get "Your SSH client doesn't support PTY"

## What I tried that DIDN'T work (so I don't waste time again)

1. ✗ `ssh-keygen -R <host>` — clearing known_hosts doesn't help and can introduce new failures
2. ✗ `-o IdentitiesOnly=yes` alone (without disabling the agent) — the agent still gets consulted
3. ✗ `ssh-add -D && ssh-add ~/.ssh/id_ed25519` — putting the SAME key in the agent fresh; still fails because the agent itself is the problem
4. ✗ Removing `-tt` — the proxy refuses with "doesn't support PTY"
5. ✗ Single `-t` instead of `-tt` — same proxy refusal
6. ✗ `-o BatchMode=yes` — irrelevant, doesn't address the agent
7. ✗ `-o ControlMaster=no` — no control master was active
8. ✗ Direct TCP (`ssh root@<ip> -p <port>`) — the test pods only expose SSH via the proxy, not via public TCP. Only Pod A (now gone) had public TCP.
9. ✗ `ssh user@host 'bash -s' < script.sh` — RunPod's proxy still demands PTY
10. ✗ Trying SSH from a fresh shell — the agent is per-user, not per-shell

## Verification command (to know your sweep will work)

```bash
unset SSH_AUTH_SOCK
ssh -tt -F /dev/null -i ~/.ssh/id_ed25519 \
    -o IdentitiesOnly=yes -o IdentityAgent=none \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=10 \
    <pod-id>-<hash>@ssh.runpod.io 'echo OK; exit'
```

If this prints `OK` and exits cleanly, the fix is working. If it prints `Permission denied (publickey)`, the agent is still being consulted somewhere — check `ssh-add -l` and verify `SSH_AUTH_SOCK` is empty in your environment.

## The .sh files in this directory that have the fix applied

- `runpod_tests/loop/podsweep.sh` — heartbeat sweep (uses `unset SSH_AUTH_SOCK` + `IdentityAgent=none`)
- `runpod_tests/loop/podphase2.sh` — phase 2 (n-gram build) dispatcher
- `runpod_tests/loop/podbootstrap.sh` — fresh-pod bootstrap
- `runpod_tests/loop/podkillbuild.sh` — emergency kill of stuck `05_build_dc500.py`

If you copy any of these into a new script, **don't strip the `unset SSH_AUTH_SOCK` line**. It looks unnecessary but it's load-bearing.

## Last verified working

2026-04-08T0405Z by C5 monitor sweep on Pod E (`9g10r6i4rst296-644114cb`), commit `28f3edc` baseline. After the fix, sweep returned exit 0 and pulled fresh `results.jsonl` rows showing `L06_ln_x_L07_byte_seed42 = 2.2276`.
