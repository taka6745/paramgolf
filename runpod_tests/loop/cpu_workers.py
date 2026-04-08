#!/usr/bin/env python3
"""cpu_workers.py — CPU-side worker pool that runs in parallel with GPU training.

Spawned ONCE per pod by run_forever.sh's preflight(). Each worker process loops:
  1. Look in `data/cpu_jobs/pending/` for *.json job files
  2. Atomically rename to `data/cpu_jobs/in_progress/<worker_id>_<jobname>.json`
  3. Dispatch to a handler based on job["type"]
  4. Write result to `data/cpu_jobs/done/<jobname>.result.json`
  5. Sleep 5s if no work

Job types currently supported:
  - "brotli_sweep": given a path to a .ptz checkpoint, run brotli levels 0-11
    on a sample of bytes and report optimal level + size. Useful for finding
    the best brotli level for the int8 quant payload (vs. the default zlib).
  - "ngram_table_inspect": load a .npy n-gram table and report nnz, mean, max,
    sparsity. Useful for choosing tile sizes / hash widths.
  - "noop": for smoke testing the pool.

Goal: address the user's PD8 directive ("max out CPU+RAM, not just GPU") and
gap #2 from the 0648Z status report (CPU sitting idle while GPU trains).

Idempotent: launching twice is safe — the second launcher will see the .pid file
and exit. Jobs are pulled atomically via rename(), so two workers can't grab the
same job.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "data" / "cpu_jobs"
PENDING = JOBS / "pending"
INPROG = JOBS / "in_progress"
DONE = JOBS / "done"
PID = REPO / "runpod_tests" / "loop" / "cpu_workers.pid"
LOG = REPO / "runpod_tests" / "loop" / "cpu_workers.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{ts} cpu_workers: {msg}"
    try:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


# ---------------- Job handlers ----------------

def job_brotli_sweep(spec: dict) -> dict:
    """Sweep brotli levels 0-11 on a sample of bytes from a file.

    Inputs:
      path:        absolute or repo-relative file path
      sample_bytes: optional, default 4 MiB
    Output:
      best_level: int 0-11
      sizes: list of (level, size) tuples
      original_bytes: int
    """
    import brotli  # pip install brotli (already on pods)

    path = Path(spec["path"])
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return {"ok": False, "error": f"path not found: {path}"}
    sample_bytes = int(spec.get("sample_bytes", 4 * 1024 * 1024))
    raw = path.read_bytes()[:sample_bytes]
    if not raw:
        return {"ok": False, "error": "empty file"}
    sizes = []
    for level in range(12):
        try:
            t0 = time.time()
            comp = brotli.compress(raw, quality=level)
            sizes.append((level, len(comp), time.time() - t0))
        except Exception as e:
            sizes.append((level, -1, str(e)))
    best = min((s for s in sizes if s[1] > 0), key=lambda s: s[1])
    return {
        "ok": True,
        "path": str(path),
        "original_bytes": len(raw),
        "best_level": best[0],
        "best_size": best[1],
        "best_ratio": best[1] / len(raw),
        "all_sizes": sizes,
    }


def job_ngram_table_inspect(spec: dict) -> dict:
    """Inspect an n-gram table .npy file: nnz, mean, max, sparsity, shape."""
    import numpy as np

    path = Path(spec["path"])
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return {"ok": False, "error": f"path not found: {path}"}
    arr = np.load(str(path), mmap_mode="r")
    flat = np.asarray(arr).reshape(-1)
    nnz = int((flat != 0).sum())
    return {
        "ok": True,
        "path": str(path),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "n_total": int(flat.size),
        "n_nonzero": nnz,
        "sparsity": 1.0 - nnz / max(flat.size, 1),
        "mean": float(flat.mean()),
        "abs_max": float(abs(flat).max()) if flat.size else 0.0,
    }


def job_noop(spec: dict) -> dict:
    return {"ok": True, "echo": spec}


HANDLERS = {
    "brotli_sweep": job_brotli_sweep,
    "ngram_table_inspect": job_ngram_table_inspect,
    "noop": job_noop,
}


# ---------------- Worker loop ----------------

def worker_loop(worker_id: int, idle_sleep: float = 5.0) -> None:
    log(f"worker {worker_id} starting (pid={os.getpid()})")
    while True:
        try:
            jobs = sorted(PENDING.glob("*.json"))
            if not jobs:
                time.sleep(idle_sleep)
                continue
            grabbed = None
            for j in jobs:
                target = INPROG / f"w{worker_id}_{j.name}"
                try:
                    j.rename(target)
                    grabbed = target
                    break
                except (FileNotFoundError, OSError):
                    continue  # another worker beat us to it
            if grabbed is None:
                time.sleep(idle_sleep)
                continue
            try:
                spec = json.loads(grabbed.read_text())
            except Exception as e:
                log(f"worker {worker_id} bad json {grabbed.name}: {e}")
                grabbed.unlink(missing_ok=True)
                continue
            jtype = spec.get("type", "noop")
            handler = HANDLERS.get(jtype)
            if handler is None:
                result = {"ok": False, "error": f"unknown job type: {jtype}"}
            else:
                t0 = time.time()
                try:
                    result = handler(spec)
                    result["_runtime_s"] = round(time.time() - t0, 2)
                except Exception:
                    result = {
                        "ok": False,
                        "error": "handler raised",
                        "trace": traceback.format_exc(),
                        "_runtime_s": round(time.time() - t0, 2),
                    }
            result["_worker_id"] = worker_id
            result["_job_name"] = grabbed.stem
            result["_completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            out_path = DONE / f"{grabbed.stem}.result.json"
            out_path.write_text(json.dumps(result, indent=2))
            grabbed.unlink(missing_ok=True)
            log(f"worker {worker_id} done {jtype} {grabbed.stem} -> {out_path.name} ok={result.get('ok')}")
        except KeyboardInterrupt:
            log(f"worker {worker_id} KeyboardInterrupt — exiting")
            return
        except Exception:
            log(f"worker {worker_id} top-level error:\n{traceback.format_exc()}")
            time.sleep(idle_sleep)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="number of worker processes (default cpu_count-2)")
    p.add_argument("--idle-sleep", type=float, default=5.0)
    p.add_argument("--once", action="store_true",
                   help="run one worker in foreground (smoke test); ignore --n")
    args = p.parse_args()

    PENDING.mkdir(parents=True, exist_ok=True)
    INPROG.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)

    if args.once:
        worker_loop(0, idle_sleep=args.idle_sleep)
        return 0

    # PID-file guard so run_forever.sh doesn't fork-bomb the pool on each loop iter.
    if PID.exists():
        try:
            existing = int(PID.read_text().strip())
            os.kill(existing, 0)
            log(f"already running (pid={existing}), exiting")
            return 0
        except (ValueError, ProcessLookupError, OSError):
            log(f"stale pid file (pid={PID.read_text().strip()}), removing")
            PID.unlink(missing_ok=True)

    PID.write_text(str(os.getpid()))
    log(f"launching {args.n} workers (master pid={os.getpid()})")
    procs = []
    for i in range(args.n):
        proc = mp.Process(target=worker_loop, args=(i, args.idle_sleep), daemon=True)
        proc.start()
        procs.append(proc)
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        log("KeyboardInterrupt — terminating workers")
        for proc in procs:
            proc.terminate()
    finally:
        PID.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
