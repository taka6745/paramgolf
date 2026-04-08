#!/usr/bin/env python3
"""cpu_jobs_emitter.py — emit useful jobs into data/cpu_jobs/pending/ so the
CPU worker pool always has something productive to do in parallel with GPU
training.

Called once per run_forever.sh outer loop iteration. Idempotent: if a job for
a given target already exists in pending/in_progress/done, it is skipped.

Currently emits:
  1. brotli_sweep on the most recent N final_model.int8.ptz logs (helps L10
     find a better compression level than zlib).
  2. ngram_table_inspect on every data/*_logprobs*.npy table (helps L09 plan
     tile cache + hash width tuning).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PENDING = REPO / "data" / "cpu_jobs" / "pending"
INPROG = REPO / "data" / "cpu_jobs" / "in_progress"
DONE = REPO / "data" / "cpu_jobs" / "done"

PENDING.mkdir(parents=True, exist_ok=True)
INPROG.mkdir(parents=True, exist_ok=True)
DONE.mkdir(parents=True, exist_ok=True)


def already_seen(name: str) -> bool:
    if (PENDING / f"{name}.json").exists():
        return True
    if (DONE / f"{name}.result.json").exists():
        return True
    for p in INPROG.glob(f"*_{name}.json"):
        return True
    return False


def emit_job(name: str, spec: dict) -> bool:
    if already_seen(name):
        return False
    spec["_emitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (PENDING / f"{name}.json").write_text(json.dumps(spec, indent=2))
    return True


def emit_brotli_sweeps(max_n: int = 3) -> int:
    """Emit a brotli_sweep job for the most-recent N int8 ptz files."""
    logs_dir = REPO / "runpod_tests" / "loop" / "logs"
    if not logs_dir.exists():
        return 0
    candidates = sorted(logs_dir.glob("*.int8.ptz"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_n]
    n = 0
    for c in candidates:
        name = f"brotli_{c.stem.replace('.int8','')}"
        if emit_job(name, {"type": "brotli_sweep", "path": str(c.relative_to(REPO))}):
            n += 1
    return n


def emit_ngram_inspections() -> int:
    """Emit a ngram_table_inspect job for every n-gram table currently on disk."""
    data_dir = REPO / "data"
    if not data_dir.exists():
        return 0
    n = 0
    for p in sorted(data_dir.glob("*_logprobs*.npy")):
        name = f"ngrinsp_{p.stem}"
        if emit_job(name, {"type": "ngram_table_inspect", "path": str(p.relative_to(REPO))}):
            n += 1
    return n


def main() -> int:
    n_brotli = emit_brotli_sweeps()
    n_ngram = emit_ngram_inspections()
    print(f"cpu_jobs_emitter: +{n_brotli} brotli +{n_ngram} ngram_inspect "
          f"pending={len(list(PENDING.glob('*.json')))} done={len(list(DONE.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
