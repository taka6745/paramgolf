#!/usr/bin/env python3
"""
09_compute_shard_entropy.py — Per-shard byte-entropy proxy via zlib

Produces a JSON map {"<shard_basename.bin>": <zlib_ratio_float>} that the
USE_DAT_BYTE_ENTROPY_CURRICULUM (Patch 32) data loader picks up to reorder
shards low-to-high (easy bytes first → faster early loss drop).

Lower ratio = more compressible = lower-entropy = "easier" curriculum stage.

Idempotent: skips existing entries unless BEC_FORCE=1.
Output written to: data/datasets/<dataset>/shard_entropy.json

Env vars:
    BEC_SAMPLE_BYTES   sample size per shard (default 4 MiB)
    BEC_FORCE          1 = recompute even if file exists
    BEC_DATASETS       comma-separated dataset dirs (default sp1024 + sp8192)
"""

import json
import os
import sys
import time
import zlib
from pathlib import Path

SAMPLE_BYTES = int(os.environ.get("BEC_SAMPLE_BYTES", str(4 * 1024 * 1024)))
FORCE = bool(int(os.environ.get("BEC_FORCE", "0")))
DEFAULT_DATASETS = [
    Path("data/datasets/fineweb10B_sp1024"),
    Path("data/datasets/fineweb10B_sp8192"),
]
_envds = os.environ.get("BEC_DATASETS", "").strip()
DATASETS = [Path(p) for p in _envds.split(",") if p] if _envds else DEFAULT_DATASETS

HEADER_BYTES = 1024


def shard_entropy(path: Path, sample_bytes: int) -> float:
    try:
        size = path.stat().st_size
    except OSError:
        return 1.0
    if size <= HEADER_BYTES:
        return 1.0
    take = min(sample_bytes, size - HEADER_BYTES)
    if take <= 0:
        return 1.0
    with open(path, "rb") as fh:
        fh.seek(HEADER_BYTES)
        sample = fh.read(take)
    if len(sample) == 0:
        return 1.0
    return float(len(zlib.compress(sample, 1))) / float(len(sample))


def process_dataset(ds_dir: Path) -> int:
    if not ds_dir.exists():
        print(f"  skip: {ds_dir} does not exist")
        return 0
    out_path = ds_dir / "shard_entropy.json"
    existing = {}
    if out_path.exists() and not FORCE:
        try:
            existing = json.loads(out_path.read_text())
            print(f"  loaded existing {out_path} with {len(existing)} entries")
        except Exception as e:
            print(f"  warn: failed to load {out_path}: {e}")
    shards = sorted(ds_dir.glob("fineweb_train_*.bin")) + sorted(ds_dir.glob("fineweb_val_*.bin"))
    if not shards:
        print(f"  skip: no shards in {ds_dir}")
        return 0
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    for shard in shards:
        if shard.name in existing and not FORCE:
            n_skipped += 1
            continue
        existing[shard.name] = shard_entropy(shard, SAMPLE_BYTES)
        n_done += 1
        if n_done % 25 == 0:
            print(f"    {n_done} done, last={shard.name}")
    sorted_map = dict(sorted(existing.items(), key=lambda kv: kv[1]))
    out_path.write_text(json.dumps(sorted_map, indent=2))
    elapsed = time.time() - t0
    if existing:
        ratios = sorted(existing.values())
        print(
            f"  wrote {out_path}: {len(existing)} entries (new={n_done} cached={n_skipped}) "
            f"min={ratios[0]:.4f} median={ratios[len(ratios)//2]:.4f} max={ratios[-1]:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
    return n_done


def main() -> int:
    print(f"=== 09_compute_shard_entropy.py — sample={SAMPLE_BYTES}B/shard force={FORCE} ===")
    total = 0
    for ds in DATASETS:
        print(f"--- {ds} ---")
        total += process_dataset(ds)
    print(f"=== done: {total} shards (re)processed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
