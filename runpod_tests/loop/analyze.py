#!/usr/bin/env python3
"""
analyze.py — read results.jsonl and produce a grouped/averaged leaderboard.
Groups results by name (so multi-seed runs are averaged), then prints
both individual rows and group summaries sorted by mean train_loss.
"""
import json
import statistics
import sys
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.jsonl"


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def main() -> None:
    results = load(RESULTS_FILE)
    if not results:
        print("(no results yet)")
        return

    # Group by name (or by config family if you prefer)
    by_name: dict[str, list[dict]] = {}
    for r in results:
        if r.get("exit_code") != 0 or r.get("train_loss") is None:
            continue
        by_name.setdefault(r["name"], []).append(r)

    rows = []
    for name, runs in by_name.items():
        losses = [r["train_loss"] for r in runs if r["train_loss"] is not None]
        if not losses:
            continue
        n = len(losses)
        mean = statistics.fmean(losses)
        std = statistics.stdev(losses) if n > 1 else 0.0
        best = min(losses)
        steps = [r.get("max_step", 0) for r in runs]
        ms = [r.get("ms_step") for r in runs if r.get("ms_step")]
        env = runs[0].get("env_overrides", {})
        rows.append(
            (mean, std, best, name, n, max(steps), (sum(ms) / len(ms)) if ms else 0, env)
        )
    rows.sort(key=lambda r: r[0])

    print(f"# Aggregated results — {len(rows)} unique configs, "
          f"{sum(r[4] for r in rows)} total runs")
    print()
    print(f"{'name':<30} {'mean_tl':>9} {'std':>7} {'best':>9} {'n':>3} {'steps':>6} {'ms':>7}  config")
    print("-" * 130)
    for mean, std, best, name, n, mxs, msa, env in rows:
        env_s = " ".join(f"{k}={v}" for k, v in env.items() if k != "MAX_WALLCLOCK_SECONDS")
        print(f"{name:<30} {mean:>9.4f} {std:>7.4f} {best:>9.4f} {n:>3} {mxs:>6} {msa:>7.1f}  {env_s}")

    # Family grouping (e.g. all D* are bigram weight sweep)
    families: dict[str, list[float]] = {}
    for r in results:
        if r.get("train_loss") is None:
            continue
        family = r["name"][:1] if r["name"] else "?"
        families.setdefault(family, []).append(r["train_loss"])
    print()
    print("# By family")
    for family, vals in sorted(families.items()):
        if not vals:
            continue
        m = statistics.fmean(vals)
        print(f"  {family}: n={len(vals)} mean={m:.4f} best={min(vals):.4f}")


if __name__ == "__main__":
    main()
