"""Summary of a reproduce.sh results directory: one row per run, then the best configuration per framework.

Usage: python summarize.py <results dir>   (e.g. $BENCH_ROOT/results/s16384) -> <dir>/summary.tsv and a table on stdout.
Every framework's result JSON carries cluster_tokens_per_second (tokens trained in steps 6..N / their wall time) and
peak_mem_allocated_gib (max over ranks); <run>.load holds the host's 1-minute load before and after the run.
The step-1 loss is fixed by construction (same weights, same rows); a run whose step-1 loss differs from the first run
of its configuration by more than 1% trained something else and is marked invalid and left out of the means.
"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for result in sorted(root.glob("*/*.json")):
    framework, run = result.parent.name, result.stem
    data = json.loads(result.read_text())
    tps = data.get("cluster_tokens_per_second")
    if tps is None:
        continue
    load = root / framework / f"{run}.load"
    loads = [line.split()[1] for line in load.read_text().splitlines()] if load.exists() else []
    losses = data.get("losses") or []
    rows.append(
        {
            "framework": framework,
            "run": run,
            "config": run.rsplit("_r", 1)[0],
            "tokens_per_second": round(tps),
            "peak_alloc_gib": round(data.get("peak_mem_allocated_gib", 0), 1),
            "loss_step1": round(losses[0], 4) if losses else "",
            "loss_last": round(losses[-1], 4) if losses else "",
            "load_before": loads[0] if loads else "",
            "load_after": loads[1] if len(loads) > 1 else "",
        }
    )
first = {}
for r in rows:
    key = (r["framework"], r["config"])
    first.setdefault(key, r["loss_step1"])
    ref = first[key]
    r["valid"] = "yes" if r["loss_step1"] == "" or abs(r["loss_step1"] - ref) <= 0.01 * ref else "no: step-1 loss"
header = list(rows[0]) if rows else []
lines = ["\t".join(header)] + ["\t".join(str(r[k]) for k in header) for r in rows]
(root / "summary.tsv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print()
best = {}
for r in rows:
    if r["valid"] != "yes":
        continue
    runs = [
        x["tokens_per_second"]
        for x in rows
        if x["framework"] == r["framework"] and x["config"] == r["config"] and x["valid"] == "yes"
    ]
    mean = sum(runs) / len(runs)
    if r["framework"] not in best or mean > best[r["framework"]][1]:
        best[r["framework"]] = (r["config"], mean, r["peak_alloc_gib"], runs)
for framework, (config, mean, peak, runs) in sorted(best.items(), key=lambda kv: -kv[1][1]):
    print(f"{framework:16s} {config:24s} {mean:8.0f} tok/s  runs {runs}  peak {peak} GiB")
