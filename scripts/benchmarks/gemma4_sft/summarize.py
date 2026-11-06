"""Summary of a reproduce.sh results directory: one row per run, then the best configuration per framework.

Usage: python summarize.py <results dir>   (e.g. $BENCH_ROOT/results/s16384) -> <dir>/summary.tsv and a table on stdout.
Every framework's result JSON carries cluster_tokens_per_second (tokens trained in steps 6..N / their wall time) and
peak_mem_allocated_gib (max over ranks).
The step-1 loss is fixed by construction (same weights, same rows; frameworks agree within about 1%, the spread of their
bf16 kernels): a run whose token-weighted step-1 loss differs from the median of all runs in the directory by more than 2%
trained something else and is marked invalid and left out of the means.
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
    losses = data.get("losses") or []
    weighted = data.get("losses_token_weighted") or []
    step1 = weighted[0] if weighted and weighted[0] is not None else (losses[0] if losses else None)
    rows.append(
        {
            "framework": framework,
            "run": run,
            "config": run.rsplit("_r", 1)[0],
            "tokens_per_second": round(tps),
            "peak_alloc_gib": round(data.get("peak_mem_allocated_gib", 0), 1),
            "loss_step1": round(step1, 4) if step1 is not None else "",
            "loss_last": round(losses[-1], 4) if losses else "",
        }
    )
step1 = sorted(r["loss_step1"] for r in rows if r["loss_step1"] != "")
ref = step1[len(step1) // 2] if step1 else 0.0
for r in rows:
    r["valid"] = "no: step-1 loss" if r["loss_step1"] != "" and abs(r["loss_step1"] - ref) > 0.02 * ref else "yes"
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
