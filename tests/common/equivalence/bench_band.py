"""Bench band: a ``make bench`` run held to the committed baselines ``tests/baselines/bench/<key>.json``.

A run is one log per repeat of the rows ``tests/gpu/profiling/run_baseline_benchmarks.sh`` launches, and every
rank of a row prints one line per key (``tests/common/reporting.py``). Per key, each gated metric
(:data:`GATES`) is judged against the band of its baseline's repeats: a rate or a duration must stay within it,
and a peak must not rise above it. A key compares only with a baseline measured by the same command on as many
ranks of the same GPU model; a measured key with no baseline, and a baseline the run did not measure, fail too.

``check`` writes ``evidence/bench_band-<revision>.json`` and exits non-zero on a failure. ``record`` makes the
baselines the run's: one file per key from at least :data:`MIN_REPEATS` repeats of one clean commit, refusing
repeats that do not each fall inside the band of the others. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.common.equivalence.revisions import EVIDENCE_DIR, write_evidence
from tests.common.repo import REPO_ROOT
from tests.common.reporting import BENCH_DECIMALS, parse_benchmarks

BASELINE_DIR = REPO_ROOT / "tests" / "baselines" / "bench"
# Keeps the store in git while it holds no baseline; record removes it.
STORE_PLACEHOLDER = ".gitkeep"
# A band is the baseline repeats' mean, give or take the larger of a margin (a fraction of the mean) and
# SPREAD_FACTOR times their spread (highest minus lowest). A peak's margin is narrower, but a peak gets one: held
# to the highest of three repeats exactly, a peak that varies between runs fails one run in four.
RATE_MARGIN = 0.02
PEAK_MARGIN = 0.005
SPREAD_FACTOR = 2
MIN_REPEATS = 3
PROVENANCE_FIELDS = ("commit", "image", "measured", "world_size", "command")
LINE_FIELDS = frozenset({"key", "rank", "world_size", "command", "metrics"})
_COMMIT = re.compile(r"[0-9a-f]{40}")


def band(repeats: Sequence[float], margin: float) -> tuple[float, float]:
    """The ``(low, high)`` band a metric's baseline repeats span, at the precision the metrics carry."""
    center = statistics.fmean(repeats)
    half_width = max(margin * center, SPREAD_FACTOR * (max(repeats) - min(repeats)))
    return round(center - half_width, BENCH_DECIMALS), round(center + half_width, BENCH_DECIMALS)


def outliers(repeats: Sequence[float], margin: float) -> list[int]:
    """The repeats outside the band of the others."""
    found = []
    for index, value in enumerate(repeats):
        low, high = band([*repeats[:index], *repeats[index + 1 :]], margin)
        if not low <= value <= high:
            found.append(index)
    return found


@dataclass(frozen=True)
class Banded:
    """A rate or a duration, the mean over ranks and repeats. It fails outside its band on either side: a band a
    speedup left behind would pass the regression that undoes it."""

    margin: float

    def reduce(self, values: Sequence[float]) -> float:
        return round(statistics.fmean(values), BENCH_DECIMALS)

    def judge(self, repeats: Sequence[float], value: float) -> dict:
        low, high = band(repeats, self.margin)
        verdict = "below" if value < low else "above" if value > high else "within"
        return {"run": value, "band": [low, high], "verdict": verdict}


@dataclass(frozen=True)
class Ceiling:
    """A high-water mark, the highest over ranks and repeats. It fails above its band's upper edge, which is never
    below the highest repeat; a lower peak passes."""

    margin: float

    def reduce(self, values: Sequence[float]) -> float:
        return max(values)

    def judge(self, repeats: Sequence[float], value: float) -> dict:
        _, high = band(repeats, self.margin)
        return {"run": value, "ceiling": high, "verdict": "above" if value > high else "within"}


GATES: dict[str, Banded | Ceiling] = {
    "tokens_per_second": Banded(margin=RATE_MARGIN),
    "peak_allocated_gb": Ceiling(margin=PEAK_MARGIN),
    "save_seconds": Banded(margin=RATE_MARGIN),
    "save_peak_rss_gib": Ceiling(margin=PEAK_MARGIN),
}


@dataclass(frozen=True)
class Measurement:
    """One key as one log measured it: rank 0's metrics with every gated metric reduced over all ranks."""

    command: str
    world_size: int
    metrics: dict


def _measurement(log: Path, key: str, lines: Mapping[int, dict]) -> Measurement:
    world_sizes = {line["world_size"] for line in lines.values()}
    commands = {line["command"] for line in lines.values()}
    not_one_launch = (
        f"{log}: the lines of {key} are not one launch's ranks: ranks {sorted(lines)}, world sizes "
        f"{sorted(world_sizes)}, commands {sorted(commands)}"
    )
    if len(world_sizes) != 1 or len(commands) != 1:
        raise ValueError(not_one_launch)
    (world_size,), (command,) = world_sizes, commands
    if set(lines) != set(range(world_size)):
        raise ValueError(not_one_launch)
    metrics = dict(lines[0]["metrics"])
    for name, gate in GATES.items():
        if name in metrics:
            metrics[name] = gate.reduce([line["metrics"][name] for line in lines.values()])
    return Measurement(command=command, world_size=world_size, metrics=metrics)


def read_logs(logs: Sequence[Path]) -> dict[str, list[Measurement]]:
    """Per key, its measurement in each log, in log order: one log per repeat of the rows."""
    run: dict[str, list[Measurement]] = defaultdict(list)
    for log in logs:
        by_key: dict[str, dict[int, dict]] = defaultdict(dict)
        for line in parse_benchmarks(log.read_text(encoding="utf-8", errors="replace")):
            if missing := sorted(LINE_FIELDS - line.keys()):
                raise ValueError(f"{log}: a benchmark line lacks {missing}: {json.dumps(line)[:200]}")
            ranks = by_key[line["key"]]
            if line["rank"] in ranks:
                raise ValueError(f"{log}: rank {line['rank']} printed {line['key']} twice: two rows emit one key")
            ranks[line["rank"]] = line
        for key, lines in by_key.items():
            run[key].append(_measurement(log, key, lines))
    return dict(run)


def _inconsistencies(repeats: Sequence[Mapping]) -> list[str]:
    """Each gated metric some repeats lack, and each repeat outside the band of the others."""
    problems = []
    for name, gate in GATES.items():
        holding = [name in repeat for repeat in repeats]
        if any(holding) and not all(holding):
            problems.append(f"{name} is missing from some repeats")
        elif all(holding):
            values = [repeat[name] for repeat in repeats]
            problems += [
                f"repeat {index + 1}'s {name} {values[index]} lies outside the band of the others"
                for index in outliers(values, gate.margin)
            ]
    return problems


@dataclass(frozen=True)
class Baseline:
    """A committed baseline: how it was measured, and the metrics of each repeat."""

    key: str
    provenance: dict
    repeats: list[dict]


def load_baselines(directory: Path) -> dict[str, Baseline]:
    """Every baseline in ``directory`` by key. A file whose key is not its name, whose provenance is incomplete,
    or whose repeats are too few or do not each fall inside the band of the others raises."""
    baselines = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        repeats = data.get("repeats", [])
        problems = _inconsistencies(repeats)
        if data.get("key") != path.stem:
            problems.append(f"its key {data.get('key')!r} is not its file name")
        if missing := [field for field in PROVENANCE_FIELDS if field not in data.get("provenance", {})]:
            problems.append(f"its provenance lacks {missing}")
        if len(repeats) < MIN_REPEATS:
            problems.append(f"it holds {len(repeats)} repeats, fewer than {MIN_REPEATS}")
        if problems:
            raise ValueError(f"{path}: {'; '.join(problems)}")
        baselines[path.stem] = Baseline(path.stem, data["provenance"], repeats)
    return baselines


def _gpu_model(metrics: Mapping) -> str | None:
    return metrics.get("diagnostics", {}).get("gpu_model")


def _incomparability(baseline: Baseline, measurements: Sequence[Measurement]) -> str | None:
    """Why ``measurements`` cannot be held to ``baseline``, or ``None`` when they can."""
    provenance = baseline.provenance
    for measurement in measurements:
        if measurement.command != provenance["command"]:
            return f"the run's command {measurement.command!r} is not the baseline's {provenance['command']!r}"
        if measurement.world_size != provenance["world_size"]:
            return f"the run spans {measurement.world_size} ranks, the baseline {provenance['world_size']}"
        run_gpu, baseline_gpu = _gpu_model(measurement.metrics), _gpu_model(baseline.repeats[0])
        if run_gpu != baseline_gpu:
            return f"the run measured on {run_gpu}, the baseline on {baseline_gpu}"
    return None


def _bound(judgment: Mapping) -> str:
    if "band" in judgment:
        low, high = judgment["band"]
        return f"band {low} .. {high}"
    return f"ceiling {judgment['ceiling']}"


def check(run: Mapping[str, Sequence[Measurement]], baselines: Mapping[str, Baseline]) -> dict:
    """``run`` held to ``baselines``: the verdict, every failure, and each key's judged metrics."""
    failures = [f"{key}: has a baseline the run did not measure" for key in sorted(baselines.keys() - run.keys())]
    failures += [f"{key}: measured, but has no baseline" for key in sorted(run.keys() - baselines.keys())]
    judged = {}
    for key in sorted(run.keys() & baselines.keys()):
        baseline, measurements = baselines[key], run[key]
        if reason := _incomparability(baseline, measurements):
            failures.append(f"{key}: not comparable with its baseline: {reason}")
            continue
        judged[key] = {"commit": baseline.provenance["commit"], "metrics": {}}
        for name, gate in GATES.items():
            in_baseline = name in baseline.repeats[0]
            if in_baseline != all(name in measurement.metrics for measurement in measurements):
                failures.append(f"{key}: {name} is in the {'baseline' if in_baseline else 'run'} only")
                continue
            if not in_baseline:
                continue
            value = gate.reduce([measurement.metrics[name] for measurement in measurements])
            judgment = gate.judge([repeat[name] for repeat in baseline.repeats], value)
            judged[key]["metrics"][name] = judgment
            if judgment["verdict"] != "within":
                failures.append(f"{key}: {name} {value} is {judgment['verdict']} its {_bound(judgment)}")
    return {"verdict": "fail" if failures else "pass", "failures": failures, "keys": judged}


def record(
    run: Mapping[str, Sequence[Measurement]],
    *,
    commit: str,
    image: str,
    measured: datetime.date,
    directory: Path,
) -> list[Path]:
    """Make ``directory`` hold ``run``'s baselines, measured at ``commit``: one file per key, every other baseline
    in it removed. Writes nothing and raises when ``commit`` is not a full commit hash (a modified tree's revision
    is not), or a key has fewer than :data:`MIN_REPEATS` repeats, repeats of differing commands or world sizes, or
    a repeat outside the band of the others."""
    if not _COMMIT.fullmatch(commit):
        raise ValueError(f"a baseline names the clean commit it was measured at, and {commit!r} is not one")
    if not run:
        raise ValueError("the logs hold no benchmark lines")
    problems, baselines = [], {}
    for key, measurements in sorted(run.items()):
        setups = {(measurement.command, measurement.world_size) for measurement in measurements}
        if len(measurements) < MIN_REPEATS:
            problems.append(f"{key}: {len(measurements)} repeats, fewer than {MIN_REPEATS}")
        elif len(setups) > 1:
            problems.append(f"{key}: its repeats differ in command or world size: {sorted(setups)}")
        else:
            repeats = [measurement.metrics for measurement in measurements]
            problems += [f"{key}: {problem}" for problem in _inconsistencies(repeats)]
            command, world_size = setups.pop()
            provenance = {
                "commit": commit,
                "image": image,
                "measured": measured.isoformat(),
                "world_size": world_size,
                "command": command,
            }
            baselines[key] = {"key": key, "provenance": provenance, "repeats": repeats}
    if problems:
        raise ValueError("not recording the baselines:\n  " + "\n  ".join(problems))
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for key, baseline in baselines.items():
        path = directory / f"{key}.json"
        path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    for stale in set(directory.glob("*.json")) - set(written):
        stale.unlink()
    (directory / STORE_PLACEHOLDER).unlink(missing_ok=True)
    return written


def _print_verdict(verdict: Mapping) -> None:
    for key, judged in verdict["keys"].items():
        print(f"{key} (baseline {judged['commit'][:12]})")
        for name, judgment in judged["metrics"].items():
            print(f"  {name:<20} {judgment['run']:>12}  {_bound(judgment):<28} {judgment['verdict']}")
    for failure in verdict["failures"]:
        print(f"FAIL {failure}")
    print(f"bench band: {verdict['verdict']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold make bench logs to tests/baselines/bench, or record them.")
    parser.add_argument("mode", choices=("check", "record"))
    parser.add_argument("logs", nargs="+", type=Path, help="make bench logs, one per repeat")
    parser.add_argument("--revision", required=True, help="the commit the logs measured, suffixed -dirty if modified")
    parser.add_argument("--image", required=True, help="the image the rows ran in")
    args = parser.parse_args(argv)
    try:
        run = read_logs(args.logs)
        if args.mode == "record":
            written = record(
                run, commit=args.revision, image=args.image, measured=datetime.date.today(), directory=BASELINE_DIR
            )
            for path in written:
                print(f"wrote {path}")
            return 0
        baselines = load_baselines(BASELINE_DIR)
    except ValueError as error:
        raise SystemExit(f"bench band: {error}") from error
    verdict = check(run, baselines)
    _print_verdict(verdict)
    evidence = write_evidence(
        "bench_band",
        base=" ".join(sorted({baseline.provenance["commit"] for baseline in baselines.values()})),
        head=args.revision,
        image=args.image,
        verdict=verdict["verdict"],
        findings={
            "logs": [str(log) for log in args.logs],
            **{name: verdict[name] for name in ("failures", "keys")},
        },
        directory=EVIDENCE_DIR,
    )
    print(f"evidence: {evidence}")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
