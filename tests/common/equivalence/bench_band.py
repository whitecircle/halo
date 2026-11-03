"""Bench band: a ``make bench`` run held to committed ``tests/baselines/bench/<key>.json`` baselines. Standard library
only. A key compares only with its own command, rank count and GPU model. ``check`` flags, but never fails, a mean far
off its median; ``record`` refuses outlying repeats."""

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
# A band is the repeats' mean, give or take the larger of a margin and SPREAD_FACTOR times their spread. A peak's
# margin is narrower, but it needs one: held to its highest repeat, a varying peak fails one run in MIN_REPEATS + 1.
RATE_MARGIN = 0.02
PEAK_MARGIN = 0.005
# Host I/O into the page cache moves the save's time by up to 6% between runs of one commit, its peak RSS in ~1% steps.
SAVE_TIME_MARGIN = 0.06
HOST_PEAK_MARGIN = 0.01
SPREAD_FACTOR = 2
# A few repeats' spread is a noisy estimate: five keep a band a later run passes, where three refuse almost every run.
MIN_REPEATS = 5
# A disturbed repeat (a row sharing the host with another job), not one at the edge of four repeats' spread.
OUTLIER_FACTOR = 4
STEP_TIME_SKEW = 0.02
PROVENANCE_FIELDS = ("commit", "image", "measured", "world_size", "command")
LINE_FIELDS = frozenset({"key", "rank", "world_size", "command", "metrics"})
_COMMIT = re.compile(r"[0-9a-f]{40}")


def band(repeats: Sequence[float], margin: float, spread_factor: float = SPREAD_FACTOR) -> tuple[float, float]:
    """The ``(low, high)`` band a metric's repeats span, at the precision the metrics carry."""
    center = statistics.fmean(repeats)
    half_width = max(margin * center, spread_factor * (max(repeats) - min(repeats)))
    return round(center - half_width, BENCH_DECIMALS), round(center + half_width, BENCH_DECIMALS)


@dataclass(frozen=True)
class Banded:
    """A rate or duration, the mean over ranks and repeats; fails on either side, since a band a speedup left behind
    would pass the regression undoing it."""

    margin: float

    def reduce(self, values: Sequence[float]) -> float:
        return round(statistics.fmean(values), BENCH_DECIMALS)

    def judge(self, repeats: Sequence[float], value: float) -> dict:
        low, high = band(repeats, self.margin)
        verdict = "below" if value < low else "above" if value > high else "within"
        return {"run": value, "band": [low, high], "verdict": verdict}

    def outlying(self, others: Sequence[float], value: float) -> str | None:
        low, high = band(others, self.margin, OUTLIER_FACTOR)
        return None if low <= value <= high else "lies outside the band of the others"


@dataclass(frozen=True)
class Ceiling:
    """A high-water mark, the highest over ranks and repeats; fails above its band's upper edge, and a lower peak
    passes."""

    margin: float

    def reduce(self, values: Sequence[float]) -> float:
        return max(values)

    def judge(self, repeats: Sequence[float], value: float) -> dict:
        _, high = band(repeats, self.margin)
        return {"run": value, "ceiling": high, "verdict": "above" if value > high else "within"}

    def outlying(self, others: Sequence[float], value: float) -> str | None:
        """Only a repeat above the others' ceiling: a lower one widens the spread, so the recorded ceiling still clears
        every repeat."""
        _, high = band(others, self.margin, OUTLIER_FACTOR)
        return None if value <= high else "lies above the ceiling of the others"


GATES: dict[str, Banded | Ceiling] = {
    "tokens_per_second": Banded(margin=RATE_MARGIN),
    "peak_allocated_gb": Ceiling(margin=PEAK_MARGIN),
    "save_seconds": Banded(margin=SAVE_TIME_MARGIN),
    "save_peak_rss_gib": Ceiling(margin=HOST_PEAK_MARGIN),
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
    """Per key, its measurement in each log, in order: one log per repeat."""
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
    problems = []
    for name, gate in GATES.items():
        holding = [name in repeat for repeat in repeats]
        if any(holding) and not all(holding):
            problems.append(f"{name} is missing from some repeats")
        elif all(holding):
            values = [repeat[name] for repeat in repeats]
            for index, value in enumerate(values):
                if reason := gate.outlying([*values[:index], *values[index + 1 :]], value):
                    problems.append(f"repeat {index + 1}'s {name} {value} {reason}")
    return problems


@dataclass(frozen=True)
class Baseline:
    key: str
    provenance: dict
    repeats: list[dict]


def load_baselines(directory: Path) -> dict[str, Baseline]:
    """Every baseline in ``directory`` by key; a mismatched key, incomplete provenance, or too few or outlying repeats
    raises."""
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


def _skewed_step_times(run: Mapping[str, Sequence[Measurement]]) -> list[str]:
    """Each repeat whose mean step time is more than :data:`STEP_TIME_SKEW` off its median."""
    flags = []
    for key, measurements in sorted(run.items()):
        for index, measurement in enumerate(measurements):
            metrics = measurement.metrics
            if "median_step_time_ms" not in metrics:
                continue
            mean, median = metrics["mean_step_time_ms"], metrics["median_step_time_ms"]
            if abs(mean - median) > STEP_TIME_SKEW * median:
                flags.append(
                    f"{key}: repeat {index + 1}'s mean step time {mean} ms is {mean / median - 1:+.1%} off its "
                    f"median {median} ms"
                )
    return flags


def _bound(judgment: Mapping) -> str:
    if "band" in judgment:
        low, high = judgment["band"]
        return f"band {low} .. {high}"
    return f"ceiling {judgment['ceiling']}"


def check(run: Mapping[str, Sequence[Measurement]], baselines: Mapping[str, Baseline]) -> dict:
    """``run`` held to ``baselines``: the verdict, every failure, flagged skews and each key's judged metrics."""
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
    return {
        "verdict": "fail" if failures else "pass",
        "failures": failures,
        "flags": _skewed_step_times(run),
        "keys": judged,
    }


def record(
    run: Mapping[str, Sequence[Measurement]],
    *,
    commit: str,
    image: str,
    measured: datetime.date,
    directory: Path,
) -> list[Path]:
    """Make ``directory`` hold ``run``'s baselines at ``commit``, one file per key, removing every other baseline.
    Writes nothing and raises on a non-full commit hash, too few repeats, differing commands or world sizes, or an
    outlying repeat."""
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
    for flag in verdict["flags"]:
        print(f"FLAG {flag}")
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
            **{name: verdict[name] for name in ("failures", "flags", "keys")},
        },
        directory=EVIDENCE_DIR,
    )
    print(f"evidence: {evidence}")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
