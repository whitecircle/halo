#!/usr/bin/env python
"""
Offline re-grading of recorded code-contest trajectories — contention-free and reproducible.

Online/large-scale evaluation generates trajectories with many concurrent rollout workers per process
and several processes per host. Grading runs untrusted solutions in subprocesses, so at extreme
aggregate concurrency the *wall-clock* a solution is measured against is inflated by host load — a
correct, fast solution can be spuriously TIME-LIMIT-EXCEEDED. The robust pattern is to **decouple
generation from grading**: generate in parallel (cheap, API-bound), then re-grade the recorded
submissions here in a single process with bounded concurrency, where each run gets an effectively
dedicated core and the wall-clock limit is real.

This reads the JSONL files written by ``write_trajectories_jsonl`` (a ``{"type": "meta", ...}`` line
then one episode per line), rebuilds each problem's hidden tests from its dataset by index — the same
order ``build_examples`` used — and re-runs every recorded ``submit_solution`` through the *same*
``grade_solution`` the environment uses, reproducing the env's comparison / checker / time-limit
exactly. It reports first-submission and within-budget solve rates (``s@1`` / ``s@2``).

Usage::

    python scripts/environments/inference/regrade_trajectories.py /path/to/trajectories/*.jsonl \
        --workers 64 --output /path/to/regraded.jsonl
"""

import argparse
import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS
from src.environments.envs.tasks.coding.grading import grade_solution
from src.environments.eval_runner import load_hf_split
from src.environments.registry import resolve_environment

# Meta keys a re-grade needs: env_type/adapter/dataset select the environment and rebuild the hidden
# tests, model/language name the row of the report. Only run_code_contests.py stamps the full set —
# run_env.py writes the generic eval meta, without adapter/language.
_REQUIRED_META_KEYS = ("env_type", "adapter", "dataset", "model", "language")


def validate_meta(path: str, meta: dict[str, Any]) -> None:
    """Fail before any grading when the trajectory's meta line lacks a key the re-grade needs.

    Unvalidated, a missing selector surfaces as a bare ``KeyError`` and a missing report field as a
    ``TypeError`` in the summary line — the latter only AFTER the whole file has been graded.
    """
    missing = [key for key in _REQUIRED_META_KEYS if meta.get(key) is None]
    if missing:
        raise ValueError(
            f"{path}: trajectory meta is missing {missing}. Re-grading needs the code-contest meta "
            f"stamped by scripts/environments/inference/run_code_contests.py; trajectories from "
            f"run_env.py carry the generic eval meta (no adapter/language) and cannot be re-graded."
        )


def read_trajectories(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(meta, episodes)`` from a trajectory JSONL (line 1 is the meta record)."""
    meta: dict[str, Any] = {}
    episodes: list[dict[str, Any]] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            record = json.loads(line)
            if i == 0 and record.get("type") == "meta":
                meta = record
            else:
                episodes.append(record)
    return meta, episodes


@cache
def _load_payloads(adapter_name: str, dataset: str, config: str | None, split: str) -> tuple[dict[str, Any], ...]:
    """Rebuild a dataset's per-problem grading payloads (tests, checker, time limit) in the same order
    ``build_examples`` produced — so an episode's ``index`` selects its problem. Cached so re-grading a
    whole model × language matrix loads each dataset once."""
    adapter = CODE_DATASET_ADAPTERS[adapter_name]
    rows = adapter.load(dataset, config, split) if adapter.load else load_hf_split(dataset, config, split)
    return tuple(adapter.pack_verification(row) for row in rows if adapter.keep(row))


def build_payloads(meta: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Grading payloads for a trajectory's dataset, keyed by example index (see :func:`_load_payloads`)."""
    return _load_payloads(meta["adapter"], meta["dataset"], meta.get("config"), meta.get("split", "test"))


def episode_submission_budget(episode: dict[str, Any], env: Any) -> int:
    """The submission cap this episode actually ran under.

    Effort profiles stamp it per episode (``episode_max_submissions``); an unstamped trajectory falls
    back to the env's constructor value.
    """
    info = episode.get("info") or {}
    stamped = info.get("episode_max_submissions")
    return int(stamped) if isinstance(stamped, int) and stamped > 0 else env.max_submissions


def submitted_solutions(episode: dict[str, Any]) -> list[str | None]:
    """The ``submit_solution`` payloads of an episode, in submission order."""
    solutions: list[str | None] = []
    for message in episode.get("messages", []):
        for call in message.get("tool_calls") or []:
            if (call.get("function") or {}).get("name") == "submit_solution":
                try:
                    solutions.append(json.loads(call["function"]["arguments"]).get("code"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    solutions.append(None)
    return solutions


def regrade_file(path: str, workers: int) -> dict[str, Any]:
    """Re-grade one trajectory file and return its corrected metrics."""
    meta, episodes = read_trajectories(path)
    validate_meta(path, meta)
    payloads = build_payloads(meta)
    # Reproduce the environment the run used, not a default one: env_kwargs carries the interaction
    # budgets (max_submissions, reasoning_effort_profiles) that decide how many submissions count.
    # Without them a run that raised max_submissions re-grades against the class default and reports
    # a lower s@k than the online number.
    env_config = {
        **(meta.get("env_kwargs") or {}),
        "language": meta.get("language", "python"),
        "reasoning_effort": meta.get("reasoning_effort", "medium"),
    }
    env = resolve_environment(meta["env_type"], env_config)
    # The run's own grading contract off its meta line, minus the two knobs an offline re-grade must
    # not inherit. ``stop_on_first_failure``: s@1/s@2 need the all-pass verdict, not the pass
    # *fraction*, so short-circuiting is identical to full grading while sparing the remaining
    # (expensive) tests of an already-failed solution. ``max_grading_seconds`` exists to stop one
    # episode pinning a rollout worker; applying it here would score a correct-but-slow solution as
    # unsolved. An absent block (a run that stamped none) leaves the rebuilt env's own spec standing.
    spec = env.grading_spec.with_meta(
        meta.get("env_grading") or {}, stop_on_first_failure=True, max_grading_seconds=None
    )

    def grade(idx: int, code: str | None) -> bool:
        """True iff ``code`` passes all of problem ``idx``'s tests."""
        payload = payloads[idx] if 0 <= idx < len(payloads) else None
        if not code or not payload or not payload.get("tests"):
            return False
        passed, total, *_ = grade_solution(
            code,
            payload["tests"],
            spec,
            checker=payload.get("checker"),
            time_limit=payload.get("time_limit"),
        )
        return total > 0 and passed == total

    # One grading task per (episode, submission ≤ budget); bounded concurrency keeps wall-clock real.
    tasks = [
        (ep_i, code)
        for ep_i, episode in enumerate(episodes)
        for code in submitted_solutions(episode)[: episode_submission_budget(episode, env)]
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda t: grade(episodes[t[0]]["index"], t[1]), tasks))

    per_episode: dict[int, list[bool]] = {}
    for (ep_i, _), ok in zip(tasks, verdicts, strict=False):
        per_episode.setdefault(ep_i, []).append(ok)

    n = len(episodes)
    solved_first = sum(1 for v in per_episode.values() if v and v[0])
    solved_within = sum(1 for v in per_episode.values() if any(v))
    return {
        "model": meta.get("model"),
        "adapter": meta.get("adapter"),
        "language": meta.get("language"),
        "n": n,
        "s@1": solved_first / n if n else 0.0,
        "s@2": solved_within / n if n else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectories", nargs="+", help="Trajectory JSONL file(s) to re-grade.")
    p.add_argument(
        "--workers",
        type=int,
        default=min(64, (os.cpu_count() or 8)),
        help="Concurrent gradings in this single process. Keep at/below the core count so each run "
        "gets a dedicated core and the wall-clock limit stays real.",
    )
    p.add_argument("--output", help="Optional JSONL to append the corrected per-file metrics to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with contextlib.ExitStack() as stack:
        output = stack.enter_context(open(args.output, "a")) if args.output else None
        for path in args.trajectories:
            metrics = regrade_file(path, args.workers)
            # print (not logging — third-party imports reconfigure the root logger) and persist each
            # result as it lands, so progress is visible and partial runs are not lost.
            print(
                f"{metrics['model']:28s} {metrics['adapter']:13s} {metrics['language']:6s} "
                f"n={metrics['n']} s@1={metrics['s@1']:.0%} s@2={metrics['s@2']:.0%}",
                flush=True,
            )
            if output:
                output.write(json.dumps(metrics) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
