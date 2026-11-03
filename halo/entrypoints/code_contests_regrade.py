"""Offline re-grading of recorded code-contest trajectories: problems rebuilt from the dataset by index under the run's
selection, the environment from the meta line, every admitted ``submit_solution`` through ``grade_solution``."""

import contextlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

from halo.data.sources.loading import load_dataset_split
from halo.environments.eval_runner import GENERATION_ERROR_KEY
from halo.environments.registry import build_environment
from halo.environments.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    DEFAULT_REASONING_EFFORT,
    CodeContestsEnvironment,
)
from halo.environments.tasks.coding.code_contests_prompts import SUBMIT_TOOL
from halo.environments.tasks.coding.contest_datasets import CODE_DATASET_ADAPTERS, ContestSelection
from halo.environments.tasks.coding.grading import grade_solution
from halo.environments.tasks.coding.program_source import normalized_program
from halo.environments.tasks.coding.verdicts import FloatTolerance, find_stated_float_tolerance
from halo.environments.tool_accounting import EPISODE_TOOL_BUDGETS_KEY
from halo.environments.tools.definitions import NativeToolCall, ToolArgumentError
from halo.log import info_logger

# INFO: the report lines are the tool's output, printed under the WARNING root it leaves unconfigured.
logger = info_logger(__name__)

# env_type/adapter/dataset/split select the environment and rebuild the hidden tests, model/language name the report
# row. Only eval_code_contests.py stamps the full set; eval_env.py's generic meta has no adapter/language.
_REQUIRED_META_KEYS = ("env_type", "adapter", "dataset", "split", "model", "language")
# Recorded at the meta line's top level as the env resolved them; ``env_kwargs`` supplies every other option.
_TOP_LEVEL_ENV_KEYS = ("language", "reasoning_effort", "eval_protocol")


def validate_meta(path: str, meta: dict[str, Any]) -> None:
    """Fail before any grading when the meta line lacks a needed key: a missing report field would otherwise be a
    ``TypeError`` only after the whole file was graded."""
    missing = [key for key in _REQUIRED_META_KEYS if meta.get(key) is None]
    if missing:
        raise ValueError(
            f"{path}: trajectory meta is missing {missing}. Re-grading needs the code-contest meta "
            f"stamped by scripts/environments/inference/eval_code_contests.py; trajectories from "
            f"eval_env.py carry the generic eval meta (no adapter/language) and cannot be re-graded."
        )


def read_trajectories(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(meta, episodes)``; line 1 is the meta record."""
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
def _load_payloads(
    adapter_name: str, dataset: str, config: str | None, split: str, selection: ContestSelection
) -> tuple[dict[str, Any], ...]:
    """Grading payloads in the order ``build_examples`` produced, under the same selection, so an episode's ``index``
    selects its problem. Cached so a model × language matrix loads each dataset once."""
    adapter = CODE_DATASET_ADAPTERS[adapter_name]
    if not adapter.scores_raw_rows:
        raise SystemExit(
            f"adapter {adapter_name!r} scores prepared pools only; its raw rows carry no gradable payload"
        )
    rows = (
        adapter.load(dataset, config, split)
        if adapter.load
        else load_dataset_split(dataset, split=split, config=config)
    )
    payloads = tuple(adapter.pack_verification(row) for row in adapter.scored_rows(rows, selection))
    if not payloads:
        raise SystemExit(f"{dataset} ({adapter_name}) yielded no gradable problem")
    return payloads


def build_payloads(meta: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return _load_payloads(
        meta["adapter"],
        meta["dataset"],
        meta.get("config"),
        meta["split"],
        ContestSelection.from_meta(meta.get("selection")),
    )


def rebuild_environment(meta: dict[str, Any]) -> CodeContestsEnvironment:
    """The environment the run used: ``env_kwargs`` carries the budgets (``max_submissions``,
    ``reasoning_effort_profiles``) that decide how many submissions count, so a class default would report a lower s@k
    than the online number. A meta line naming no protocol ran the harness."""
    env_kwargs = {k: v for k, v in (meta.get("env_kwargs") or {}).items() if k not in _TOP_LEVEL_ENV_KEYS}
    return build_environment(
        meta["env_type"],
        {
            **env_kwargs,
            "language": meta["language"],
            "reasoning_effort": meta.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
            "eval_protocol": meta.get("eval_protocol") or DEFAULT_EVAL_PROTOCOL,
        },
    )


def episode_submission_budget(episode: dict[str, Any], env: Any) -> int:
    """The env stamps the cap per episode as the ``submit_solution`` tool budget; an unstamped trajectory falls back to
    the env's constructor value."""
    info = episode.get("info") or {}
    stamped = (info.get(EPISODE_TOOL_BUDGETS_KEY) or {}).get(SUBMIT_TOOL)
    return int(stamped) if isinstance(stamped, int) and stamped > 0 else env.max_submissions


def submitted_solutions(episode: dict[str, Any], env: CodeContestsEnvironment) -> list[tuple[str, str | None]]:
    """The ``submit_solution`` payloads ``env`` admitted, as ``(code, language)`` in order (``None`` where the run
    fixed the language), read the way the environment read them. A call the tool refuses to bind, code refused as
    another language (:meth:`CodeContestsEnvironment.find_mislabel`) or code identical comments aside to one already
    graded (:func:`normalized_program`) was never graded and spent no budget, so it takes no slot here."""
    tool = env.registry.get(SUBMIT_TOOL)
    solutions: list[tuple[str, str | None]] = []
    graded: set[str] = set()
    for message in episode.get("messages", []):
        for raw in message.get("tool_calls") or []:
            call = NativeToolCall.from_openai_format(raw)
            if call.name != SUBMIT_TOOL:
                continue
            try:
                bound = tool.bind(call.arguments)
            except ToolArgumentError:
                continue
            language = bound.get("language")
            program = normalized_program(bound["code"], language or env.language)
            if env.find_mislabel(bound["code"], language) is None and program not in graded:
                graded.add(program)
                solutions.append((bound["code"], language))
    return solutions


def episode_stated_tolerance(episode: dict[str, Any]) -> FloatTolerance | None:
    """The float tolerance an episode's own user turns state, as the environment read it from the prompt."""
    return find_stated_float_tolerance(
        "\n".join(
            str(message.get("content") or "")
            for message in episode.get("messages", [])
            if message.get("role") == "user"
        )
    )


def display_language(language: str | list[str]) -> str:
    return ",".join(language) if isinstance(language, list) else str(language)


def regrade_file(path: str, workers: int) -> dict[str, Any]:
    """Re-grade one file. An episode recorded with a ``generation_error`` stopped on the driver's fault, carries no
    verdict and leaves ``n``, as it leaves the eval's scores."""
    meta, recorded = read_trajectories(path)
    validate_meta(path, meta)
    episodes = [episode for episode in recorded if not episode.get(GENERATION_ERROR_KEY)]
    payloads = build_payloads(meta)
    # An index past the rebuilt problems means the dataset or its selection no longer matches the run's.
    stray = sorted({episode["index"] for episode in episodes if not 0 <= episode["index"] < len(payloads)})
    if stray:
        raise SystemExit(
            f"{path}: episode index(es) {stray} address none of the {len(payloads)} problems the rebuilt "
            f"dataset yields; it no longer matches the one the run was scored on"
        )
    env = rebuild_environment(meta)
    # The run's own grading contract minus two knobs. ``stop_on_first_failure``: s@1/s@2 need only the all-pass
    # verdict, so short-circuiting spares an already-failed solution's tests. ``max_grading_seconds`` only stops an
    # episode pinning a rollout worker; here it would score a correct-but-slow solution unsolved.
    spec = env.grading_spec.with_meta(
        meta.get("env_grading") or {}, stop_on_first_failure=True, max_grading_seconds=None
    )

    def grade(idx: int, code: str | None, language: str | None, stated_tolerance: FloatTolerance | None) -> bool:
        """True iff ``code`` passes all of problem ``idx``'s tests, in its language under its stated tolerance."""
        payload = payloads[idx]
        if not code or not payload.get("tests"):
            return False
        passed, total, *_ = grade_solution(
            code,
            payload["tests"],
            spec,
            checker=payload.get("checker"),
            time_limit=payload.get("time_limit"),
            stated_tolerance=stated_tolerance,
            language=language,
        )
        return total > 0 and passed == total

    # Bounded concurrency keeps wall-clock real.
    tasks = [
        (ep_i, code, language)
        for ep_i, episode in enumerate(episodes)
        for code, language in submitted_solutions(episode, env)[: episode_submission_budget(episode, env)]
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tolerances = [episode_stated_tolerance(episode) for episode in episodes]
        verdicts = list(pool.map(lambda t: grade(episodes[t[0]]["index"], t[1], t[2], tolerances[t[0]]), tasks))

    per_episode: dict[int, list[bool]] = {}
    for (ep_i, _, _), ok in zip(tasks, verdicts, strict=True):
        per_episode.setdefault(ep_i, []).append(ok)

    n = len(episodes)
    solved_first = sum(1 for v in per_episode.values() if v and v[0])
    solved_within = sum(1 for v in per_episode.values() if any(v))
    return {
        "model": meta.get("model"),
        "adapter": meta.get("adapter"),
        "language": meta.get("language"),
        "eval_protocol": env.eval_protocol,
        "n": n,
        "s@1": solved_first / n if n else 0.0,
        "s@2": solved_within / n if n else 0.0,
        "generation_errors": len(recorded) - n,
    }


def regrade_trajectory_files(paths: list[str], workers: int, output_path: str | None) -> None:
    """Print each file's corrected metrics and append them to ``output_path`` as each lands."""
    with contextlib.ExitStack() as stack:
        output = stack.enter_context(open(output_path, "a")) if output_path else None
        for path in paths:
            metrics = regrade_file(path, workers)
            # As each result lands, so progress is visible and partial runs are not lost.
            logger.info(
                f"{metrics['model']:28s} {metrics['adapter']:13s} {display_language(metrics['language']):6s} "
                f"{metrics['eval_protocol']:11s} n={metrics['n']} s@1={metrics['s@1']:.0%} s@2={metrics['s@2']:.0%} "
                f"generation_errors={metrics['generation_errors']}"
            )
            if output:
                output.write(json.dumps(metrics) + "\n")
                output.flush()
