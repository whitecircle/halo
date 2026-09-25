"""Standalone evaluation rollout: drive any environment over a set of examples against an
OpenAI-compatible endpoint (vLLM or OpenRouter) and aggregate rewards.

Offline counterpart to :mod:`~src.environments.ray_actors`: same env ``reset``/``step`` loop, episodes
run concurrently via ``asyncio`` (``env.step`` offloaded to a worker thread because tool/submit
handlers block on sandboxed execution). Eval scripts build examples and call :func:`collect_results` +
:func:`report`.

An *example* is ``{"prompt": str | list, "context": dict, "group": Any}``: ``context`` carries the
env's per-episode payload, ``group`` is an optional report bucketing key.
"""

import asyncio
import json
import logging
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from functools import partial
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from openai import NOT_GIVEN, APIStatusError, APITimeoutError, AsyncOpenAI

from src.configs.rollout_config import RolloutConfig
from src.data.sources.paths import parse_dataset_source
from src.environments.base import (
    EPISODE_ERROR_KEY,
    EPISODE_INVALID_REASON_KEY,
    BaseEnvironment,
    Trajectory,
    solve_verdict,
)
from src.environments.engine_wire import generation_control_fields
from src.environments.episode import (
    EpisodeDispatcher,
    EpisodeEffort,
    TurnGeneration,
    bind_episode_effort,
    describe_exception,
    generate_turn,
    is_context_overflow,
    is_engine_fault,
    step_context_from_generation,
    validate_thinking_budget_scope,
)
from src.inference.openai_client import generate_openai_response
from src.inference.response import FINISH_REASON_LENGTH, get_finish_reason

logger = logging.getLogger(__name__)

# Per-generation HTTP timeout (seconds). Generous by default: eval runs many episodes concurrently
# against one endpoint, and a long reasoning turn queued behind them takes minutes to come back.
DEFAULT_REQUEST_TIMEOUT_S = 180.0
# ``info`` keys a persisted trajectory leaves out: the row payload and the raw tool-call log, which the
# messages already carry; ``_``-prefixed grading stamps (hidden tests, checker source) go with them.
_SERIALIZED_INFO_DROP = frozenset({"tool_calls", "context"})
# The sample-record key of an episode that lost a generation on the driver's side past every retry: it
# carries no verdict. Stamped on the trajectory under the private key, which the serializer drops.
GENERATION_ERROR_KEY = "generation_error"
_DRIVER_FAULT_KEY = "_driver_fault"


def load_hf_split(dataset: str, config: str | None, split: str) -> Dataset:
    """Load one split from a local ``save_to_disk`` directory or the HuggingFace Hub.

    The source is classified by :func:`parse_dataset_source` rather than by whether the path exists on
    this host: an existence probe would read a stale local directory that shadows a Hub id, treat a
    mistyped local path as a Hub id (reporting a network failure instead of the typo), and answer
    differently per rank on a non-shared filesystem.
    """
    source_type, _bucket, path = parse_dataset_source(dataset)
    if source_type == "s3":
        raise ValueError(
            f"'{dataset}' is an S3 URI; the eval runner reads a local save_to_disk directory or a "
            f"Hub dataset id. Download it first, or point --dataset at 'org/name'."
        )
    if source_type == "hf_hub":
        return load_dataset(path, config, split=split)

    ds = load_from_disk(path)
    # A bare saved Dataset holds one unnamed split; membership would iterate rows, then fail.
    if isinstance(ds, Dataset):
        return ds
    return ds[split] if split in ds else ds[next(iter(ds.keys()))]


def serialize_trajectory(traj: Trajectory | None) -> dict[str, Any] | None:
    """Serialize a finished episode for persistence: the full message list plus ``info`` without the
    ``_``-prefixed grading stamps and :data:`_SERIALIZED_INFO_DROP`."""
    if traj is None:
        return None
    info = {k: v for k, v in traj.info.items() if not k.startswith("_") and k not in _SERIALIZED_INFO_DROP}
    info["eval_stats"] = traj.info.get("_eval_stats")
    return {
        "messages": [m.to_dict() for m in traj.messages],
        "total_reward": traj.total_reward,
        "done": traj.done,
        "truncated": traj.truncated,
        # The generation contract this episode ran under. Recorded per episode because a ``random``
        # effort setting draws per episode, so the meta line cannot say which level produced a row.
        "reasoning_effort": traj.reasoning_effort,
        "reasoning_budget": traj.reasoning_budget,
        "info": info,
    }


def trajectory_path(directory: str, *parts: str) -> str:
    """Build ``<directory>/<part>__<part>__….jsonl``, each part slugified to a filesystem-safe token,
    so a model × dataset × language matrix lands as distinct, self-describing files."""
    name = "__".join(re.sub(r"[^A-Za-z0-9._-]+", "-", str(p)).strip("-") or "x" for p in parts if p)
    return os.path.join(directory, f"{name}.jsonl")


def write_trajectories_jsonl(path: str, meta: dict[str, Any], results: list[dict[str, Any]]) -> int:
    """Write a run's trajectories to ``path`` as JSONL, returning the episode count.

    Line 1 is a ``{"type": "meta", ...}`` record; each following line is a ``{"type": "episode", ...}``
    with one sample's messages and grading verdict, addressable by ``index`` and dataset ``id``, and
    :data:`GENERATION_ERROR_KEY` naming the failure of a sample that carries no verdict (``None`` on a
    scored one). Episodes present only when ``collect_results`` ran with ``collect_trajectories=True``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "meta", **meta}, default=str) + "\n")
        for index, r in enumerate(results):
            for i, s in enumerate(r.get("samples", [])):
                rec: dict[str, Any] = {
                    "type": "episode",
                    "index": index,
                    "id": r.get("id"),
                    "group": r.get("group"),
                    "sample_index": i,
                    "reward": s.get("reward"),
                    "success": s.get("success"),
                    GENERATION_ERROR_KEY: s.get(GENERATION_ERROR_KEY),
                    "stats": s.get("stats"),
                }
                traj = s.get("trajectory")
                if traj is not None:
                    rec.update(traj)
                fh.write(json.dumps(rec, default=str) + "\n")
                count += 1
    return count


def serialize_tool_calls(tool_calls) -> list[dict[str, Any]]:
    """Convert OpenAI SDK tool-call objects into the dicts the environment's tool parser expects."""
    out = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        out.append(
            {
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", "") or "",
                },
            }
        )
    return out


def require_answers(env: BaseEnvironment, examples: list[dict[str, Any]], source: str) -> None:
    """Refuse examples an answer-graded environment cannot grade, before any episode runs: the
    trainer's dataset gate (``requires_answer``) for the eval drivers, which would otherwise generate
    every episode in full and then fail to grade it. ``source`` names where the answer was read from."""
    if env.requires_answer and any("answer" not in example["context"] for example in examples):
        raise ValueError(
            f"{type(env).__name__} grades each episode against an expected answer (requires_answer), but "
            f"{source} carries none."
        )


def _is_terminal_for_eval(exc: BaseException) -> bool:
    """Give-up predicate of the eval's turn retries: everything but an engine fault is terminal here.

    The OpenAI client has already retried the transport (connection errors, 408/409/429, 5xx) under its
    own ``max_retries``; the engine fault it cannot tell from a client error is the one left to retry.
    """
    return not (isinstance(exc, APIStatusError) and is_engine_fault(exc.status_code, exc.message))


def _is_request_fault(exc: BaseException) -> bool:
    """Whether a generation failed on the request the episode itself built: the conversation outgrew the
    served context (:func:`is_context_overflow`), or a turn outran ``request_timeout`` on every client
    retry. Both follow the episode's own length, so the sample is graded on what it earned. Any other
    failure is the driver's, a rejected key, an unknown model or route and a malformed request among
    them, and the sample carries no verdict."""
    if isinstance(exc, APITimeoutError):
        return True
    return isinstance(exc, APIStatusError) and is_context_overflow(exc.status_code, exc.message)


async def _request_turn(
    client: AsyncOpenAI,
    messages: str | list[dict[str, Any]],
    tools: list[dict] | None,
    rollout: RolloutConfig,
    effort: EpisodeEffort,
) -> TurnGeneration:
    """One generation request for the turn ``rollout`` caps, as the turn the environment steps with."""
    resp = await generate_openai_response(
        model=rollout.model_name or NOT_GIVEN,
        user_message=messages,
        temperature=rollout.temperature,
        max_tokens=rollout.max_tokens,
        top_p=rollout.top_p,
        custom_client=client,
        tools=tools,
        request_timeout=rollout.request_timeout,
        extra_body=generation_control_fields(rollout, effort.level, effort.thinking_budget),
    )
    # The same stamp the training rollout uses, so an eval treats a turn cut off at the token cap the
    # way training does instead of grading the fragment.
    return TurnGeneration(
        text=resp.answer or "",
        tool_calls=serialize_tool_calls(resp.tool_calls),
        reasoning=resp.reasoning or "",
        tokens=resp.completion_tokens or 0,
        finish_reason=get_finish_reason(resp, completion_tokens=resp.completion_tokens, max_tokens=rollout.max_tokens),
        token_ids=resp.token_ids,
    )


async def run_episode(
    env: BaseEnvironment,
    prompt: str | list[dict[str, Any]],
    context: dict[str, Any],
    client: AsyncOpenAI,
    *,
    rollout: RolloutConfig,
) -> Trajectory | None:
    """Drive one episode: ``env.reset`` → (generate → ``env.step``)* until done; return the trajectory.

    ``env.reset``/``env.step`` run in a worker thread because submit/tool handlers block on sandboxed
    execution; offloading keeps the event loop free so episodes overlap. A turn runs under the training
    rollout's retry policy (:func:`generate_turn`); a generation that still fails ends the episode
    early, stamped :data:`EPISODE_ERROR_KEY`, and also :data:`_DRIVER_FAULT_KEY` unless the request
    itself caused it (:func:`_is_request_fault`). ``rollout.model_name`` unset means the endpoint
    serves exactly one model.
    """
    # Bound through the same helper as the training rollout, so the level and its implied budget match
    # what the policy was trained under. The resolved draw is stamped back into the reset context so
    # per-episode effort-conditioned setup (interaction budgets) sees the level being used.
    effort = bind_episode_effort(
        context,
        env,
        max_tokens=rollout.max_tokens,
        max_thinking_tokens=rollout.max_thinking_tokens,
        scope=rollout.thinking_budget_scope,
        turn_reserve=rollout.thinking_turn_reserve,
    )
    if effort.level is not None:
        context = {**(context or {}), "reasoning_effort": effort.level}
    # The per-episode contract, narrowed as the training actor narrows it, so the engine enforces the
    # level's CoT budget here too rather than the trajectory only recording it.
    episode_rollout = replace(rollout, max_tokens=effort.max_tokens)
    reasoning_spent = 0

    episode = EpisodeDispatcher(env)
    episode_ids, steps = await episode.reset([prompt], [context])
    eid, step = episode_ids[0], steps[0]
    # Read after reset, as in the training actor: an MCP environment learns its tools only when the
    # first reset connects to the server, so an earlier read would run the episode without tools.
    tools = env.get_tools_schema() or None

    try:
        finish_reasons: list[str | None] = []
        completion_tokens = 0
        generation_error: str | None = None
        driver_fault = False

        for _ in range(env.max_turns):
            if step.done:
                break
            # The engine cap this turn: the level's budget, or under the episode scope what it has left.
            turn_rollout = replace(episode_rollout, max_thinking_tokens=effort.turn_thinking_cap(reasoning_spent))
            try:
                gen = await generate_turn(
                    partial(_request_turn, client, step.observation, tools, turn_rollout, effort),
                    rollout,
                    retry_on=(APIStatusError,),
                    giveup=_is_terminal_for_eval,
                    log_prefix="eval episode",
                )
            except Exception as exc:
                # Unlogged, this break yields an all-zero eval indistinguishable from a bad endpoint.
                logger.warning("generation failed, ending episode early", exc_info=True)
                generation_error = describe_exception(exc)
                driver_fault = not _is_request_fault(exc)
                break

            finish_reasons.append(gen.finish_reason)
            completion_tokens += gen.tokens
            reasoning_spent += effort.spend_of(gen, rollout.reasoning_end_token_id)
            steps = await episode.step([eid], [gen.text], [step_context_from_generation(context, gen)])
            step = steps[0]

        traj = env.get_trajectories([eid])[0]
        # An episode can exit the loop still open (generation raised, or the turn cap hit). Closed by
        # explicit truncation rather than a synthetic empty turn, which would mark it ``completed``
        # and pay completion-rewarded envs; reward already accrued still counts. A lost generation is
        # stamped first, so the env prices the truncation as the driver's fault, not a turn overflow.
        if traj is not None and not traj.done:
            if generation_error is not None:
                traj.info[EPISODE_ERROR_KEY] = generation_error
            steps = await episode.finalize_truncated([eid])
            traj = steps[0].trajectory
            # After the close: the environment drops the private info keys when it finalizes.
            if driver_fault:
                traj.info[_DRIVER_FAULT_KEY] = generation_error

        effort.stamp(traj, reasoning_spent)
        if traj is not None:
            traj.info["_eval_stats"] = {
                "generations": len(finish_reasons),
                "tool_calls": traj.info.get("total_tool_calls", 0),
                "completion_tokens": completion_tokens,
                "length_capped": any(fr == FINISH_REASON_LENGTH for fr in finish_reasons),
                "empty_turns": traj.info.get("empty_turns", 0),
            }

        return traj
    finally:
        # On every path: an exception mid-episode would otherwise leak the trajectory and its sandbox.
        env.cleanup([eid])


async def collect_results(
    env: BaseEnvironment,
    examples: list[dict[str, Any]],
    client: AsyncOpenAI,
    *,
    rollout: RolloutConfig,
    num_samples: int = 1,
    success_threshold: float = 1.0,
    max_workers: int = 32,
    collect_trajectories: bool = False,
) -> list[dict[str, Any]]:
    """Run every ``example × num_samples`` episode under a concurrency cap; return per-example results.

    The generation contract is a single :class:`RolloutConfig` (the object the training rollout hands
    its actors), reaching the request through the training payload's own
    :func:`~src.environments.engine_wire.generation_control_fields`, so an eval samples the policy the
    way training does.

    Each result is ``{"group", "id", "samples": [{"reward", "success", "stats"}, ...]}``. A sample is a
    success on the environment's own solve verdict where it reports one (:func:`_solved`), else when
    reward ≥ ``success_threshold``. A sample whose generation failed on the driver's side past every
    retry carries :data:`GENERATION_ERROR_KEY` and no verdict (``reward`` and ``success`` are ``None``).
    ``collect_trajectories=True`` adds a ``"trajectory"`` per sample for :func:`write_trajectories_jsonl`.

    The contract passes the trainer's thinking-scope gate first, so a gap refuses the run rather than
    scoring every episode that lands on it as an error sample.
    """
    validate_thinking_budget_scope(
        env,
        scope=rollout.thinking_budget_scope,
        max_thinking_tokens=rollout.max_thinking_tokens,
        turn_reserve=rollout.thinking_turn_reserve,
    )
    semaphore = asyncio.Semaphore(max_workers)

    async def one(example) -> dict[str, Any]:
        async def sample():
            # Isolated: an escaping exception would cancel the sibling tasks in the gather.
            try:
                async with semaphore:
                    traj = await run_episode(env, example["prompt"], example["context"], client, rollout=rollout)
                reward = traj.total_reward if traj and traj.done else 0.0
                stats = (traj.info.get("_eval_stats") if traj else None) or {}
                rec: dict[str, Any]
                if traj is not None and _DRIVER_FAULT_KEY in traj.info:
                    # The episode stopped on the driver's fault, not the policy's: what it earned before
                    # is no verdict, so the sample leaves the scores, counted apart.
                    rec = {"reward": None, "success": None, "stats": stats}
                    rec[GENERATION_ERROR_KEY] = traj.info[_DRIVER_FAULT_KEY]
                elif traj is not None and traj.episode_invalid:
                    # A grade with no signal (a grader outage, a dead judge) is an error row, not a
                    # score: the training baseline drops it. The eval keeps the row and scores it
                    # zero, so `error` is what separates a failed grade from a genuine miss.
                    reason = str(traj.info.get(EPISODE_INVALID_REASON_KEY, "episode invalid"))
                    rec = {"reward": 0.0, "success": False, "stats": stats, "error": reason}
                else:
                    rec = {"reward": reward, "success": _solved(env, traj, reward, success_threshold), "stats": stats}
                if collect_trajectories:
                    rec["trajectory"] = serialize_trajectory(traj)
                return rec
            except Exception as e:  # one bad episode must not sink the batch
                reason = describe_exception(e)
                logger.warning(f"Eval episode failed (id={example.get('id')}): {reason}")
                return {"reward": 0.0, "success": False, "stats": {}, "error": reason}

        samples = await asyncio.gather(*[sample() for _ in range(num_samples)])
        return {"group": example.get("group"), "id": example.get("id"), "samples": samples}

    return await asyncio.gather(*[one(ex) for ex in examples])


def _solved(env: BaseEnvironment, traj: Trajectory | None, reward: float, success_threshold: float) -> bool:
    """Whether a sample solved its task: the environment's solve verdict (:func:`solve_verdict`, the flag
    training's ``outcome/solve_rate`` averages) where it reports one, else ``reward >= success_threshold``.

    A shaped total mixes the objective with prices a solve does not depend on — a tool-error or
    length-cutoff penalty sinks a solved episode below the threshold, a submission bonus lifts a partial
    one over it — so it decides only where the environment has no verdict of its own.
    """
    verdict = solve_verdict(env.rollout_metrics(traj)) if traj is not None else None
    return reward >= success_threshold if verdict is None else verdict


def _mean_or_nan(values: Iterable[float]) -> float:
    """The mean, or NaN when a bucket has nothing left to average (every sample a generation error)."""
    values = list(values)
    return statistics.mean(values) if values else math.nan


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, float]:
    """Mean reward, ``success@1`` (first scored sample), ``success@k`` (any scored sample), and the
    ``invalid`` and ``generation_errors`` sample counts over ``rows``.

    A generation-error sample carries no verdict and leaves every score; each score reads the rows with
    a scored sample left, one row set for all, so ``success@1`` never exceeds ``success@k``. A score
    with no sample left reads NaN.
    """
    if not rows:
        # Raise with the actual cause; the bare StatisticsError from mean([]) does not name it.
        raise ValueError("summarize() got no results — the eval produced zero episodes (empty dataset or all failed)")
    scored = [[s for s in r["samples"] if GENERATION_ERROR_KEY not in s] for r in rows]
    graded = [samples for samples in scored if samples]
    out = {
        "n": len(rows),
        "mean_reward": _mean_or_nan(statistics.mean(s["reward"] for s in samples) for samples in graded),
        "success@1": _mean_or_nan(float(samples[0]["success"]) for samples in graded),
        # Samples scored 0 with no signal (an invalid grade, a run that raised): kept in the means, counted apart.
        "invalid": sum(1 for r in rows for s in r["samples"] if "error" in s),
        "generation_errors": sum(len(r["samples"]) - len(samples) for r, samples in zip(rows, scored, strict=True)),
    }
    if num_samples > 1:
        out[f"success@{num_samples}"] = _mean_or_nan(float(any(s["success"] for s in samples)) for samples in graded)
    return out


def report(results: list[dict[str, Any]], *, num_samples: int, title: str, group_label: str | None = None) -> None:
    """Log overall mean-reward / success@k, per-episode trajectory telemetry, and an optional
    per-``group`` breakdown. The whole report is emitted as one ``logger.info`` record (no ``print``)."""
    k = num_samples
    lines = [f"=== {title} ({len(results)} examples × {k} samples) ==="]

    overall = summarize(results, k)
    line = f"overall: n={overall['n']}  mean_reward={overall['mean_reward']:.3f}  success@1={overall['success@1']:.3f}"
    if k > 1:
        line += f"  success@{k}={overall[f'success@{k}']:.3f}"
    lines.append(f"{line}  invalid={overall['invalid']}  generation_errors={overall['generation_errors']}")

    stats = [s["stats"] for r in results for s in r["samples"] if s.get("stats")]
    if stats:
        episodes = len(stats)
        used_tools = sum(1 for st in stats if st.get("tool_calls", 0) > 0)
        capped = sum(1 for st in stats if st.get("length_capped"))
        empty = sum(1 for st in stats if st.get("empty_turns"))
        lines.append(
            f"trajectory: mean_turns={statistics.mean(st.get('generations', 0) for st in stats):.2f}  "
            f"used_tools={100 * used_tools / episodes:.0f}%  "
            f"mean_tool_calls={statistics.mean(st.get('tool_calls', 0) for st in stats):.2f}  "
            f"mean_completion_tokens={int(statistics.mean(st.get('completion_tokens', 0) for st in stats))}  "
            f"length_capped={100 * capped / episodes:.0f}%  "
            f"empty_turns={100 * empty / episodes:.0f}%"
        )

    if group_label and any(r.get("group") is not None for r in results):
        buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            buckets[r.get("group")].append(r)
        lines.append(f"\nby {group_label}:")
        for key in sorted(buckets, key=lambda x: (x is None, x)):
            s = summarize(buckets[key], k)
            extra = f"  success@{k}={s[f'success@{k}']:.3f}" if k > 1 else ""
            lines.append(
                f"  {str(key):>12}: n={s['n']:4d}  mean_reward={s['mean_reward']:.3f}  "
                f"success@1={s['success@1']:.3f}{extra}"
            )
    logger.info("\n%s", "\n".join(lines))
