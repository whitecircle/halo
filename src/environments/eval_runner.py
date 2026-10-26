"""Standalone evaluation: the training rollout's episode loop over ``{"prompt", "context", "group"}`` examples
against an OpenAI-compatible endpoint (vLLM or OpenRouter), the records it persists, and the report."""

import asyncio
import json
import logging
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from typing import Any

import httpx
from datasets import Dataset, load_dataset, load_from_disk
from openai import NOT_GIVEN, APIStatusError, APITimeoutError, AsyncOpenAI

from src.data.sources.paths import parse_dataset_source
from src.environments.engine_wire import (
    capture_request_fields,
    generation_control_fields,
    turn_from_chat_completion,
    with_cut_call_as_written,
)
from src.environments.environment_base import EnvironmentBase
from src.environments.episode import (
    TurnGeneration,
    describe_exception,
    generate_turn,
    is_context_overflow,
    is_engine_fault,
    play_episode,
)
from src.environments.episode_keys import ANSWER_KEY, CONTEXT_KEY, EPISODE_INVALID_REASON_KEY, solve_verdict
from src.environments.messages import Message, Trajectory, task_prompt
from src.environments.reasoning_effort import (
    EpisodeEffort,
    bind_episode_effort,
    efforts_by_level,
    stable_reasoning_effort,
)
from src.environments.rollout_config import RANDOM_REASONING_EFFORT, RolloutConfig
from src.environments.tool_accounting import TOTAL_TOOL_CALLS_KEY
from src.inference.openai_client import chat_completion
from src.inference.response import FINISH_REASON_LENGTH
from src.rewards.samples import CUT_TOOL_CALLS_KEY

logger = logging.getLogger(__name__)

# Message fields a persisted turn records beside its render (which must not carry them): the turn's reasoning cap,
# the reasoning it sampled, and the calls a cut turn never ran, apart from the ``tool_calls`` the re-grader replays.
_SERIALIZED_TURN_FIELDS = ("thinking_cap", "reasoning_tokens", CUT_TOOL_CALLS_KEY)
# The sample-record key of an episode the driver lost past every retry, which carries no verdict, and the private
# trajectory stamp it is read from.
GENERATION_ERROR_KEY = "generation_error"
_DRIVER_FAULT_KEY = "_driver_fault"
# The driver's per-episode telemetry, persisted as ``eval_stats``.
EVAL_STATS_KEY = "_eval_stats"


def load_hf_split(dataset: str, config: str | None, split: str) -> Dataset:
    """Load one split from a local ``save_to_disk`` directory or the HuggingFace Hub, the source classified by
    :func:`parse_dataset_source`: an existence probe would let a stale directory shadow a Hub id and read a
    mistyped path as one."""
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
    if split not in ds:
        raise ValueError(f"--split {split!r} is not in {path}, which holds {list(ds)}")
    return ds[split]


def _serialized_message(message: Message) -> dict[str, Any]:
    """A message's render plus the :data:`_SERIALIZED_TURN_FIELDS` it recorded."""
    record = message.to_dict()
    record.update({name: value for name in _SERIALIZED_TURN_FIELDS if (value := getattr(message, name)) is not None})
    return record


def serialize_trajectory(traj: Trajectory | None) -> dict[str, Any] | None:
    """A finished episode as persisted: every message, and ``info`` without the row payload and the private
    ``_``-prefixed grading stamps."""
    if traj is None:
        return None
    info = {k: v for k, v in traj.info.items() if not k.startswith("_") and k != CONTEXT_KEY}
    info["eval_stats"] = traj.info.get(EVAL_STATS_KEY)
    return {
        "messages": [_serialized_message(m) for m in traj.messages],
        "total_reward": traj.total_reward,
        "done": traj.done,
        "truncated": traj.truncated,
        # Per episode: a ``random`` setting draws per episode, so the meta line cannot say which level ran.
        "reasoning_effort": traj.reasoning_effort,
        "reasoning_budget": traj.reasoning_budget,
        "reasoning_budget_soft": traj.reasoning_budget_soft,
        "info": info,
    }


def trajectory_path(directory: str, *parts: str) -> str:
    """Build ``<directory>/<part>__<part>__….jsonl``, each part slugified to a filesystem-safe token,
    so a model × dataset × language matrix lands as distinct, self-describing files."""
    name = "__".join(re.sub(r"[^A-Za-z0-9._-]+", "-", str(p)).strip("-") or "x" for p in parts if p)
    return os.path.join(directory, f"{name}.jsonl")


def write_trajectories_jsonl(path: str, meta: dict[str, Any], results: list[dict[str, Any]]) -> int:
    """Write a run's trajectories to ``path`` as JSONL and return the episode count: a ``{"type": "meta"}`` line,
    then one ``{"type": "episode"}`` line per sample (``collect_trajectories=True``) with its messages and verdict,
    addressed by ``index`` and dataset ``id``, :data:`GENERATION_ERROR_KEY` naming a lost sample's failure."""
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


def reject_answerless_examples(env: EnvironmentBase, examples: list[dict[str, Any]], source: str) -> None:
    """The trainer's ``requires_answer`` dataset gate for the eval: refuse examples without the answer an
    answer-graded environment needs, before any episode runs; ``source`` names where it was read from."""
    if env.requires_answer and any(ANSWER_KEY not in example["context"] for example in examples):
        raise ValueError(
            f"{type(env).__name__} grades each episode against an expected answer (requires_answer), but "
            f"{source} carries none."
        )


def _is_terminal_for_eval(exc: BaseException) -> bool:
    """Give-up predicate of the eval's turn retries: everything but an engine fault, the one failure the OpenAI
    client's own transport retries cannot tell from a client error."""
    return not (isinstance(exc, APIStatusError) and is_engine_fault(exc.status_code, exc.message))


async def _detokenize(client: AsyncOpenAI, rollout: RolloutConfig, ids: list[int]) -> str:
    """vLLM's ``/detokenize`` of ``ids``, at the server root the client's ``/v1`` base sits under."""
    root = client.base_url.path.rstrip("/").removesuffix("/v1")
    reply = await client.post(
        str(client.base_url.copy_with(path=f"{root}/detokenize")),
        cast_to=httpx.Response,
        body={"model": rollout.model_name, "tokens": ids},
        options={"timeout": rollout.request_timeout},
    )
    return reply.json()["prompt"]


async def _chat_turn(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    rollout: RolloutConfig,
    effort: EpisodeEffort,
) -> TurnGeneration:
    """One generation request for the turn ``rollout`` caps, read back by the training rollout's own parser, a
    call cut at the cap decoded as the policy wrote it (:func:`with_cut_call_as_written`)."""
    request: dict[str, Any] = {
        "model": rollout.model_name or NOT_GIVEN,
        "messages": messages,
        "max_tokens": rollout.max_tokens,
        "timeout": rollout.request_timeout,
        "temperature": rollout.temperature,
        "top_p": rollout.top_p,
        "extra_body": {
            **capture_request_fields(rollout),
            **generation_control_fields(rollout, effort.level, effort.thinking_budget, effort.budget_soft),
        },
    }
    if tools is not None:
        request["tools"] = tools
    completion = await chat_completion(client, **request)
    gen = turn_from_chat_completion(completion.model_dump(), rollout)
    return await with_cut_call_as_written(gen, rollout, partial(_detokenize, client, rollout), "eval episode")


async def _request_turn(
    client: AsyncOpenAI,
    env: EnvironmentBase,
    messages: list[dict[str, Any]],
    rollout: RolloutConfig,
    effort: EpisodeEffort,
) -> TurnGeneration:
    """One turn under the training rollout's retry policy (:func:`generate_turn`). The tools are read per turn,
    after the reset: an MCP environment learns its tools only when the first reset connects to the server."""
    try:
        return await generate_turn(
            partial(_chat_turn, client, messages, env.get_tools_schema() or None, rollout, effort),
            rollout,
            retry_on=(APIStatusError,),
            giveup=_is_terminal_for_eval,
            log_prefix="eval episode",
        )
    except Exception:
        # Unlogged, an early end yields an all-zero eval indistinguishable from a bad endpoint.
        logger.warning("generation failed, ending episode early", exc_info=True)
        raise


def _bind_eval_effort(
    env: EnvironmentBase, prompt: str | list[dict[str, Any]], context: dict[str, Any], rollout: RolloutConfig
) -> tuple[dict[str, Any], EpisodeEffort]:
    """The episode's effort, bound as the training rollout binds it, and the reset context carrying its level; a
    ``random`` level is drawn from the problem, so every checkpoint scores a problem at the same level."""
    if env.reasoning_effort == RANDOM_REASONING_EFFORT and not (context or {}).get("reasoning_effort"):
        context = {**(context or {}), "reasoning_effort": stable_reasoning_effort(prompt)}
    effort = bind_episode_effort(
        context,
        env,
        max_tokens=rollout.max_tokens,
        max_thinking_tokens=rollout.max_thinking_tokens,
        max_episode_tokens=rollout.max_episode_tokens,
        max_answer_tokens=rollout.max_answer_tokens,
    )
    if effort.level is not None:
        context = {**(context or {}), "reasoning_effort": effort.level}
    return context, effort


async def run_episode(
    env: EnvironmentBase,
    prompt: str | list[dict[str, Any]],
    context: dict[str, Any],
    client: AsyncOpenAI,
    *,
    rollout: RolloutConfig,
) -> Trajectory | None:
    """Play one episode against ``client``; a generation lost past the retries closes it on what it earned,
    stamped :data:`_DRIVER_FAULT_KEY` unless the request itself caused it."""
    context, effort = _bind_eval_effort(env, prompt, context, rollout)
    played = await play_episode(
        env,
        prompt,
        context,
        effort,
        partial(_request_turn, client, env),
        rollout,
        log_prefix="eval episode",
        keep_lost_episode=True,
    )
    traj, lost = played.trajectory, played.lost_turn
    if traj is None:
        return None
    # A request the episode's own length broke (the served context, ``request_timeout`` on every client retry)
    # is graded on what it earned; any other failure is the driver's, and the sample carries no verdict.
    request_fault = isinstance(lost, APITimeoutError) or (
        isinstance(lost, APIStatusError) and is_context_overflow(lost.status_code, lost.message)
    )
    if lost is not None and not request_fault:
        # Stamped after the close: the environment drops the private info keys when it finalizes.
        traj.info[_DRIVER_FAULT_KEY] = describe_exception(lost)
    traj.info[EVAL_STATS_KEY] = {
        "generations": len(played.turns),
        "tool_calls": traj.info.get(TOTAL_TOOL_CALLS_KEY, 0),
        "completion_tokens": played.generated,
        "length_capped": any(turn.finish_reason == FINISH_REASON_LENGTH for turn in played.turns),
        "empty_turns": traj.info.get("empty_turns", 0),
    }
    return traj


def _solved(env: EnvironmentBase, traj: Trajectory | None, reward: float, success_threshold: float) -> bool:
    """Whether a sample solved its task: the environment's :func:`solve_verdict` where it reports one, else
    ``reward >= success_threshold``, since a shaped total mixes in prices a solve does not depend on."""
    verdict = solve_verdict(env.rollout_metrics(traj)) if traj is not None else None
    return reward >= success_threshold if verdict is None else verdict


def _sample_record(env: EnvironmentBase, traj: Trajectory | None, success_threshold: float) -> dict[str, Any]:
    """One sample's verdict: its reward and success (:func:`_solved`), none where the driver lost the episode,
    and a zero with the cause where the grade carries no signal; the driver's stats beside them."""
    reward = traj.total_reward if traj and traj.done else 0.0
    stats = (traj.info.get(EVAL_STATS_KEY) if traj else None) or {}
    if traj is not None and _DRIVER_FAULT_KEY in traj.info:
        # Stopped on the driver's fault: no verdict, so the sample leaves the scores, counted apart.
        return {"reward": None, "success": None, "stats": stats, GENERATION_ERROR_KEY: traj.info[_DRIVER_FAULT_KEY]}
    if traj is not None and traj.episode_invalid:
        # A grade with no signal (a grader outage) scores zero, `error` telling it from a genuine miss.
        reason = str(traj.info.get(EPISODE_INVALID_REASON_KEY, "episode invalid"))
        return {"reward": 0.0, "success": False, "stats": stats, "error": reason}
    return {"reward": reward, "success": _solved(env, traj, reward, success_threshold), "stats": stats}


async def collect_results(
    env: EnvironmentBase,
    examples: list[dict[str, Any]],
    client: AsyncOpenAI,
    *,
    rollout: RolloutConfig,
    num_samples: int = 1,
    success_threshold: float = 1.0,
    max_workers: int = 32,
    collect_trajectories: bool = False,
) -> list[dict[str, Any]]:
    """Run every ``example × num_samples`` episode under a concurrency cap; return one ``{"group", "id",
    "samples"}`` result per example, each sample carrying its ``"trajectory"`` under ``collect_trajectories``. The
    trainer's turn-cap gate (:func:`efforts_by_level`) runs first, so a bad level refuses the run up front."""
    efforts_by_level(
        env,
        max_tokens=rollout.max_tokens,
        max_thinking_tokens=rollout.max_thinking_tokens,
        max_episode_tokens=rollout.max_episode_tokens,
        max_answer_tokens=rollout.max_answer_tokens,
    )
    tasks = [task_prompt(example["prompt"]) for example in examples]
    semaphore = asyncio.Semaphore(max_workers)

    async def sample(example: dict[str, Any], task: Any) -> dict[str, Any]:
        # Isolated: an escaping exception would cancel the sibling tasks in the gather.
        try:
            async with semaphore:
                traj = await run_episode(env, task, example["context"], client, rollout=rollout)
            rec = _sample_record(env, traj, success_threshold)
            if collect_trajectories:
                rec["trajectory"] = serialize_trajectory(traj)
            return rec
        except Exception as e:  # one bad episode must not sink the batch
            reason = describe_exception(e)
            logger.warning(f"Eval episode failed (id={example.get('id')}): {reason}")
            return {"reward": 0.0, "success": False, "stats": {}, "error": reason}

    async def one(example: dict[str, Any], task: Any) -> dict[str, Any]:
        samples = await asyncio.gather(*[sample(example, task) for _ in range(num_samples)])
        return {"group": example.get("group"), "id": example.get("id"), "samples": samples}

    return await asyncio.gather(*[one(example, task) for example, task in zip(examples, tasks, strict=True)])


def _mean_or_nan(values: Iterable[float]) -> float:
    """The mean, or NaN when a bucket has nothing left to average (every sample a generation error)."""
    values = list(values)
    return statistics.mean(values) if values else math.nan


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, float]:
    """Mean reward, ``success@1`` (first scored sample), ``success@k`` (any scored sample), and the ``invalid`` and
    ``generation_errors`` counts over ``rows``. A generation-error sample leaves every score, each read over the
    rows with a scored sample left, so ``success@1`` never exceeds ``success@k``; with none left a score is NaN."""
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
    lines = [f"{title} ({len(results)} examples × {k} samples)"]

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
