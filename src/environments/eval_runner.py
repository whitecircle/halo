"""Standalone evaluation rollout: drive any environment over a set of examples against an
OpenAI-compatible endpoint (vLLM or OpenRouter) and aggregate rewards.

Offline counterpart to :mod:`~src.environments.ray_actors`: same env ``reset``/``step`` loop, episodes
run concurrently via ``asyncio`` (``env.step`` offloaded to a worker thread because tool/submit
handlers block on sandboxed execution). Eval scripts build examples and call :func:`collect_results` +
:func:`report`.

An *example* is ``{"prompt": str | list, "context": dict, "group": Any}`` — ``context`` carries the
env's per-episode payload; ``group`` is an optional report bucketing key.
"""

import asyncio
import json
import logging
import os
import re
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from openai import NOT_GIVEN, AsyncOpenAI

from src.configs.rollout_config import RolloutConfig
from src.data.sources.paths import parse_dataset_source
from src.environments.base import BaseEnvironment, Trajectory
from src.environments.engine_wire import generation_control_fields
from src.environments.episode import (
    EpisodeDispatcher,
    TurnGeneration,
    bind_episode_effort,
    step_context_from_generation,
)
from src.inference.openai_client import generate_openai_response
from src.inference.response import FINISH_REASON_LENGTH

logger = logging.getLogger(__name__)

# Per-generation HTTP timeout (seconds). Generous by default: eval runs many episodes concurrently
# against one endpoint, and a long reasoning turn queued behind them takes minutes to come back.
DEFAULT_REQUEST_TIMEOUT_S = 180.0


def load_hf_split(dataset: str, config: str | None, split: str) -> Dataset:
    """Load one split from a local ``save_to_disk`` directory or the HuggingFace Hub.

    Classified by :func:`parse_dataset_source` — the toolkit's own convention — rather than by
    whether the path happens to exist on this host: an existence probe silently reads a stale local
    directory that shadows a Hub id, reads a mistyped local path as a Hub id whose failure names the
    network instead of the typo, and answers per rank on a non-shared filesystem.
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
    """Serialize a finished episode for persistence: full message list plus a curated ``info`` view.
    Drops answer-key and bookkeeping fields so the file neither bloats nor leaks the expected output:
    ``_``-prefixed keys (hidden tests, checker source), the raw ``tool_calls`` log, and ``context``."""
    if traj is None:
        return None
    _DROP = {"tool_calls", "context"}
    info = {k: v for k, v in traj.info.items() if not k.startswith("_") and k not in _DROP}
    info["eval_stats"] = traj.info.get("_eval_stats")
    return {
        "messages": [m.to_dict() for m in traj.messages],
        "total_reward": traj.total_reward,
        "done": traj.done,
        "truncated": traj.truncated,
        # The generation contract this episode ran under. Per-episode, not per-run: a ``random`` effort
        # setting draws per episode, so the meta line cannot say which level produced a given row.
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
    with one sample's messages and grading verdict, addressable by ``index`` and dataset ``id``.
    Episodes present only when ``collect_results`` ran with ``collect_trajectories=True``."""
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
    execution; offloading keeps the event loop free so episodes overlap. A failed generation ends the
    episode early with whatever reward it accrued. ``rollout.model_name`` unset means the endpoint
    serves exactly one model.
    """
    # The same seam the training rollout binds through, so the level steering generation here — and
    # the budget the level implies — are the ones the policy was trained under. The resolved draw is
    # stamped back into the reset context so per-episode effort-conditioned setup (interaction
    # budgets) sees the level generation is steered with.
    effort = bind_episode_effort(
        context, env, max_tokens=rollout.max_tokens, max_thinking_tokens=rollout.max_thinking_tokens
    )
    if effort.level is not None:
        context = {**(context or {}), "reasoning_effort": effort.level}
    # The per-episode contract, narrowed exactly as the training actor narrows it, so the engine
    # enforces the level's CoT budget here too instead of the trajectory merely recording it.
    episode_rollout = replace(rollout, max_tokens=effort.max_tokens, max_thinking_tokens=effort.thinking_budget)

    episode = EpisodeDispatcher(env)
    episode_ids, steps = await episode.reset([prompt], [context])
    eid, step = episode_ids[0], steps[0]
    # After reset, like the training actor: an MCP environment only learns its tools when the first
    # reset connects to the server, so a schema read before it would run the episode toolless.
    tools = env.get_tools_schema() or None

    try:
        finish_reasons: list[str | None] = []
        completion_tokens = 0

        for _ in range(env.max_turns):
            if step.done:
                break
            try:
                resp = await generate_openai_response(
                    model=rollout.model_name or NOT_GIVEN,
                    user_message=step.observation,
                    temperature=rollout.temperature,
                    max_tokens=episode_rollout.max_tokens,
                    top_p=rollout.top_p,
                    custom_client=client,
                    tools=tools,
                    request_timeout=rollout.request_timeout,
                    extra_body=generation_control_fields(episode_rollout, effort.level),
                )
            except Exception:
                # An unlogged break yields an all-zero eval indistinguishable from a bad URL/auth/model.
                logger.warning("generation failed, ending episode early", exc_info=True)
                break

            finish_reasons.append(resp.finish_reason)
            completion_tokens += resp.completion_tokens or 0
            # Through the same stamp the training rollout uses, so an eval reads a turn the engine cut
            # off at its token cap exactly as training does instead of grading the fragment.
            gen = TurnGeneration(
                text=resp.answer or "",
                tool_calls=serialize_tool_calls(resp.tool_calls),
                reasoning=resp.reasoning or "",
                tokens=resp.completion_tokens or 0,
                finish_reason=resp.finish_reason,
            )
            steps = await episode.step([eid], [gen.text], [step_context_from_generation(context, gen)])
            step = steps[0]

        traj = env.get_trajectories([eid])[0]
        # An episode can exit the loop still open (generation raised, or the turn cap hit). Explicit
        # truncation, never a synthetic empty turn: that would mark it ``completed`` and pay
        # completion-rewarded envs, whereas here reward already accrued still counts.
        if traj is not None and not traj.done:
            steps = await episode.finalize_truncated([eid])
            traj = steps[0].trajectory

        effort.stamp(traj)
        if traj is not None:
            traj.info["_eval_stats"] = {
                "generations": len(finish_reasons),
                "tool_calls": traj.info.get("total_tool_calls", 0),
                "completion_tokens": completion_tokens,
                "length_capped": any(fr == FINISH_REASON_LENGTH for fr in finish_reasons),
            }

        return traj
    finally:
        # On every path (mirrors EnvironmentActor.run_episode): an exception mid-episode would
        # otherwise leak the trajectory and its sandbox session.
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

    The generation contract is one :class:`RolloutConfig` — the object the training rollout hands its
    actors — reaching the request through the training payload's own
    :func:`~src.environments.engine_wire.generation_control_fields`, so an eval samples the policy the
    way training does rather than through a re-listing that quietly drops a knob.

    Each result is ``{"group", "id", "samples": [{"reward", "success", "stats"}, ...]}`` — a sample is a
    success when reward ≥ ``success_threshold``; ``collect_trajectories=True`` adds a ``"trajectory"``
    per sample for :func:`write_trajectories_jsonl`.
    """
    semaphore = asyncio.Semaphore(max_workers)

    async def one(example) -> dict[str, Any]:
        async def sample():
            # Isolated: an escaping exception would cancel every sibling in the gather.
            try:
                async with semaphore:
                    traj = await run_episode(env, example["prompt"], example["context"], client, rollout=rollout)
                reward = traj.total_reward if traj and traj.done else 0.0
                stats = (traj.info.get("_eval_stats") if traj else None) or {}
                rec: dict[str, Any] = {"reward": reward, "success": reward >= success_threshold, "stats": stats}
                if collect_trajectories:
                    rec["trajectory"] = serialize_trajectory(traj)
                return rec
            except Exception as e:  # one bad episode must not sink the batch
                logger.warning(f"Eval episode failed (id={example.get('id')}): {e}")
                return {"reward": 0.0, "success": False, "stats": {}, "error": str(e)}

        samples = await asyncio.gather(*[sample() for _ in range(num_samples)])
        return {"group": example.get("group"), "id": example.get("id"), "samples": samples}

    return await asyncio.gather(*[one(ex) for ex in examples])


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, float]:
    """Mean reward, ``success@1`` (first sample), and ``success@k`` (any sample) over ``rows``."""
    if not rows:
        # Fail loud with the actual cause; the bare StatisticsError from mean([]) hides it.
        raise ValueError("summarize() got no results — the eval produced zero episodes (empty dataset or all failed)")
    mean_reward = statistics.mean(statistics.mean(s["reward"] for s in r["samples"]) for r in rows)
    pass1 = statistics.mean(float(r["samples"][0]["success"]) for r in rows)
    out = {"n": len(rows), "mean_reward": mean_reward, "success@1": pass1}
    if num_samples > 1:
        out[f"success@{num_samples}"] = statistics.mean(float(any(s["success"] for s in r["samples"])) for r in rows)
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
    lines.append(line)

    stats = [s["stats"] for r in results for s in r["samples"] if s.get("stats")]
    if stats:
        episodes = len(stats)
        used_tools = sum(1 for st in stats if st.get("tool_calls", 0) > 0)
        capped = sum(1 for st in stats if st.get("length_capped"))
        lines.append(
            f"trajectory: mean_turns={statistics.mean(st.get('generations', 0) for st in stats):.2f}  "
            f"used_tools={100 * used_tools / episodes:.0f}%  "
            f"mean_tool_calls={statistics.mean(st.get('tool_calls', 0) for st in stats):.2f}  "
            f"mean_completion_tokens={int(statistics.mean(st.get('completion_tokens', 0) for st in stats))}  "
            f"length_capped={100 * capped / episodes:.0f}%"
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
