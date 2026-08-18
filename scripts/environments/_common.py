"""Shared CLI surface for the environment eval scripts: the flags every runner registers and the
output writer that consumes them.

The eval driver itself lives in :mod:`src.environments.eval_runner`; only the argparse-shaped half —
the flags and the ``Namespace``-reading output writer — lives here with the scripts that own it.
"""

import argparse
import json
import logging
import os
from typing import Any

from src.configs.rollout_config import DEFAULT_ROLLOUT_TOP_P, RolloutConfig
from src.environments.base import BaseEnvironment
from src.environments.eval_runner import DEFAULT_REQUEST_TIMEOUT_S, trajectory_path, write_trajectories_jsonl
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, resolve_local_api_key

logger = logging.getLogger(__name__)


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    """Register the dataset-source, endpoint, and output flags every eval script shares.

    Task-specific flags (env type, adapter, language, sampling budgets, concurrency) stay on the
    script's own parser — only the flags whose meaning and defaults are identical across the eval
    scripts live here.
    """
    parser.add_argument("--dataset", required=True, help="HF Hub id or local save_to_disk dir.")
    parser.add_argument("--config", default=None, help="Dataset config (e.g. 'all', 'verifiable', 'taco').")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--base_url", default=DEFAULT_LOCAL_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--api_key",
        default=resolve_local_api_key(),
        help="API key (default: $VLLM_API_KEY, else $OPENAI_API_KEY, else the placeholder a keyless local "
        "server accepts; OpenRouter requires a real key).",
    )
    parser.add_argument("--model", required=True, help="Served/model name.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_ROLLOUT_TOP_P,
        help="Nucleus-sampling cutoff. Defaults to the training rollout's own value, so an eval "
        "samples the policy the way training did rather than at whatever the server defaults to.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-generation HTTP timeout (s); raise it when running many concurrent episodes so long reasoning turns are not cut off.",
    )
    parser.add_argument("--output", default=None, help="Optional path to dump per-example JSON results.")
    parser.add_argument(
        "--save_trajectories",
        default=None,
        help="Optional explicit JSONL path to record full trajectories (system prompt + tool schemas in "
        "a meta line, then one episode per line with its messages, tool calls/results, and reward).",
    )
    parser.add_argument(
        "--trajectory_dir",
        default=None,
        help="Optional output folder for trajectories; the file is auto-named per run from the model "
        "and the dataset/split it ran on. Ignored when --save_trajectories is given.",
    )


def rollout_config_from_args(args: argparse.Namespace, *, temperature: float, max_tokens: int) -> RolloutConfig:
    """The generation contract an eval run samples under, from the shared endpoint flags.

    The same object the training rollout hands its actors, so the two paths cannot drift on a
    sampling knob. ``temperature`` and ``max_tokens`` are explicit for the same reason as in
    :func:`write_eval_outputs`: each script registers its own flag and default for them.
    """
    return RolloutConfig(
        model_name=args.model,
        temperature=temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        request_timeout=args.request_timeout,
    )


def resolve_trajectory_path(args: argparse.Namespace, *parts: str) -> str | None:
    """The JSONL file this run records trajectories to, or ``None`` when it records none.

    An explicit ``--save_trajectories`` wins; otherwise ``--trajectory_dir`` gets one auto-named file
    per run, built from ``parts`` — the knobs that identify this run within the folder's matrix.
    Both flags are registered by :func:`add_endpoint_args`, so which one wins is decided here rather
    than re-derived by each runner.
    """
    if args.save_trajectories is not None:
        return args.save_trajectories
    if args.trajectory_dir:
        return trajectory_path(args.trajectory_dir, args.model, *parts)
    return None


def write_eval_outputs(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    *,
    env: BaseEnvironment,
    traj_path: str | None,
    env_type: str,
    max_turns: int | None,
    max_tokens: int,
    temperature: float,
    num_samples: int,
    meta_extra: dict[str, Any] | None = None,
) -> None:
    """Write an eval run's outputs: the ``--output`` JSON dump and the trajectory JSONL.

    The trajectory meta line carries the run's dataset/endpoint/sampling knobs plus the env's system
    prompt and tool schemas; ``meta_extra`` adds the task-specific keys (adapter, language, env-level
    grading knobs, …) an offline re-grade needs to reproduce the same verdicts.

    Only the flags :func:`add_endpoint_args` registers are read off ``args``; everything else is an
    explicit parameter, so a script pairing the two helpers without those flags fails at the call
    instead of with an ``AttributeError`` after the whole eval has run.
    """
    if args.output:
        # The dump runs after every episode has been paid for, so a missing parent directory must
        # not discard the run; ``default=str`` matches the trajectory writer, which serializes the
        # same per-episode records.
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Wrote per-example results to %s", args.output)

    if not traj_path:
        return
    meta = {
        "model": args.model,
        "env_type": env_type,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        # The effective cap, not the flag: an omitted --max_turns leaves the env's own value, and a
        # null here would leave a trajectory with no record of the budget it actually ran under.
        "max_turns": max_turns if max_turns is not None else env.max_turns,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": args.top_p,
        "num_samples": num_samples,
        "system_prompt": env.system_prompt,
        "tools": env.get_tools_schema(),
        **(meta_extra or {}),
    }
    count = write_trajectories_jsonl(traj_path, meta, results)
    logger.info("Wrote %d trajectories to %s", count, traj_path)
