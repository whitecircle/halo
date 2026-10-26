"""The CLI surface the environment eval scripts share: the flags every runner registers, the ``Namespace``
readers that turn them into a generation contract, and the output writer. The eval driver is
:mod:`halo.environments.eval_runner`; the training YAML's contract is :mod:`scripts.environments.training_contract`.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, replace
from typing import Any

from halo.environments.environment_base import EnvironmentBase
from halo.environments.eval_runner import trajectory_path, write_trajectories_jsonl
from halo.environments.rollout_config import RolloutConfig
from scripts.cli_arguments import add_openai_endpoint_args
from scripts.environments.training_contract import TrainingContract

logger = logging.getLogger(__name__)

# The sampling flags an explicit CLI value lays over the training config, by ``RolloutConfig`` field.
_SAMPLING_FLAGS = ("temperature", "top_p", "max_tokens", "request_timeout")
# The split an eval reads when --split is omitted, unless its dataset ships a single split of its own.
DEFAULT_SPLIT = "test"
# Per-generation HTTP timeout (seconds) without --training_config. Generous: an eval runs many episodes
# concurrently against one endpoint, and a long reasoning turn queued behind them takes minutes to return.
EVAL_REQUEST_TIMEOUT_SECONDS = 180.0


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    """Register the dataset-source, endpoint and output flags every eval script shares; ``--split`` and the
    task flags stay the runner's own. The sampling flags default to ``None``, so :func:`rollout_config_from_args`
    tells an explicit value from an omitted one: explicit flag > ``--training_config`` > the script's default."""
    parser.add_argument("--dataset", required=True, help="HF Hub id or local save_to_disk dir.")
    parser.add_argument("--config", default=None, help="Dataset config (e.g. 'all', 'verifiable', 'taco').")
    add_openai_endpoint_args(parser, model_help="Served/model name.")
    parser.add_argument(
        "--training_config",
        default=None,
        help="Environmental-GRPO training YAML to evaluate under: its rollout contract (chat-template "
        "variables, stop tokens, thinking budget, episode output budget, backend, sampling) and its environment config become "
        "the eval's, with any sampling flag passed explicitly laid over them. Without it the eval samples "
        "under the flags alone.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help=f"Nucleus-sampling cutoff. Default: the training config's rollout_top_p under --training_config, "
        f"else the training rollout's own default ({RolloutConfig.top_p}), so an eval samples the policy the "
        f"way training did rather than at whatever the server defaults to.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=None,
        help=f"Per-generation HTTP timeout (s). Default: the training config's request_timeout under "
        f"--training_config, else {EVAL_REQUEST_TIMEOUT_SECONDS:.0f}; raise it when running many concurrent "
        f"episodes so long reasoning turns are not cut off.",
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


def resolve_setting(explicit: Any, trained: Any, default: Any) -> Any:
    """Explicit CLI flag > the training config's value > the script's default (``None`` = unset)."""
    if explicit is not None:
        return explicit
    return default if trained is None else trained


def rollout_config_from_args(
    args: argparse.Namespace,
    contract: TrainingContract | None,
    env: EnvironmentBase,
    *,
    default_temperature: float,
    default_max_tokens: int,
) -> RolloutConfig:
    """The generation contract an eval of ``env`` samples under: the training run's own :class:`RolloutConfig`
    under ``--training_config``, else one built from the script's defaults, with the served model name and every
    explicitly passed sampling flag laid over it."""
    if contract is not None:
        base = contract.rollout_config(env)
    else:
        base = RolloutConfig(
            temperature=default_temperature,
            top_p=RolloutConfig.top_p,
            max_tokens=default_max_tokens,
            request_timeout=EVAL_REQUEST_TIMEOUT_SECONDS,
        )
    explicit = {name: getattr(args, name) for name in _SAMPLING_FLAGS if getattr(args, name) is not None}
    return replace(base, model_name=args.model, **explicit)


def resolve_trajectory_path(args: argparse.Namespace, *parts: str) -> str | None:
    """The JSONL file this run records trajectories to, or ``None``: an explicit ``--save_trajectories``, else one
    file under ``--trajectory_dir`` named from the model and ``parts``, the knobs identifying the run."""
    if args.save_trajectories is not None:
        return args.save_trajectories
    if args.trajectory_dir:
        return trajectory_path(args.trajectory_dir, args.model, *parts)
    return None


def write_eval_outputs(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    *,
    env: EnvironmentBase,
    traj_path: str | None,
    env_type: str,
    split: str,
    rollout: RolloutConfig,
    num_samples: int,
    meta_extra: dict[str, Any] | None = None,
) -> None:
    """Write the ``--output`` JSON dump and the trajectory JSONL, whose meta line records the run's whole contract
    plus ``meta_extra`` (what an offline re-grade needs); only :func:`add_endpoint_args`' flags are read."""
    if args.output:
        # Written after every episode ran, so a missing parent directory must not discard the run.
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
        "split": split,
        # The env's resolved cap, not a flag: an omitted --max_turns leaves the env class's own value.
        "max_turns": env.max_turns,
        "rollout": asdict(rollout),
        "training_config": args.training_config,
        "num_samples": num_samples,
        "system_prompt": env.system_prompt,
        "tools": env.get_tools_schema(),
        **(meta_extra or {}),
    }
    count = write_trajectories_jsonl(traj_path, meta, results)
    logger.info("Wrote %d trajectories to %s", count, traj_path)
