"""The flags every environment eval script registers; the readers they feed live in
:mod:`halo.entrypoints.eval_contract`."""

import argparse

from halo.environments.rollout_config import RolloutConfig
from halo.inference.openai_client import REQUEST_TIMEOUT_SECONDS
from scripts.cli_arguments import add_openai_endpoint_args


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """``--split`` and the task flags stay the runner's own. Sampling flags default to ``None``, so
    :func:`rollout_config_from_args` tells explicit from omitted: flag > ``--training_config`` > default."""
    parser.add_argument(
        "--dataset", required=True, help="HF Hub id, s3:// URI, local data file or local save_to_disk dir."
    )
    parser.add_argument("--config", default=None, help="Dataset config (e.g. 'all', 'verifiable', 'taco').")
    add_openai_endpoint_args(parser, model_help="Served/model name.")
    parser.add_argument(
        "--training_config",
        default=None,
        help="Environmental-GRPO training YAML to evaluate under: its rollout contract (chat-template "
        "variables, stop tokens, reasoning cap, episode output budget, backend, sampling) and its environment config "
        "become "
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
        f"--training_config, else {REQUEST_TIMEOUT_SECONDS:.0f}; raise it when running many concurrent "
        f"episodes so long reasoning turns are not cut off.",
    )
    parser.add_argument("--output_path", default=None, help="Optional JSON file to dump per-example results to.")
    trajectories = parser.add_mutually_exclusive_group()
    trajectories.add_argument(
        "--save_trajectories",
        default=None,
        help="Optional explicit JSONL path to record full trajectories (system prompt + tool schemas in "
        "a meta line, then one episode per line with its messages, tool calls/results, and reward).",
    )
    trajectories.add_argument(
        "--trajectory_dir",
        default=None,
        help="Optional output folder for trajectories; the file is auto-named per run from the model "
        "and the dataset/split it ran on.",
    )
