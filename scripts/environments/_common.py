"""Shared CLI surface for the environment eval scripts: the flags every runner registers, the
training-config contract an eval can sample under, and the output writer that consumes them.

The eval driver itself lives in :mod:`src.environments.eval_runner`; only the argparse half (the
flags, the ``--training_config`` loader and the ``Namespace``-reading output writer) lives here with
the scripts that use it.
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl import ModelConfig

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.environment_config import EnvironmentConfig
from src.configs.rollout_config import DEFAULT_ROLLOUT_TOP_P, THINKING_SCOPE_EPISODE, RolloutConfig
from src.environments.base import BaseEnvironment
from src.environments.episode import resolve_reasoning_end_token_id
from src.environments.eval_runner import DEFAULT_REQUEST_TIMEOUT_S, trajectory_path, write_trajectories_jsonl
from src.inference.openai_client import DEFAULT_LOCAL_BASE_URL, resolve_local_api_key
from src.training.parser import H4ArgumentParser

logger = logging.getLogger(__name__)

# The training script's config classes an eval reads its contract from: the environment's, the
# rollout's, and the model's (its tokenizer resolves ``rollout_stop_tokens``). The YAML's other keys
# (training arguments, dataset fields) belong to classes the eval never instantiates.
TRAINING_CONTRACT_CLASSES = (EnvironmentConfig, AsyncTrainingConfig, ModelConfig)

# The sampling flags an explicit CLI value lays over the training config, by ``RolloutConfig`` field.
_SAMPLING_FLAGS = ("temperature", "top_p", "max_tokens", "request_timeout")


def add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    """Register the dataset-source, endpoint, and output flags every eval script shares.

    Task-specific flags (env type, adapter, language, sampling budgets, concurrency) stay on the
    script's own parser; only flags whose meaning and defaults are identical across the eval scripts
    live here. The sampling flags default to ``None`` so :func:`rollout_config_from_args` can tell an
    explicit value from an omitted one: explicit CLI > ``--training_config`` > default.
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
        "--training_config",
        default=None,
        help="Environmental-GRPO training YAML to evaluate under: its rollout contract (chat-template "
        "variables, stop tokens, thinking budget, backend, sampling) and its environment config become "
        "the eval's, with any sampling flag passed explicitly laid over them. Without it the eval samples "
        "under the flags alone.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help=f"Nucleus-sampling cutoff. Default: the training config's rollout_top_p under --training_config, "
        f"else the training rollout's own default ({DEFAULT_ROLLOUT_TOP_P}), so an eval samples the policy the "
        f"way training did rather than at whatever the server defaults to.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=None,
        help=f"Per-generation HTTP timeout (s). Default: the training config's request_timeout under "
        f"--training_config, else {DEFAULT_REQUEST_TIMEOUT_S:.0f}; raise it when running many concurrent "
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


def resolve_stop_token_ids(tokenizer: PreTrainedTokenizerBase, names: list[str]) -> list[int]:
    """The ids of ``rollout_stop_tokens`` under the training tokenizer, every name required to resolve.

    Stricter than the trainer, which warns and skips a partially unresolved set: an eval that silently
    dropped a terminator would sample under a contract the policy was not trained with, the drift
    ``--training_config`` exists to close.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    ids = {name: tokenizer.convert_tokens_to_ids(name) for name in names}
    unresolved = sorted(name for name, tid in ids.items() if tid is None or tid == unk)
    if unresolved:
        raise ValueError(
            f"rollout_stop_tokens {unresolved} are not tokens of the training config's tokenizer, so the "
            f"eval cannot stop turns where training did; fix the YAML or the tokenizer it names."
        )
    return list(ids.values())


@dataclass(frozen=True)
class TrainingContract:
    """What a training YAML fixes about generation and the environment, for an eval to sample under.

    Parsed with the training script's own config classes, so the eval reads the YAML the way the run
    did rather than through a hand-picked subset of its keys.
    """

    path: str
    env_config: EnvironmentConfig
    async_config: AsyncTrainingConfig
    stop_token_ids: list[int] | None
    reasoning_end_token_id: int | None = None

    @classmethod
    def load(cls, path: str) -> "TrainingContract":
        """Parse ``path``; the stop tokens and the reasoning-end marker go through the tokenizer of the
        model the YAML trains."""
        parser = H4ArgumentParser(TRAINING_CONTRACT_CLASSES)
        env_config, async_config, model_config = parser.parse_yaml_file(path, allow_extra_keys=True)
        episode_scope = async_config.rollout_thinking_budget_scope == THINKING_SCOPE_EPISODE
        stop_token_ids = reasoning_end_token_id = None
        if async_config.rollout_stop_tokens or episode_scope:
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
            )
            if async_config.rollout_stop_tokens:
                stop_token_ids = resolve_stop_token_ids(tokenizer, async_config.rollout_stop_tokens)
            if episode_scope:
                reasoning_end_token_id = resolve_reasoning_end_token_id(
                    tokenizer, async_config.rollout_reasoning_end_token
                )
        return cls(
            path=path,
            env_config=env_config,
            async_config=async_config,
            stop_token_ids=stop_token_ids,
            reasoning_end_token_id=reasoning_end_token_id,
        )

    def rollout_config(self) -> RolloutConfig:
        """The training run's own ``RolloutConfig``, minus the engine captures the eval transport never
        requests (ids, logprobs, routing) — recorded as off so the meta line does not claim them. The
        episode thinking scope still gets the ids it counts with: its own request flag asks for them. The
        eval joins no process group, so the NCCL watchdog bound on the run's timeouts does not apply."""
        rollout = self.async_config.get_rollout_config(
            stop_token_ids=self.stop_token_ids,
            reasoning_end_token_id=self.reasoning_end_token_id,
            in_process_group=False,
        )
        return replace(rollout, capture_token_ids=False, capture_routed_experts=False)

    def env_config_dict(self) -> dict[str, Any]:
        """The dict the training run hands its environment factory (rewards, turn cap, env kwargs)."""
        return self.env_config.to_env_config()


def load_training_contract(path: str | None) -> TrainingContract | None:
    """:meth:`TrainingContract.load` for an optional ``--training_config``; ``None`` without the flag."""
    return TrainingContract.load(path) if path else None


def resolve_setting(explicit: Any, trained: Any, default: Any) -> Any:
    """Explicit CLI flag > the training config's value > the script's default (``None`` = unset)."""
    if explicit is not None:
        return explicit
    return default if trained is None else trained


def rollout_config_from_args(
    args: argparse.Namespace,
    contract: TrainingContract | None,
    *,
    default_temperature: float,
    default_max_tokens: int,
) -> RolloutConfig:
    """The generation contract an eval run samples under.

    Under ``--training_config`` it is the training run's own :class:`RolloutConfig` — the object the
    training rollout hands its actors, template variables and stop tokens included — with the served
    model name and every sampling flag passed explicitly laid over it. Without the flag it is built
    from the shared endpoint flags and the script's defaults, as before. ``default_temperature`` and
    ``default_max_tokens`` are the script's own, for the same reason as in :func:`write_eval_outputs`.
    """
    if contract is not None:
        base = contract.rollout_config()
    else:
        base = RolloutConfig(
            temperature=default_temperature,
            top_p=DEFAULT_ROLLOUT_TOP_P,
            max_tokens=default_max_tokens,
            request_timeout=DEFAULT_REQUEST_TIMEOUT_S,
        )
    explicit = {name: getattr(args, name) for name in _SAMPLING_FLAGS if getattr(args, name) is not None}
    return replace(base, model_name=args.model, **explicit)


def resolve_trajectory_path(args: argparse.Namespace, *parts: str) -> str | None:
    """The JSONL file this run records trajectories to, or ``None`` when it records none.

    An explicit ``--save_trajectories`` wins; otherwise ``--trajectory_dir`` gets one auto-named file
    per run, built from ``parts``, the knobs identifying this run within the folder's matrix. Both
    flags are registered by :func:`add_endpoint_args`, so precedence is decided here rather than
    re-derived by each runner.
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
    rollout: RolloutConfig,
    num_samples: int,
    meta_extra: dict[str, Any] | None = None,
) -> None:
    """Write an eval run's outputs: the ``--output`` JSON dump and the trajectory JSONL.

    The trajectory meta line carries the run's dataset/endpoint knobs, the whole generation contract
    (``rollout``, every :class:`RolloutConfig` field, so a knob added to it cannot be left out of the
    record), the training config it came from, and the env's system prompt and tool schemas;
    ``meta_extra`` adds the task-specific keys (adapter, language, env-level grading knobs, …) an
    offline re-grade needs to reproduce the same verdicts.

    Only the flags :func:`add_endpoint_args` registers are read off ``args``; everything else is an
    explicit parameter, so a script pairing the two helpers without those flags fails at the call
    rather than after the whole eval has run.
    """
    if args.output:
        # The dump runs after every episode has been generated, so a missing parent directory must
        # not discard the run. ``default=str`` matches the trajectory writer, which serializes the
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
        # The effective cap rather than the flag: an omitted --max_turns leaves the env's own value,
        # and a null here would leave a trajectory with no record of the budget it ran under.
        "max_turns": max_turns if max_turns is not None else env.max_turns,
        "rollout": asdict(rollout),
        "training_config": args.training_config,
        "num_samples": num_samples,
        "system_prompt": env.system_prompt,
        "tools": env.get_tools_schema(),
        **(meta_extra or {}),
    }
    count = write_trajectories_jsonl(traj_path, meta, results)
    logger.info("Wrote %d trajectories to %s", count, traj_path)
