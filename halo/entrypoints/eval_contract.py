"""The environment evals' training contract and run plumbing: a training YAML read the way the run reads it, the
generation contract, the trajectory path and the output writer."""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig
from halo.configs.parser import HaloArgumentParser
from halo.configs.script_arguments import ModelArguments
from halo.environments.engine_wire import reject_unenforceable_thinking_budget
from halo.environments.environment_base import EnvironmentBase
from halo.environments.eval_runner import trajectory_path, write_trajectories_jsonl
from halo.environments.reasoning_effort import efforts_by_level, has_thinking_cap
from halo.environments.rollout_config import (
    VLLM_BACKEND,
    RolloutConfig,
    find_reasoning_end_ids,
    resolve_reasoning_end_token_id,
    resolve_rollout_stop_token_ids,
)
from halo.inference.openai_client import REQUEST_TIMEOUT_SECONDS
from halo.models.config_access import config_model_types
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# The config classes an eval reads its contract from; the YAML's other keys belong to classes it never instantiates.
TRAINING_CONTRACT_CLASSES = (EnvironmentConfig, AsyncRolloutConfig, ModelArguments)

_SAMPLING_FLAGS = ("temperature", "top_p", "max_tokens", "request_timeout")

# The split an eval reads when --split is omitted, unless its dataset ships a single split of its own.
DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class TrainingContract:
    """What a training YAML fixes about generation and the environment, parsed with the script's own config classes, so
    the eval reads it as the run did."""

    path: str
    env_config: EnvironmentConfig
    async_config: AsyncRolloutConfig
    model_config: ModelArguments

    @classmethod
    def load(cls, path: str) -> "TrainingContract":
        parser = HaloArgumentParser(TRAINING_CONTRACT_CLASSES)
        return cls(path, *parser.parse_yaml_file(path, allow_extra_keys=True))

    def rollout_config(self, env: EnvironmentBase) -> RolloutConfig:
        """The run's ``RolloutConfig`` for an eval of ``env``, stop tokens resolved by the trained tokenizer. A vLLM
        reasoning cap the run binds is refused where vLLM cannot enforce it, and has the eval capture
        sampled ids to count reasoning as training does; routing is never captured. The eval joins no process group, so
        no NCCL watchdog bound."""
        cfg, model = self.async_config, self.model_config
        efforts = efforts_by_level(
            env,
            max_tokens=cfg.rollout_max_tokens,
            max_thinking_tokens=cfg.rollout_max_thinking_tokens,
            max_episode_tokens=cfg.rollout_max_episode_tokens,
            max_answer_tokens=cfg.rollout_max_answer_tokens,
        )
        capped = cfg.rollout_backend == VLLM_BACKEND and has_thinking_cap(efforts)
        if capped:
            hf_config = AutoConfig.from_pretrained(
                model.model_name_or_path, revision=model.model_revision, trust_remote_code=model.trust_remote_code
            )
            reject_unenforceable_thinking_budget(
                config_model_types(hf_config),
                resolve_weight_sync_client(cfg.rollout_backend).UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES,
            )
        tokenizer = None
        if capped or cfg.rollout_stop_tokens:
            tokenizer = AutoTokenizer.from_pretrained(
                model.model_name_or_path, trust_remote_code=model.trust_remote_code
            )
        reasoning_end_token_id = (
            resolve_reasoning_end_token_id(find_reasoning_end_ids(tokenizer, cfg.rollout_reasoning_end_token))
            if capped
            else None
        )
        rollout = cfg.get_rollout_config(
            stop_token_ids=resolve_rollout_stop_token_ids(tokenizer, cfg.rollout_stop_tokens),
            reasoning_end_token_id=reasoning_end_token_id,
            in_process_group=False,
        )
        return replace(rollout, capture_token_ids=reasoning_end_token_id is not None, capture_routed_experts=False)


def load_training_contract(path: str | None) -> TrainingContract | None:
    return TrainingContract.load(path) if path else None


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
) -> RolloutConfig:
    """The training run's :class:`RolloutConfig` under ``--training_config``, else one from the script's temperature and
    the training rollout's other defaults, with the served model name and every explicit sampling flag laid over it."""
    if contract is not None:
        base = contract.rollout_config(env)
    else:
        base = RolloutConfig(
            temperature=default_temperature,
            top_p=RolloutConfig.top_p,
            request_timeout=REQUEST_TIMEOUT_SECONDS,
        )
    explicit = {name: getattr(args, name) for name in _SAMPLING_FLAGS if getattr(args, name) is not None}
    return replace(base, model_name=args.model, **explicit)


def reject_flag_owned_env_kwargs(env_kwargs: dict, flag_owned: tuple[str, ...]) -> None:
    """Such an option in ``--env_kwargs`` would silently override its flag, the JSON being laid over the flags."""
    for key in flag_owned:
        if key in env_kwargs:
            raise SystemExit(f"set the {key} with --{key}, not --env_kwargs, which would override the flag")


def resolve_trajectory_path(args: argparse.Namespace, *parts: str) -> str | None:
    """``--save_trajectories``, or a file under ``--trajectory_dir`` named from the model and parts."""
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
    """Write the ``--output_path`` dump and the trajectory JSONL, whose meta line records the run's whole contract plus
    ``meta_extra``; only :func:`add_eval_args`' flags are read."""
    if args.output_path:
        # Written after every episode ran, so a missing parent directory must not discard the run.
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Wrote per-example results to %s", args.output_path)

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
