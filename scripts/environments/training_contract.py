"""The generation and environment contract of an environmental-GRPO training YAML, read the way the training run
reads it, for an eval to sample under (``--training_config``)."""

from dataclasses import dataclass, replace

from transformers import AutoConfig, AutoTokenizer

from src.configs.grpo_environmental import AsyncTrainingConfig, EnvironmentConfig
from src.configs.parser import HaloArgumentParser
from src.configs.script_arguments import ModelArguments
from src.environments.engine_wire import reject_unenforceable_thinking_budget
from src.environments.environment_base import EnvironmentBase
from src.environments.reasoning_effort import efforts_by_level, has_thinking_budget
from src.environments.rollout_config import (
    VLLM_BACKEND,
    RolloutConfig,
    find_reasoning_end_ids,
    resolve_reasoning_end_token_id,
    resolve_rollout_stop_token_ids,
)
from src.models.config_access import config_model_types

# The training script's config classes an eval reads its contract from: the environment's, the rollout's, and the
# model's (its tokenizer resolves the stop tokens and the reasoning close). The YAML's other keys (training
# arguments, dataset fields) belong to classes the eval never instantiates.
TRAINING_CONTRACT_CLASSES = (EnvironmentConfig, AsyncTrainingConfig, ModelArguments)


@dataclass(frozen=True)
class TrainingContract:
    """What a training YAML fixes about generation and the environment, parsed with the training script's own
    config classes, so the eval reads the YAML the way the run did rather than through a subset of its keys."""

    path: str
    env_config: EnvironmentConfig
    async_config: AsyncTrainingConfig
    model_config: ModelArguments

    @classmethod
    def load(cls, path: str) -> "TrainingContract":
        parser = HaloArgumentParser(TRAINING_CONTRACT_CLASSES)
        return cls(path, *parser.parse_yaml_file(path, allow_extra_keys=True))

    def rollout_config(self, env: EnvironmentBase) -> RolloutConfig:
        """The training run's own ``RolloutConfig`` for an eval of ``env``, its stop tokens resolved by the trained
        model's tokenizer. A vLLM thinking budget some level of ``env`` draws is refused where vLLM cannot enforce
        it, and has the eval capture the sampled ids to count each turn's reasoning as training does; the routing
        is never captured. The eval joins no process group, so the NCCL watchdog bound on the timeouts does not
        apply."""
        cfg, model = self.async_config, self.model_config
        efforts = efforts_by_level(
            env,
            max_tokens=cfg.rollout_max_tokens,
            max_thinking_tokens=cfg.rollout_max_thinking_tokens,
            max_episode_tokens=cfg.rollout_max_episode_tokens,
            max_answer_tokens=cfg.rollout_max_answer_tokens,
        )
        budgeted = cfg.rollout_backend == VLLM_BACKEND and has_thinking_budget(efforts)
        if budgeted:
            hf_config = AutoConfig.from_pretrained(
                model.model_name_or_path, revision=model.model_revision, trust_remote_code=model.trust_remote_code
            )
            reject_unenforceable_thinking_budget(config_model_types(hf_config), cfg.rollout_backend)
        tokenizer = None
        if budgeted or cfg.rollout_stop_tokens:
            tokenizer = AutoTokenizer.from_pretrained(
                model.model_name_or_path, trust_remote_code=model.trust_remote_code
            )
        reasoning_end_token_id = (
            resolve_reasoning_end_token_id(find_reasoning_end_ids(tokenizer, cfg.rollout_reasoning_end_token))
            if budgeted
            else None
        )
        rollout = cfg.get_rollout_config(
            stop_token_ids=resolve_rollout_stop_token_ids(tokenizer, cfg.rollout_stop_tokens),
            reasoning_end_token_id=reasoning_end_token_id,
            in_process_group=False,
        )
        return replace(rollout, capture_token_ids=reasoning_end_token_id is not None, capture_routed_experts=False)


def load_training_contract(path: str | None) -> TrainingContract | None:
    """:meth:`TrainingContract.load` for an optional ``--training_config``; ``None`` without the flag."""
    return TrainingContract.load(path) if path else None
