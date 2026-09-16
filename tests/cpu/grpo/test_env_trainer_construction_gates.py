#!/usr/bin/env python
"""Construction-time gates of the environmental GRPO trainer that used to fail late or not at all.

* TRL's ``off_policy_mask_threshold`` masks on ``sampling_per_token_logps``, a batch key this trainer
  never emits; TRL then thresholds a KL of exactly 0 and the knob is a silent no-op. Refused, pointing
  at ``isr_opsm_delta``.
* The default eval round is ``per_device_eval_batch_size`` rows per rank; TRL validates only the
  GLOBAL eval batch against ``num_generations_eval``, so a per-rank batch that does not hold whole
  groups used to raise N steps in, on the first evaluation.
* ``carry_reasoning`` on an SGLang rollout backend is refused until the engine's handling of an
  assistant message carrying ``reasoning_content`` is verified.
* A dataset with no ``answer`` column under an environment that grades against one scores a single
  constant — zero advantage in every GRPO group, nothing in the logs — so it is refused here, and
  ``remove_unused_columns`` is forced off because the rollout context IS the row's other columns.

    python tests/cpu/grpo/test_env_trainer_construction_gates.py
"""

import types

import pytest
from datasets import Dataset
from trl import GRPOConfig

from src.configs.async_training_config import AsyncTrainingConfig
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.environments.engine_wire import SGLANG_BACKEND, VLLM_BACKEND
from src.environments.registry import resolve_environment
from src.trainers.grpo.environmental import (
    DistributedAsyncEnvironmentalGRPOTrainer,
    reject_off_policy_mask_threshold,
)


def _grpo_config(tmp_path, **overrides) -> GRPOConfig:
    return GRPOConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, **overrides)


def test_off_policy_mask_threshold_is_refused_with_the_working_knob_named(tmp_path):
    with pytest.raises(ValueError, match="isr_opsm_delta"):
        reject_off_policy_mask_threshold(_grpo_config(tmp_path, off_policy_mask_threshold=0.5))


def test_the_trl_default_passes(tmp_path):
    reject_off_policy_mask_threshold(_grpo_config(tmp_path))


class _Args:
    def __init__(self, per_device: int, drop_last: bool = False):
        self.per_device_eval_batch_size = per_device
        self.dataloader_drop_last = drop_last


def _eval_host(rows: int | None, per_device: int, num_generations_eval: int):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.args = _Args(per_device)
    host.async_config = AsyncTrainingConfig(eval_rollout_batch_size=rows)
    host.num_generations_eval = num_generations_eval
    return host


def test_the_default_eval_round_must_hold_whole_groups_per_rank():
    with pytest.raises(
        ValueError, match="per_device_eval_batch_size \\(6\\) must be divisible by num_generations_eval"
    ):
        _eval_host(None, per_device=6, num_generations_eval=4)._validate_eval_round()
    _eval_host(None, per_device=8, num_generations_eval=4)._validate_eval_round()


def test_an_explicit_round_is_the_geometry_that_is_checked():
    """With ``eval_rollout_batch_size`` set, the loader draws that many rows per rank and the eval batch
    is only the loss forward's chunk, so it is the round that must hold whole groups."""
    _eval_host(8, per_device=6, num_generations_eval=4)._validate_eval_round()
    with pytest.raises(ValueError, match="eval_rollout_batch_size \\(6\\)"):
        _eval_host(6, per_device=8, num_generations_eval=4)._validate_eval_round()


def _carry_host(backend: str, carry_reasoning: bool):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(rollout_backend=backend)
    host._rollout_env = types.SimpleNamespace(carry_reasoning=carry_reasoning)
    return host


def test_carried_reasoning_on_sglang_is_refused_as_unverified():
    with pytest.raises(ValueError, match="unverified"):
        _carry_host("sglang", carry_reasoning=True)._reject_unverified_carried_reasoning()


def test_carried_reasoning_on_vllm_and_plain_sglang_pass():
    _carry_host("vllm", carry_reasoning=True)._reject_unverified_carried_reasoning()
    _carry_host("sglang", carry_reasoning=False)._reject_unverified_carried_reasoning()


def test_the_wire_backend_keys_are_the_weight_sync_clients_keys():
    """The tokenize path and the carried-reasoning gate compare against the wire module's spellings,
    so those must be the keys the client registry resolves ``rollout_backend`` by."""
    assert VLLM_BACKEND == VLLMWeightSyncClient.BACKEND_KEY
    assert SGLANG_BACKEND == SGLangWeightSyncClient.BACKEND_KEY


_PROMPT = [[{"role": "user", "content": "solve it"}]]


def _dataset(**columns) -> Dataset:
    return Dataset.from_dict({"prompt": _PROMPT, **columns})


def _answer_host(env_type: str, env_kwargs: dict, train, eval_dataset=None):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._rollout_env = resolve_environment(env_type, env_kwargs)
    host.train_dataset = train
    host.eval_dataset = eval_dataset
    return host


def test_a_grading_environment_refuses_a_dataset_without_the_answer_column():
    """code_contests reads its hidden tests out of ``answer``: with no column every submission grades
    against zero tests and grades 0, which reads as a policy that never solves anything."""
    host = _answer_host("code_contests", {"sandbox_backend": "local"}, _dataset())
    with pytest.raises(ValueError, match="CodeContestsEnvironment"):
        host._reject_answerless_datasets()


def test_a_grading_environment_accepts_the_dataset_that_carries_it():
    """Guards the refusal above from being satisfied by refusing everything."""
    _answer_host("code_contests", {"sandbox_backend": "local"}, _dataset(answer=["{}"]))._reject_answerless_datasets()


def test_the_eval_dataset_is_held_to_the_same_column():
    """An eval split without the column scores nothing on every evaluation round, N steps in."""
    host = _answer_host("exam_qa", {}, _dataset(answer=["A"]), eval_dataset=_dataset())
    with pytest.raises(ValueError, match="eval_dataset"):
        host._reject_answerless_datasets()


def test_a_non_grading_environment_accepts_an_answer_less_dataset():
    """native_math pays for completing the task, so prompts alone are a complete dataset for it."""
    _answer_host("native_math", {}, _dataset(), eval_dataset=_dataset())._reject_answerless_datasets()


def test_column_pruning_is_forced_off(tmp_path):
    """The rollout context is every non-``prompt`` column of the row, and pruning keeps only what the
    model's forward signature names — so it would drop ``answer`` and every context field, leaving
    the environment grading episodes it was handed no ground truth for."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.args = _grpo_config(tmp_path, remove_unused_columns=True)
    host._force_full_dataset_columns()
    assert host.args.remove_unused_columns is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
