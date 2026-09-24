#!/usr/bin/env python
"""Construction-time gates of the environmental GRPO trainer — each refuses before the run starts.

* TRL's ``off_policy_mask_threshold`` masks on ``sampling_per_token_logps``, a batch key this trainer
  never emits; TRL then thresholds a KL of exactly 0 and the knob is a silent no-op. Refused, pointing
  at ``isr_opsm_delta``.
* ``carry_reasoning`` on an SGLang rollout backend is refused until the engine's handling of an
  assistant message carrying ``reasoning_content`` is verified.
* A dataset with no ``answer`` column under an environment that grades against one scores a single
  constant — zero advantage in every GRPO group, nothing in the logs — so it is refused here, and
  ``remove_unused_columns`` is forced off because the rollout context IS the row's other columns.
* The episode thinking scope needs a budget for the turns to share (a level's ``thinking_tokens`` or the
  run's ceiling) and a per-turn reserve no level's budget falls below; either gap is refused.

    python tests/cpu/grpo/test_env_trainer_construction_gates.py
"""

import ast
import inspect
import textwrap
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


# Every gate below is driven as a bound method on a bare host, which pins its logic but not its
# wiring: a deleted call site would leave all of those green and the gate dead.
_INIT_GATES = (
    "_reject_unverified_carried_reasoning",
    "_validate_eval_round",
    "_force_full_dataset_columns",
    "_reject_answerless_datasets",
    "_validate_effort_length_terms",
    "_validate_thinking_budget_scope",
    "reject_off_policy_mask_threshold",
)


def _called_names(fn: ast.FunctionDef) -> set[str]:
    """Every name called anywhere in ``fn``, ``self.gate()`` and bare ``gate()`` alike."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def test_every_gate_is_still_called_from_the_trainers_init():
    source = textwrap.dedent(inspect.getsource(DistributedAsyncEnvironmentalGRPOTrainer.__init__))
    called = _called_names(ast.parse(source).body[0])
    missing = [gate for gate in _INIT_GATES if gate not in called]
    assert not missing, f"DistributedAsyncEnvironmentalGRPOTrainer.__init__ no longer calls: {missing}"


def test_off_policy_mask_threshold_is_refused_with_the_working_knob_named(tmp_path):
    with pytest.raises(ValueError, match="isr_opsm_delta"):
        reject_off_policy_mask_threshold(_grpo_config(tmp_path, off_policy_mask_threshold=0.5))


def test_the_trl_default_passes(tmp_path):
    reject_off_policy_mask_threshold(_grpo_config(tmp_path))


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


def _scope_host(budgets: dict, **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(**config)
    host._rollout_env = types.SimpleNamespace(thinking_budget_for_effort=budgets.get)
    return host


def test_episode_scope_refuses_a_run_with_no_budget_to_share():
    """With no level setting ``thinking_tokens`` and no run ceiling the scope has no total: every
    episode would bind at its first turn instead of here, after the servers are up."""
    episode = {"rollout_thinking_budget_scope": "episode"}
    with pytest.raises(ValueError, match="nothing to share"):
        _scope_host({}, **episode)._validate_thinking_budget_scope()
    # One budgeted level is enough, and so is the run-wide ceiling.
    _scope_host({"high": 16384}, **episode)._validate_thinking_budget_scope()
    _scope_host({}, **episode, rollout_max_thinking_tokens=8000)._validate_thinking_budget_scope()
    # The per-turn scope shares nothing and runs uncapped as before.
    _scope_host({})._validate_thinking_budget_scope()


def test_episode_scope_refuses_a_reserve_no_level_budget_can_hold():
    """A level whose whole budget sits below the reserve would be clamped to it silently, so the model
    would be promised a budget it never gets a turn to spend."""
    episode = {"rollout_thinking_budget_scope": "episode"}
    with pytest.raises(ValueError, match="exceeds the thinking_tokens"):
        _scope_host(
            {"low": 256, "high": 16384}, **episode, rollout_thinking_turn_reserve=512
        )._validate_thinking_budget_scope()
    _scope_host(
        {"low": 256, "high": 16384}, **episode, rollout_thinking_turn_reserve=256
    )._validate_thinking_budget_scope()
    # Not a per-turn-scope concern: there the reserve is never read.
    _scope_host({"low": 256}, rollout_thinking_turn_reserve=512)._validate_thinking_budget_scope()


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
