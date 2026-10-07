#!/usr/bin/env python
"""Construction-time gates of the environmental GRPO trainer — each refuses before the run starts.

* TRL's ``off_policy_mask_threshold`` masks on ``sampling_per_token_logps``, a batch key this trainer
  never emits; TRL then thresholds a KL of exactly 0 and the knob is a silent no-op. Refused, pointing
  at ``isr_opsm_delta``, ahead of the ``balance_token_mass`` gate that would name the pair instead.
* ``carry_reasoning`` on an SGLang rollout backend is refused until the engine's handling of an
  assistant message carrying ``reasoning_content`` is verified.
* A dataset with no ``answer`` column under an environment that grades against one scores a single
  constant — zero advantage in every GRPO group, nothing in the logs — so it is refused here, and
  ``remove_unused_columns`` is forced off because the rollout context IS the row's other columns.
* A drawable effort level whose ``thinking_tokens`` reach ``rollout_max_tokens`` leaves its turns no
  answer room, so every one is cut mid-reasoning; refused before the servers are up.
* A vLLM thinking budget forces reasoning closes the loss must not train on: the run needs the IS
  correction, and a close marker the tokenizer lacks is warned, the forced closes left in the loss.

    python tests/cpu/grpo/test_env_trainer_construction_gates.py
"""

import ast
import inspect
import logging
import textwrap
import types

import pytest
from accelerate import PartialState
from datasets import Dataset
from trl import GRPOConfig

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.rollout_config import DEFAULT_REASONING_END_TOKEN
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.environments.engine_wire import SGLANG_BACKEND, VLLM_BACKEND
from src.environments.registry import resolve_environment
from src.trainers.grpo.environmental import (
    DistributedAsyncEnvironmentalGRPOTrainer,
    reject_off_policy_mask_threshold,
)

PartialState()  # the per-turn close-marker warning logs through accelerate, which refuses to log without it


def _grpo_config(tmp_path, **overrides) -> GRPOConfig:
    return GRPOConfig(output_dir=str(tmp_path / "out"), bf16=False, use_cpu=True, **overrides)


# Every gate below is driven as a bound method on a bare host, which pins its logic but not its
# wiring: a deleted call site would leave all of those green and the gate dead.
_INIT_GATES = (
    "_reject_unverified_carried_reasoning",
    "_validate_eval_round",
    "_force_full_dataset_columns",
    "_reject_answerless_datasets",
    "_validate_reasoning_terms",
    "_require_forced_close_neutralized",
    "reject_off_policy_mask_threshold",
    "reject_inert_std_floor",
    "resolve_rollout_stop_token_ids",
    "_arm_update_breaker",
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


def _init_source() -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(DistributedAsyncEnvironmentalGRPOTrainer.__init__))).body[0]


def test_every_gate_is_still_called_from_the_trainers_init():
    called = _called_names(_init_source())
    missing = [gate for gate in _INIT_GATES if gate not in called]
    assert not missing, f"DistributedAsyncEnvironmentalGRPOTrainer.__init__ no longer calls: {missing}"


def test_the_off_policy_mask_is_refused_before_the_balance_gate_reads_it():
    """Run first, the balance gate refuses the pair ("unset one of them") over a knob this trainer refuses
    on its own: a user who unsets the balance meets the real refusal only on the next launch."""
    first_call = {}
    for node in ast.walk(_init_source()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            first_call[node.func.id] = min(node.lineno, first_call.get(node.func.id, node.lineno))
    assert first_call["reject_off_policy_mask_threshold"] < first_call["validate_token_mass_balance"]


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


def _level_host(budgets: dict, effort: str | None = "random", **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(**config)
    host._rollout_env = types.SimpleNamespace(reasoning_effort=effort, thinking_budget_for_effort=budgets.get)
    return host


_EVERY_LEVEL = {"low": 8192, "medium": 12288, "high": 16384}


def test_a_drawable_levels_budget_must_sit_below_the_turn_cap():
    """A level's ``thinking_tokens`` at or above ``rollout_max_tokens`` leaves its turns no answer room:
    each is cut mid-reasoning and trains as a length-cut turn forever. Only the levels the environment
    can draw are checked, since a budget elsewhere binds no episode."""
    turn = {"rollout_max_tokens": 16384}
    with pytest.raises(
        ValueError, match=r"'high' level's thinking_tokens \(16384\) must sit below rollout_max_tokens"
    ):
        _level_host(_EVERY_LEVEL, **turn)._validate_reasoning_terms()
    with pytest.raises(ValueError, match="'high' level's thinking_tokens"):
        _level_host(_EVERY_LEVEL, effort="high", **turn)._validate_reasoning_terms()
    _level_host(_EVERY_LEVEL, effort="low", **turn)._validate_reasoning_terms()
    _level_host(_EVERY_LEVEL, effort=None, **turn)._validate_reasoning_terms()
    _level_host(_EVERY_LEVEL, rollout_max_tokens=16385)._validate_reasoning_terms()


def test_column_pruning_is_forced_off(tmp_path):
    """The rollout context is every non-``prompt`` column of the row, and pruning keeps only what the
    model's forward signature names — so it would drop ``answer`` and every context field, leaving
    the environment grading episodes it was handed no ground truth for."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.args = _grpo_config(tmp_path, remove_unused_columns=True)
    host._force_full_dataset_columns()
    assert host.args.remove_unused_columns is False


def test_an_enforced_thinking_budget_needs_the_is_correction():
    """Forced reasoning closes are neutralized through the IS ratio; with the correction off they would train
    with the episode's advantage and teach the model to stop closing its reasoning."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._forced_close_ids, host._is_correction = (7,), False
    with pytest.raises(ValueError, match="importance-sampling correction is off"):
        host._require_forced_close_neutralized()
    host._is_correction = True
    host._require_forced_close_neutralized()
    host._forced_close_ids, host._is_correction = None, False
    host._require_forced_close_neutralized()


_CLOSE_ID = 7
# gpt-oss's final-channel opener: added tokens around two plain words, the shape a harmony close takes.
_OPENER = "<|start|>assistant<|channel|>final<|message|>"
_OPENER_IDS = (70, 71, 72, 73, 74)


class _Tokenizer:
    """Encodes the markers a family writes to their ids; ``knows_close`` decides whether the default
    ``</think>`` is one of its added tokens or splits into plain text, as on Gemma 4 and gpt-oss."""

    def __init__(self, knows_close: bool):
        self._encodings = {
            DEFAULT_REASONING_END_TOKEN: [_CLOSE_ID] if knows_close else [60, 61, 62],
            _OPENER: _OPENER_IDS,
        }
        self.added_tokens_decoder = dict.fromkeys(({_CLOSE_ID} if knows_close else set()) | {70, 72, 74})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens, "the engine encodes its reasoning end string without special tokens"
        return list(self._encodings[text])


def _close_host(budgets: dict, knows_close: bool = True, **config):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(**config)
    host._rollout_env = types.SimpleNamespace(reasoning_effort="random", thinking_budget_for_effort=budgets.get)
    host._tokenizer = _Tokenizer(knows_close)
    return host


def test_a_vllm_thinking_budget_resolves_the_close_the_engine_forces():
    """A level's ``thinking_tokens`` or the run's ceiling each enforce a budget, and either way the loss
    needs the marker's ids to find the closes the engine forced."""
    assert _close_host({"high": 16384})._resolve_forced_close_ids() == (_CLOSE_ID,)
    assert _close_host({}, rollout_max_thinking_tokens=8192)._resolve_forced_close_ids() == (_CLOSE_ID,)


def test_a_multi_token_close_resolves_to_its_whole_sequence():
    """gpt-oss's budget forces a five-token opener; the loss needs every id of it, in the engine's order."""
    host = _close_host({"high": 16384}, knows_close=False, rollout_reasoning_end_token=_OPENER)
    assert host._resolve_forced_close_ids() == _OPENER_IDS


def test_no_forced_close_where_no_budget_can_be_enforced():
    """SGLang enforces no thinking budget and an unbudgeted run caps nothing, so neither forces a close: a
    resolved marker would demand the IS correction of a run that needs none, and zero the ratio at a close
    the model itself emitted with certainty."""
    assert _close_host({"high": 16384}, rollout_backend=SGLANG_BACKEND)._resolve_forced_close_ids() is None
    assert _close_host({})._resolve_forced_close_ids() is None


def test_a_marker_the_tokenizer_does_not_write_warns_and_trains_on_the_forced_closes(caplog):
    """Nothing else reads the marker, so a family whose reasoning ends otherwise still runs, told that
    its forced closes stay in the loss."""
    host = _close_host({"high": 16384}, knows_close=False)
    with caplog.at_level(logging.WARNING, logger="src.trainers.grpo.environmental"):
        assert host._resolve_forced_close_ids() is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stay in the policy loss" in w and DEFAULT_REASONING_END_TOKEN in w for w in warnings), warnings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
