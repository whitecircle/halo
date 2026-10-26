#!/usr/bin/env python
"""``--training_config`` makes an eval sample under the generation contract the policy was trained with.

The shared eval flags carry temperature, top-p, max tokens and a request timeout; the training
rollout also fixes chat-template variables, stop tokens, a thinking budget and the backend. A policy
trained with ``preserve_thinking`` or ``rollout_stop_tokens`` and evaluated without them is measured
under a different contract; the meta line records the whole contract.

    python tests/cpu/environments/test_eval_scripts_training_contract.py
"""

import argparse
import json
from types import SimpleNamespace

import pytest
from transformers import CONFIG_MAPPING

import halo.entrypoints.environment_eval as training_contract
import halo.environments.rollout_config as rollout_config
from halo.entrypoints.environment_eval import (
    TrainingContract,
    load_training_contract,
    rollout_config_from_args,
    write_eval_outputs,
)
from halo.env import nccl_timeout
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig
from halo.environments.tasks.coding.code_contests import DEFAULT_REASONING_EFFORT
from halo.inference.openai_client import REQUEST_TIMEOUT_SECONDS
from scripts.environments.inference.run_code_contests import CODING_ENV_TYPES, resolve_env_config
from tests.common.code_contests import StubSandbox
from tests.common.repo import REPO_ROOT

_CALL_TOKEN_ID = 200012
_THINK_END_ID = 151668
# A shipped recipe whose episode_timeout needs the raised NCCL watchdog its launch line exports.
_CODEFORCES_RECIPE = (
    REPO_ROOT
    / "examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-full-ep1-stage1-codeforces.yaml"
)

_TRAINING_YAML = """\
model_name_or_path: dummy/model
output_dir: /tmp/contract-eval
learning_rate: 1.0e-6
environment_type: code_contests
max_turns: 7
environment_kwargs:
  language: cpp
  timeout_per_test: 3
rollout_backend: vllm
rollout_temperature: 1.0
rollout_top_p: 1.0
rollout_max_tokens: 4096
rollout_max_episode_tokens: 16384
rollout_max_thinking_tokens: 2048
rollout_chat_template_kwargs:
  preserve_thinking: true
rollout_stop_tokens: ["<|call|>"]
request_timeout: 300
"""


class _Tokenizer:
    unk_token_id = None
    added_tokens_decoder = {_CALL_TOKEN_ID: "<|call|>", _THINK_END_ID: "</think>"}

    def convert_tokens_to_ids(self, name):
        return _CALL_TOKEN_ID if name == "<|call|>" else None

    def encode(self, text, add_special_tokens=True):
        return [_THINK_END_ID] if text == "</think>" else [1, 2]


@pytest.fixture(autouse=True)
def dense_model_config(monkeypatch):
    """The model config the contract reads to check a vLLM thinking budget is enforceable."""
    monkeypatch.setattr(training_contract.AutoConfig, "from_pretrained", lambda *a, **k: CONFIG_MAPPING["qwen3"]())


@pytest.fixture
def contract(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML)
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    return TrainingContract.load(str(path))


def _env(contract: TrainingContract):
    """The environment the eval builds from ``contract``, a coding env's sandbox stubbed."""
    config = contract.env_config.to_env_config()
    if contract.env_config.environment_type in CODING_ENV_TYPES:
        config["sandbox"] = StubSandbox()
    return build_environment(contract.env_config.environment_type, config)


def _args(**overrides):
    flags = {"model": "served-name", "temperature": None, "top_p": None, "max_tokens": None, "request_timeout": None}
    return argparse.Namespace(**{**flags, **overrides})


def test_the_yaml_rollout_contract_reaches_the_eval(contract):
    rollout = rollout_config_from_args(
        _args(), contract, _env(contract), default_temperature=0.2, default_max_tokens=99
    )
    assert rollout.chat_template_kwargs == {"preserve_thinking": True}
    assert rollout.stop_token_ids == [_CALL_TOKEN_ID]
    assert rollout.max_thinking_tokens == 2048
    assert rollout.max_episode_tokens == 16384, "the episode output budget is part of the trained contract"
    assert rollout.backend == "vllm"
    assert (rollout.temperature, rollout.top_p, rollout.max_tokens, rollout.request_timeout) == (1.0, 1.0, 4096, 300.0)
    assert rollout.model_name == "served-name", "the served name comes from --model, never the YAML"
    assert rollout.reasoning_end_token_id == _THINK_END_ID, "the eval counts reasoning up to the close training does"
    assert rollout.capture_token_ids, "the sampled ids the reasoning count reads"
    assert not rollout.capture_routed_experts, "the eval transport captures no routing"


def test_an_explicit_sampling_flag_overrides_the_yaml_alone(contract):
    rollout = rollout_config_from_args(
        _args(temperature=0.3, max_tokens=512),
        contract,
        _env(contract),
        default_temperature=0.2,
        default_max_tokens=99,
    )
    assert (rollout.temperature, rollout.max_tokens) == (0.3, 512)
    assert (rollout.top_p, rollout.request_timeout) == (1.0, 300.0)
    assert rollout.stop_token_ids == [_CALL_TOKEN_ID]


def test_without_the_flag_the_script_defaults_stand():
    assert load_training_contract(None) is None
    rollout = rollout_config_from_args(_args(), None, None, default_temperature=0.2, default_max_tokens=99)
    assert (rollout.temperature, rollout.top_p, rollout.max_tokens, rollout.request_timeout) == (
        0.2,
        RolloutConfig.top_p,
        99,
        REQUEST_TIMEOUT_SECONDS,
    )
    assert rollout.stop_token_ids is None and rollout.chat_template_kwargs == {}


def test_the_yaml_environment_config_reaches_the_eval(contract):
    assert contract.env_config.environment_type == "code_contests"
    assert contract.env_config.to_env_config() == {
        "reward_terms": [{"source": "environment"}],
        "max_turns": 7,
        "language": "cpp",
        "timeout_per_test": 3,
    }


def test_a_mixed_case_environment_type_reaches_the_coding_eval_as_the_registry_name(tmp_path, monkeypatch):
    """The registry resolves ``environment_type`` case-insensitively, so a ``Code_Contests`` YAML trains;
    the coding eval compares the name against its coding envs, and must accept the run it trained."""
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace("environment_type: code_contests", "environment_type: Code_Contests"))
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    environment_type = TrainingContract.load(str(path)).env_config.environment_type
    assert environment_type == "code_contests" and environment_type in CODING_ENV_TYPES


@pytest.fixture
def default_watchdog_recipe(monkeypatch):
    """The codeforces recipe's contract under the default watchdog, which its episode_timeout exceeds."""
    monkeypatch.delenv("DIST_NCCL_TIMEOUT_MINUTES", raising=False)
    contract = TrainingContract.load(str(_CODEFORCES_RECIPE))
    assert contract.async_config.episode_timeout > nccl_timeout().total_seconds()
    return contract


def test_a_shipped_recipe_evaluates_without_the_training_watchdog(default_watchdog_recipe):
    """The eval joins no process group, so the recipe's contract builds on the default watchdog."""
    rollout = default_watchdog_recipe.rollout_config(_env(default_watchdog_recipe))
    assert rollout.episode_timeout == default_watchdog_recipe.async_config.episode_timeout
    trained_budget = default_watchdog_recipe.async_config.rollout_max_episode_tokens
    assert trained_budget is not None and rollout.max_episode_tokens == trained_budget


def test_training_still_refuses_the_recipe_on_the_default_watchdog(default_watchdog_recipe):
    with pytest.raises(ValueError, match="NCCL collective watchdog"):
        default_watchdog_recipe.async_config.get_rollout_config()


def test_an_unresolvable_stop_token_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace('["<|call|>"]', '["<|call|>", "<|nope|>"]'))
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    contract = TrainingContract.load(str(path))
    with pytest.raises(ValueError, match="<\\|nope\\|>"):
        contract.rollout_config(_env(contract))


def test_a_thinking_budget_vllm_cannot_enforce_is_refused(tmp_path, monkeypatch):
    """vLLM encodes Mistral 4's [THINK] markers as plain text, so the budget the YAML sets never counts."""
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML)
    mistral_small_4 = CONFIG_MAPPING["mistral3"](text_config=CONFIG_MAPPING["mistral4"]())
    monkeypatch.setattr(training_contract.AutoConfig, "from_pretrained", lambda *a, **k: mistral_small_4)
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    contract = TrainingContract.load(str(path))
    with pytest.raises(ValueError, match="cannot enforce a thinking budget"):
        contract.rollout_config(_env(contract))


_LEVEL_BUDGET_YAML = """\
model_name_or_path: dummy/model
output_dir: /tmp/contract-eval
learning_rate: 1.0e-6
environment_type: native_math
environment_kwargs:
  reasoning_effort: low
  reasoning_effort_profiles:
    low: {thinking_tokens: 1024}
rollout_backend: vllm
rollout_max_tokens: 4096
"""


def test_a_level_budget_vllm_cannot_enforce_is_refused_without_a_run_wide_one(tmp_path, monkeypatch):
    """A drawable level's ``thinking_tokens`` reaches vLLM as the turn's budget just as the run-wide cap does,
    so a Mistral 4 run budgeted per level alone is refused too; the same run without the level budget samples."""
    mistral_small_4 = CONFIG_MAPPING["mistral3"](text_config=CONFIG_MAPPING["mistral4"]())
    monkeypatch.setattr(training_contract.AutoConfig, "from_pretrained", lambda *a, **k: mistral_small_4)
    budgeted = tmp_path / "budgeted.yaml"
    budgeted.write_text(_LEVEL_BUDGET_YAML)
    contract = TrainingContract.load(str(budgeted))
    assert contract.async_config.rollout_max_thinking_tokens is None
    with pytest.raises(ValueError, match="cannot enforce a thinking budget"):
        contract.rollout_config(_env(contract))
    unbudgeted = tmp_path / "unbudgeted.yaml"
    unbudgeted.write_text(
        _LEVEL_BUDGET_YAML.replace("  reasoning_effort_profiles:\n    low: {thinking_tokens: 1024}\n", "")
    )
    contract = TrainingContract.load(str(unbudgeted))
    assert contract.rollout_config(_env(contract)).max_thinking_tokens is None


def test_the_contract_loads_the_tokenizer_for_stop_tokens_or_a_vllm_thinking_budget(tmp_path, monkeypatch):
    """The tokenizer resolves the stop tokens and, under a vLLM thinking budget, the reasoning close the eval
    counts each turn's reasoning up to: a contract with neither never loads one, and captures no ids."""
    budgeted = tmp_path / "budgeted.yaml"
    budgeted.write_text(_TRAINING_YAML.replace('rollout_stop_tokens: ["<|call|>"]\n', ""))
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    contract = TrainingContract.load(str(budgeted))
    rollout = contract.rollout_config(_env(contract))
    assert rollout.stop_token_ids is None and rollout.reasoning_end_token_id == _THINK_END_ID

    def never(*a, **k):
        raise AssertionError("the contract loaded a tokenizer with nothing to resolve")

    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", never)
    plain = tmp_path / "plain.yaml"
    plain.write_text(
        _LEVEL_BUDGET_YAML.replace("  reasoning_effort_profiles:\n    low: {thinking_tokens: 1024}\n", "")
    )
    unbudgeted = TrainingContract.load(str(plain))
    rollout = unbudgeted.rollout_config(_env(unbudgeted))
    assert rollout.reasoning_end_token_id is None and not rollout.capture_token_ids


def test_the_eval_resolves_stop_tokens_as_the_trainer_does():
    """One resolver, one policy: a recipe the trainer accepts is one its eval accepts."""
    assert training_contract.resolve_rollout_stop_token_ids is rollout_config.resolve_rollout_stop_token_ids


@pytest.mark.parametrize(
    ("effort_line", "level"),
    [("  reasoning_effort: null\n", None), ("", DEFAULT_REASONING_EFFORT)],
    ids=["null", "absent"],
)
def test_the_coding_eval_takes_the_level_the_training_env_was_built_with(tmp_path, monkeypatch, effort_line, level):
    """A YAML's ``reasoning_effort: null`` trains at no level and one without the key at the env
    class's default: the eval resolves each to the level training built its env with, instead of
    reading the null as unset and grading a no-level policy at the default."""
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace("  timeout_per_test: 3\n", f"  timeout_per_test: 3\n{effort_line}"))
    monkeypatch.setattr(training_contract.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    contract = TrainingContract.load(str(path))
    trained_env = contract.env_config.to_env_config()
    training = build_environment(contract.env_config.environment_type, {**trained_env, "sandbox": StubSandbox()})
    flags = SimpleNamespace(eval_protocol=None, language=None, reasoning_effort=None, max_turns=None)
    assert resolve_env_config(flags, trained_env, {})["reasoning_effort"] == training.reasoning_effort == level


def test_the_meta_line_records_the_whole_generation_contract(contract, tmp_path):
    rollout = rollout_config_from_args(
        _args(), contract, _env(contract), default_temperature=0.2, default_max_tokens=99
    )
    traj_path = tmp_path / "trajectories.jsonl"
    args = _args(output_path=None, dataset="d", config=None, training_config=contract.path)
    env = SimpleNamespace(max_turns=7, system_prompt="sp", get_tools_schema=lambda: None)

    write_eval_outputs(
        args,
        [],
        env=env,
        traj_path=str(traj_path),
        env_type="code_contests",
        split="train",
        rollout=rollout,
        num_samples=1,
    )

    meta = json.loads(traj_path.read_text().splitlines()[0])
    assert meta["split"] == "train"
    assert meta["max_turns"] == 7, "the meta line records the cap the env resolved"
    assert meta["training_config"] == contract.path
    assert meta["rollout"]["chat_template_kwargs"] == {"preserve_thinking": True}
    assert meta["rollout"]["stop_token_ids"] == [_CALL_TOKEN_ID]
    assert meta["rollout"]["max_thinking_tokens"] == 2048
    assert meta["rollout"]["max_episode_tokens"] == 16384
    assert meta["rollout"]["temperature"] == 1.0 and meta["rollout"]["max_tokens"] == 4096


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
