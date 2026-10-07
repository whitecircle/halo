#!/usr/bin/env python
"""``--training_config`` makes an eval sample under the generation contract the policy was trained with.

The shared eval flags carry temperature, top-p, max tokens and a request timeout; the training
rollout also fixes chat-template variables, stop tokens, a thinking budget and the backend. A policy
trained with ``preserve_thinking`` or ``rollout_stop_tokens`` and evaluated without them is measured
under a different contract; the meta line records the whole contract.

    python tests/cpu/environments/test_eval_scripts_training_contract.py
"""

import argparse
import dataclasses
import json
from types import SimpleNamespace

import pytest

import scripts.environments._common as common
import src.environments.episode as episode
from scripts.environments._common import (
    TrainingContract,
    load_training_contract,
    rollout_config_from_args,
    write_eval_outputs,
)
from scripts.environments.inference.run_code_contests import CODING_ENV_TYPES, resolve_env_config
from src.configs.rollout_config import DEFAULT_ROLLOUT_TOP_P
from src.env import resolve_nccl_timeout_minutes
from src.environments.envs.tasks.coding.code_contests import DEFAULT_REASONING_EFFORT
from src.environments.eval_runner import DEFAULT_REQUEST_TIMEOUT_S
from src.environments.registry import resolve_environment
from tests.common.code_contests import StubSandbox
from tests.common.utils import REPO_ROOT

_CALL_TOKEN_ID = 200012
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

    def convert_tokens_to_ids(self, name):
        return _CALL_TOKEN_ID if name == "<|call|>" else None


@pytest.fixture
def contract(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML)
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    return TrainingContract.load(str(path))


def _args(**overrides):
    flags = {"model": "served-name", "temperature": None, "top_p": None, "max_tokens": None, "request_timeout": None}
    return argparse.Namespace(**{**flags, **overrides})


def test_the_yaml_rollout_contract_reaches_the_eval(contract):
    rollout = rollout_config_from_args(_args(), contract, default_temperature=0.2, default_max_tokens=99)
    assert rollout.chat_template_kwargs == {"preserve_thinking": True}
    assert rollout.stop_token_ids == [_CALL_TOKEN_ID]
    assert rollout.max_thinking_tokens == 2048
    assert rollout.max_episode_tokens == 16384, "the episode output budget is part of the trained contract"
    assert rollout.backend == "vllm"
    assert (rollout.temperature, rollout.top_p, rollout.max_tokens, rollout.request_timeout) == (1.0, 1.0, 4096, 300.0)
    assert rollout.model_name == "served-name", "the served name comes from --model, never the YAML"
    assert not rollout.capture_token_ids and not rollout.capture_routed_experts, "the eval transport captures nothing"


def test_an_explicit_sampling_flag_overrides_the_yaml_alone(contract):
    rollout = rollout_config_from_args(
        _args(temperature=0.3, max_tokens=512), contract, default_temperature=0.2, default_max_tokens=99
    )
    assert (rollout.temperature, rollout.max_tokens) == (0.3, 512)
    assert (rollout.top_p, rollout.request_timeout) == (1.0, 300.0)
    assert rollout.stop_token_ids == [_CALL_TOKEN_ID]


def test_without_the_flag_the_script_defaults_stand():
    assert load_training_contract(None) is None
    rollout = rollout_config_from_args(_args(), None, default_temperature=0.2, default_max_tokens=99)
    assert (rollout.temperature, rollout.top_p, rollout.max_tokens, rollout.request_timeout) == (
        0.2,
        DEFAULT_ROLLOUT_TOP_P,
        99,
        DEFAULT_REQUEST_TIMEOUT_S,
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
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    environment_type = TrainingContract.load(str(path)).env_config.environment_type
    assert environment_type == "code_contests" and environment_type in CODING_ENV_TYPES


@pytest.fixture
def default_watchdog_recipe(monkeypatch):
    """The codeforces recipe's contract under the default watchdog, which its episode_timeout exceeds."""
    monkeypatch.delenv("DIST_NCCL_TIMEOUT_MINUTES", raising=False)
    contract = TrainingContract.load(str(_CODEFORCES_RECIPE))
    assert contract.async_config.episode_timeout > resolve_nccl_timeout_minutes() * 60
    return contract


def test_a_shipped_recipe_evaluates_without_the_training_watchdog(default_watchdog_recipe):
    """The eval joins no process group, so the recipe's contract builds on the default watchdog."""
    rollout = default_watchdog_recipe.rollout_config()
    assert rollout.episode_timeout == default_watchdog_recipe.async_config.episode_timeout
    trained_budget = default_watchdog_recipe.async_config.rollout_max_episode_tokens
    assert trained_budget is not None and rollout.max_episode_tokens == trained_budget


def test_training_still_refuses_the_recipe_on_the_default_watchdog(default_watchdog_recipe):
    with pytest.raises(ValueError, match="NCCL collective watchdog"):
        default_watchdog_recipe.async_config.get_rollout_config()


def test_an_unresolvable_stop_token_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace('["<|call|>"]', '["<|call|>", "<|nope|>"]'))
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    with pytest.raises(ValueError, match="<\\|nope\\|>"):
        TrainingContract.load(str(path))


def test_the_contract_loads_the_tokenizer_for_the_stop_tokens_alone(tmp_path, monkeypatch):
    """Only the stop tokens go through the tokenizer: a contract without them never loads one, and no
    reasoning-end id is resolved for the eval (the trainer's overlong charge is the one reader of it)."""
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace('rollout_stop_tokens: ["<|call|>"]\n', ""))

    def never(*a, **k):
        raise AssertionError("the contract loaded a tokenizer with no stop tokens to resolve")

    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", never)
    contract = TrainingContract.load(str(path))
    assert contract.stop_token_ids is None
    assert "reasoning_end_token_id" not in {f.name for f in dataclasses.fields(contract)}
    assert contract.rollout_config().reasoning_end_token_id is None


def test_the_eval_resolves_stop_tokens_as_the_trainer_does():
    """One resolver, one policy: a recipe the trainer accepts is one its eval accepts."""
    assert common.resolve_rollout_stop_token_ids is episode.resolve_rollout_stop_token_ids


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
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    contract = TrainingContract.load(str(path))
    trained_env = contract.env_config.to_env_config()
    training = resolve_environment(contract.env_config.environment_type, {**trained_env, "sandbox": StubSandbox()})
    flags = SimpleNamespace(eval_protocol=None, language=None, reasoning_effort=None, max_turns=None)
    assert resolve_env_config(flags, trained_env, {})["reasoning_effort"] == training.reasoning_effort == level


def test_the_meta_line_records_the_whole_generation_contract(contract, tmp_path):
    rollout = rollout_config_from_args(_args(), contract, default_temperature=0.2, default_max_tokens=99)
    traj_path = tmp_path / "trajectories.jsonl"
    args = _args(output=None, dataset="d", config=None, training_config=contract.path)
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
