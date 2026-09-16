#!/usr/bin/env python
"""``--training_config`` makes an eval sample under the generation contract the policy was trained with.

The shared eval flags carry temperature, top-p, max tokens and a request timeout; the training
rollout also fixes chat-template variables, stop tokens, a thinking budget and the backend. A policy
trained with ``preserve_thinking`` or ``rollout_stop_tokens`` and evaluated without them is measured
under a different contract, and the meta line recorded three knobs of it.

    python tests/cpu/environments/test_eval_scripts_training_contract.py
"""

import argparse
import json
from types import SimpleNamespace

import pytest

import scripts.environments._common as common
from scripts.environments._common import (
    TrainingContract,
    load_training_contract,
    rollout_config_from_args,
    write_eval_outputs,
)
from src.configs.rollout_config import DEFAULT_ROLLOUT_TOP_P
from src.environments.eval_runner import DEFAULT_REQUEST_TIMEOUT_S

_CALL_TOKEN_ID = 200012

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
    assert contract.env_config_dict() == {
        "reward_terms": [{"source": "environment"}],
        "max_turns": 7,
        "language": "cpp",
        "timeout_per_test": 3,
    }


def test_an_unresolvable_stop_token_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    path.write_text(_TRAINING_YAML.replace('["<|call|>"]', '["<|call|>", "<|nope|>"]'))
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer())
    with pytest.raises(ValueError, match="<\\|nope\\|>"):
        TrainingContract.load(str(path))


def test_the_meta_line_records_the_whole_generation_contract(contract, tmp_path):
    rollout = rollout_config_from_args(_args(), contract, default_temperature=0.2, default_max_tokens=99)
    traj_path = tmp_path / "trajectories.jsonl"
    args = _args(output=None, dataset="d", config=None, split="test", training_config=contract.path)
    env = SimpleNamespace(max_turns=7, system_prompt="sp", get_tools_schema=lambda: None)

    write_eval_outputs(
        args,
        [],
        env=env,
        traj_path=str(traj_path),
        env_type="code_contests",
        max_turns=None,
        rollout=rollout,
        num_samples=1,
    )

    meta = json.loads(traj_path.read_text().splitlines()[0])
    assert meta["training_config"] == contract.path
    assert meta["rollout"]["chat_template_kwargs"] == {"preserve_thinking": True}
    assert meta["rollout"]["stop_token_ids"] == [_CALL_TOKEN_ID]
    assert meta["rollout"]["max_thinking_tokens"] == 2048
    assert meta["rollout"]["temperature"] == 1.0 and meta["rollout"]["max_tokens"] == 4096


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
