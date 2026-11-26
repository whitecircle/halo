#!/usr/bin/env python
"""``packed_microbatch_tokens`` is one field of one bundle that both on-policy GRPO configs mix in, validated there, and
the online script hands it to the trainer."""

import ast
import dataclasses

import pytest

from halo.configs.grpo import PackedMicrobatchArguments
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from tests.common.repo import REPO_ROOT

_KNOB = "packed_microbatch_tokens"
_CONFIGS = (OnlineGRPOScriptArguments, AsyncRolloutConfig)


@pytest.mark.parametrize("config", _CONFIGS)
def test_both_configs_take_the_knob_from_the_bundle_alone(config):
    assert issubclass(config, PackedMicrobatchArguments)
    (field,) = [f for f in dataclasses.fields(config) if f.name == _KNOB]
    (bundled,) = [f for f in dataclasses.fields(PackedMicrobatchArguments) if f.name == _KNOB]
    assert field.default is None and field.metadata is bundled.metadata


@pytest.mark.parametrize("config", _CONFIGS)
def test_the_budget_is_unset_or_an_integer_of_at_least_two(config):
    assert getattr(config(), _KNOB) is None
    assert getattr(config(**{_KNOB: 2}), _KNOB) == 2
    for bad in (1, 0, -8, 4096.0, True):
        with pytest.raises(ValueError, match=_KNOB):
            config(**{_KNOB: bad})


def test_the_online_script_hands_the_knob_to_the_trainer():
    tree = ast.parse((REPO_ROOT / "scripts" / "training" / "grpo" / "online.py").read_text())
    forwarded = {
        keyword.arg: ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "trainer_cls"
        for keyword in node.keywords
    }
    assert forwarded[_KNOB] == f"args.{_KNOB}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
