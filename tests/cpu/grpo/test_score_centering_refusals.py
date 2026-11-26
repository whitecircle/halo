#!/usr/bin/env python
"""Every setup score centering cannot honor is refused at construction, the rollout and scorer ones also by the script
ahead of its preflights; each reason named, several at once."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.environmental_rules import (
    reject_unsupported_score_centering,
    reject_unsupported_score_centering_loss,
)
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.repo import REPO_ROOT

_SUPPORTED = {
    "score_centering_top_k": 16,
    "rollout_backend": "sglang",
    "train_on_sampled_tokens": True,
    "use_chunked_grpo_logprobs": True,
    "rollout_top_p": 1.0,
}


def _args(**overrides):
    return SimpleNamespace(**{"num_iterations": 1, "beta": 0.0, "importance_sampling_level": "token", **overrides})


def _reject(overrides=None, args=None, misaligned_accumulation=False):
    config = AsyncRolloutConfig(**{**_SUPPORTED, **(overrides or {})})
    reject_unsupported_score_centering(config)
    reject_unsupported_score_centering_loss(
        config.score_centering_top_k, _args(**(args or {})), misaligned_accumulation=misaligned_accumulation
    )


def test_a_supported_run_and_a_run_with_the_knob_off_pass():
    _reject()
    off = AsyncRolloutConfig(rollout_backend="vllm", train_on_sampled_tokens=False, rollout_top_p=0.9)
    reject_unsupported_score_centering(off)


@pytest.mark.parametrize(
    ("overrides", "args", "match"),
    [
        ({"rollout_backend": "vllm"}, {}, "single API-server process"),
        ({"train_on_sampled_tokens": False}, {}, "train_on_sampled_tokens: false"),
        ({"rollout_top_p": 0.9}, {}, r"\{'rollout_top_p': 0.9\}"),
        ({"rollout_top_k": 20}, {}, r"\{'rollout_top_k': 20\}"),
        ({"rollout_min_p": 0.05}, {}, r"\{'rollout_min_p': 0.05\}"),
        ({"rollout_repetition_penalty": 1.1}, {}, r"\{'rollout_repetition_penalty': 1.1\}"),
        ({"use_chunked_grpo_logprobs": False}, {}, "use_chunked_grpo_logprobs: false"),
        ({"loss_forward_importance_weights": True}, {}, "loss_forward_importance_weights"),
        ({"balance_token_mass": True}, {}, "balance_token_mass"),
        ({"isr_engine_reference": True}, {}, "isr_engine_reference"),
        ({}, {"num_iterations": 2}, "num_iterations=2"),
        ({}, {"beta": 0.04}, "beta=0.04"),
        ({}, {"importance_sampling_level": "sequence"}, "importance_sampling_level='sequence'"),
    ],
)
def test_each_unsupported_setup_is_refused_naming_the_setting(overrides, args, match):
    with pytest.raises(ValueError, match=match):
        _reject(overrides, args)


def test_accumulation_straddling_a_round_is_refused():
    with pytest.raises(ValueError, match="straddling a generation round"):
        _reject(misaligned_accumulation=True)


def test_a_temperature_alone_leaves_the_sampler_plain():
    _reject({"rollout_temperature": 0.7})


def test_several_reasons_are_reported_together():
    with pytest.raises(ValueError, match=r"rollout_backend: 'vllm'.*train_on_sampled_tokens: false"):
        _reject({"rollout_backend": "vllm", "train_on_sampled_tokens": False})
    with pytest.raises(ValueError, match=r"num_iterations=2.*beta=0.04"):
        _reject(args={"num_iterations": 2, "beta": 0.04})


def _called(method) -> set[str]:
    source = textwrap.dedent(inspect.getsource(method))
    return {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_the_trainer_and_the_script_both_refuse_and_the_script_ahead_of_its_preflights():
    """A dropped trainer call leaves a hand-built trainer unrefused; a script call after the preflights sends a vLLM run
    into the candidate probe, whose refusal names the wrong remedy."""
    trainer = DistributedAsyncEnvironmentalGRPOTrainer
    assert "reject_unsupported_score_centering" in _called(trainer._configure_rollout_state)
    assert "reject_unsupported_score_centering_loss" in _called(trainer._configure_score_centering)
    first_line = {}
    for node in ast.walk(ast.parse((REPO_ROOT / "scripts/training/grpo/environmental.py").read_text())):
        if isinstance(node, ast.Call):
            first_line.setdefault(ast.unparse(node.func), node.lineno)
    assert first_line["reject_unsupported_score_centering"] < first_line["verify_server_flags"]
    assert first_line["reject_unsupported_score_centering"] < first_line["load_model_and_tokenizer"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
