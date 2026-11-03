#!/usr/bin/env python
"""The online arm's sampler-logprob preflight. A nucleus-renormalized reference lifts each token's sampling logprob,
so a sequence-level IS mode or VESPO, which sum the log-ratios, drive the ratio to 0 and the run stalls silently."""

import ast
import re
import types

import pytest
import torch
from trl import GRPOTrainer

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.rollout import engine_preflight
from halo.trainers.grpo.rollout.engine_preflight import (
    _verify_sampler_logprob_reference_on_rank0,
    verify_sampler_logprob_reference,
)
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import FakeVLLMServer
from tests.common.repo import REPO_ROOT

RLVR_SCRIPT = REPO_ROOT / "scripts/training/grpo/online.py"
ENV_SCRIPT = REPO_ROOT / "scripts/training/grpo/environmental.py"


def _preflight_argument(script, name: str = "sequence_ratio_active") -> str:
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "verify_sampler_logprob_reference":
            for keyword in node.keywords:
                if keyword.arg == name:
                    return ast.unparse(keyword.value)
    raise AssertionError(f"{script.name} no longer runs the sampler-logprob preflight with {name}=")


@pytest.fixture
def nucleus_server():
    server = FakeVLLMServer()
    server.logprobs_mode = "processed_logprobs"
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize(
    ("correction", "mode", "loss_type", "expected"),
    [
        (True, "sequence_mask", "dapo", True),
        (True, "sequence_truncate", "dapo", True),
        (True, "token_mask", "dapo", False),
        (True, "token_truncate", "dapo", False),
        (False, "sequence_mask", "dapo", False),
        (True, "token_truncate", "vespo", True),
        (True, "token_mask", "vespo", True),
        (False, "token_truncate", "vespo", False),
    ],
)
def test_the_sequence_sum_follows_the_mode_the_loss_and_the_correction_switch(correction, mode, loss_type, expected):
    """VESPO skips TRL's per-token IS multiply and folds every token's log-ratio into one sequence weight, so it sums
    in either token mode; with the correction off TRL hands it no ratio."""
    grpo_args = types.SimpleNamespace(
        vllm_importance_sampling_correction=correction, vllm_importance_sampling_mode=mode, loss_type=loss_type
    )
    assert DistributedOnlineGRPOTrainer.sums_sequence_logratio(grpo_args) is expected


def test_vespo_folds_every_tokens_engine_ratio_into_one_sequence_weight():
    """The premise behind counting VESPO: a nucleus-lifted reference shrinks the sequence's weight with its length."""
    advantages, log_ratio, mask = torch.ones(1, 1), torch.zeros(1, 64), torch.ones(1, 64)
    lifted = GRPOTrainer.get_gamma_weights(advantages, log_ratio, mask, torch.full((1, 64), 0.8))
    exact = GRPOTrainer.get_gamma_weights(advantages, log_ratio, mask, torch.ones(1, 64))
    assert lifted.item() < 1e-6 * exact.item()


def test_a_nucleus_reference_is_refused_for_any_sequence_summing_consumer(nucleus_server):
    """The refusal names both remedies, since either consumer may be the one summing."""
    with pytest.raises(ValueError, match="top_p: 1.0") as excinfo:
        _verify_sampler_logprob_reference_on_rank0(
            VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=True
        )
    message = str(excinfo.value)
    assert "vllm_importance_sampling_mode" in message and "isr_geo_band" in message, message
    # A token-level consumer takes the ratio per token, so the lift cancels nothing and it passes.
    _verify_sampler_logprob_reference_on_rank0(
        VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=False
    )


def test_both_refusals_point_each_arm_at_the_remedy_it_honors(nucleus_server):
    """Online GRPO takes the ratio per token under any loss but VESPO. The environmental correction is per token in
    every mode and loss its gates accept, so its remedy is to drop the ``isr_*`` stages, the only ones that sum."""
    per_arm = re.compile(
        r"on online GRPO a token_\* vllm_importance_sampling_mode with a loss_type other than vespo; on the "
        r"environmental arm[^;.]*drop isr_geo_band_min/max and isr_opsm_delta"
    )
    with pytest.raises(ValueError) as penalty:
        _verify_sampler_logprob_reference_on_rank0(
            VLLMWeightSyncClient, [], 1.0, 1.0, 0, 0.0, 1.1, sequence_ratio_active=True
        )
    with pytest.raises(ValueError) as nucleus:
        _verify_sampler_logprob_reference_on_rank0(
            VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=True
        )
    for refusal in (penalty, nucleus):
        assert per_arm.search(str(refusal.value)), str(refusal.value)


def test_the_collective_form_takes_the_consumer_flag(nucleus_server):
    with pytest.raises(ValueError, match="summed over each sequence"):
        verify_sampler_logprob_reference(
            [nucleus_server.url],
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            sequence_ratio_active=True,
            backend=VLLMWeightSyncClient.BACKEND_KEY,
        )


@pytest.mark.parametrize(
    ("error", "message"),
    [(KeyError("url"), r"^KeyError: 'url'$"), (ValueError("refused"), r"^refused$")],
    ids=["other-failure", "refusal"],
)
def test_a_preflight_failure_reaches_every_rank_with_its_type(monkeypatch, error, message):
    """A failure other than the preflight's refusal keeps its type in the text; ``KeyError('url')`` reads ``'url'``."""

    def fail(*args):
        raise error

    monkeypatch.setattr(engine_preflight, "_verify_sampler_logprob_reference_on_rank0", fail)
    with pytest.raises(ValueError, match=message) as raised:
        verify_sampler_logprob_reference(
            ["http://unused"],
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            sequence_ratio_active=True,
            backend=VLLMWeightSyncClient.BACKEND_KEY,
        )
    assert raised.value.__cause__ is error


def test_the_rlvr_script_feeds_the_gate_from_the_is_mode():
    """A pinned False leaves TRL's default ``sequence_mask`` running against a renormalized reference unchecked."""
    assert _preflight_argument(RLVR_SCRIPT) == "DistributedOnlineGRPOTrainer.sums_sequence_logratio(grpo_config)"


@pytest.mark.parametrize(
    ("knobs", "expected"),
    [
        ({}, False),
        ({"isr_geo_band_min": 0.99, "isr_geo_band_max": 1.01}, True),
        ({"isr_opsm_delta": 0.1}, True),
        ({"isr_veto_min": 1e-4}, False),
    ],
    ids=["neither", "geo_band_only", "opsm_only", "veto_only"],
)
def test_the_env_script_arms_the_gate_for_every_sequence_summing_consumer(knobs, expected):
    """OPSM sums the per-token log-ratios as the geometric band does (``apply_opsm`` thresholds ``|mean log-ratio|``),
    so an expression naming only the band leaves an OPSM-only run's reference unchecked."""
    async_config = AsyncRolloutConfig(**knobs)
    assert eval(_preflight_argument(ENV_SCRIPT), {"async_config": async_config}) is expected  # noqa: S307


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (
            ENV_SCRIPT,
            {
                "top_k": "async_config.rollout_top_k",
                "min_p": "async_config.rollout_min_p",
                "repetition_penalty": "async_config.rollout_repetition_penalty",
            },
        ),
        (
            RLVR_SCRIPT,
            {
                "top_k": "grpo_config.top_k",
                "min_p": "0.0 if grpo_config.min_p is None else grpo_config.min_p",
                "repetition_penalty": "grpo_config.repetition_penalty",
            },
        ),
    ],
    ids=["env", "rlvr"],
)
def test_each_script_feeds_the_preflight_its_own_sampler_filters(script, expected):
    """A pinned off value here would pass every preflight test while a run's own cut went unchecked."""
    assert {name: _preflight_argument(script, name) for name in expected} == expected


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"isr_geo_band_max": 1.01}, "set together"),
        ({"isr_geo_band_min": 0.99}, "set together"),
        ({"isr_geo_band_min": 1.01, "isr_geo_band_max": 1.02}, r"isr_geo_band_min must be in \(0, 1\)"),
        ({"isr_geo_band_min": 0.99, "isr_geo_band_max": 0.999}, "isr_geo_band_max must be > 1"),
        ({"isr_veto_min": 1.5}, "isr_veto_min"),
        ({"isr_opsm_delta": 0.0}, "isr_opsm_delta"),
        ({"isr_opsm_delta": float("nan")}, "isr_opsm_delta"),
    ],
)
def test_the_isr_bounds_are_refused_at_parse(knobs, match):
    """A lone upper band bound must not reach the sampler preflight reading as no sequence consumer."""
    with pytest.raises(ValueError, match=match):
        AsyncRolloutConfig(**knobs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
