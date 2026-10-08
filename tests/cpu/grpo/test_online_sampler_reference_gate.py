#!/usr/bin/env python
"""The online arm's share of the sampler-logprob preflight: a sequence-level IS mode, and VESPO's
sequence weight in any mode, sum the per-token log-ratios, so a nucleus-renormalized reference is
refused for them exactly as for the env trainer's geometric band.

TRL's ``sequence_mask`` (its default) multiplies the sequence loss by ``exp(Σ(old − sampling))``.
Against vLLM ``processed_logprobs`` with ``top_p < 1`` every uncertain token's sampling logprob is
renormalized over its nucleus — lifted — so the sum drives the ratio toward 0, and ``sequence_mask``
only zeroes ratios ABOVE the cap: the run stalls with no error. The gate is the CONSUMER ("sums
per-token log-ratios over a sequence"), not one trainer's knob, and the RLVR script feeds it from the
IS mode and the loss. Each refusal points each arm at the remedy it honors.

    python tests/cpu/grpo/test_online_sampler_reference_gate.py
"""

import ast
import re
import types

import pytest
import torch
from trl import GRPOTrainer

from src.configs.async_training_config import AsyncTrainingConfig
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout import weight_sync_clients
from src.trainers.grpo.rollout.weight_sync_clients import (
    verify_sampler_logprob_reference,
    verify_sampler_logprob_reference_synced,
)
from tests.common.utils import REPO_ROOT
from tests.cpu.grpo.test_weight_sync_protocol import FakeVLLMServer

RLVR_SCRIPT = REPO_ROOT / "scripts/training/online_grpo/rlvr.py"
ENV_SCRIPT = REPO_ROOT / "scripts/training/environmental_grpo.py"


def _preflight_argument(script, name: str = "sequence_ratio_active") -> str:
    """The source of the ``name=`` argument the script feeds the preflight, from its own AST."""
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "verify_sampler_logprob_reference_synced":
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
    """VESPO skips TRL's per-token IS multiply and folds the log of every token's ratio into one sequence
    weight, so it sums in either token mode TRL lets it run; with the correction off TRL hands it no ratio."""
    grpo_args = types.SimpleNamespace(
        vllm_importance_sampling_correction=correction, vllm_importance_sampling_mode=mode, loss_type=loss_type
    )
    assert DistributedGRPOTrainer.sums_sequence_logratio(grpo_args) is expected


def test_vespo_folds_every_tokens_engine_ratio_into_one_sequence_weight():
    """The premise behind counting VESPO, read off the installed TRL: a nucleus-lifted sampling reference
    (each token's ratio a little under 1) shrinks the whole sequence's weight with its length."""
    advantages, log_ratio, mask = torch.ones(1, 1), torch.zeros(1, 64), torch.ones(1, 64)
    lifted = GRPOTrainer.get_gamma_weights(advantages, log_ratio, mask, torch.full((1, 64), 0.8))
    exact = GRPOTrainer.get_gamma_weights(advantages, log_ratio, mask, torch.ones(1, 64))
    assert lifted.item() < 1e-6 * exact.item()


def test_a_nucleus_reference_is_refused_for_any_sequence_summing_consumer(nucleus_server):
    """The refusal names both remedies, since either consumer may be the one summing."""
    with pytest.raises(ValueError, match="top_p: 1.0") as excinfo:
        verify_sampler_logprob_reference(
            VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=True
        )
    message = str(excinfo.value)
    assert "vllm_importance_sampling_mode" in message and "isr_geo_band" in message, message
    # A token-level consumer takes the ratio per token: the lift cancels nothing, so it passes.
    verify_sampler_logprob_reference(
        VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=False
    )


def test_both_refusals_point_each_arm_at_the_remedy_it_honors(nucleus_server):
    """Online GRPO takes the ratio per token through TRL's mode, under any loss but VESPO. The environmental
    correction is per token in every mode and loss its gates accept and refuses the others (``token_mask`` and
    ``vespo`` among them), so there the remedy is dropping the ``isr_*`` stages, the only consumers that sum."""
    per_arm = re.compile(
        r"on online GRPO a token_\* vllm_importance_sampling_mode with a loss_type other than vespo; on the "
        r"environmental arm[^;.]*drop isr_geo_band_min/max and isr_opsm_delta"
    )
    with pytest.raises(ValueError) as penalty:
        verify_sampler_logprob_reference(VLLMWeightSyncClient, [], 1.0, 1.0, 0, 0.0, 1.1, sequence_ratio_active=True)
    with pytest.raises(ValueError) as nucleus:
        verify_sampler_logprob_reference(
            VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, 0, 0.0, 1.0, sequence_ratio_active=True
        )
    for refusal in (penalty, nucleus):
        assert per_arm.search(str(refusal.value)), str(refusal.value)


def test_the_synced_form_takes_the_consumer_flag(nucleus_server):
    with pytest.raises(ValueError, match="summed over each sequence"):
        verify_sampler_logprob_reference_synced(
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
    """Every rank raises ``ValueError``; a failure other than the preflight's own refusal keeps its type in the
    text (``KeyError('url')`` alone reads ``'url'``), and rank 0 chains the original."""

    def fail(*args):
        raise error

    monkeypatch.setattr(weight_sync_clients, "verify_sampler_logprob_reference", fail)
    with pytest.raises(ValueError, match=message) as raised:
        verify_sampler_logprob_reference_synced(
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
    """The script must derive the flag from the config, not pin it: a pinned False leaves TRL's
    default ``sequence_mask`` running against a renormalized reference unchecked."""
    assert _preflight_argument(RLVR_SCRIPT) == "DistributedGRPOTrainer.sums_sequence_logratio(grpo_config)"


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
    """OPSM sums the per-token log-ratios over a trajectory exactly as the geometric band does
    (``apply_opsm`` thresholds ``|mean log-ratio|``), so a nucleus-renormalized reference biases it the
    same way. The env script's flag is evaluated from its own source: an expression naming only the
    geometric band leaves an OPSM-only run's reference unchecked."""
    async_config = AsyncTrainingConfig(**knobs)
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
        ({"isr_geo_band_min": 1.01, "isr_geo_band_max": 1.02}, "0 < min < 1 < max"),
        ({"isr_veto_min": 1.5}, "isr_veto_min"),
        ({"isr_opsm_delta": 0.0}, "isr_opsm_delta"),
        ({"isr_opsm_delta": float("nan")}, "isr_opsm_delta"),
    ],
)
def test_the_isr_bounds_are_refused_at_parse(knobs, match):
    """A bad bound fails before the model loads, and a lone upper band bound never reaches the sampler
    preflight reading as no sequence consumer."""
    with pytest.raises(ValueError, match=match):
        AsyncTrainingConfig(**knobs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
