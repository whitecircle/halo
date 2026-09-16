#!/usr/bin/env python
"""The online arm's share of the sampler-logprob preflight: a sequence-level IS mode sums the per-token
log-ratios, so a nucleus-renormalized reference is refused for it exactly as for the env trainer's
geometric band.

TRL's ``sequence_mask`` (its default) multiplies the sequence loss by ``exp(Σ(old − sampling))``.
Against vLLM ``processed_logprobs`` with ``top_p < 1`` every uncertain token's sampling logprob is
renormalized over its nucleus — lifted — so the sum drives the ratio toward 0, and ``sequence_mask``
only zeroes ratios ABOVE the cap: the run stalls with no error. The gate is the CONSUMER ("sums
per-token log-ratios over a sequence"), not one trainer's knob, and the RLVR script feeds it from the
IS mode.

    python tests/cpu/grpo/test_online_sampler_reference_gate.py
"""

import ast
import types

import pytest

from src.configs.async_training_config import AsyncTrainingConfig
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.weight_sync_clients import (
    verify_sampler_logprob_reference,
    verify_sampler_logprob_reference_synced,
)
from tests.common.utils import REPO_ROOT
from tests.cpu.grpo.test_weight_sync_protocol import FakeVLLMServer

RLVR_SCRIPT = REPO_ROOT / "scripts/training/online_grpo/rlvr.py"
ENV_SCRIPT = REPO_ROOT / "scripts/training/environmental_grpo.py"


def _gate_expression(script) -> str:
    """The ``sequence_ratio_active=`` source the script feeds the preflight, from its own AST."""
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "verify_sampler_logprob_reference_synced":
            for keyword in node.keywords:
                if keyword.arg == "sequence_ratio_active":
                    return ast.unparse(keyword.value)
    raise AssertionError(f"{script.name} no longer runs the sampler-logprob preflight")


@pytest.fixture
def nucleus_server():
    server = FakeVLLMServer()
    server.logprobs_mode = "processed_logprobs"
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize(
    ("correction", "mode", "expected"),
    [
        (True, "sequence_mask", True),
        (True, "sequence_truncate", True),
        (True, "token_mask", False),
        (True, "token_truncate", False),
        (False, "sequence_mask", False),
    ],
)
def test_sequence_level_is_follows_the_mode_and_the_correction_switch(correction, mode, expected):
    grpo_args = types.SimpleNamespace(
        vllm_importance_sampling_correction=correction, vllm_importance_sampling_mode=mode
    )
    assert DistributedGRPOTrainer.sequence_level_importance_sampling(grpo_args) is expected


def test_a_nucleus_reference_is_refused_for_any_sequence_summing_consumer(nucleus_server):
    """The refusal names both remedies, since either consumer may be the one summing."""
    with pytest.raises(ValueError, match="top_p: 1.0") as excinfo:
        verify_sampler_logprob_reference(
            VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, sequence_ratio_active=True
        )
    message = str(excinfo.value)
    assert "vllm_importance_sampling_mode" in message and "isr_geo_band" in message, message
    # A token-level consumer takes the ratio per token: the lift cancels nothing, so it passes.
    verify_sampler_logprob_reference(
        VLLMWeightSyncClient, [nucleus_server.url], 1.0, 0.95, sequence_ratio_active=False
    )


def test_the_synced_form_takes_the_consumer_flag_under_its_new_name(nucleus_server):
    with pytest.raises(ValueError, match="summed over each sequence"):
        verify_sampler_logprob_reference_synced(
            [nucleus_server.url],
            temperature=1.0,
            top_p=0.95,
            sequence_ratio_active=True,
            backend=VLLMWeightSyncClient.BACKEND_KEY,
        )


def test_the_rlvr_script_feeds_the_gate_from_the_is_mode():
    """The script must derive the flag from the config, not pin it — a pinned False is the bug this
    fixes: TRL's default ``sequence_mask`` ran against a renormalized reference unchecked."""
    assert _gate_expression(RLVR_SCRIPT) == "DistributedGRPOTrainer.sequence_level_importance_sampling(grpo_config)"


@pytest.mark.parametrize(
    ("knobs", "expected"),
    [
        ({}, False),
        ({"isr_geo_band_min": 0.99, "isr_geo_band_max": 1.01}, True),
        ({"isr_opsm_delta": 0.1}, True),
    ],
    ids=["neither", "geo_band_only", "opsm_only"],
)
def test_the_env_script_arms_the_gate_for_every_sequence_summing_consumer(knobs, expected):
    """OPSM sums the per-token log-ratios over a trajectory exactly as the geometric band does
    (``apply_opsm`` thresholds ``|mean log-ratio|``), so a nucleus-renormalized reference biases it the
    same way. The env script's flag is evaluated from its own source: an expression naming only the
    geometric band leaves an OPSM-only run's reference unchecked."""
    async_config = AsyncTrainingConfig(**knobs)
    assert eval(_gate_expression(ENV_SCRIPT), {"async_config": async_config}) is expected  # noqa: S307


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
