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

from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.grpo.rollout.weight_sync_clients import (
    verify_sampler_logprob_reference,
    verify_sampler_logprob_reference_synced,
)
from tests.common.utils import REPO_ROOT
from tests.cpu.grpo.test_weight_sync_protocol import FakeVLLMServer

RLVR_SCRIPT = REPO_ROOT / "scripts/training/online_grpo/rlvr.py"


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
    for node in ast.walk(ast.parse(RLVR_SCRIPT.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "verify_sampler_logprob_reference_synced":
            keywords = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            assert keywords["sequence_ratio_active"] == (
                "DistributedGRPOTrainer.sequence_level_importance_sampling(grpo_config)"
            ), keywords
            return
    raise AssertionError("rlvr.py no longer runs the sampler-logprob preflight")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
