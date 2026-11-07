#!/usr/bin/env python
"""``rollout_weight_precision: fp8_block`` is refused where the push cannot honor it: beside adapter sync, on a family without a GPU row, and from the script's first read of the checkpoint config."""

import pytest
from transformers import LlamaConfig, Qwen3Config, Qwen3MoeConfig

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_config
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient


def test_the_default_is_bf16_on_both_on_policy_arms():
    assert AsyncRolloutConfig().rollout_weight_precision == "bf16"
    assert OnlineGRPOScriptArguments().rollout_weight_precision == "bf16"


def test_fp8_beside_adapter_sync_is_refused():
    with pytest.raises(ValueError, match="rollout_adapter_sync"):
        AsyncRolloutConfig(rollout_weight_precision="fp8_block", rollout_adapter_sync=True)
    AsyncRolloutConfig(rollout_weight_precision="fp8_block")


@pytest.mark.parametrize("config", [Qwen3Config(), Qwen3MoeConfig()], ids=["qwen3", "qwen3-moe"])
def test_a_validated_family_on_vllm_passes(config):
    reject_unsyncable_config(config, "vllm", "fp8_block")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_both_engines_take_block_fp8(backend):
    reject_unsyncable_config(Qwen3Config(), backend, "fp8_block")


def test_a_failed_sglang_chunk_names_the_deepgemm_remedy_only_for_the_engines_partial_update_answer():
    refused = RuntimeError("Failed to update weights: partially updated, discard the whole weights")
    assert "SGLANG_ENABLE_JIT_DEEPGEMM=0" in SGLangWeightSyncClient.chunk_failure_remedy([refused])
    assert SGLangWeightSyncClient.chunk_failure_remedy([RuntimeError("connection reset")]) == ""
    assert VLLMWeightSyncClient.chunk_failure_remedy([refused]) == ""


def test_a_family_without_a_gpu_row_is_refused_with_its_reason():
    with pytest.raises(ValueError, match="no GPU row has validated block-FP8 rollouts"):
        reject_unsyncable_config(LlamaConfig(), "vllm", "fp8_block")
    reject_unsyncable_config(LlamaConfig(), "vllm", "bf16")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
