#!/usr/bin/env python
"""The engine rosters spell shared model types once: the Step-3 spellings come off the family's EP
layer class, the two Bailing loader gaps off one table both clients quote for their own release.

    python tests/cpu/grpo/test_weight_sync_unservable_roster_derivation.py
"""

import pytest

from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.distributed.nccl.clients.base import unregistered_bailing_model_types
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient


def test_sglang_refuses_every_step3_spelling_the_trainer_knows_with_one_fact():
    """A spelling added to the class must be refused too, or the gate admits it into a dead sync."""
    facts = {SGLangWeightSyncClient.UNSERVABLE_MODEL_TYPES[mt] for mt in EPStep3p7MoELayer.HF_MODEL_TYPES}
    assert len(facts) == 1 and "step3p5 loader" in next(iter(facts))


def test_both_clients_quote_the_shared_bailing_gaps_for_their_own_release():
    shared = set(unregistered_bailing_model_types("x"))
    assert shared == {"bailing_hybrid", "bailing_moe_linear"}
    assert shared < set(EPBailingMoELayer.HF_MODEL_TYPES), "the gaps are spellings the toolkit trains"
    for client, release in ((SGLangWeightSyncClient, "SGLang 0.5.17"), (VLLMWeightSyncClient, "vLLM 0.26.0")):
        for model_type in shared:
            assert client.UNSERVABLE_MODEL_TYPES[model_type] == unregistered_bailing_model_types(release)[model_type]
            assert release in client.UNSERVABLE_MODEL_TYPES[model_type]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
