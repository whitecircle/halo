#!/usr/bin/env python
"""The engine rosters spell shared facts once: the Step-3 spellings come off the family's EP layer class,
the two Bailing loader gaps off one table both clients quote, and the release off each client's
``ENGINE_RELEASE``.

    python tests/cpu/weight_sync/test_unservable_roster_derivation.py
"""

import pytest

from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.weight_sync.clients.base import unregistered_bailing_model_types
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.clients.vllm import VLLMWeightSyncClient

_CLIENTS = (VLLMWeightSyncClient, SGLangWeightSyncClient)


def test_sglang_refuses_every_step3_spelling_the_trainer_knows_with_one_fact():
    """A spelling added to the class must be refused too, or the gate admits it into a dead sync."""
    facts = {SGLangWeightSyncClient.UNSERVABLE_MODEL_TYPES[mt] for mt in EPStep3p7MoELayer.HF_MODEL_TYPES}
    assert len(facts) == 1 and "step3p5 loader" in next(iter(facts))


def test_both_clients_quote_the_shared_bailing_gaps_for_their_own_release():
    shared = set(unregistered_bailing_model_types("x"))
    assert shared == {"bailing_hybrid", "bailing_moe_linear"}
    assert shared < set(EPBailingMoELayer.HF_MODEL_TYPES), "the gaps are spellings the toolkit trains"
    for client in _CLIENTS:
        quoted = unregistered_bailing_model_types(client.ENGINE_RELEASE)
        for model_type in shared:
            assert client.UNSERVABLE_MODEL_TYPES[model_type] == quoted[model_type]


@pytest.mark.parametrize("client", _CLIENTS, ids=lambda client: client.BACKEND_KEY)
def test_every_roster_fact_names_the_release_its_client_declares(client):
    """Each fact describes one pinned server, so each one quotes ``ENGINE_RELEASE``, the value the stack
    version gate holds to the image pin; a release spelled by hand inside one fact drifts on a bump."""
    assert client.ENGINE_RELEASE.startswith(f"{client.BACKEND_NAME} ")
    stale = {
        model_type: fact
        for model_type, fact in client.UNSERVABLE_MODEL_TYPES.items()
        if client.ENGINE_RELEASE not in fact
    }
    assert not stale, f"{client.__name__} roster facts that do not name {client.ENGINE_RELEASE!r}: {stale}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
