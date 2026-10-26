#!/usr/bin/env python
"""The engine rosters spell shared facts once: the Step-3 spellings come off the family spec,
the two Bailing gaps off one table both clients quote, and the release off each client's
``ENGINE_RELEASE``; the DeepSeek-V4 fact names the per-expert keys the spec declares, and only vLLM declares
a thinking budget it cannot enforce.

    python tests/cpu/weight_sync/test_unservable_roster_derivation.py
"""

import pytest

from halo.models.moe.family_specs import BAILING, DEEPSEEK_V4, STEP3P7
from halo.weight_sync.clients.base import unservable_bailing_model_types
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient

_CLIENTS = (VLLMWeightSyncClient, SGLangWeightSyncClient)


def test_sglang_refuses_every_step3_spelling_the_trainer_knows_with_one_fact():
    """A spelling added to the spec must be refused too, or the gate admits it into a dead sync."""
    facts = {SGLangWeightSyncClient.UNSERVABLE_MODEL_TYPES[mt] for mt in STEP3P7.model_types}
    assert len(facts) == 1 and "step3p5 loader" in next(iter(facts))


def test_both_clients_quote_the_shared_bailing_gaps_for_their_own_release():
    shared = set(unservable_bailing_model_types("x"))
    assert shared == {"bailing_hybrid", "bailing_moe_linear"}
    assert shared < set(BAILING.model_types), "the gaps are spellings the toolkit trains"
    for client in _CLIENTS:
        quoted = unservable_bailing_model_types(client.ENGINE_RELEASE)
        for model_type in shared:
            assert client.UNSERVABLE_MODEL_TYPES[model_type] == quoted[model_type]


@pytest.mark.parametrize("roster", ["UNSERVABLE_MODEL_TYPES", "UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES"])
@pytest.mark.parametrize("client", _CLIENTS, ids=lambda client: client.BACKEND_KEY)
def test_every_roster_fact_names_the_release_its_client_declares(client, roster):
    """Each fact describes one pinned server, so each one quotes ``ENGINE_RELEASE``, the value the stack
    version gate holds to the image pin; a release spelled by hand inside one fact drifts on a bump."""
    assert client.ENGINE_RELEASE.startswith(f"{client.BACKEND_NAME} ")
    stale = {
        model_type: fact for model_type, fact in getattr(client, roster).items() if client.ENGINE_RELEASE not in fact
    }
    assert not stale, f"{client.__name__}.{roster} facts that do not name {client.ENGINE_RELEASE!r}: {stale}"


def test_vllm_alone_cannot_enforce_a_mistral4_thinking_budget():
    """vLLM serves Mistral 4 with its mistral_common tokenizer, which encodes the budget markers as plain text;
    the SGLang path enforces no budget through those markers, so it declares none."""
    assert set(VLLMWeightSyncClient.UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES) == {"mistral4"}
    assert not SGLangWeightSyncClient.UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES


@pytest.mark.parametrize("client", _CLIENTS, ids=lambda client: client.BACKEND_KEY)
def test_the_deepseek_v4_fact_names_the_per_expert_keys_against_the_fused_gather(client):
    """Both pinned loaders map the release's per-expert names, which the spec declares, while the gather emits the
    fused pair; a fact naming another layout sends a reader after a gap the engine no longer has."""
    fact = client.UNSERVABLE_MODEL_TYPES["deepseek_v4"]
    assert all(key in fact for key in DEEPSEEK_V4.expert_layout.per_expert_keys), fact
    assert "fused pair" in fact


def test_vllm_alone_drops_the_laguna_balancing_bias():
    """vLLM's Laguna loader skips the exported balancing bias; the run warning names the dropping engine by
    its ``ENGINE_RELEASE``, read off this roster."""
    assert {"laguna"} == VLLMWeightSyncClient.DROPPED_BALANCING_BIAS_MODEL_TYPES
    assert frozenset() == SGLangWeightSyncClient.DROPPED_BALANCING_BIAS_MODEL_TYPES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
