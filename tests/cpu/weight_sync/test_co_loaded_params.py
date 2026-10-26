#!/usr/bin/env python
"""CPU tests for the co-load groups a weight-sync client declares.

SGLang's MLA loaders fuse ``q_a_proj`` and ``kv_a_proj_with_mqa`` from a cache local to one
``load_weights`` call, so a half arriving in a request without the other is dropped without error;
every request is one chunk, and the chunk boundary falls wherever the byte budget fills. What is
pinned: a group a boundary would split waits whole for the next chunk; a buffer holding nothing but
such members sends nothing; a group the sync never completes refuses the close rather than closing
over a dropped half; and the groups are scoped to the modules the model declares, since the engine
fuses nothing where a half does not exist.

    python tests/cpu/weight_sync/test_co_loaded_params.py
"""

import pytest
import torch

import src.weight_sync.clients.base as base_module
from src.weight_sync.chunking import declared_co_load_groups, split_co_loaded
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.weight_sync import Wire, offline_client, push

GROUPS = (("self_attn.q_a_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),)
# Four bf16 elements: 8 bytes each, so a budget of 16 holds two.
_BUDGET = 16


def _param() -> torch.Tensor:
    return torch.zeros(4, dtype=torch.bfloat16)


def _layer(index: int, leaf: str) -> str:
    return f"model.layers.{index}.self_attn.{leaf}.weight"


def _names(chunk) -> list[str]:
    return [name for name, _ in chunk]


def _mla_payload() -> list[tuple[str, torch.Tensor]]:
    return [(_layer(0, leaf), _param()) for leaf in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")]


# The budget's cut falls between the halves, so the first one crosses into the next chunk.
_GROUPED_CHUNKS = [
    [_layer(0, "q_b_proj")],
    [_layer(0, "q_a_proj"), _layer(0, "kv_a_proj_with_mqa")],
    [_layer(0, "o_proj")],
]


def test_split_holds_only_the_members_of_an_incomplete_group():
    chunk = [
        (_layer(0, "q_a_proj"), _param()),
        (_layer(0, "q_b_proj"), _param()),
        (_layer(0, "kv_a_proj_with_mqa"), _param()),
        (_layer(1, "q_a_proj"), _param()),
    ]
    sendable, held = split_co_loaded(chunk, GROUPS)
    assert _names(sendable) == [_layer(0, "q_a_proj"), _layer(0, "q_b_proj"), _layer(0, "kv_a_proj_with_mqa")]
    assert _names(held) == [_layer(1, "q_a_proj")], "layer 1's pair is incomplete, so its half waits"


def test_a_suffix_claims_a_name_only_at_a_module_boundary():
    """``xself_attn.q_a_proj.weight`` is another module's tensor and must not join layer 0's group."""
    chunk = [("model.layers.0.xself_attn.q_a_proj.weight", _param()), (_layer(0, "q_a_proj"), _param())]
    sendable, held = split_co_loaded(chunk, GROUPS)
    assert _names(sendable) == ["model.layers.0.xself_attn.q_a_proj.weight"]
    assert _names(held) == [_layer(0, "q_a_proj")]


@pytest.fixture
def streamed(monkeypatch) -> tuple[SGLangWeightSyncClient, Wire]:
    """An SGLang client on a recording wire, with the chunk budget shrunk to two params."""
    monkeypatch.setattr(base_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    wire = Wire()
    return wire.attach(offline_client(SGLangWeightSyncClient)), wire


def test_a_pair_the_budget_would_split_moves_whole_into_the_next_chunk(streamed):
    client, wire = streamed
    push(client, _mla_payload())
    assert wire.chunk_names == _GROUPED_CHUNKS


def test_without_the_group_the_budget_cuts_between_the_halves(streamed):
    """The budget alone cuts between the halves, so the regrouping in ``_GROUPED_CHUNKS`` is the co-load
    group's doing."""
    client, wire = streamed
    client.scope_co_load_groups([])
    push(client, _mla_payload())
    assert wire.chunk_names == [
        [_layer(0, "q_a_proj"), _layer(0, "q_b_proj")],
        [_layer(0, "kv_a_proj_with_mqa"), _layer(0, "o_proj")],
    ]


def test_a_buffer_of_nothing_but_a_waiting_half_sends_nothing(streamed):
    client, wire = streamed
    client.update_named_param(_layer(0, "q_a_proj"), _param())
    client.flush_chunk()
    assert wire.chunks == [], "a lone half must wait for its partner, not go out as a chunk"
    assert _names(client.drain_param_buffer()) == [_layer(0, "q_a_proj")]


def test_the_close_refuses_a_pair_the_sync_never_completed(streamed):
    client, wire = streamed
    client.update_named_param(_layer(0, "q_a_proj"), _param())
    with pytest.raises(RuntimeError, match="missing their partners"):
        client.finish_weight_update()
    assert wire.chunks == [], "nothing may reach the engine once the close is refused"


def test_groups_are_scoped_to_the_modules_the_model_declares():
    """An MLA block without ``q_lora_rank`` has no ``q_a_proj``: the engine fuses nothing there, so the
    surviving half must travel on its own rather than wait for a partner that never comes."""
    without_q_a = ["model.layers.0.self_attn.kv_a_proj_with_mqa", "model.layers.0.self_attn.q_b_proj"]
    assert declared_co_load_groups(SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS, without_q_a) == (
        ("self_attn.kv_a_proj_with_mqa.weight",),
    )
    # A PEFT-wrapped tree keeps the module names, so the pair stays whole under an adapter.
    peft = [f"base_model.model.model.layers.0.self_attn.{leaf}" for leaf in ("q_a_proj", "kv_a_proj_with_mqa")]
    assert declared_co_load_groups(SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS, peft) == GROUPS
    assert declared_co_load_groups(SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS, []) == ((),)


def test_a_scoped_client_sends_a_partnerless_half_and_closes(streamed):
    client, wire = streamed
    client.scope_co_load_groups(["model.layers.0.self_attn.kv_a_proj_with_mqa"])
    client.update_named_param(_layer(0, "kv_a_proj_with_mqa"), _param())
    client.finish_weight_update()
    assert wire.chunk_names == [[_layer(0, "kv_a_proj_with_mqa")]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
