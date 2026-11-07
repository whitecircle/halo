#!/usr/bin/env python
"""SGLang's MLA loaders fuse ``q_a_proj`` and ``kv_a_proj_with_mqa`` from a cache local to one ``load_weights`` call,
so a half arriving without the other is dropped without error, and a chunk boundary must never split a pair."""

import pytest
import torch

import halo.weight_sync.clients.base as base_module
from halo.weight_sync.chunking import declared_co_load_groups, split_co_loaded
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
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
    chunk = [("model.layers.0.xself_attn.q_a_proj.weight", _param()), (_layer(0, "q_a_proj"), _param())]
    sendable, held = split_co_loaded(chunk, GROUPS)
    assert _names(sendable) == ["model.layers.0.xself_attn.q_a_proj.weight"]
    assert _names(held) == [_layer(0, "q_a_proj")]


@pytest.fixture
def streamed(monkeypatch) -> tuple[SGLangWeightSyncClient, Wire]:
    monkeypatch.setattr(base_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    wire = Wire()
    return wire.attach(offline_client(SGLangWeightSyncClient)), wire


def test_a_pair_the_budget_would_split_moves_whole_into_the_next_chunk(streamed):
    client, wire = streamed
    push(client, _mla_payload())
    assert wire.chunk_names == _GROUPED_CHUNKS


def test_without_the_group_the_budget_cuts_between_the_halves(streamed):
    """The control: the regrouping in ``_GROUPED_CHUNKS`` is the co-load group's doing."""
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
    """An MLA block without ``q_lora_rank`` has no ``q_a_proj``, so the surviving half must not wait for a partner."""
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


# A block-FP8 weight and its scale: found by dtype and name, in any client, with no declared group.
def _fp8_pair(layer: int) -> list[tuple[str, torch.Tensor]]:
    name = _layer(layer, "q_proj")
    return [(name, torch.zeros(8, dtype=torch.float8_e4m3fn)), (name + "_scale_inv", torch.zeros(2))]


def test_a_block_fp8_weight_waits_for_its_scale_and_the_scale_for_its_weight():
    weight, scale = _fp8_pair(0)
    assert _names(split_co_loaded([weight], ())[0]) == []
    assert _names(split_co_loaded([weight, scale], ())[0]) == [weight[0], scale[0]]
    assert _names(split_co_loaded([scale], ())[1]) == [scale[0]]


def test_a_bf16_weight_and_a_name_that_only_ends_alike_are_not_pairs():
    chunk = [(_layer(0, "q_proj"), _param()), ("model.layers.0.hc_attn_scale_inv", torch.zeros(1))]
    sendable, held = split_co_loaded(chunk, ())
    assert _names(sendable) == _names(chunk) and held == []


def test_a_budget_cut_between_a_weight_and_its_scale_moves_the_pair_whole(monkeypatch):
    monkeypatch.setattr(base_module, "WEIGHT_SYNC_CHUNK_BYTES", 16)
    wire = Wire()
    client = wire.attach(offline_client(VLLMWeightSyncClient))
    other = (_layer(1, "k_proj"), _param())
    push(client, [other, *_fp8_pair(0)])
    assert wire.chunk_names == [[other[0]], [_layer(0, "q_proj"), _layer(0, "q_proj") + "_scale_inv"]]


def test_the_close_refuses_a_weight_whose_scale_never_came(monkeypatch):
    wire = Wire()
    client = wire.attach(offline_client(VLLMWeightSyncClient))
    client.update_named_param(*_fp8_pair(0)[0])
    with pytest.raises(RuntimeError, match="missing their partners"):
        client.finish_weight_update()
    assert wire.chunks == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
