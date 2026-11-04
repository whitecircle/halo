#!/usr/bin/env python
"""A MoE head as an EP layer on the loopback transport, and each rank's slice under expert TP: one dispatch per forward
under checkpointing with hard and soft issuing the same ops, a row with no target keeping the head's ops, and the head
reusing the capacity the first MoE layer all-reduced."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from halo.distributed.expert_parallel.deepep import backends as ep_backends
from halo.distributed.expert_parallel.deepep import dispatcher as ep_dispatcher
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.mtp.head import find_mtp_head
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from tests.common.ep_stubs import LoopbackBackend, no_ep_dispatcher
from tests.common.mtp_heads import LABEL_IGNORE, MTP_TINY_FAMILIES, padded_rows, write_checkpoint_with_tail
from tests.common.parallelism import single_process_ep_config, stubbed_rank_ep_config
from tests.common.tiny_models import tiny_family_model

_ROUTED = ("qwen3_5_moe_text", "qwen3_next", "glm4_moe_lite")
_WEIGHT = 0.7
_EXPERTS = 8
_FORWARD_ORDER = ["dispatch_forward", "combine_forward"]
_BACKWARD_ORDER = ["combine_backward", "dispatch_backward"]


@pytest.fixture
def loopback_transport(monkeypatch):
    monkeypatch.setattr(ep_dispatcher, "deep_ep", lambda: SimpleNamespace(topk_idx_t=torch.int64))
    monkeypatch.setattr(ep_dispatcher, "_live_dispatchers", weakref.WeakValueDictionary())


def _experts_of(model) -> dict[str, torch.Tensor]:
    [(_path, layer)] = named_ep_layers(find_mtp_head(model))
    return dict(layer.expert_named_params())


def _attach(model, path, *, objective: MtpObjective, ep_config) -> None:
    source = ModelSource(str(path), None, str(path))
    attach_mtp_head(
        model,
        MtpHeadRequest(objective),
        pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: ep_config),
        source=source,
        dtype=torch.float32,
        wraps_experts=ep_config is not None,
        init_from_scratch=False,
        preserve_checkpoint_precision=False,
    )


def _live(name: str, tmp_path, log: list, *, gc: bool = False, backend_cls=LoopbackBackend, **objective):
    ep_config = single_process_ep_config()
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).float()
    apply_ep_layers(model, ep_config)
    _attach(model, tmp_path / "ckpt", objective=MtpObjective(_WEIGHT, **objective), ep_config=ep_config)
    model.float().train()
    if gc:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
        model.config.use_cache = False
    for _, layer in named_ep_layers(model):
        layer.dispatcher._noop = False
        layer.dispatcher.backend = backend_cls(layer.dispatcher, log)
    return model


def _head_ops(model, log: list) -> list[str]:
    head_dispatchers = {layer.dispatcher for _path, layer in named_ep_layers(find_mtp_head(model))}
    assert head_dispatchers, "the head holds no EP layer, so its ops would count nothing"
    return [op for op, dispatcher, _handle in log if dispatcher in head_dispatchers]


@pytest.mark.parametrize("scope", ["joint", "detached"])
@pytest.mark.parametrize("name", _ROUTED)
def test_checkpointing_replays_the_heads_dispatch_and_soft_issues_the_ops_of_hard(
    name, scope, tmp_path, loopback_transport
):
    """A second dispatch in the recompute would invalidate the backward's handle; the head's ops must not depend on the
    loss type, or an EP group running one rank's row through a different path would mispair its collectives."""
    ops = {}
    for loss_type in ("hard", "soft"):
        log: list = []
        model = _live(name, tmp_path / loss_type, log, gc=True, loss_type=loss_type, scope=scope)
        model(**padded_rows()).loss.backward()
        ops[loss_type] = _head_ops(model, log)
    assert ops["hard"] == _FORWARD_ORDER + _BACKWARD_ORDER
    assert ops["soft"] == ops["hard"]


@pytest.mark.parametrize(
    "rows",
    [
        {"input_ids": padded_rows()["input_ids"][:, :2], "labels": padded_rows()["input_ids"][:, :2]},
        {"input_ids": padded_rows()["input_ids"], "labels": torch.full_like(padded_rows()["input_ids"], LABEL_IGNORE)},
    ],
    ids=["too_short", "all_masked"],
)
@pytest.mark.parametrize("scope", ["joint", "detached", "head_only"])
@pytest.mark.parametrize("loss_type", ["hard", "soft"])
def test_a_row_with_no_head_target_keeps_the_heads_ops(rows, scope, loss_type, tmp_path, loopback_transport):
    """Its peers' combines and dispatches wait on this rank's, so an empty rank must walk the head's graph too."""
    log: list = []
    model = _live("qwen3_5_moe_text", tmp_path, log, loss_type=loss_type, scope=scope)
    output = model(**rows, use_cache=False, num_items_in_batch=torch.tensor(1))
    assert output.mtp_loss_split.head_targets == 0
    output.loss.backward()
    assert _head_ops(model, log) == _FORWARD_ORDER + _BACKWARD_ORDER


@pytest.mark.parametrize(
    "rows",
    [
        {"input_ids": padded_rows()["input_ids"], "labels": padded_rows()["labels"]},
        {"input_ids": padded_rows()["input_ids"], "labels": torch.full_like(padded_rows()["input_ids"], LABEL_IGNORE)},
    ],
    ids=["scored", "all_masked"],
)
@pytest.mark.parametrize("name", _ROUTED)
def test_a_held_out_forward_walks_the_heads_dispatch_with_no_backward_on_every_row(
    name, rows, tmp_path, loopback_transport
):
    """Every rank scores its own rows, so a rank whose rows score nothing still pairs with its peers' dispatches."""
    log: list = []
    model = _live(name, tmp_path, log, loss_type="soft").eval()
    output = model(**rows, use_cache=False)
    assert _head_ops(model, log) == _FORWARD_ORDER
    assert int(output.mtp_acceptance.positions) == int((rows["labels"][:, 2:] != LABEL_IGNORE).sum())


class _CapacityBackend(LoopbackBackend):
    """A loopback whose ``ensure`` sizes through the real elastic resolver, over a group every layer shares."""

    group = object()
    sized: list[tuple[int, int]]

    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        resolver = SimpleNamespace(_d=SimpleNamespace(dispatch_group=self.group))
        needed = ep_backends._ElasticBackend._resolve_capacity(resolver, num_tokens, num_topk)
        type(self).sized.append((num_tokens, needed))


@pytest.mark.parametrize("name", _ROUTED)
def test_the_head_dispatches_within_the_capacity_the_first_layer_all_reduced(
    name, tmp_path, monkeypatch, loopback_transport
):
    """Same tokens per rank and the same forward's generation, so the head adds no collective and raises no overrun."""
    all_reduces: list = []
    real_tensor = torch.tensor
    monkeypatch.setattr(ep_backends, "EP_CAPACITY_DEDUP", True)
    monkeypatch.setattr(
        ep_backends.torch, "tensor", lambda data, **kwargs: real_tensor(data, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(ep_backends.dist, "all_reduce", lambda *args, **kwargs: all_reduces.append(args))
    monkeypatch.setattr(_CapacityBackend, "sized", [], raising=False)
    log: list = []
    model = _live(name, tmp_path, log, backend_cls=_CapacityBackend)
    model(**padded_rows()).loss.backward()
    layers = len(named_ep_layers(model))
    assert layers > 1 and len(_CapacityBackend.sized) == layers, "not every MoE layer sized a dispatch"
    assert len(set(_CapacityBackend.sized)) == 1, f"the layers sized apart: {_CapacityBackend.sized}"
    assert len(all_reduces) == 1, f"{len(all_reduces)} capacity all-reduces in one forward"


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_under_expert_tp_a_rank_streams_the_slice_the_eager_path_would_cut(name, rank, tmp_path):
    """Two EP ways by two ETP ways on four ranks: the lazy planner reads what a full bank patched in memory would keep."""
    world, ways = 4, 2
    ep_config = stubbed_rank_ep_config(world, ways, rank, _EXPERTS)
    assert ep_config.expert_tp_size == ways, "this row is meant to slice each expert's width"
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).to(torch.bfloat16)
    with no_ep_dispatcher():
        _attach(model, tmp_path / "ckpt", objective=MtpObjective(1.0), ep_config=ep_config)
    streamed = _experts_of(model)

    torch.manual_seed(1)
    whole = tiny_family_model(MTP_TINY_FAMILIES[name]).to(torch.bfloat16)
    _attach(whole, tmp_path / "ckpt", objective=MtpObjective(1.0), ep_config=None)
    head = find_mtp_head(whole)
    shell = nn.Module()
    shell.add_module("head", head)
    bank = {key: value.numel() for key, value in head.named_parameters() if ".experts." in key}
    with no_ep_dispatcher():
        apply_ep_layers(shell, ep_config)
    eager = _experts_of(whole)

    assert streamed.keys() == eager.keys() and streamed
    for key, value in eager.items():
        assert torch.equal(streamed[key].float(), value.float()), key
    assert sum(value.numel() for value in streamed.values()) == sum(bank.values()) // world, "not this rank's share"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
