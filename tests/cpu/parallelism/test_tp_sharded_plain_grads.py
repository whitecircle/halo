#!/usr/bin/env python
"""CPU tests: params the TP plan sharded as PLAIN tensors must not be treated as replicas.

GptOss attention sinks cannot be DTensors — the forward concatenates them with already-sharded
logits — so ``apply_tp_to_attention_only`` slices them by hand and records the suffix on
``model._tp_sharded_non_dtensor`` (the same registry the save path all-gathers). They therefore
reach the gradient path looking exactly like a replicated LayerNorm weight while each TP rank owns
a DISJOINT head slice:

  * a TP-group AVG mixes another rank's heads into every sink gradient;
  * counting each slice as a full replica under-counts the global grad norm AND makes it
    rank-dependent, so TP peers clip the same shard by different coefficients.

Run: ``pytest -m cpu tests/cpu/parallelism/test_tp_sharded_plain_grads.py``
"""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from src.distributed.tensor_parallel.parallelize_attention import shard_sinks_param
from src.trainers.mixins import grad_sync as grad_sync_module
from src.trainers.mixins.base import DistributedTrainerMixin

TP_SIZE = 2
SINKS_SUFFIX = "self_attn.sinks"


class _Attn(nn.Module):
    def __init__(self, local_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.sinks = nn.Parameter(torch.zeros(local_heads))


class _Layer(nn.Module):
    def __init__(self, local_heads: int):
        super().__init__()
        self.self_attn = _Attn(local_heads)
        self.input_layernorm = nn.LayerNorm(4)


class _Model(nn.Module):
    """Stand-in for a TP-parallelized causal LM: sinks sliced to this rank's heads, everything
    else replicated. ``register=False`` models a family whose TP plan shards nothing by hand."""

    def __init__(self, n_layers: int = 2, local_heads: int = 2, register: bool = True):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(local_heads) for _ in range(n_layers))
        if register:
            self._tp_sharded_non_dtensor = [(SINKS_SUFFIX, 0)]


def _trainer(model: nn.Module) -> SimpleNamespace:
    trainer = SimpleNamespace(
        _top_level_model=lambda: model,
        _get_tp_process_group=lambda: "tp-group",
        _sharded_grad_bucket=DistributedTrainerMixin._sharded_grad_bucket,
        # The PP scope the norm path reads: None on every non-PP run (the stage IS the world, and
        # there is no chain to reduce over), which is what this stub models.
        _pp_stage_group=None,
        _pp_chain_group=None,
    )
    trainer._tp_sharded_plain_param_ids = MethodType(DistributedTrainerMixin._tp_sharded_plain_param_ids, trainer)
    trainer._tp_per_head_norm_param_ids = MethodType(DistributedTrainerMixin._tp_per_head_norm_param_ids, trainer)
    trainer.parallelism_config = SimpleNamespace(fp32_grad_reduce=False)
    trainer.state = SimpleNamespace(global_step=0)
    return trainer


def _fill_grads(model: nn.Module, *, sink_grad: float, replicated_grad: float) -> None:
    for name, param in model.named_parameters():
        param.grad = torch.full_like(param, sink_grad if name.endswith(SINKS_SUFFIX) else replicated_grad)


def _sink_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.endswith(SINKS_SUFFIX)]


def test_ids_are_derived_from_the_tp_plan_registry():
    model = _Model(n_layers=3, local_heads=2)
    ids = _trainer(model)._tp_sharded_plain_param_ids()
    assert ids == {id(p) for p in _sink_params(model)}
    assert len(ids) == 3, "one entry per layer's sinks"


def test_ids_empty_without_a_registered_hand_shard():
    model = _Model(n_layers=2, local_heads=2, register=False)
    assert _trainer(model)._tp_sharded_plain_param_ids() == set()


def test_a_tp_plan_marks_nothing():
    """transformers populates ``_tp_plan`` from the model CLASS on every load, TP or not — and when
    it IS applied, every planned projection is a DTensor whose grad reduces itself.

    Reading the plan here would mark every projection of a plainly-loaded model disjoint. On an
    EP-only run — where ``_device_mesh`` is None, so the TP bucket reduce never runs — their norms
    would then be dropped from the global grad norm entirely. Only the hand-shard registry records
    a plain slice.
    """
    model = _Model(n_layers=2, local_heads=2, register=False)
    model._tp_plan = {"layers.*.self_attn.q_proj": "colwise"}

    assert _trainer(model)._tp_sharded_plain_param_ids() == set()


class _StubMesh:
    """The two DeviceMesh methods ``shard_sinks_param`` calls."""

    def __init__(self, size: int, rank: int):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _NamedAttnModel(nn.Module):
    """A model holding its attention under a family-chosen attribute name."""

    def __init__(self, attn_name: str, n_layers: int = 2, total_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in range(n_layers))
        for layer in self.layers:
            setattr(layer, attn_name, _Attn(total_heads))


@pytest.mark.parametrize("attn_name", ("self_attn", "attention", "attn"))
def test_hand_sharded_sinks_register_under_the_attribute_they_actually_live_on(attn_name):
    """The registered suffix must be the parameter's real FQN tail, not one family's spelling.

    ``apply_tp_to_attention_only`` resolves the attention attribute per family (``self_attn`` /
    ``attention`` / ``attn``). A fixed ``self_attn.sinks`` entry matches no parameter on the others,
    so their disjoint head slices read back as replicas: the TP group AVGs another rank's heads into
    every sink gradient, the grad norm counts one slice as the whole tensor, and the save path
    all-gathers nothing — writing rank 0's partial sinks into the checkpoint.
    """
    model = _NamedAttnModel(attn_name)
    mesh = _StubMesh(TP_SIZE, 0)
    for layer in model.layers:
        shard_sinks_param(model, getattr(layer, attn_name), attn_name, mesh)

    assert model._tp_sharded_non_dtensor == [(f"{attn_name}.sinks", 0)]
    sinks = [p for name, p in model.named_parameters() if name.endswith("sinks")]
    assert [p.shape[0] for p in sinks] == [2, 2], "sinks must be sliced to this rank's head range"
    assert _trainer(model)._tp_sharded_plain_param_ids() == {id(p) for p in sinks}


def test_sharded_sinks_are_excluded_from_the_tp_average(monkeypatch):
    """The TP-group AVG must cover the replicated params only — never the disjoint sink slices."""
    model = _Model(n_layers=2, local_heads=2)
    _fill_grads(model, sink_grad=1.0, replicated_grad=2.0)
    trainer = _trainer(model)

    reduced: dict[object, list[torch.Tensor]] = {}
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: TP_SIZE)
    monkeypatch.setattr(
        grad_sync_module, "reduce_grads_bucketed", lambda grads, **kw: reduced.setdefault(kw["op"], []).extend(grads)
    )

    params = list(model.parameters())
    DistributedTrainerMixin._sync_tp_replicated_grads(trainer, params)

    # The per-head-norm SUM bucket is the other collective; this model has none, so the AVG bucket
    # must carry every replicated grad and nothing else.
    assert not reduced.get(dist.ReduceOp.SUM), "this model has no per-head attention norms to SUM"
    synced = reduced[dist.ReduceOp.AVG]
    synced_ids = {id(g) for g in synced}
    sink_grad_ids = {id(p.grad) for p in _sink_params(model)}
    assert not (synced_ids & sink_grad_ids), "TP-sharded sinks were averaged across the TP group"
    replicated_ids = {id(p.grad) for p in params if id(p.grad) not in sink_grad_ids}
    assert synced_ids == replicated_ids, "every genuinely replicated grad must still be averaged"


def test_grad_norm_sums_sharded_sinks_over_the_tp_group(monkeypatch):
    """Sink shard norms belong in the TP-summed bucket, not the un-reduced replicated total."""
    monkeypatch.setattr(grad_sync_module, "current_device", lambda: torch.device("cpu"))
    model = _Model(n_layers=2, local_heads=2)
    _fill_grads(model, sink_grad=3.0, replicated_grad=0.5)
    trainer = _trainer(model)

    captured: dict[str, float] = {}

    def _fake_reduce(tp1d_sq, tp2d_sq, dp_sq):
        captured.update(tp1d=float(tp1d_sq), tp2d=float(tp2d_sq), dp=float(dp_sq))
        return tp1d_sq * TP_SIZE + tp2d_sq + dp_sq  # stand-in for the TP-group SUM over 2 ranks

    trainer._reduce_shard_norm_buckets = _fake_reduce

    params = list(model.parameters())
    norm = DistributedTrainerMixin._compute_tp_grad_norm(trainer, params)

    sink_sq = sum(float(p.grad.norm() ** 2) for p in _sink_params(model))
    sink_ids = {id(p) for p in _sink_params(model)}
    replicated_sq = sum(float(p.grad.norm() ** 2) for p in params if id(p) not in sink_ids)

    assert captured["tp1d"] == pytest.approx(sink_sq), "sink shard norms must land in the TP-summed bucket"
    assert norm == pytest.approx(math.sqrt(TP_SIZE * sink_sq + replicated_sq)), (
        "the global norm must count every TP rank's sink shard, not just this rank's slice"
    )


def test_grad_norm_unchanged_without_hand_sharded_params(monkeypatch):
    """Families whose TP plan shards nothing by hand keep the plain replicated accounting."""
    monkeypatch.setattr(grad_sync_module, "current_device", lambda: torch.device("cpu"))
    model = _Model(n_layers=2, local_heads=2, register=False)
    _fill_grads(model, sink_grad=3.0, replicated_grad=0.5)
    trainer = _trainer(model)
    trainer._reduce_shard_norm_buckets = lambda tp1d_sq, tp2d_sq, dp_sq: tp1d_sq + tp2d_sq + dp_sq

    params = list(model.parameters())
    norm = DistributedTrainerMixin._compute_tp_grad_norm(trainer, params)
    total_sq = sum(float(p.grad.norm() ** 2) for p in params)
    assert norm == pytest.approx(math.sqrt(total_sq))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
