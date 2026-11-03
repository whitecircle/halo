#!/usr/bin/env python
"""Gradient-checkpoint recompute replays EP's dispatch and combine through a live dispatcher. A second dispatch
reuses the buffer the backward node names and corrupts gradients silently."""

import re
import weakref
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from halo.distributed.expert_parallel.deepep import dispatcher as ep_dispatcher
from halo.distributed.expert_parallel.deepep.backends import DeepEPBackend
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.gradient_checkpointing import active_recompute_scope, enable_scoped_gradient_checkpointing
from halo.models.moe.family_specs import find_moe_family_spec
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

BATCH, SEQ = 2, 8
CHECKPOINT_MODES = {"reentrant": True, "non_reentrant": False}
CHECKPOINTED_FAMILIES = tuple(
    name for name in TINY_MOE_FAMILIES if find_moe_family_spec(name).supports_gradient_checkpointing
)


class LoopbackBackend(DeepEPBackend):
    """Every token returns to the rank that sent it, so each op is the identity; each is logged as ``(op,
    dispatcher, handle)``."""

    def __init__(self, dispatcher: ep_dispatcher.DeepEPDispatcher, log: list):
        super().__init__(dispatcher)
        self._log = log
        self.counts: list[tuple[int, ...]] = []

    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        pass

    def dispatch_forward(self, x, topk_idx, topk_weights):
        handle = f"handle{len(self._log)}"
        self._log.append(("dispatch_forward", self._d, handle))
        counts = tuple(torch.bincount(topk_idx[topk_idx >= 0], minlength=self._d.num_experts).tolist())
        self.counts.append(counts)
        return x.clone(), topk_idx.clone(), topk_weights.clone(), handle, counts

    def dispatch_backward(self, grad_recv_x, grad_recv_topk_weights, handle):
        self._log.append(("dispatch_backward", self._d, handle))
        grad_topk_weights = None if grad_recv_topk_weights is None else grad_recv_topk_weights.clone()
        return grad_recv_x.clone(), grad_topk_weights

    def combine_forward(self, x, topk_weights, handle):
        self._log.append(("combine_forward", self._d, handle))
        return x.clone()

    def combine_backward(self, grad_combined_x, handle):
        self._log.append(("combine_backward", self._d, handle))
        return grad_combined_x.clone()


@pytest.fixture(autouse=True)
def loopback_transport(monkeypatch):
    """DeepEP's index dtype without the native extension."""
    monkeypatch.setattr(ep_dispatcher, "deep_ep", lambda: SimpleNamespace(topk_idx_t=torch.int64))
    monkeypatch.setattr(ep_dispatcher, "_live_dispatchers", weakref.WeakValueDictionary())


def _live_model(name: str, log: list, *, use_reentrant: bool | None) -> torch.nn.Module:
    """``use_reentrant=None`` is unchecked."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[name]).train()
    apply_ep_layers(model, single_process_ep_config())
    layers = named_ep_layers(model)
    assert layers, f"{name}: the EP patch wrapped no MoE block"
    for _, layer in layers:
        layer.dispatcher._noop = False
        layer.dispatcher.backend = LoopbackBackend(layer.dispatcher, log)
    if use_reentrant is not None:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    return model


def _microbatch(model: torch.nn.Module, seed: int) -> torch.Tensor:
    vocab_size = model.get_input_embeddings().num_embeddings
    return torch.randint(0, vocab_size, (BATCH, SEQ), generator=torch.Generator().manual_seed(seed))


def _loss(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids).logits.float().square().mean()


def _ops_by_layer(model: torch.nn.Module, log: list) -> dict[str, list[tuple[str, str]]]:
    path_of = {layer.dispatcher: path for path, layer in named_ep_layers(model)}
    ops = {path: [] for path in path_of.values()}
    for op, dispatcher, handle in log:
        ops[path_of[dispatcher]].append((op, handle))
    return ops


def _record_passes(model: torch.nn.Module) -> dict[str, list]:
    """``(checkpoint scope, is recompute)`` of every forward pass."""
    passes = defaultdict(list)
    for path, layer in named_ep_layers(model):

        def record(_module, _args, path=path):
            scope = active_recompute_scope()
            passes[path].append((scope, scope is not None and scope.is_recompute))

        layer.register_forward_pre_hook(record)
    return passes


def _assert_same_gradients(model: torch.nn.Module, reference: torch.nn.Module) -> None:
    for (name, param), (_, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (param.grad is None) == (expected.grad is None), name
        if param.grad is not None:
            assert torch.equal(param.grad, expected.grad), name
    for path, layer in named_ep_layers(model):
        ungraded = [name for name, param in layer.expert_named_params() if param.grad is None]
        assert not ungraded, f"{path}: no gradient reached the experts {ungraded}"


@pytest.mark.parametrize("mode", CHECKPOINT_MODES)
@pytest.mark.parametrize("name", CHECKPOINTED_FAMILIES)
def test_the_recompute_replays_the_forwards_dispatch_and_combine(name, mode):
    log = []
    model = _live_model(name, log, use_reentrant=CHECKPOINT_MODES[mode])
    reference = _live_model(name, [], use_reentrant=None)
    passes = _record_passes(model)
    input_ids = _microbatch(model, seed=1)
    _loss(model, input_ids).backward()
    _loss(reference, input_ids).backward()

    for path, ops in _ops_by_layer(model, log).items():
        handle = ops[0][1]
        assert ops == [
            ("dispatch_forward", handle),
            ("combine_forward", handle),
            ("combine_backward", handle),
            ("dispatch_backward", handle),
        ], path
        (scope, recompute), (replay_scope, replay_recompute) = passes[path]
        assert scope is not None and replay_scope is scope and (recompute, replay_recompute) == (False, True), path
    _assert_same_gradients(model, reference)


@pytest.mark.parametrize("mode", CHECKPOINT_MODES)
@pytest.mark.parametrize("name", CHECKPOINTED_FAMILIES)
def test_the_recompute_replays_the_forwards_receive_counts(name, mode, monkeypatch):
    """The grouped sort sizes its compaction off these host-side counts."""
    model = _live_model(name, [], use_reentrant=CHECKPOINT_MODES[mode])
    seen = defaultdict(list)
    compute = EPMoELayerBase._compute_experts

    def spy(self, tokens, experts, weights, output_dtype, *, recv_counts):
        seen[id(self)].append(recv_counts)
        return compute(self, tokens, experts, weights, output_dtype, recv_counts=recv_counts)

    monkeypatch.setattr(EPMoELayerBase, "_compute_experts", spy)
    _loss(model, _microbatch(model, seed=1)).backward()

    for path, layer in named_ep_layers(model):
        (reported,) = layer.dispatcher.backend.counts
        assert sum(reported) > 0, path
        forward, recompute = seen[id(layer)]
        assert forward is reported and recompute is reported, path


@pytest.mark.parametrize("mode", CHECKPOINT_MODES)
@pytest.mark.parametrize("name", CHECKPOINTED_FAMILIES)
def test_microbatches_in_flight_replay_their_own_frames(name, mode):
    log = []
    model = _live_model(name, log, use_reentrant=CHECKPOINT_MODES[mode])
    reference = _live_model(name, [], use_reentrant=None)
    microbatches = [_microbatch(model, seed) for seed in (1, 2)]
    losses = [_loss(model, input_ids) for input_ids in microbatches]
    for loss in losses:
        loss.backward()
    for input_ids in microbatches:
        _loss(reference, input_ids).backward()

    for path, ops in _ops_by_layer(model, log).items():
        first, second = ops[0][1], ops[2][1]
        assert first != second, path
        assert ops == [
            ("dispatch_forward", first),
            ("combine_forward", first),
            ("dispatch_forward", second),
            ("combine_forward", second),
            ("combine_backward", first),
            ("dispatch_backward", first),
            ("combine_backward", second),
            ("dispatch_backward", second),
        ], path
    _assert_same_gradients(model, reference)


@pytest.mark.parametrize("mode", CHECKPOINT_MODES)
@pytest.mark.parametrize("name", CHECKPOINTED_FAMILIES)
def test_checkpointing_without_the_ep_scopes_raises_naming_the_family(name, mode):
    model = _live_model(name, [], use_reentrant=None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": CHECKPOINT_MODES[mode]})
    (family_class,) = {type(layer).__name__ for _, layer in named_ep_layers(model)}
    with pytest.raises(RuntimeError, match=re.escape(f"{family_class}: training forward inside backward")):
        _loss(model, _microbatch(model, seed=1)).backward()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
