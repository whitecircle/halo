#!/usr/bin/env python
"""Gradient-checkpoint recompute replays EP's dispatch and combine through a live dispatcher.

A recompute that dispatched again would reuse the DeepEP buffer the forward's backward node still
names and corrupt every gradient without raising, so an EP layer caches its first pass's dispatch and
combine in the checkpoint frame's scope and the recompute replays them. Here every MoE family that
supports checkpointing runs its tiny model with each dispatcher live over a loopback transport that
stands in for DeepEP: a one-rank EP group owning every expert, so each transport op is the identity,
and every op is logged with the handle it ran on. Under reentrant and non-reentrant checkpointing
installed by ``enable_scoped_gradient_checkpointing``:

- each layer dispatches and combines once per microbatch, its backward ops run on that dispatch's
  handle, and the recompute runs in the forward's scope;
- two microbatches in flight, as under 1F1B, each replay their own frame;
- the gradients equal those of the same model unchecked;
- checkpointing installed without the EP scopes raises, naming the family's layer class.

``tests/gpu/parallelism/ep/test_ep_replay_cache.py`` runs the replay on real DeepEP dispatch.

Run: python tests/cpu/parallelism/test_ep_gc_replay.py
"""

import re
import weakref
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from src.distributed.expert_parallel import autograd as ep_autograd
from src.distributed.expert_parallel import dispatcher as ep_dispatcher
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.models.gradient_checkpointing import active_recompute_scope, enable_scoped_gradient_checkpointing
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

BATCH, SEQ = 2, 8
CHECKPOINT_MODES = {"reentrant": True, "non_reentrant": False}
CHECKPOINTED_FAMILIES = tuple(
    name for name in TINY_MOE_FAMILIES if ep_layer_class_by_model_type()[name]._supports_gradient_checkpointing
)


class LoopbackBackend(ep_dispatcher._DeepEPBackend):
    """DeepEP's transport ops on a one-rank EP group owning every expert: every token returns to the
    rank that sent it, so each op is the identity. Each op is appended to ``log`` as
    ``(op, dispatcher, handle)``, and each dispatch hands out a handle of its own."""

    def __init__(self, dispatcher: ep_dispatcher.DeepEPDispatcher, log: list):
        super().__init__(dispatcher)
        self._log = log

    def prepare(self) -> None:
        pass

    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        pass

    def dispatch_fwd(self, x, topk_idx, topk_weights):
        handle = f"handle{len(self._log)}"
        self._log.append(("dispatch_fwd", self._d, handle))
        return x.clone(), topk_idx.clone(), topk_weights.clone(), handle

    def combine_grad(self, grad_recv_x, grad_recv_topk_weights, handle):
        self._log.append(("combine_grad", self._d, handle))
        grad_topk_weights = None if grad_recv_topk_weights is None else grad_recv_topk_weights.clone()
        return grad_recv_x.clone(), grad_topk_weights

    def combine_fwd(self, x, topk_weights, handle):
        self._log.append(("combine_fwd", self._d, handle))
        return x.clone()

    def dispatch_grad(self, grad_combined_x, handle):
        self._log.append(("dispatch_grad", self._d, handle))
        return grad_combined_x.clone()


@pytest.fixture(autouse=True)
def loopback_transport(monkeypatch):
    """DeepEP's index dtype without the native extension, and a live-dispatcher registry of the test's own."""
    monkeypatch.setattr(ep_autograd, "deep_ep", lambda: SimpleNamespace(topk_idx_t=torch.int64))
    monkeypatch.setattr(ep_dispatcher, "_LIVE_DISPATCHERS", weakref.WeakValueDictionary())


def _live_model(name: str, log: list, *, use_reentrant: bool | None) -> torch.nn.Module:
    """``name``'s seeded tiny model in training mode, EP-patched with every dispatcher live over a
    :class:`LoopbackBackend` logging to ``log``, checkpointed in the given mode (``None``: unchecked)."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[name]).train()
    patch_moe_model_for_ep(model, single_process_ep_config())
    layers = find_ep_layers(model)
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
    """Each EP layer's logged transport ops, in call order, as ``(op, handle)``."""
    path_of = {layer.dispatcher: path for path, layer in find_ep_layers(model)}
    ops = {path: [] for path in path_of.values()}
    for op, dispatcher, handle in log:
        ops[path_of[dispatcher]].append((op, handle))
    return ops


def _record_passes(model: torch.nn.Module) -> dict[str, list]:
    """Per EP layer, the ``(checkpoint scope, is recompute)`` of every forward pass it enters."""
    passes = defaultdict(list)
    for path, layer in find_ep_layers(model):

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
    for path, layer in find_ep_layers(model):
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
            ("dispatch_fwd", handle),
            ("combine_fwd", handle),
            ("dispatch_grad", handle),
            ("combine_grad", handle),
        ], path
        (scope, recompute), (replay_scope, replay_recompute) = passes[path]
        assert scope is not None and replay_scope is scope and (recompute, replay_recompute) == (False, True), path
    _assert_same_gradients(model, reference)


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
            ("dispatch_fwd", first),
            ("combine_fwd", first),
            ("dispatch_fwd", second),
            ("combine_fwd", second),
            ("dispatch_grad", first),
            ("combine_grad", first),
            ("dispatch_grad", second),
            ("combine_grad", second),
        ], path
    _assert_same_gradients(model, reference)


@pytest.mark.parametrize("mode", CHECKPOINT_MODES)
@pytest.mark.parametrize("name", CHECKPOINTED_FAMILIES)
def test_checkpointing_without_the_ep_scopes_raises_naming_the_family(name, mode):
    model = _live_model(name, [], use_reentrant=None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": CHECKPOINT_MODES[mode]})
    (family_class,) = {type(layer).__name__ for _, layer in find_ep_layers(model)}
    with pytest.raises(RuntimeError, match=re.escape(f"{family_class}: training forward inside backward")):
        _loss(model, _microbatch(model, seed=1)).backward()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
