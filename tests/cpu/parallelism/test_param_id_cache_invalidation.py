#!/usr/bin/env python3
"""``id(param)`` memos must not survive the wrapping that replaces the Parameter objects.

``EPLayersMixin`` classifies EP parameters by identity: ``_get_ep_param_ids`` and
``_get_sharded_expert_param_ids`` memoize sets of ``id(param)``, and the grad-norm buckets plus the
deferred cross-node sweep test membership against them. FSDP2 REPLACES managed Parameter objects
rather than swapping ``.data``, so any id cached before wrapping names an object no consumer will
ever see again — every EP param then reads as non-EP, its grad lands in the wrong bucket, and
nothing raises.

The PEFT dtype alignment and the fp32 upcast (``_upcast_non_ep_params_to_fp32``) call
``_get_ep_param_ids`` during setup, BEFORE wrapping, so the invariant cannot be left to convention:
clearing ``_param_id_caches`` after the wrap has to enforce it.

Usage:
    python tests/cpu/parallelism/test_param_id_cache_invalidation.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.trainers.parallelism.ep_layers import EPLayersMixin
from src.trainers.parallelism.grad_sync import GradientSyncMixin
from tests.common.ep_stubs import StubEPLayerBase

E, H, M = 2, 4, 8


class _StubEPLayer(StubEPLayerBase):
    """An EP layer with the real ``expert_named_params`` machinery over distributed experts."""

    def __init__(self):
        super().__init__()
        self.ep_config = SimpleNamespace(
            fsdp_shard_ep1_experts=False, ep_group_size=2, experts_fsdp_managed=False, defer_grad_sync=False
        )
        self.gate = nn.Linear(H, E, bias=False)  # the base's default _ROUTER_ATTR
        self.gate_proj = nn.Parameter(torch.randn(E, H, M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))


class _StubTrainer(EPLayersMixin, GradientSyncMixin):
    """Only what the introspection reads: ``self.model`` and the memo store."""

    def __init__(self):
        self.model = nn.Sequential(nn.Linear(H, H), _StubEPLayer())
        self._param_id_caches = {}

    def _top_level_model(self) -> nn.Module:
        return self.model


def _rewrap_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Stand in for FSDP2: replace every Parameter OBJECT, preserving values.

    This is the property that invalidates the memos — not a dtype change or a ``.data`` swap.

    Returns the displaced Parameters, which the caller MUST keep referenced: CPython recycles the
    address of a freed object, so dropping them lets a new Parameter land on a retired ``id`` and
    the stale set then matches by coincidence. (FSDP2 likewise keeps the originals alive as the
    sharded local pieces, so holding them is also the faithful simulation.)
    """
    displaced = []
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            displaced.append(param)
            setattr(module, name, nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad))
    return displaced


def _live_ep_ids(model: nn.Module) -> set:
    return {id(p) for m in model.modules() if isinstance(m, EPMoELayerBase) for p in m.parameters()}


def test_the_ep_layer_list_follows_the_model_object():
    """Memoized per model, so a replaced model (the PP split hands the trainer its stage) is walked again."""
    trainer = _StubTrainer()
    layers = trainer._ep_layers()
    assert layers == [trainer.model[1]] and trainer._ep_layers() is layers
    trainer.model = nn.Sequential(nn.Linear(H, H))
    assert trainer._ep_layers() == []


def test_stale_ids_would_misclassify_every_ep_param():
    """The failure this guards against, pinned: without invalidation the memo names dead objects."""
    trainer = _StubTrainer()
    pre_wrap = set(trainer._get_ep_param_ids())
    assert pre_wrap, "fixture must expose EP params"

    displaced = _rewrap_parameters(trainer.model)  # noqa: F841 — held so the ids cannot be recycled

    # The memo is still the pre-wrap set, and it names NOTHING the model now holds.
    assert trainer._get_ep_param_ids() == pre_wrap, "memo must genuinely be sticky (else this test proves nothing)"
    assert not (pre_wrap & _live_ep_ids(trainer.model)), (
        "the rewrap must actually replace the objects for this test to be meaningful"
    )


def test_invalidation_makes_the_memo_name_the_surviving_objects():
    """After invalidation the recomputed set must be exactly the live EP parameters."""
    trainer = _StubTrainer()
    trainer._get_ep_param_ids()
    trainer._get_sharded_expert_param_ids()

    displaced = _rewrap_parameters(trainer.model)  # noqa: F841 — held so the ids cannot be recycled
    trainer._param_id_caches.clear()

    assert trainer._get_ep_param_ids() == _live_ep_ids(trainer.model)
    # The sharded-expert set stays a strict subset: the router is replicated, not distributed.
    sharded = trainer._get_sharded_expert_param_ids()
    assert sharded < trainer._get_ep_param_ids()
    assert sharded == {id(trainer.model[1].gate_proj), id(trainer.model[1].down_proj)}


def test_every_memoized_id_set_lives_in_the_one_store_the_wrap_clears():
    """The EP and TP id sets all memoize through ``_param_id_caches``, so the one ``clear()`` after the
    wrap drops every one of them; a memo kept on its own attribute would survive it and fail silently,
    in the grad-norm bucket."""
    trainer = _StubTrainer()
    trainer._get_ep_param_ids()
    trainer._get_sharded_expert_param_ids()
    trainer._tp_sharded_plain_param_ids()
    trainer._tp_per_head_norm_param_ids()
    assert len(trainer._param_id_caches) == 4, sorted(trainer._param_id_caches)
    assert set(vars(trainer)) == {"model", "_param_id_caches", "_ep_layers_of"}, sorted(vars(trainer))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
