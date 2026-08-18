#!/usr/bin/env python3
"""``id(param)`` memos must not survive the wrapping that replaces the Parameter objects.

``EpIntrospectionMixin`` classifies EP parameters by identity: ``_get_ep_param_ids`` and
``_get_sharded_expert_param_ids`` memoize sets of ``id(param)``, and the grad-norm buckets plus the
deferred cross-node sweep test membership against them. FSDP2 REPLACES managed Parameter objects
rather than swapping ``.data``, so any id cached before wrapping names an object no consumer will
ever see again — every EP param then reads as non-EP, its grad lands in the wrong bucket, and
nothing raises.

The memos are documented as lazily populated by step-time consumers only, but the fp32 upcast
(``_upcast_non_ep_params_to_fp32``) calls ``_get_ep_param_ids`` during setup, BEFORE wrapping. So
the invariant cannot be left to convention — ``_invalidate_param_id_caches`` has to enforce it.

Usage:
    python tests/cpu/parallelism/test_param_id_cache_invalidation.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.trainers.mixins.ep_introspection import EpIntrospectionMixin
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


class _StubTrainer(EpIntrospectionMixin):
    """Only what the introspection reads: ``self.model``."""

    def __init__(self):
        self.model = nn.Sequential(nn.Linear(H, H), _StubEPLayer())
        self._ep_config = None


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
    trainer._invalidate_param_id_caches()

    assert trainer._get_ep_param_ids() == _live_ep_ids(trainer.model)
    # The sharded-expert set stays a strict subset: the router is replicated, not distributed.
    sharded = trainer._get_sharded_expert_param_ids()
    assert sharded < trainer._get_ep_param_ids()
    assert sharded == {id(trainer.model[1].gate_proj), id(trainer.model[1].down_proj)}


def test_every_memoized_id_cache_is_dropped():
    """Derived from the attributes themselves, so a memo added later cannot escape invalidation.

    A third ``id(param)`` cache that ``_invalidate_param_id_caches`` forgets fails exactly the way
    the two existing ones would — silently, in the grad-norm bucket — so the guard must not be a
    hand-listed pair of assignments this test mirrors by hand.
    """
    trainer = _StubTrainer()
    trainer._get_ep_param_ids()
    trainer._get_sharded_expert_param_ids()
    populated = {name for name, value in vars(trainer).items() if name.endswith("_cache") and value is not None}
    assert populated, "fixture must populate the memos"

    trainer._invalidate_param_id_caches()

    still_set = {name for name in populated if getattr(trainer, name) is not None}
    assert not still_set, f"memo(s) survived invalidation: {sorted(still_set)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
