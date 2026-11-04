#!/usr/bin/env python
"""``moe_balancing: auto`` may only resolve to a mode the enabled tree can serve.
With no EP layer and a forward without ``output_router_logits`` no balancing route exists, so ``auto`` is ``none``."""

import logging
from types import MethodType

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from halo.callbacks.balancing import apply_balancing_mode
from halo.model_loading.liger.cross_entropy import build_lce_forward
from halo.models.moe.balancing import _warned_unservable_auto, resolve_balancing_mode

_RESOLVER_LOGGER = "halo.models.moe.balancing"


class _MoEProbeConfig(PretrainedConfig):
    """A usable aux-loss coefficient is the branch that raises."""

    model_type = "auto_balancing_probe"

    def __init__(self, num_experts: int = 8, num_experts_per_tok: int = 2, **kwargs):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = 0.001
        super().__init__(**kwargs)


class _NoRouterLogitFlag(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _MoEProbeConfig()
        self.experts = nn.Linear(4, 4)

    def forward(self, input_ids=None, **kwargs):
        return input_ids


class _HonorsRouterLogitFlag(_NoRouterLogitFlag):
    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        return input_ids


def _resolve(model):
    _warned_unservable_auto.discard(type(model).__name__)  # warn-once state is keyed per model class
    return resolve_balancing_mode("auto", model, is_moe=True)


def test_auto_never_resolves_to_the_mode_this_tree_refuses():
    """Whatever ``auto`` returns must survive ``apply_balancing_mode``, the call ``configure_moe_balancing`` makes."""
    model = _NoRouterLogitFlag()
    mode = _resolve(model)
    assert mode == "none", f"auto resolved to {mode!r} on a tree that can serve neither balancing route"
    apply_balancing_mode(model, mode, is_moe=True)


def test_auto_says_why_it_gave_up_on_balancing(caplog):
    """Resolving to ``none`` silently would leave a large-expert-count run unbalanced with no trace."""
    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        _resolve(_NoRouterLogitFlag())
    message = caplog.text
    assert "output_router_logits" in message, f"the reason must name the missing forward parameter: {message}"
    assert "UNBALANCED" in message, f"the consequence must be stated: {message}"


def test_the_fused_loss_remedy_is_named_only_when_a_liger_head_replaced_the_forward(caplog):
    """On a forward that never declared the flag the remedy is false advice.
    The probe must look through a PEFT wrapper, whose own forward is never Liger's, at the patched base model."""
    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        _resolve(_NoRouterLogitFlag())
    assert "fused_linear_cross_entropy: false" not in caplog.text, (
        "advised turning off a fused loss that is not installed"
    )

    class _LigerHeaded(_NoRouterLogitFlag):
        forward = build_lce_forward()

    peft_wrapped = get_peft_model(_LigerHeaded(), LoraConfig(r=2, target_modules=["experts"]))
    for model in (_LigerHeaded(), peft_wrapped):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
            assert _resolve(model) == "none"
        assert "fused_linear_cross_entropy: false" in caplog.text, f"{type(model).__name__}: the remedy is missing"


def test_a_fused_loss_bound_on_the_instance_alone_is_seen():
    """Re-applied to a built model (TRL does this), Liger binds the fused loss on the instance alone.
    The class still declares the flag, so a verdict read off it picks ``aux_loss``, and a head that assembles
    its aux loss after the projection the fused loss replaces raises on the first step."""
    model = _HonorsRouterLogitFlag()
    model.forward = MethodType(build_lce_forward(router_aux_loss_in_head=True), model)
    mode = _resolve(model)
    assert mode == "none", f"auto resolved to {mode!r} off the class while the instance runs a fused loss"
    apply_balancing_mode(model, mode, is_moe=True)
    assert not getattr(model.config, "output_router_logits", False), (
        "the strategy enabled a flag the fused forward refuses"
    )


def test_auto_still_picks_aux_loss_where_the_forward_honours_the_flag():
    model = _HonorsRouterLogitFlag()
    assert _resolve(model) == "aux_loss"
    apply_balancing_mode(model, "aux_loss", is_moe=True)
    assert model.config.output_router_logits is True, "the aux-loss strategy must still enable router logits"


def test_an_explicit_aux_loss_still_raises_loudly():
    """Raised at config time: this is the call ``configure_moe_balancing`` makes while wiring the callbacks."""
    with pytest.raises(ValueError, match="does not take output_router_logits"):
        apply_balancing_mode(_NoRouterLogitFlag(), "aux_loss", is_moe=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
