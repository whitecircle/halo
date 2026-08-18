#!/usr/bin/env python
"""``moe_balancing: auto`` may only resolve to a mode the ENABLED tree can actually serve.

The window: a MoE checkpoint whose forward never takes ``output_router_logits`` (the multimodal
wrappers, and every family reading the flag out of ``**kwargs``) loaded with no EP MoE wrapper —
``use_grouped_gemm: false``, or a launcher that never patches the layers. Nothing there accepts a
routing bias and the aux-loss term can never reach the loss, so the model has NO balancing route at
all: ``auto`` resolving to ``aux_loss`` hands the balancing strategy a mode it refuses, and the run
dies on a ValueError the user did not ask for. It must resolve to ``none`` and say why.

The refusal itself stays: an EXPLICIT ``aux_loss`` there is a user assertion about a model that
cannot honour it, and it must fail loudly at config time rather than train unbalanced in silence.

    python tests/cpu/models/test_moe_balancing_auto_resolution.py
"""

import logging

import pytest
import torch.nn as nn
from transformers import PretrainedConfig

from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.models.moe_balancing import _WARNED_UNSERVABLE_AUTO, resolve_balancing_mode

_RESOLVER_LOGGER = "src.models.moe_balancing"


class _MoEProbeConfig(PretrainedConfig):
    """A tiny MoE config with a USABLE aux-loss coefficient — the branch that raises."""

    model_type = "auto_balancing_probe"

    def __init__(self, num_experts: int = 8, num_experts_per_tok: int = 2, **kwargs):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = 0.001
        super().__init__(**kwargs)


class _NoRouterLogitFlag(nn.Module):
    """MoE weights, no EP wrapper, and a forward that never consults the router-logit flag."""

    def __init__(self):
        super().__init__()
        self.config = _MoEProbeConfig()
        self.experts = nn.Linear(4, 4)

    def forward(self, input_ids=None, **kwargs):
        return input_ids


class _HonorsRouterLogitFlag(_NoRouterLogitFlag):
    """The same tree whose forward DOES declare the flag — HF's config-backed parameter."""

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        return input_ids


def _resolve(model):
    _WARNED_UNSERVABLE_AUTO.discard(type(model).__name__)  # warn-once state is keyed per model class
    return resolve_balancing_mode("auto", model, is_moe=True)


def test_auto_never_resolves_to_the_mode_this_tree_refuses():
    """The bug: ``auto`` picked ``aux_loss``, which the strategy then raises on.

    Pinned as the invariant rather than the string: whatever ``auto`` returns must survive
    ``apply_balancing_strategy``, the same call ``build_perf_callbacks`` makes.
    """
    model = _NoRouterLogitFlag()
    mode = _resolve(model)
    assert mode == "none", f"auto resolved to {mode!r} on a tree that can serve neither balancing route"
    apply_balancing_strategy(model, mode, is_moe=True)  # must not raise: auto's verdict is applicable


def test_auto_says_why_it_gave_up_on_balancing(caplog):
    """Resolving to ``none`` silently would leave a large-expert-count run unbalanced with no trace."""
    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        _resolve(_NoRouterLogitFlag())
    message = caplog.text
    assert "output_router_logits" in message, f"the reason must name the missing forward parameter: {message}"
    assert "UNBALANCED" in message, f"the consequence must be stated: {message}"


def test_auto_still_picks_aux_loss_where_the_forward_honours_the_flag():
    """Anti-over-rejection: the fix must not disarm balancing for the families that do honour it."""
    model = _HonorsRouterLogitFlag()
    assert _resolve(model) == "aux_loss"
    apply_balancing_strategy(model, "aux_loss", is_moe=True)
    assert model.config.output_router_logits is True, "the aux-loss strategy must still enable router logits"


def test_an_explicit_aux_loss_still_raises_loudly():
    """The guard is not weakened: an explicit mode this model cannot honour dies at config time.

    Config time, not step 1 — this is the call ``build_perf_callbacks`` makes while wiring the
    callbacks, before any training step runs.
    """
    with pytest.raises(ValueError, match="does not take output_router_logits"):
        apply_balancing_strategy(_NoRouterLogitFlag(), "aux_loss", is_moe=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
