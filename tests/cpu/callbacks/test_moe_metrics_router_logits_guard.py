#!/usr/bin/env python
"""MoEMetricsCallback must respect the balancing strategy's router-logits off decision.

``apply_balancing_strategy`` deliberately keeps ``output_router_logits`` OFF in several modes
(bias_update's EP bypass, aux-severed EP wrappers, coef-0 crash avoidance, GRPO inert-aux) and
stamps that decision on the config via ``mark_router_logits_forced_off``. An ``on_train_begin`` that
unconditionally flips ``output_router_logits = True`` undoes the decision (and re-arms the very crash
the strategy avoided). The callback must check the stamp and skip.

Run: pytest tests/cpu/callbacks/test_moe_metrics_router_logits_guard.py
"""

import pytest
import torch

from src.callbacks.moe_metrics import MoEMetricsCallback
from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.models.moe_balancing import mark_router_logits_forced_off, router_logits_forced_off


class _Cfg:
    def __init__(self, output_router_logits=False, router_aux_loss_coef=0.0, model_type="test_moe"):
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_type = model_type


class _Model(torch.nn.Module):
    """Stands in for a causal-LM MoE family: the ``output_router_logits`` forward parameter is how
    HF marks a class that resolves the flag against the config, and hence one ``aux_loss`` can reach."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")


class _SeveredEPLayer(torch.nn.Module):
    """EP-layer stand-in whose class severs the HF aux-loss path (DeepSeek-V4-like)."""

    _ep_severs_aux_loss = True
    _supports_bias_balancing = True


class _AdoptedBiasEPLayer(torch.nn.Module):
    """EP-layer stand-in whose bias-update state is a native buffer (adopted-slot families): the
    buffer lives in ``_buffers``, not ``__dict__``, so the strict ``bias_update`` export contract
    accepts it."""

    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None

    def enable_bias_balancing(self):
        return True


def _run_callback(model):
    # enable_router_logits=True is the aux_loss-mode permission — the strongest case for the stamp to
    # override: even when the callback IS allowed to enable router logits, the stamp must still win.
    cb = MoEMetricsCallback(topk=2, enable_router_logits=True)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    return cb


def test_callback_enables_router_logits_on_plain_moe():
    """Baseline: without a forced-off stamp, a permitted callback wires the hook and enables logits."""
    model = _Model(_Cfg(output_router_logits=False))
    cb = _run_callback(model)
    assert model.config.output_router_logits is True
    assert cb._hook_handles
    cb.on_train_end(args=None, state=None, control=None)


def test_callback_respects_strategy_forced_off_stamp():
    model = _Model(_Cfg(output_router_logits=False))
    mark_router_logits_forced_off(model.config)
    cb = _run_callback(model)
    assert model.config.output_router_logits is False, "callback undid the strategy's off decision"
    assert cb._hook_handles == [], "callback wired a hook that can never capture router logits"


@pytest.mark.parametrize(
    ("mode", "kwargs", "adopted_router"),
    [
        ("aux_loss", {"policy_gradient_loss": True}, False),  # GRPO: aux loss inert, must stay off
        # EP bias path (native-slot acceptor, MoE): router-logit recording bypassed
        ("bias_update", {"is_moe": True}, True),
        ("aux_loss", {}, False),  # coef=0: enabling would crash TRL on outputs.aux_loss
    ],
)
def test_strategy_off_decisions_survive_the_callback(mode, kwargs, adopted_router):
    """End-to-end: every strategy path that leaves router logits off must stamp the config so the
    callback's train-begin cannot flip them back on."""
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    if adopted_router:
        model.ep_layer = _AdoptedBiasEPLayer()
    apply_balancing_strategy(model, mode, **{"is_moe": False, **kwargs})
    assert router_logits_forced_off(model.config), f"{mode} {kwargs} did not stamp the config"
    _run_callback(model)
    assert model.config.output_router_logits is False


def test_explicit_bias_mode_on_a_dense_model_raises():
    """An explicit bias mode with no acceptor anywhere is a misconfiguration, not a silently inert
    controller — the strategy must refuse rather than stamp and balance nothing."""
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    with pytest.raises(ValueError, match="dense model"):
        apply_balancing_strategy(model, "bias_update", is_moe=False)


def test_severed_ep_aux_loss_stays_off_through_callback():
    """DeepSeek-V4-under-EP shape: aux_loss requested but the EP wrappers sever the path — the
    strategy leaves logits off and the callback must not re-enable them (that would crash the HF
    aux-loss reduction on the empty router_logits tuple)."""
    cfg = _Cfg(output_router_logits=False, router_aux_loss_coef=0.01)
    model = _Model(cfg)
    model.ep_layer = _SeveredEPLayer()
    apply_balancing_strategy(model, "aux_loss", is_moe=True)
    assert router_logits_forced_off(cfg)
    _run_callback(model)
    assert cfg.output_router_logits is False


def test_aux_loss_with_usable_coef_not_stamped():
    """The normal aux_loss path must NOT stamp the config — metrics observation stays possible."""
    cfg = _Cfg(output_router_logits=False, router_aux_loss_coef=0.01)
    model = _Model(cfg)
    apply_balancing_strategy(model, "aux_loss", is_moe=True)
    assert cfg.output_router_logits is True
    assert not router_logits_forced_off(cfg)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
