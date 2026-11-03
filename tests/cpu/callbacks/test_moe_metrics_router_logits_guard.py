#!/usr/bin/env python
"""``apply_balancing_mode`` keeps ``output_router_logits`` off in several modes and stamps that with
``mark_router_logits_forced_off``. A callback flipping it back re-arms the crash the strategy avoided."""

import pytest
import torch

from halo.callbacks.balancing import apply_balancing_mode
from halo.callbacks.moe_metrics import MoEMetricsCallback
from halo.models.moe.router_logits import mark_router_logits_forced_off, router_logits_forced_off


class _Cfg:
    def __init__(self, output_router_logits=False, router_aux_loss_coef=0.0, model_type="test_moe"):
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.model_type = model_type


class _Model(torch.nn.Module):
    """The ``output_router_logits`` forward parameter is how HF marks a class ``aux_loss`` can reach."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")


class _SeveredEPLayer(torch.nn.Module):
    """Severs the HF aux-loss path (DeepSeek-V4-like)."""

    SEVERS_AUX_LOSS = True
    SUPPORTS_BIAS_BALANCING = True

    def can_adopt_native_balancing(self):
        return False


class _AdoptedBiasEPLayer(torch.nn.Module):
    """The buffer lives in ``_buffers``, not ``__dict__``, so the strict ``bias_update`` export contract accepts it."""

    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None

    def enable_bias_balancing(self):
        return True


def _run_callback(model):
    # The aux_loss-mode permission is the strongest case for the stamp to override.
    cb = MoEMetricsCallback(topk=2, enable_router_logits=True)
    cb.on_train_begin(args=None, state=None, control=None, model=model)
    return cb


def test_callback_enables_router_logits_on_plain_moe():
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
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    if adopted_router:
        model.ep_layer = _AdoptedBiasEPLayer()
    apply_balancing_mode(model, mode, **{"is_moe": False, **kwargs})
    assert router_logits_forced_off(model.config), f"{mode} {kwargs} did not stamp the config"
    _run_callback(model)
    assert model.config.output_router_logits is False


def test_explicit_bias_mode_on_a_dense_model_raises():
    model = _Model(_Cfg(output_router_logits=False, router_aux_loss_coef=0.0))
    with pytest.raises(ValueError, match="dense model"):
        apply_balancing_mode(model, "bias_update", is_moe=False)


def test_severed_ep_aux_loss_stays_off_through_callback():
    """DeepSeek-V4-under-EP shape: re-enabling logits would crash the HF aux-loss reduction on the empty tuple."""
    cfg = _Cfg(output_router_logits=False, router_aux_loss_coef=0.01)
    model = _Model(cfg)
    model.ep_layer = _SeveredEPLayer()
    apply_balancing_mode(model, "aux_loss", is_moe=True)
    assert router_logits_forced_off(cfg)
    _run_callback(model)
    assert cfg.output_router_logits is False


def test_aux_loss_with_usable_coef_not_stamped():
    cfg = _Cfg(output_router_logits=False, router_aux_loss_coef=0.01)
    model = _Model(cfg)
    apply_balancing_mode(model, "aux_loss", is_moe=True)
    assert cfg.output_router_logits is True
    assert not router_logits_forced_off(cfg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
