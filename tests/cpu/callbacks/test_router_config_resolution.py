"""Router balancing must read and write the config the modeling code reads. A wrapper-only read of a composite
config returns the default, so ``auto`` sees no aux-loss term and MoE VLMs silently get no balancing or ``moe/*``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from halo.callbacks.balancing import apply_balancing_mode
from halo.models.introspection.config_access import get_config_field, set_config_field
from halo.models.moe.config_fields import resolve_expert_count, resolve_router_topk
from halo.models.moe.router_logits import mark_router_logits_forced_off, router_logits_forced_off


class _TextConfig:
    def __init__(self, **fields):
        self.model_type = "fake_moe_text"
        self.__dict__.update(fields)

    def get_text_config(self):
        return self


class _CompositeConfig:
    """Declares no router fields and no ``__getattr__`` delegation, like ``Qwen3_5MoeConfig``."""

    def __init__(self, text_config: _TextConfig):
        self.model_type = "fake_moe"
        self.text_config = text_config

    def get_text_config(self):
        return self.text_config


class _Model(nn.Module):
    """The ``output_router_logits`` forward parameter marks a class HF resolves the flag on. Without it this would be a
    multimodal wrapper, where ``aux_loss`` is rejected."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_ids=None, output_router_logits=None, **kwargs):
        raise NotImplementedError("stand-in: only the signature is read")


def _composite(**text_fields) -> _Model:
    return _Model(_CompositeConfig(_TextConfig(**text_fields)))


def test_aux_loss_enables_router_logits_on_the_text_config():
    model = _composite(router_aux_loss_coef=0.001, output_router_logits=False)

    apply_balancing_mode(model, "aux_loss", is_moe=True)

    text_cfg = model.config.text_config
    assert text_cfg.output_router_logits is True, (
        "aux_loss did not enable output_router_logits on the text config — the HF forward reads it "
        "there, so the aux loss is never added and balancing is silently dead."
    )
    assert not router_logits_forced_off(model.config), (
        "aux_loss wrongly concluded the model has no aux-loss term. This is the top-level-config "
        "read regressing: the coef lives on text_config."
    )


class _NativeBiasRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.zeros(4))
        self.expert_load_counter = None


def test_bias_update_zeroes_the_text_config_coef():
    model = _composite(router_aux_loss_coef=0.001, output_router_logits=True)
    model.router = _NativeBiasRouter()  # without an acceptor the balance-nothing gate raises first

    apply_balancing_mode(model, "bias_update", is_moe=True)

    text_cfg = model.config.text_config
    assert text_cfg.router_aux_loss_coef == 0.0, (
        "bias_update left the text config's router_aux_loss_coef non-zero — the model still adds "
        "the aux loss on top of the bias update (double-balancing)."
    )
    assert text_cfg.output_router_logits is False, (
        "bias_update left output_router_logits on for the text config — the EP bias path never "
        "populates router_logits, so the aux-loss reduction IndexErrors an empty tuple."
    )


def test_forced_off_stamp_is_visible_from_either_config_object():
    """Consumers hold whichever object ``model.config`` returned; both must agree."""
    model = _composite(router_aux_loss_coef=0.001)

    mark_router_logits_forced_off(model.config)

    assert router_logits_forced_off(model.config)
    assert router_logits_forced_off(model.config.text_config), (
        "the forced-off verdict is invisible from the text config, so a consumer resolving the "
        "text config would re-enable router logits the strategy deliberately turned off"
    )


def test_plain_config_is_untouched_by_the_resolver():
    model = _Model(_TextConfig(router_aux_loss_coef=0.001, output_router_logits=False))

    apply_balancing_mode(model, "aux_loss", is_moe=True)

    assert model.config.output_router_logits is True
    assert not router_logits_forced_off(model.config)


def test_get_config_field_prefers_a_real_value_over_the_wrapper_default():
    cfg = _CompositeConfig(_TextConfig(router_aux_loss_coef=0.001))

    assert get_config_field(cfg, "router_aux_loss_coef") == 0.001
    assert get_config_field(cfg, "not_a_field", "fallback") == "fallback"


def test_set_config_field_writes_both_configs():
    cfg = _CompositeConfig(_TextConfig(output_router_logits=False))

    set_config_field(cfg, "output_router_logits", True, only_declared=False)

    assert cfg.output_router_logits is True
    assert cfg.text_config.output_router_logits is True


def test_gemma4_style_top_k_experts_is_detected():
    """A zero top-k makes MoEMetricsCallback measure a top-1 distribution the model never routes, and makes
    EfficiencyCallback's sparsity factor 1.0, so S-MFU degenerates to dense MFU."""
    model = _Model(_TextConfig(num_experts=128, top_k_experts=4))

    num_experts, top_k = (resolve_expert_count(model.config), resolve_router_topk(model.config))

    assert (num_experts, top_k) == (128, 4)


def test_composite_expert_fields_are_detected():
    model = _composite(num_experts=512, num_experts_per_tok=10)

    assert (resolve_expert_count(model.config), resolve_router_topk(model.config)) == (512, 10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
