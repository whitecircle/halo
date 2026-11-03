#!/usr/bin/env python
"""Class-declared EP-layer contracts: GC support, activation constants, router and container probes, mixin rules."""

from __future__ import annotations

import logging
from abc import ABC
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3MoeConfig

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.distributed.expert_parallel.moe_layer.expert_lora import EPExpertLoraMixin
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.fused_glu import GluCombine
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.parallelism.ep_layers import EPLayersMixin
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def _parallelism_config(world: int = 2, **kwargs) -> ParallelismConfig:
    kwargs.setdefault("gpus_per_node", world)
    return make_parallelism_config(world_size=world, ep_size=2, **kwargs)


class _EPStub(EPMoELayerBase):
    """A real subclass since the mixin identifies EP modules by ``isinstance``; ``__init__`` needs a wrapped HF
    layer + DeepEP, so it is bypassed."""

    def __init__(self, supports_gc: bool = True):
        nn.Module.__init__(self)
        self.ep_config = SimpleNamespace(is_deferred_dp=False)
        self.FAMILY = MoEFamilySpec(model_types=(), supports_gradient_checkpointing=supports_gc)

    def forward(self, hidden_states):  # Abstract on the base; never called by these tests.
        raise NotImplementedError


class _GCSequential(nn.Sequential):
    """Installing ``_gradient_checkpointing_func`` mirrors ``PreTrainedModel``: the EP path re-points it at the
    scoped wrapper its dispatch replay hangs off."""

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
    ):
        self.gc_kwargs = gradient_checkpointing_kwargs
        self.every_n_layers = every_n_layers
        self.offload = offload
        self._gradient_checkpointing_func = checkpoint


class _Trainer(EPLayersMixin):
    def __init__(
        self, layer: nn.Module, gradient_checkpointing: bool, gradient_checkpointing_kwargs=None, **config_kwargs
    ):
        self.model = _GCSequential(layer)
        self._ep_config = layer.ep_config
        self.parallelism_config = _parallelism_config(**config_kwargs)
        self.args = SimpleNamespace(
            gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def _top_level_model(self) -> nn.Module:
        return self.model


def test_zaya_declares_gc_unsupported_and_base_default_is_supported():
    assert EPZayaMoELayer.FAMILY.supports_gradient_checkpointing is False
    assert EPGptOssMoELayer.FAMILY.supports_gradient_checkpointing is True


def test_gc_on_declaring_false_layer_raises():
    trainer = _Trainer(_EPStub(supports_gc=False), gradient_checkpointing=True)
    with pytest.raises(ValueError, match="supports_gradient_checkpointing=False"):
        trainer._install_ep_gradient_checkpointing()


def test_gc_disabled_skips_the_contract_check():
    trainer = _Trainer(_EPStub(supports_gc=False), gradient_checkpointing=False)
    trainer._install_ep_gradient_checkpointing()


def test_gc_on_supporting_layer_passes():
    trainer = _Trainer(_EPStub(), gradient_checkpointing=True)
    with patch("halo.trainers.parallelism.ep_layers.enable_scoped_gradient_checkpointing") as enable:
        trainer._install_ep_gradient_checkpointing()
    enable.assert_called_once_with(trainer.model, gradient_checkpointing_kwargs={"use_reentrant": True})
    assert trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is True
    assert trainer.args.gradient_checkpointing is False


def test_gc_on_grouped_gemm_only_moe_stays_reentrant():
    """A router re-run in non-reentrant recompute can flip a near-tie pick."""
    trainer = _Trainer(_EPStub(), gradient_checkpointing=True)
    trainer.parallelism_config = make_parallelism_config(world_size=2, gpus_per_node=2)
    trainer.model.config = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)
    assert not trainer.parallelism_config.has_ep_group, "fixture must leave only the config to decide"
    with patch("halo.trainers.parallelism.ep_layers.enable_scoped_gradient_checkpointing") as enable:
        trainer._install_ep_gradient_checkpointing()
    enable.assert_called_once_with(trainer.model, gradient_checkpointing_kwargs={"use_reentrant": True})


def test_gc_under_pp_is_non_reentrant():
    # PP needs non-reentrant GC: a reentrant forward runs under no_grad, so FSDP2 registers no pre-backward hooks.
    trainer = _Trainer(_EPStub(), gradient_checkpointing=True, world=4, gpus_per_node=2, pp_size=2)
    with patch("halo.trainers.parallelism.ep_layers.enable_scoped_gradient_checkpointing") as enable:
        trainer._install_ep_gradient_checkpointing()
    enable.assert_called_once_with(trainer.model, gradient_checkpointing_kwargs={"use_reentrant": False})
    assert trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is False


def test_enable_keywords_are_lifted_out_of_the_checkpoint_kwargs():
    """Left inside the kwargs dict they reach ``torch.utils.checkpoint``."""
    trainer = _Trainer(
        _EPStub(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"every_n_layers": 2, "offload": True},
    )
    trainer._install_ep_gradient_checkpointing()
    model = trainer.model
    assert (model.gc_kwargs, model.every_n_layers, model.offload) == ({"use_reentrant": True}, 2, True)
    assert trainer.args.gradient_checkpointing_kwargs == {"every_n_layers": 2, "offload": True, "use_reentrant": True}

    model.every_n_layers, model.offload = None, None
    model.gradient_checkpointing_enable(trainer.args.gradient_checkpointing_kwargs)  # TRL's raw re-enable.
    assert (model.gc_kwargs, model.every_n_layers, model.offload) == ({"use_reentrant": True}, 2, True)


def test_gc_enable_requires_a_real_checkpoint_function():
    # A no-op gradient_checkpointing_enable leaves EP issuing a second dispatch inside backward.
    class _Silent(nn.Sequential):
        def gradient_checkpointing_enable(
            self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
        ):
            pass

    with pytest.raises(RuntimeError, match="installed no checkpoint function"):
        enable_scoped_gradient_checkpointing(_Silent(_EPStub()), gradient_checkpointing_kwargs={"use_reentrant": True})


def test_repeated_gc_reenables_neither_stack_nor_lose_the_scope():
    """TRL's ``disable_gradient_checkpointing`` exit re-enables every step of online GRPO, and an EP layer entering
    backward without a scope raises."""
    model = _GCSequential(_EPStub())
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    wrapper = model.gradient_checkpointing_enable

    for _ in range(3):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    assert model.gradient_checkpointing_enable is wrapper, "the re-enable wrapper nested itself"
    assert getattr(model._gradient_checkpointing_func, "_ep_scoped", False)


def test_a_second_full_enable_is_idempotent_not_a_raise():
    """The rescope hook has already wrapped the fresh install, so already-scoped functions read as scopes in place."""
    model = _GCSequential(_EPStub())
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    assert getattr(model._gradient_checkpointing_func, "_ep_scoped", False)


class _GptOssRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.bias = nn.Parameter(torch.randn(E))
        self.top_k = K
        self.num_experts = E


class _GptOssExperts(nn.Module):
    def __init__(self, with_activation_constants: bool = True):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, 2 * M))
        self.gate_up_proj_bias = nn.Parameter(torch.randn(E, 2 * M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        self.down_proj_bias = nn.Parameter(torch.randn(E, H))
        self.num_experts = E
        if with_activation_constants:
            self.alpha = 1.702
            self.limit = 7.0


class _GptOssBlock(nn.Module):
    def __init__(self, with_activation_constants: bool = True):
        super().__init__()
        self.router = _GptOssRouter()
        self.experts = _GptOssExperts(with_activation_constants)


def test_gptoss_layer_reads_activation_constants_from_experts():
    layer = EPGptOssMoELayer(_GptOssBlock(), single_process_ep_config(E))
    assert layer._gptoss_glu == GluCombine.gptoss(1.702, 7.0)


def test_gptoss_layer_fails_loud_on_missing_activation_constants():
    with pytest.raises(AttributeError, match="alpha"):
        EPGptOssMoELayer(_GptOssBlock(with_activation_constants=False), single_process_ep_config(E))


def _grouped_mm_probe(use_grouped_mm: bool, expert_tp_size: int) -> EPGptOssMoELayer:
    """A real ``__init__`` needs DeepEP groups and a device."""
    layer = EPGptOssMoELayer.__new__(EPGptOssMoELayer)
    layer._use_grouped_mm = use_grouped_mm
    layer.expert_tp_size = expert_tp_size
    layer._gptoss_glu = GluCombine.gptoss(1.702, 7.0)
    return layer


@pytest.mark.parametrize("use_grouped_mm", [True, False])
def test_init_summary_reports_the_effective_grouped_mm_decision(caplog, use_grouped_mm):
    layer = _grouped_mm_probe(use_grouped_mm=use_grouped_mm, expert_tp_size=2)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 0, 0, E, E
    layer.fp32_router = layer.fp32_experts = False

    with caplog.at_level(logging.INFO, logger=EPMoELayerBase.__module__):
        layer._log_init_summary("extra=1")

    (record,) = [r for r in caplog.records if r.levelno == logging.INFO]
    assert f"grouped_mm={use_grouped_mm}" in record.getMessage(), record.getMessage()
    assert "extra=1" in record.getMessage(), record.getMessage()


@pytest.mark.parametrize("is_main", [True, False])
def test_init_summary_is_rank_gated(caplog, is_main):
    """~47k lines on a 92-layer model at 512 ranks, and it may carry only rank-uniform fields."""
    layer = _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=1)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 0, 0, E, E
    layer.fp32_router = layer.fp32_experts = False

    with (
        patch("halo.distributed.expert_parallel.moe_layer.layer.is_global_main_process", return_value=is_main),
        caplog.at_level(logging.INFO, logger=EPMoELayerBase.__module__),
    ):
        layer._log_init_summary()

    emitted = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(emitted) == (1 if is_main else 0)
    assert not any("owns experts" in r.getMessage() for r in emitted), "rank-specific field on a rank-0-only line"


@pytest.mark.parametrize("is_main", [True, False])
def test_every_rank_debug_logs_the_expert_range_it_owns(caplog, is_main):
    """Printed only on rank 0 the range says nothing about the actual split."""
    layer = _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=1)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 1, E // 2, E, E
    layer.fp32_router = layer.fp32_experts = False

    with (
        patch("halo.distributed.expert_parallel.moe_layer.layer.is_global_main_process", return_value=is_main),
        caplog.at_level(logging.DEBUG, logger=EPMoELayerBase.__module__),
    ):
        layer._log_init_summary()

    (record,) = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert f"EP rank 1 owns experts {E // 2}-{E - 1} of {E}" in record.getMessage(), record.getMessage()


def test_router_probe_reads_the_declared_attribute_not_a_name_ladder():
    """A ``("gate", "router")`` ladder would hand GptOss the ``gate`` submodule, hooking a different module than the
    forward reads."""
    both = SimpleNamespace(gate=nn.Linear(2, 2), router=nn.Linear(2, 2))

    assert EPGptOssMoELayer._declared_router(both) is both.router
    assert EPMoELayerBase._declared_router(both) is both.gate, "the default declaration is 'gate'"


def test_router_probe_fails_loud_and_names_the_declaration():
    """A renamed upstream router must say which declaration went stale."""
    with pytest.raises(AttributeError, match="_ROUTER_ATTR='router'"):
        EPGptOssMoELayer._declared_router(SimpleNamespace(gate=nn.Linear(2, 2)))


def test_top_k_probe_reads_the_declared_router_not_a_name_ladder():
    """A name ladder would read top-k off a ``gate`` submodule while the forward routes through ``router``."""
    both = SimpleNamespace(
        gate=SimpleNamespace(num_experts_per_tok=2),
        router=SimpleNamespace(num_experts_per_tok=7),
    )

    assert EPGptOssMoELayer._find_top_k(both) == 7
    assert EPMoELayerBase._find_top_k(both) == 2, "the default declaration is 'gate'"


def test_expert_container_probe_reads_the_family_declaration_in_its_own_order():
    """GLM-4 (and Laguna) is the one family served under two container spellings."""
    both = SimpleNamespace(experts=nn.Linear(2, 2), routed_experts=nn.Linear(2, 2))

    assert EPGlm4MoELayer._find_experts_container(both) is both.experts
    assert EPQwen3MoELayer._find_experts_container(both) is both.experts, "the default declaration is 'experts'"


def test_expert_container_probe_accepts_only_the_spellings_the_family_declares():
    """The lazy loader's key regexes union these declarations, so a container outside them builds a layer whose keys
    the planner never matches."""
    routed_only = SimpleNamespace(routed_experts=nn.Linear(2, 2))
    assert EPGlm4MoELayer._find_experts_container(routed_only) is routed_only.routed_experts

    with pytest.raises(AttributeError, match="experts_container_attrs"):
        EPQwen3MoELayer._find_experts_container(routed_only)


def test_the_gptoss_wrapper_stores_its_router_under_the_declared_attribute():
    """The fp32 upcast and grad-sync hook key off ``_ROUTER_ATTR``, so both must reach the module the forward routes
    through."""
    layer = EPGptOssMoELayer(_GptOssBlock(), single_process_ep_config(E))
    assert getattr(layer, EPGptOssMoELayer._ROUTER_ATTR) is layer.router


def test_the_layer_mixins_define_no_init():
    """A mixin ``__init__`` would run out of order or be skipped by the ``nn.Module.__init__`` the base calls."""
    mixins = [base for base in EPMoELayerBase.__bases__ if base not in (nn.Module, ABC)]
    assert len(mixins) == 6, f"premise: EPMoELayerBase composes its six concerns as mixins, got {mixins}"
    assert [m.__name__ for m in mixins if "__init__" in vars(m)] == []


def test_expert_lora_mixin_is_mixed_only_into_the_layer_base():
    """Adapter helpers select by ``isinstance(m, EPExpertLoraMixin)``, equal to ``EPMoELayerBase`` only while the
    base is its sole direct subclass."""
    assert EPExpertLoraMixin.__subclasses__() == [EPMoELayerBase]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
