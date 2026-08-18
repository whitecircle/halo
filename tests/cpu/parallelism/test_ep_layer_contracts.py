#!/usr/bin/env python
"""CPU tests for class-declared EP-layer contracts.

* ``_supports_gradient_checkpointing``: Zaya declares ``False`` (EDA/CCA recompute recursion) and
  ``EpIntrospectionMixin._setup_ep_gradient_checkpointing`` must RAISE when GC is enabled on a
  declaring layer instead of silently corrupting backward; supporting families pass through.
* GptOss activation constants: ``EPGptOssMoELayer`` must read ``alpha``/``limit`` from the wrapped
  experts (1.702 / swiglu_limit) — a missing attribute is an ``AttributeError``, never a silently
  substituted wrong constant.
* ``_ROUTER_ATTR`` / ``_EXPERTS_CONTAINER_ATTRS``: the router and expert-container probes must read
  the family's own declaration, not guess from a global name ladder, and the per-layer construction
  INFO must be rank-gated.

Run: ``python tests/cpu/parallelism/test_ep_layer_contracts.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.ep_introspection import EpIntrospectionMixin
from tests.common.parallelism import make_parallelism_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def _parallelism_config(world: int = 2, **kwargs) -> ParallelismConfig:
    kwargs.setdefault("gpus_per_node", world)
    return make_parallelism_config(world_size=world, ep_size=2, **kwargs)


class _EPStub(EPMoELayerBase):
    """Minimal EP layer the introspection mixin recognizes.

    A real ``EPMoELayerBase`` subclass, not a duck-typed stand-in: the mixin identifies EP modules by
    ``isinstance``, so a bare ``nn.Module`` carrying ``ep_config`` would not exercise the live path.
    ``EPMoELayerBase.__init__`` needs a wrapped HF layer + DeepEP, so it is bypassed deliberately.
    """

    def __init__(self, supports_gc: bool | None):
        nn.Module.__init__(self)
        self.ep_config = SimpleNamespace(is_deferred_dp=False)
        if supports_gc is not None:
            self._supports_gradient_checkpointing = supports_gc

    def forward(self, hidden_states):  # abstract on the base; never called by these tests
        raise NotImplementedError


class _GCSequential(nn.Sequential):
    """Stub policy model: records the GC kwargs the EP path forces on it.

    ``enable_ep_gradient_checkpointing`` raises on a model with no
    ``gradient_checkpointing_enable`` — returning quietly would leave GC off on BOTH sides,
    since the EP path has already disabled HF Trainer's own. Installing
    ``_gradient_checkpointing_func`` mirrors ``PreTrainedModel``: the EP path re-points it at the
    scoped wrapper its dispatch replay hangs off.
    """

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
        self.gc_kwargs = gradient_checkpointing_kwargs
        self.every_n_layers = every_n_layers
        self._gradient_checkpointing_func = checkpoint


class _Trainer(EpIntrospectionMixin):
    def __init__(
        self, layer: nn.Module, gradient_checkpointing: bool, gradient_checkpointing_kwargs=None, **config_kwargs
    ):
        self.model = _GCSequential(layer)
        self._ep_config = layer.ep_config
        self.parallelism_config = _parallelism_config(**config_kwargs)
        self.args = SimpleNamespace(
            gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )


def test_zaya_declares_gc_unsupported_and_base_default_is_supported():
    assert EPZayaMoELayer._supports_gradient_checkpointing is False
    # Absent declaration = supported (base contract default).
    assert getattr(EPGptOssMoELayer, "_supports_gradient_checkpointing", True) is True


def test_gc_on_declaring_false_layer_raises():
    trainer = _Trainer(_EPStub(supports_gc=False), gradient_checkpointing=True)
    with pytest.raises(ValueError, match="_supports_gradient_checkpointing"):
        trainer._setup_ep_gradient_checkpointing()


def test_gc_disabled_skips_the_contract_check():
    trainer = _Trainer(_EPStub(supports_gc=False), gradient_checkpointing=False)
    trainer._setup_ep_gradient_checkpointing()


def test_gc_on_supporting_layer_passes():
    trainer = _Trainer(_EPStub(supports_gc=None), gradient_checkpointing=True)
    with patch("src.trainers.mixins.ep_introspection.enable_ep_gradient_checkpointing") as enable:
        trainer._setup_ep_gradient_checkpointing()
    enable.assert_called_once_with(trainer.model, gradient_checkpointing_kwargs={"use_reentrant": True})
    assert trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is True
    assert trainer.args.gradient_checkpointing is False  # HF Trainer re-enable prevented


def test_gc_under_pp_is_non_reentrant():
    # PP needs non-reentrant GC: a reentrant forward runs under no_grad, so FSDP2 registers no pre-backward hooks.
    trainer = _Trainer(_EPStub(supports_gc=None), gradient_checkpointing=True, world=4, gpus_per_node=2, pp_size=2)
    with patch("src.trainers.mixins.ep_introspection.enable_ep_gradient_checkpointing") as enable:
        trainer._setup_ep_gradient_checkpointing()
    enable.assert_called_once_with(trainer.model, gradient_checkpointing_kwargs={"use_reentrant": False})
    assert trainer.args.gradient_checkpointing_kwargs["use_reentrant"] is False


def test_every_n_layers_is_lifted_out_of_the_checkpoint_kwargs():
    """``every_n_layers`` is ``gradient_checkpointing_enable``'s own keyword (which decoder layers
    checkpoint); left inside the kwargs dict it reaches ``torch.utils.checkpoint``. The EP enable
    lifts it, and so does the re-enable seam TRL drives with the raw args dict after generation."""
    trainer = _Trainer(
        _EPStub(supports_gc=None), gradient_checkpointing=True, gradient_checkpointing_kwargs={"every_n_layers": 2}
    )
    trainer._setup_ep_gradient_checkpointing()
    model = trainer.model
    assert (model.gc_kwargs, model.every_n_layers) == ({"use_reentrant": True}, 2)
    assert trainer.args.gradient_checkpointing_kwargs == {"every_n_layers": 2, "use_reentrant": True}

    model.every_n_layers = None
    model.gradient_checkpointing_enable(trainer.args.gradient_checkpointing_kwargs)  # TRL's raw re-enable
    assert (model.gc_kwargs, model.every_n_layers) == ({"use_reentrant": True}, 2)


def test_gc_enable_requires_a_real_checkpoint_function():
    # A no-op gradient_checkpointing_enable leaves EP issuing a SECOND dispatch inside backward.
    class _Silent(nn.Sequential):
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
            pass

    with pytest.raises(RuntimeError, match="installed no checkpoint function"):
        enable_ep_gradient_checkpointing(
            _Silent(_EPStub(supports_gc=None)), gradient_checkpointing_kwargs={"use_reentrant": True}
        )


def test_repeated_gc_reenables_neither_stack_nor_lose_the_scope():
    """Re-enabling GC must not strip the EP scopes, however many times it happens.

    Online GRPO re-enables every step: TRL wraps generation in ``disable_gradient_checkpointing``,
    whose exit calls the model's ``gradient_checkpointing_enable`` again. That installs HF's bare
    checkpoint function, and an EP layer entering backward without a scope raises rather than issue
    the second DeepEP dispatch that would corrupt every gradient.
    """
    model = _GCSequential(_EPStub(supports_gc=None))
    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    wrapper = model.gradient_checkpointing_enable

    for _ in range(3):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

    assert model.gradient_checkpointing_enable is wrapper, "the re-enable wrapper nested itself"
    assert getattr(model._gradient_checkpointing_func, "_ep_scoped", False)


def test_a_second_full_enable_is_idempotent_not_a_raise():
    """A mode switch re-runs the FULL ``enable_ep_gradient_checkpointing``. The rescope hook has
    already wrapped the fresh install by then, so ``install_ep_checkpoint_scopes`` finds only
    already-scoped functions — which must read as "scopes in place", never as the no-checkpoint
    RuntimeError."""
    model = _GCSequential(_EPStub(supports_gc=None))
    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    assert getattr(model._gradient_checkpointing_func, "_ep_scoped", False)


# GptOss activation constants (alpha / limit)


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


def _ep1_config() -> EPConfig:
    cfg = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    cfg.finalize_expert_assignment(E)
    return cfg


def test_gptoss_layer_reads_activation_constants_from_experts():
    layer = EPGptOssMoELayer(_GptOssBlock(), _ep1_config())
    assert layer.alpha == 1.702
    assert layer.limit == 7.0


def test_gptoss_layer_fails_loud_on_missing_activation_constants():
    with pytest.raises(AttributeError, match="alpha"):
        EPGptOssMoELayer(_GptOssBlock(with_activation_constants=False), _ep1_config())


def _grouped_mm_probe(use_grouped_mm: bool, expert_tp_size: int) -> EPGptOssMoELayer:
    """A layer carrying only what ``_grouped_mm_enabled`` reads — the method is pure, and a real
    ``__init__`` needs DeepEP groups and a device."""
    layer = EPGptOssMoELayer.__new__(EPGptOssMoELayer)
    layer._use_grouped_mm = use_grouped_mm
    layer.expert_tp_size = expert_tp_size
    return layer


def test_gptoss_etp_disables_grouped_mm():
    """Expert-TP drops GptOss expert compute to the per-expert loop: ETP stores gate/up under the
    plain names, not the de-interleaved ``*_gmm`` pair ``_compute_experts_gmm`` reads."""
    assert _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=2)._grouped_mm_enabled() is False
    assert _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=1)._grouped_mm_enabled() is True
    assert _grouped_mm_probe(use_grouped_mm=False, expert_tp_size=1)._grouped_mm_enabled() is False


def test_init_summary_reports_the_effective_grouped_mm_decision(caplog):
    """The construction-time line must report what the layer WILL run, not what was requested.

    Reporting ``_use_grouped_mm`` would advertise grouped GEMM on a GptOss ETP layer that silently
    runs the per-expert loop — an unexplained throughput cliff on a run asking for grouped GEMM.
    """
    layer = _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=2)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 0, 0, E, E
    layer.fp32_router = layer.fp32_experts = False

    with caplog.at_level(logging.INFO, logger=EPMoELayerBase.__module__):
        layer._log_init_summary("extra=1")

    (record,) = [r for r in caplog.records if r.levelno == logging.INFO]
    assert "grouped_mm=False" in record.getMessage(), record.getMessage()
    assert "extra=1" in record.getMessage(), record.getMessage()


@pytest.mark.parametrize("is_main", [True, False])
def test_init_summary_is_rank_gated(caplog, is_main):
    """One INFO line per MoE layer per rank is ~47k lines on a 92-layer model at 512 ranks, so only
    the main process emits it — and it may therefore carry only rank-uniform fields."""
    layer = _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=1)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 0, 0, E, E
    layer.fp32_router = layer.fp32_experts = False

    with (
        patch("src.distributed.expert_parallel.base_layer.is_global_main_process", return_value=is_main),
        caplog.at_level(logging.INFO, logger=EPMoELayerBase.__module__),
    ):
        layer._log_init_summary()

    emitted = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(emitted) == (1 if is_main else 0)
    assert not any("owns experts" in r.getMessage() for r in emitted), "rank-specific field on a rank-0-only line"


@pytest.mark.parametrize("is_main", [True, False])
def test_every_rank_debug_logs_the_expert_range_it_owns(caplog, is_main):
    """The one rank-specific fact must survive the INFO gate: printed only on rank 0 it says nothing
    about the actual split, so every rank emits its own range at DEBUG."""
    layer = _grouped_mm_probe(use_grouped_mm=True, expert_tp_size=1)
    layer.ep_rank, layer.expert_start, layer.expert_end, layer.num_experts = 1, E // 2, E, E
    layer.fp32_router = layer.fp32_experts = False

    with (
        patch("src.distributed.expert_parallel.base_layer.is_global_main_process", return_value=is_main),
        caplog.at_level(logging.DEBUG, logger=EPMoELayerBase.__module__),
    ):
        layer._log_init_summary()

    (record,) = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert f"EP rank 1 owns experts {E // 2}-{E - 1} of {E}" in record.getMessage(), record.getMessage()


# router resolution (_ROUTER_ATTR)


def test_router_probe_reads_the_declared_attribute_not_a_name_ladder():
    """A block exposing BOTH names must resolve to the family's declaration.

    A ``("gate", "router")`` ladder would hand GptOss the ``gate`` submodule — silently overriding
    ``_ROUTER_ATTR = "router"`` and hooking / upcasting a different module than the forward reads.
    """
    both = SimpleNamespace(gate=nn.Linear(2, 2), router=nn.Linear(2, 2))

    assert EPGptOssMoELayer._find_gate_or_router(both) is both.router
    assert EPMoELayerBase._find_gate_or_router(both) is both.gate, "the default declaration is 'gate'"


def test_router_probe_fails_loud_and_names_the_declaration():
    """A renamed upstream router must say which declaration went stale, not 'no gate/router found'."""
    with pytest.raises(AttributeError, match="_ROUTER_ATTR='router'"):
        EPGptOssMoELayer._find_gate_or_router(SimpleNamespace(gate=nn.Linear(2, 2)))


def test_top_k_probe_reads_the_declared_router_not_a_name_ladder():
    """``_find_top_k`` descends through the SAME declaration the hooks and the upcast use.

    A ``("gate", "router")`` ladder would read GptOss's top-k off a ``gate`` submodule while the
    forward routes through ``router`` — a silent expert-count mismatch on any block carrying both.
    """
    both = SimpleNamespace(
        gate=SimpleNamespace(num_experts_per_tok=2),
        router=SimpleNamespace(num_experts_per_tok=7),
    )

    assert EPGptOssMoELayer._find_top_k(both) == 7
    assert EPMoELayerBase._find_top_k(both) == 2, "the default declaration is 'gate'"


# expert-container resolution (_EXPERTS_CONTAINER_ATTRS)


def test_expert_container_probe_reads_the_family_declaration_in_its_own_order():
    """A block exposing BOTH spellings must resolve to the family's declaration, in its order.

    GLM-4 (and Laguna, which inherits it) is the one family served under two container spellings; every
    other family declares the single ``experts``. A global name ladder would let a block that happened
    to expose a second name decide for all of them.
    """
    both = SimpleNamespace(experts=nn.Linear(2, 2), routed_experts=nn.Linear(2, 2))

    assert EPGlm4MoELayer._find_experts_container(both) is both.experts
    assert EPMoELayerBase._find_experts_container(both) is both.experts, "the default declaration is 'experts'"


def test_expert_container_probe_accepts_only_the_spellings_the_family_declares():
    """A spelling the family does not declare must fail LOUD and name the declaration.

    The same declarations are unioned into the lazy loader's checkpoint-key regexes, so a container
    resolved off a name outside them builds a layer whose expert keys the planner never matches.
    """
    routed_only = SimpleNamespace(routed_experts=nn.Linear(2, 2))
    assert EPGlm4MoELayer._find_experts_container(routed_only) is routed_only.routed_experts

    with pytest.raises(AttributeError, match="_EXPERTS_CONTAINER_ATTRS"):
        EPQwen3MoELayer._find_experts_container(routed_only)


def test_the_gptoss_wrapper_stores_its_router_under_the_declared_attribute():
    """The declaration names one attribute on BOTH sides — the HF block it is read from and the
    wrapper it is re-registered on — so the fp32 upcast and the grad-sync hook (both keyed off
    ``_ROUTER_ATTR``) reach the module the forward actually routes through."""
    layer = EPGptOssMoELayer(_GptOssBlock(), _ep1_config())
    assert getattr(layer, EPGptOssMoELayer._ROUTER_ATTR) is layer.router


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
