#!/usr/bin/env python
"""EP-layer construction seams against real family ``__init__`` runs, ``EPConfig(ep_size=1)`` single-process with a
noop dispatcher."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.deepep import dispatcher as dispatcher_mod
from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def _ep_config(experts_fsdp_managed: bool = False) -> EPConfig:
    # Keeps the per-expert loop so the forward runs on CPU even when the dev box exposes a CUDA device.
    cfg = EPConfig(
        ep_size=1,
        stage_world_size=1,
        nvlink_domain_size=1,
        experts_fsdp_managed=experts_fsdp_managed,
        use_grouped_gemm=False,
    )
    cfg.finalize_expert_assignment(E)
    return cfg


class _Gate(nn.Module):
    """Hidden is read from shape[1]."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.top_k = K
        self.norm_topk_prob = True


class _FusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * M, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, M))


class _Qwen3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.experts = _FusedExperts()
        self.top_k = K


class _Qwen35Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.experts = _FusedExperts()
        self.shared_expert = None
        self.shared_expert_gate = None


class _BailingGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.top_k = K
        self.num_experts = E
        self.routed_scaling_factor = 1.0


class _BailingExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, M, bias=False)
        self.up_proj = nn.Linear(H, M, bias=False)
        self.down_proj = nn.Linear(M, H, bias=False)


class _BailingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _BailingGate()
        self.experts = nn.ModuleList([_BailingExpert() for _ in range(E)])


class _Lfm2Gate(nn.Module):
    # Mirrors Lfm2MoeTopKRouter: the wrapper reads the routing constants off the gate, no defaults.
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.num_experts = E
        self.top_k = K
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.use_expert_bias = False

    def forward(self, x):
        return F.linear(x, self.weight)


class _Lfm2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Lfm2Gate()
        self.experts = _FusedExperts()
        self.top_k = K
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.use_expert_bias = False


class _DeepseekV4Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.top_k = K
        self.num_experts = E
        self.routed_scaling_factor = 1.0
        self.score_fn = torch.sigmoid
        self.e_score_correction_bias = nn.Parameter(torch.zeros(E), requires_grad=False)


class _DeepseekV4Experts(nn.Module):
    """``missing`` drops one attribute, the layout drift a transformers rename would produce."""

    def __init__(self, missing: str | None = None):
        super().__init__()
        self.num_experts = E
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * M, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, M))
        if missing != "act_fn":
            self.act_fn = F.silu
        if missing != "limit":
            self.limit = 7.0


class _DeepseekV4Block(nn.Module):
    def __init__(self, missing: str | None = None):
        super().__init__()
        self.gate = _DeepseekV4Gate()
        self.experts = _DeepseekV4Experts(missing)
        self.shared_experts = None
        self.is_hash = False


def _build_all(fsdp_shard_ep1_experts: bool = False):
    torch.manual_seed(0)
    cfg = _ep_config(fsdp_shard_ep1_experts)
    layers = [
        EPQwen3MoELayer(_Qwen3Block(), cfg),
        EPQwen3_5MoELayer(_Qwen35Block(), cfg),
        EPBailingMoELayer(_BailingBlock(), cfg),
        EPLfm2MoELayer(_Lfm2Block(), cfg),
    ]
    # Construction places the layers on CUDA when the dev box exposes one.
    return [layer.cpu() for layer in layers]


def test_dispatcher_construction_logs_only_on_the_main_process(monkeypatch, caplog):
    """A 92-layer MoE at 512 ranks would open with ~47k identical lines."""
    caplog.set_level(logging.INFO, logger=dispatcher_mod.__name__)
    monkeypatch.setattr(dispatcher_mod, "is_global_main_process", lambda: False)

    _build_all()

    assert [r for r in caplog.records if "noop" in r.message] == [], (
        "the noop-dispatcher line fired on a non-main rank, once per MoE layer"
    )

    caplog.clear()
    monkeypatch.setattr(dispatcher_mod, "is_global_main_process", lambda: True)
    _build_all()
    assert [r for r in caplog.records if "noop" in r.message], "anti-vacuity: main rank must still report it"


def test_replay_capable_families_set_top_k_at_construction():
    """Bailing reads it in _gate_weights_at on the first replayed forward, and the injector sizes the mask from it."""
    for layer in _build_all():
        assert layer.FAMILY.supports_routing_replay, f"{type(layer).__name__} unexpectedly opted out of replay"
        assert hasattr(layer, "top_k"), f"{type(layer).__name__} never set top_k"
        assert int(layer.top_k) == K


def test_injector_reads_top_k_from_constructed_layers():
    layers = _build_all()
    injector = RoutingReplayInjector(layers, engine_layers=len(layers), layer_indices=list(range(len(layers))))
    assert injector.top_k == K
    assert injector.num_experts == E
    assert injector.num_ep_layers == len(layers)


def test_injector_rejects_layer_without_top_k():
    """Otherwise it crashes mid-step."""
    layer = _build_all()[0]
    del layer.top_k
    with pytest.raises(AttributeError, match="top_k"):
        RoutingReplayInjector([layer], engine_layers=1, layer_indices=[0])


def test_full_finetune_trainable_params_all_synced():
    """Else the mixin's unsynced-param guard would reject stock full-finetune runs."""
    for layer in _build_all(fsdp_shard_ep1_experts=False):
        synced = layer.synced_trainable_param_ids()
        offenders = [n for n, p in layer.named_parameters() if p.requires_grad and id(p) not in synced]
        assert not offenders, f"{type(layer).__name__} has unsynced trainable params: {offenders}"


def test_expert_fp32_upcast_preserves_frozen_requires_grad():
    """The default requires_grad=True would silently train the frozen base."""
    torch.manual_seed(0)
    cfg = _ep_config(experts_fsdp_managed=False)
    cfg.fp32_experts = True
    layer = EPQwen3MoELayer(_Qwen3Block(), cfg)
    assert layer.gate_proj.dtype == torch.float32

    layer.gate_proj.requires_grad_(False)
    layer.up_proj.requires_grad_(True)
    layer._upcast_experts_to_fp32()
    assert layer.gate_proj.dtype == torch.float32
    assert layer.gate_proj.requires_grad is False, "frozen expert base re-enabled by the fp32 upcast"
    assert layer.up_proj.requires_grad is True, "trainable expert param frozen by the fp32 upcast"


def test_expert_fp32_upcast_skipped_when_fsdp_manages_ep1_experts():
    """FSDP2's bf16 param_dtype policy casts the experts anyway, so the upcast is skipped instead of silently
    degrading."""
    torch.manual_seed(0)
    cfg = _ep_config(experts_fsdp_managed=True)
    cfg.fp32_experts = True
    layer = EPQwen3MoELayer(_Qwen3Block(), cfg)
    assert layer.gate_proj.dtype == torch.float32 or layer.gate_proj.dtype == torch.get_default_dtype()
    # Stub weights are fp32 already, so a bf16 param proves the skip.
    layer.down_proj = nn.Parameter(layer.down_proj.data.to(torch.bfloat16), requires_grad=False)
    layer._upcast_experts_to_fp32()
    assert layer.down_proj.dtype == torch.bfloat16, "ep1 FSDP-managed experts must not be upcast"
    assert layer.down_proj.requires_grad is False


def test_lfm2_is_shared_experts_base_with_none_shared():
    layer = _build_all()[3]
    assert isinstance(layer, EPSharedExpertsMoELayerBase)
    assert layer.shared_experts is None
    assert layer.replicated_named_params() == []
    # The forward must come from the shared base.
    assert "forward" not in vars(EPLfm2MoELayer), "LFM2 re-grew a duplicated forward"


def test_lfm2_forward_matches_dense_reference():
    torch.manual_seed(1)
    layer = _build_all()[3]
    layer.eval()
    x = torch.randn(2, 3, H)

    with torch.no_grad():
        out = layer(x)

        flat = x.view(-1, H)
        scores = torch.sigmoid(F.linear(flat, layer.gate.weight).float())
        topw, topi = torch.topk(scores, k=K, dim=-1)
        topw = topw / (topw.sum(dim=-1, keepdim=True) + 1e-6)
        ref = torch.zeros_like(flat)
        gate_up = layer.gate_up_proj.data  # [E, H, 2M] matmul convention.
        down = layer.down_proj.data
        for t in range(flat.shape[0]):
            for j in range(K):
                e = int(topi[t, j])
                gu = flat[t] @ gate_up[e]
                g, u = gu.chunk(2, dim=-1)
                ref[t] += topw[t, j] * ((F.silu(g) * u) @ down[e])

    assert out.shape == x.shape
    assert torch.allclose(out.view(-1, H), ref, atol=1e-4), (
        f"LFM2 shared-base forward diverged from dense reference (max err "
        f"{(out.view(-1, H) - ref).abs().max().item():.2e})"
    )


def test_deepseek_v4_takes_the_activation_and_limit_from_the_experts_container():
    layer = EPDeepseekV4MoELayer(_DeepseekV4Block(), _ep_config()).cpu()
    assert layer.limit == 7.0
    assert layer.act_fn is F.silu


@pytest.mark.parametrize("missing", ("limit", "act_fn"))
def test_deepseek_v4_rejects_experts_missing_an_activation_attribute(missing):
    """``limit`` clamps both GLU halves and ``act_fn`` decides the compiled combine, so a stand-in literal trains a
    different activation silently."""
    with pytest.raises(AttributeError, match=missing):
        EPDeepseekV4MoELayer(_DeepseekV4Block(missing), _ep_config())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
