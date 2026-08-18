#!/usr/bin/env python
"""CPU tests for EP-layer construction seams (real layer construction, no GPU/DeepEP).

Covers, against REAL family ``__init__`` runs (``EPConfig(ep_size=1)`` single-process mode, noop
dispatcher):

* every routing-replay-capable family sets ``self.top_k`` at construction — Qwen3 / Qwen3.5 /
  Bailing read it in ``_gate_weights_at`` / feed :class:`RoutingReplayInjector.top_k`, so a missing
  attribute is an ``AttributeError`` mid-RL-step;
* :class:`RoutingReplayInjector` REJECTS a layer without ``top_k`` at construction (clear error
  instead of the mid-step crash);
* full-finetune trainable params are covered by ``synced_trainable_param_ids`` for every constructed
  family (the mixin's unsynced-param guard must not fire on stock full-FT);
* ``_upcast_experts_to_fp32`` preserves ``requires_grad`` (an expert-LoRA-frozen base must stay
  frozen through the fp32 rebuild) and SKIPS the upcast in the FSDP-managed ep1 state
  (``fsdp_shard_ep1_experts`` at ``ep_group_size==1``), where FSDP2's bf16 policy would silently
  degrade it anyway;
* ``EPExpertGatherMixin.__init_subclass__`` rejects a family declaring BOTH ``_PER_EXPERT_UNFUSED_KEYS``
  and its own ``gather_expert_state_dict`` (the attribute would be silently ignored);
* LFM2 is a :class:`EPSharedExpertsMoELayerBase` with ``shared_experts=None`` and empty
  ``replicated_named_params`` — its forward (the shared base path) matches a dense reference;
* dispatcher construction logs on the main process only — every layer builds its own dispatcher, so
  an ungated line is one per MoE layer per rank.

Run: ``python tests/cpu/parallelism/test_ep_layer_construction_guards.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from src.distributed.expert_parallel.base_layer import EPMoELayerBase, EPSharedExpertsMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.trainers.grpo.rollout.routing_replay import RoutingReplayInjector

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def _ep_config(fsdp_shard_ep1_experts: bool = False) -> EPConfig:
    # use_grouped_gemm off: keeps the per-expert loop path so the forward runs on CPU even when
    # the dev box exposes a CUDA device to the test process.
    cfg = EPConfig(
        ep_size=1,
        world_size=1,
        gpus_per_node=1,
        fsdp_shard_ep1_experts=fsdp_shard_ep1_experts,
        use_grouped_gemm=False,
    )
    cfg.finalize_expert_assignment(E)
    return cfg


class _Gate(nn.Module):
    """Router stub: ``weight [E, H]`` (hidden read from shape[1]) + ``top_k``."""

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
    """DeepSeek-V4 fused experts. ``missing`` drops one attribute the container always declares —
    the layout drift a transformers rename would produce."""

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
    # _to_device targets CUDA when the dev box exposes one; this is a CPU test.
    return [layer.cpu() for layer in layers]


def test_dispatcher_construction_logs_only_on_the_main_process(monkeypatch, caplog):
    """Every MoE layer builds its own dispatcher, so a per-rank INFO here is per LAYER per rank.

    A 92-layer MoE at 512 ranks opens with ~47k identical lines, burying the diagnostics that
    matter. The construction lines are identical on every rank of a group, so the main process is
    the one that prints them.
    """
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
    """Qwen3 / Qwen3.5 / Bailing / LFM2 must set self.top_k in __init__ — Bailing reads it in
    _gate_weights_at on the first replayed forward, and the injector sizes the mask from it."""
    for layer in _build_all():
        assert layer._supports_routing_replay, f"{type(layer).__name__} unexpectedly opted out of replay"
        assert hasattr(layer, "top_k"), f"{type(layer).__name__} never set top_k"
        assert int(layer.top_k) == K


def test_injector_reads_top_k_from_constructed_layers():
    layers = _build_all()
    injector = RoutingReplayInjector(layers)
    assert injector.top_k == K
    assert injector.num_experts == E
    assert injector.num_layers == len(layers)


def test_injector_rejects_layer_without_top_k():
    """A replay-capable layer that never set top_k must be rejected at injector construction with a
    clear error, not crash mid-step."""
    layer = _build_all()[0]
    del layer.top_k
    with pytest.raises(AttributeError, match="top_k"):
        RoutingReplayInjector([layer])


def test_full_finetune_trainable_params_all_synced():
    """Every trainable param of a stock full-FT family layer must be in synced_trainable_param_ids
    (the in-backward hook path, fsdp_shard_ep1_experts=False) — else the mixin's unsynced-param
    guard would reject stock full-finetune runs."""
    for layer in _build_all(fsdp_shard_ep1_experts=False):
        synced = layer.synced_trainable_param_ids()
        offenders = [n for n, p in layer.named_parameters() if p.requires_grad and id(p) not in synced]
        assert not offenders, f"{type(layer).__name__} has unsynced trainable params: {offenders}"


def test_expert_fp32_upcast_preserves_frozen_requires_grad():
    """A frozen expert base (expert-LoRA freezes it before the upcast runs) must STAY frozen through
    the fp32 nn.Parameter rebuild — the default requires_grad=True would silently train the base."""
    torch.manual_seed(0)
    cfg = _ep_config(fsdp_shard_ep1_experts=False)
    cfg.fp32_experts = True
    layer = EPQwen3MoELayer(_Qwen3Block(), cfg)
    assert layer.gate_proj.dtype == torch.float32

    layer.gate_proj.requires_grad_(False)
    layer.up_proj.requires_grad_(True)
    layer._upcast_experts_to_fp32(layer.expert_named_params())
    assert layer.gate_proj.dtype == torch.float32
    assert layer.gate_proj.requires_grad is False, "frozen expert base re-enabled by the fp32 upcast"
    assert layer.up_proj.requires_grad is True, "trainable expert param frozen by the fp32 upcast"


def test_expert_fp32_upcast_skipped_when_fsdp_manages_ep1_experts():
    """fsdp_shard_ep1_experts at ep_group_size==1: FSDP2's bf16 param_dtype policy casts the experts
    for compute anyway, so the upcast is skipped (logged) instead of silently degrading."""
    torch.manual_seed(0)
    cfg = _ep_config(fsdp_shard_ep1_experts=True)
    cfg.fp32_experts = True
    layer = EPQwen3MoELayer(_Qwen3Block(), cfg)
    assert layer.gate_proj.dtype == torch.float32 or layer.gate_proj.dtype == torch.get_default_dtype()
    # Stub weights are fp32 already; prove the skip with a bf16 param instead.
    layer.down_proj = nn.Parameter(layer.down_proj.data.to(torch.bfloat16), requires_grad=False)
    layer._upcast_experts_to_fp32(layer.expert_named_params())
    assert layer.down_proj.dtype == torch.bfloat16, "ep1 FSDP-managed experts must not be upcast"
    assert layer.down_proj.requires_grad is False


def test_init_subclass_rejects_unfused_keys_with_gather_override():
    with pytest.raises(TypeError, match="_PER_EXPERT_UNFUSED_KEYS"):

        class _Bad(EPMoELayerBase):  # noqa: F841 — definition itself must raise
            _PER_EXPERT_UNFUSED_KEYS = ("gate_proj", "up_proj", "down_proj")

            def gather_expert_state_dict(self, device="cpu", merge_lora=False):
                return {}

            def forward(self, hidden_states, **kwargs):
                return hidden_states


def test_init_subclass_allows_each_alone():
    class _KeysOnly(EPMoELayerBase):
        _PER_EXPERT_UNFUSED_KEYS = ("gate_proj", "up_proj", "down_proj")

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _GatherOnly(EPMoELayerBase):
        def gather_expert_state_dict(self, device="cpu", merge_lora=False):
            return {}

        @classmethod
        def merge_shards_to_hf(cls, prefix, params):
            return {}

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    assert _KeysOnly is not None and _GatherOnly is not None


def test_init_subclass_requires_gather_and_merge_to_be_overridden_together():
    """The merge is the inverse of the gather: a custom gather with the BASE merge would write a
    sharded checkpoint whose expert layout silently differs from the family's own gathered save."""
    with pytest.raises(TypeError, match="merge_shards_to_hf"):

        class _GatherWithoutMerge(EPMoELayerBase):  # noqa: F841 — definition itself must raise
            def gather_expert_state_dict(self, device="cpu", merge_lora=False):
                return {}

            def forward(self, hidden_states, **kwargs):
                return hidden_states

    with pytest.raises(TypeError, match="gather_expert_state_dict"):

        class _MergeWithoutGather(EPMoELayerBase):  # noqa: F841 — definition itself must raise
            @classmethod
            def merge_shards_to_hf(cls, prefix, params):
                return {}

            def forward(self, hidden_states, **kwargs):
                return hidden_states


def test_lfm2_is_shared_experts_base_with_none_shared():
    layer = _build_all()[3]
    assert isinstance(layer, EPSharedExpertsMoELayerBase)
    assert layer.shared_experts is None
    assert layer.replicated_named_params() == []
    # The forward must come from the shared base, not a family duplicate.
    assert "forward" not in vars(EPLfm2MoELayer), "LFM2 re-grew a duplicated forward"


def test_lfm2_forward_matches_dense_reference():
    """The inherited shared-base forward (shared_experts=None) must equal a dense per-token
    reference of LFM2's sigmoid routing + SwiGLU experts."""
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
        gate_up = layer.gate_up_proj.data  # [E, H, 2M] matmul convention
        down = layer.down_proj.data  # [E, M, H]
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
    """A default substituted for a missing attribute is a silent numerics change, not a fallback.

    ``limit`` is ``config.swiglu_limit`` and clamps both GLU halves of every expert; ``act_fn`` is
    ``config.hidden_act`` and additionally decides ``_glu_is_silu``, which selects the compiled
    SiLU-only combine. Standing a literal in for either — the layout drift a transformers rename
    would cause — trains the model against a different activation than the checkpoint was built
    with, and nothing says so. GptOss reads ``experts.alpha`` / ``experts.limit`` directly for the
    same reason.
    """
    with pytest.raises(AttributeError, match=missing):
        EPDeepseekV4MoELayer(_DeepseekV4Block(missing), _ep_config())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
