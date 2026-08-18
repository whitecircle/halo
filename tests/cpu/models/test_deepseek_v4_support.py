#!/usr/bin/env python
"""CPU tests for DeepSeek-V4 support: registrations, tiny-config save/load roundtrip, the
eager-only guards, clamped-GLU equivalence against the HF module, hash-layer input_ids handling,
the per-rope-type inv_freq buffer fix, and the moe_balancing auto resolution.

    python tests/cpu/models/test_deepseek_v4_support.py
"""

import sys
import tempfile
from functools import partial

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from src.data.collators.factory import select_data_collator
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import resolve_ep_merge_layer_class
from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.kernels.fused_glu import clamped_silu_mul_eager, fused_clamped_silu_mul
from src.models.moe_balancing import resolve_balancing_mode
from src.models.patches.attention import _model_is_deepseek_v4
from src.models.patches.buffer_fixes import fix_rotary_inv_freq
from tests.common.models import TINY_DSV4_CONFIG

SEED = 1234


def _tiny_config(**overrides) -> DeepseekV4Config:
    return DeepseekV4Config(**{**TINY_DSV4_CONFIG, **overrides})


def _tiny_model(config: DeepseekV4Config | None = None):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(config or _tiny_config())
    randomize_tid2eid(model)
    return model.eval()


def randomize_tid2eid(model, seed: int = SEED) -> None:
    """Fill hash-layer tid2eid with DISTINCT experts per token id (random-init leaves it all-zero;
    DeepEP dispatch and the wrapper's init guard both require distinct top-k experts per token)."""
    gen = torch.Generator().manual_seed(seed)
    num_experts = model.config.n_routed_experts
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], num_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])


def _bare_ep_layer(**attrs) -> EPDeepseekV4MoELayer:
    """An EPDeepseekV4MoELayer skeleton without EP process-group state (CPU unit tests)."""
    layer = object.__new__(EPDeepseekV4MoELayer)
    nn.Module.__init__(layer)
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


# Registrations


def test_registrations():
    assert MOE_LAYER_MAP["DeepseekV4SparseMoeBlock"] is EPDeepseekV4MoELayer
    assert resolve_ep_merge_layer_class("deepseek_v4") is EPDeepseekV4MoELayer
    # Fused hub layout: the class inherits the base merge, the exact inverse of its base gather.
    assert "merge_shards_to_hf" not in vars(EPDeepseekV4MoELayer)  # inherits the base merge
    # Fused hub layout: the base fused gather is already correct, no per-expert split.
    assert EPDeepseekV4MoELayer._PER_EXPERT_UNFUSED_KEYS is None
    assert EPDeepseekV4MoELayer._supports_bias_balancing
    assert EPDeepseekV4MoELayer._supports_routing_replay
    assert EPDeepseekV4MoELayer._ep_severs_aux_loss


# Tiny-config save/load roundtrip


def test_tiny_config_save_load_roundtrip():
    """save_pretrained → from_pretrained must reproduce identical logits (guards persistent-buffer
    coverage: e_score_correction_bias and the hash tid2eid table must survive the checkpoint)."""
    model = _tiny_model()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 64))

    with torch.no_grad():
        ref = model(input_ids=input_ids).logits

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        reloaded = AutoModelForCausalLM.from_pretrained(tmpdir).eval()
        assert torch.equal(reloaded.model.layers[0].mlp.gate.tid2eid, model.model.layers[0].mlp.gate.tid2eid)
        with torch.no_grad():
            out = reloaded(input_ids=input_ids).logits

    assert torch.allclose(ref, out, atol=1e-5), f"roundtrip logits diverged: {(ref - out).abs().max()}"


# Eager-only guards


def test_model_is_deepseek_v4_detection():
    assert _model_is_deepseek_v4(_tiny_config())
    assert not _model_is_deepseek_v4(type("Cfg", (), {"model_type": "gpt_oss"})())


def test_collator_factory_rejects_padding_free_but_allows_packing():
    """DeepSeek-V4 resolves to eager: ``padding_free`` is refused, ``packing`` is not.

    The gate is derived from the RESOLVED ``_attn_implementation``, not the model family, so this
    asserts the outcome for DSv4 without pinning the mechanism to its name. Only ``padding_free`` is
    refused — it exists solely to emit cu_seqlens no eager kernel reads. ``packing`` stays allowed
    because DSv4's masked attention isolates documents on the training path (transformers >= 5.14
    creates the cache only under ``use_cache``, which training runs with ``False``); the
    compressed-attention layers still cross boundaries by construction, the documented mixer-class
    behavior — both behaviorally pinned in ``test_deepseek_v4_packed_isolation.py``.
    """
    config = _tiny_config()
    config._attn_implementation = "eager"

    class _Tok:  # minimal stub — the guard fires before any real tokenizer use
        pad_token = "<pad>"
        eos_token_id = 1

    with pytest.raises(ValueError, match="varlen"):
        select_data_collator(tokenizer=_Tok(), padding_free=True, model_config=config)

    assert select_data_collator(tokenizer=_Tok(), packing=True, model_config=config) is not None
    # Padded batches stay allowed (default TRL collator).
    assert select_data_collator(tokenizer=_Tok(), model_config=config) is None


# Clamped-GLU equivalence vs the HF experts module


def test_clamped_glu_matches_hf_apply_gate():
    """`_glu_combine` (compiled and generic fallback) must reproduce DeepseekV4Experts._apply_gate
    exactly — a missed clamp silently diverges only on out-of-range activations."""
    config = _tiny_config()
    experts = DeepseekV4Experts(config)
    torch.manual_seed(SEED)
    # Scale up so the clamps actually engage (|x| > swiglu_limit): a formula missing the clamp
    # would still pass on small inputs.
    gate_up = torch.randn(64, 2 * config.moe_intermediate_size) * 3 * config.swiglu_limit
    expected = experts._apply_gate(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)

    assert (gate.abs() > config.swiglu_limit).any(), "test inputs must exceed the clamp limit"

    # Un-compiled reference formula.
    assert torch.allclose(clamped_silu_mul_eager(gate, up, config.swiglu_limit), expected, atol=1e-6)

    # The layer seam — the compiled SiLU combine and the generic act_fn fallback, as
    # _init_expert_compute latches them into _fused_glu_mul.
    layer = _bare_ep_layer(act_fn=experts.act_fn, limit=config.swiglu_limit)
    for combine in (partial(fused_clamped_silu_mul, limit=config.swiglu_limit), layer._clamped_swiglu_eager):
        layer._fused_glu_mul = combine
        out = layer._glu_combine(gate, up)
        assert torch.allclose(out, expected, atol=1e-6), f"_glu_combine ({layer._glu_combine_name()}) diverged"

    # The clamp must be load-bearing: dropping it changes the output on these inputs.
    assert not torch.allclose(experts.act_fn(gate) * up, expected)


def _real_moe_layer(hidden_act: str) -> EPDeepseekV4MoELayer:
    """The wrapper over the library's own non-hash MoE block, built through its real ``__init__``."""
    config = _tiny_config(hidden_act=hidden_act)
    block = next(layer.mlp for layer in _tiny_model(config).model.layers if not getattr(layer.mlp, "is_hash", True))
    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    ep_config.finalize_expert_assignment(config.n_routed_experts)
    return EPDeepseekV4MoELayer(block, ep_config).cpu()


def test_real_block_delivers_non_neutral_routing_knobs():
    """Against the installed module with a non-neutral scaling factor — the wrapper resolves its
    routing knobs by ``getattr`` with defaults, and a knob that moves upstream (the 5.14
    *TopkRouter refactor did exactly this to Mistral4) silently unscales every routed weight."""
    config = _tiny_config(routed_scaling_factor=2.5)
    block = next(layer.mlp for layer in _tiny_model(config).model.layers if not getattr(layer.mlp, "is_hash", True))
    ep_config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    ep_config.finalize_expert_assignment(config.n_routed_experts)
    layer = EPDeepseekV4MoELayer(block, ep_config).cpu()
    assert layer.routed_scaling_factor == 2.5, "scaling fell back to its default — routed weights silently unscaled"


@pytest.mark.parametrize(("hidden_act", "fused"), [("silu", True), ("gelu", False)])
def test_the_real_constructor_arms_the_compiled_silu_path(hidden_act, fused, monkeypatch):
    """The latched combine decides whether the compiled clamped SwiGLU runs. The equivalence test above
    latches it by hand, so only this — the real ``__init__`` over the library block — pins which branch
    production takes, and a nominal `isinstance(act_fn, nn.SiLU)` test answers False for BOTH rows
    (`ACT2FN["silu"]` is a `SiLUActivation`), disarming the kernel with no numerical trace.

    The spy is installed BEFORE construction: the latch binds the kernel there, which is what makes the
    reported combine and the executed one the same declaration."""
    calls: list[float] = []

    def spy(gate, up, limit):
        calls.append(limit)
        return clamped_silu_mul_eager(gate, up, limit)

    monkeypatch.setattr("src.distributed.expert_parallel.layers.deepseek_v4.fused_clamped_silu_mul", spy)
    layer = _real_moe_layer(hidden_act)
    assert (getattr(layer._fused_glu_mul, "func", None) is spy) is fused

    torch.manual_seed(SEED)
    gate, up = torch.randn(16, 8) * 3 * layer.limit, torch.randn(16, 8) * 3 * layer.limit
    out = layer._glu_combine(gate, up)

    # The flag is only interesting if it actually routes the compute.
    assert calls == ([layer.limit] if fused else [])
    silu_gated = clamped_silu_mul_eager(gate, up, layer.limit)
    if fused:
        assert torch.allclose(out, silu_gated)
    else:
        assert not torch.allclose(out, silu_gated)  # same clamps, gelu gate — the eager branch really ran


# Hash-layer selection


def _selection_layer(is_hash: bool, num_experts: int = 8, top_k: int = 2):
    gate = nn.Module()
    gate.tid2eid = torch.randint(0, num_experts, (32, top_k))
    gate.e_score_correction_bias = torch.zeros(num_experts)
    return _bare_ep_layer(gate=gate, is_hash=is_hash, top_k=top_k, num_experts=num_experts)


def test_hash_layer_raises_without_input_ids():
    layer = _selection_layer(is_hash=True)
    scores = torch.rand(6, layer.num_experts)
    with pytest.raises(RuntimeError, match="input_ids"):
        layer._select_experts(scores, None)


def test_hash_layer_selects_tid2eid_rows():
    layer = _selection_layer(is_hash=True)
    input_ids = torch.tensor([[3, 7, 11]])
    indices = layer._select_experts(torch.rand(3, layer.num_experts), input_ids)
    assert torch.equal(indices, layer.gate.tid2eid[input_ids.reshape(-1)])


def test_topk_layer_selects_biased_but_gates_unbiased():
    """Selection follows scores + e_score_correction_bias; a large bias must flip the pick."""
    layer = _selection_layer(is_hash=False, top_k=1)
    scores = torch.zeros(2, layer.num_experts)
    scores[:, 0] = 1.0  # natural winner: expert 0
    layer.gate.e_score_correction_bias[5] = 10.0  # correction bias promotes expert 5
    indices = layer._select_experts(scores, None)
    assert (indices == 5).all(), "e_score_correction_bias must steer top-k selection"


# Per-rope-type inv_freq buffer fix


def test_fix_rotary_inv_freq_covers_all_rotary_instances():
    """Every DeepseekV4RotaryEmbedding (model-level + CSA/HCA compressors + indexer) must get its
    {main,compress}_inv_freq recomputed — a missed instance keeps garbage frequencies silently."""
    model = _tiny_model()
    rotaries = [m for m in model.modules() if type(m).__name__ == "DeepseekV4RotaryEmbedding"]
    assert len(rotaries) >= 4, "expected model + CSA compressor + indexer + HCA compressor rotaries"

    references = {}
    for i, rot in enumerate(rotaries):
        for lt in rot.layer_types:
            references[(i, lt)] = getattr(rot, f"{lt}_inv_freq").clone()
            getattr(rot, f"{lt}_inv_freq").fill_(-1.0)  # corrupt (as a bf16/meta load would)

    fix_rotary_inv_freq(model)

    for (i, lt), ref in references.items():
        fixed = getattr(rotaries[i], f"{lt}_inv_freq")
        assert torch.allclose(fixed, ref, atol=1e-8), f"rotary {i} {lt}_inv_freq not recomputed"
        assert fixed.dtype == torch.float32


# moe_balancing auto resolution


def test_moe_balancing_auto_resolves_bias_update_for_v4_ep():
    """auto must NOT resolve to the severed aux_loss for an EP-wrapped V4 model."""
    # The real layer always adopts the hub gate — its ``e_score_correction_bias`` is the exported
    # native slot auto's commit condition keys on; a gate-less skeleton reads as slot-less and
    # resolves to none (the transient trade-off auto never defaults into).
    gate = nn.Module()
    gate.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))
    host = nn.Module()
    host.moe = _bare_ep_layer(gate=gate)
    assert resolve_balancing_mode("auto", host, is_moe=True) == "bias_update"
    # Control: a plain module tree carries no bias, and V4's forward takes no output_router_logits
    # either, so nothing there can balance — auto resolves to none rather than to an aux_loss the
    # strategy would refuse (tests/cpu/models/test_moe_balancing_auto_resolution.py).
    assert resolve_balancing_mode("auto", nn.Module(), is_moe=True) == "none"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
