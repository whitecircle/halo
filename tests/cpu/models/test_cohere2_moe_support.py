"""Cohere2 MoE (Command A+) family contract: registrations, routing parity, balancing, CP rotary,
packing isolation, and the PP logit-scale gate.

The family is native in transformers 5.14 (``cohere2_moe`` text backbone; the Command A+ checkpoint
wraps it as ``cohere2_vision``). Its forward carries ``position_ids`` both into mask construction
and through layer kwargs to the attention interface, so packing isolates WITHOUT a toolkit patch —
the spy test here is the defect pin in reverse: it fails if transformers drops that plumbing, which
is the signal to add a ``PositionIdsInjectingRegistry`` patch like Mistral4/Zaya.
"""

from __future__ import annotations

import copy

import pytest
import torch
from safetensors import safe_open
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import (
    ALL_ATTENTION_FUNCTIONS,
    Cohere2MoeForCausalLM,
    Cohere2MoeSparseMoeBlock,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from src.distributed.context_parallel.layers.cohere2_moe import Cohere2MoeUlyssesAttention
from src.distributed.context_parallel.layers.registry import CP_SUPPORTED_ATTENTION_CLASSES, WRAPPER_CLASS_MAP
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.distributed.pipeline_parallel.split import _reject_unapplied_logit_scale
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES
from src.models.moe_balancing import (
    accepts_native_balancing_bias,
    is_transient_balancing_router,
    resolve_balancing_mode,
)
from src.models.patches.buffer_fixes import fix_rotary_inv_freq
from tests.common.models import TINY_COHERE2_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config

E = TINY_COHERE2_MOE_CONFIG["num_experts"]
NOISE_BOUND = 1e-5  # packed-isolation drift ceiling; leaks read ~1e-1, four orders above


def _tiny_config(**overrides) -> Cohere2MoeConfig:
    return Cohere2MoeConfig(**{**TINY_COHERE2_MOE_CONFIG, **overrides})


def _tiny_model(**overrides) -> Cohere2MoeForCausalLM:
    torch.manual_seed(0)
    return Cohere2MoeForCausalLM(_tiny_config(**overrides)).eval()


def _wrapped_block(**config_overrides) -> tuple[Cohere2MoeSparseMoeBlock, EPCohere2MoELayer]:
    torch.manual_seed(0)
    block = Cohere2MoeSparseMoeBlock(_tiny_config(**config_overrides))
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return copy.deepcopy(block), EPCohere2MoELayer(block, single_process_ep_config(E)).cpu()


def test_registrations():
    assert MOE_LAYER_MAP["Cohere2MoeSparseMoeBlock"] is EPCohere2MoELayer
    assert WRAPPER_CLASS_MAP["Cohere2MoeAttention"] is Cohere2MoeUlyssesAttention
    assert "Cohere2MoeAttention" in CP_SUPPORTED_ATTENTION_CLASSES
    assert "Cohere2MoeAttention" in TP_SHARDABLE_ATTENTION_CLASSES

    assert EPCohere2MoELayer._PER_EXPERT_UNFUSED_KEYS is None  # base gather emits the fused pair
    assert EPCohere2MoELayer.hub_per_expert_keys() == ("gate_proj", "up_proj", "down_proj")
    assert EPCohere2MoELayer._supports_bias_balancing
    assert EPCohere2MoELayer._supports_transient_balancing_bias
    assert EPCohere2MoELayer._NATIVE_BALANCING_BIAS_ATTR is None  # no exported slot in the family
    assert not EPCohere2MoELayer._ep_severs_aux_loss  # the wrapper calls the real gate module
    assert EPCohere2MoELayer._supports_routing_replay
    assert EPCohere2MoELayer._supports_gradient_checkpointing
    assert not EPCohere2MoELayer._supports_weight_sync
    assert not EPCohere2MoELayer._supports_lazy_loading
    assert not EPCohere2MoELayer.implements_fused_expert_layout()  # SGLang stays refused


def test_model_type_registry_resolves_both_spellings():
    registry = dict(ep_layer_class_by_model_type())
    assert registry["cohere2_moe"] is EPCohere2MoELayer
    assert registry["cohere2_vision"] is EPCohere2MoELayer


@pytest.mark.parametrize(
    ("expert_selection_fn", "norm_topk_prob"),
    [("sigmoid", True), ("sigmoid", False), ("softmax", True)],
)
def test_route_matches_hf_router(expert_selection_fn, norm_topk_prob):
    """The wrapper's re-derived selection and gating must reproduce ``Cohere2MoeTopKRouter``
    exactly: top-k on the RAW logits, activation over only the selected scores."""
    _block, layer = _wrapped_block(expert_selection_fn=expert_selection_fn, norm_topk_prob=norm_topk_prob)
    torch.manual_seed(3)
    hidden = torch.randn(11, TINY_COHERE2_MOE_CONFIG["hidden_size"])

    router_logits, hf_weights, hf_indices = layer.gate(hidden)
    indices, weights = layer.route_tokens_to_experts(router_logits.float())

    assert torch.equal(indices, hf_indices)
    assert torch.allclose(weights, hf_weights.float(), atol=1e-6)


def test_output_scale_follows_the_combination_strategy():
    _block, averaged = _wrapped_block()
    assert averaged.shared_experts is not None and averaged._output_scale == 0.5

    _block, summed = _wrapped_block(shared_expert_combination_strategy="sum")
    assert summed._output_scale == 1.0

    _block, no_shared = _wrapped_block(num_shared_experts=0)
    assert no_shared.shared_experts is None and no_shared._output_scale == 1.0


def test_balancing_auto_resolves_none_and_transient_is_the_opt_in():
    """No exported slot anywhere in the family: ``auto`` must land on ``none`` (never silently on an
    unexportable bias), adoption must be impossible, and ``enable_bias_balancing`` must yield the
    transient side-buffer that only the explicit ``bias_update_transient`` opts into."""
    model = _tiny_model()
    layer = EPCohere2MoELayer(model.model.layers[0].mlp, single_process_ep_config(E)).cpu()
    model.model.layers[0].mlp = layer

    assert resolve_balancing_mode("auto", model, is_moe=True) == "none"
    assert not accepts_native_balancing_bias(model)
    assert not layer.can_adopt_native_balancing()

    assert layer.enable_bias_balancing() is True
    assert is_transient_balancing_router(layer)
    assert layer.balancing_biases.dtype == torch.float32
    assert layer.balancing_biases.shape == (E,)


def test_balancing_bias_perturbs_selection_only():
    """A large bias on one expert must force it into every selection while the gate weights stay the
    unbiased activation of the selected raw scores."""
    _block, layer = _wrapped_block()
    layer.enable_bias_balancing()

    torch.manual_seed(5)
    logits = torch.randn(9, E)
    logits[:, 3] = logits.min() - 10.0  # never picked naturally — the forced pick cannot be vacuous
    natural, _ = layer.route_tokens_to_experts(logits)
    assert not (natural == 3).any()

    layer.balancing_biases[3] = 1e3
    indices, weights = layer.route_tokens_to_experts(logits)
    assert (indices == 3).any(dim=-1).all(), "the balancing bias never reached selection"

    unbiased = torch.sigmoid(logits.gather(-1, indices))
    unbiased = unbiased / unbiased.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights, unbiased, atol=1e-6), "the bias leaked into the gate weights"


def test_cp_rotary_matches_hf_interleaved_rope():
    """The CP wrapper's fp32 interleaved rotary must reproduce the modeling's
    ``apply_rotary_pos_emb`` bit-for-bit in bf16 (BSHD wrapper layout vs BHSD HF layout)."""
    batch, seq, heads, dim = 2, 6, 4, 8
    torch.manual_seed(7)
    q = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, dim, dtype=torch.bfloat16)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.arange(seq).float()[:, None] * inv_freq[None, :]
    emb = torch.repeat_interleave(freqs, 2, dim=-1).expand(batch, -1, -1)  # interleaved cos/sin
    cos, sin = emb.cos(), emb.sin()

    wrapper = object.__new__(Cohere2MoeUlyssesAttention)
    wrapper._use_rope = True
    got_q, got_k = wrapper._apply_rotary_core(q, k, cos.unsqueeze(2), sin.unsqueeze(2))

    hf_q, hf_k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
    assert torch.equal(got_q.transpose(1, 2), hf_q)
    assert torch.equal(got_k.transpose(1, 2), hf_k)


def test_cp_rotary_identity_on_nope_layers():
    wrapper = object.__new__(Cohere2MoeUlyssesAttention)
    wrapper._use_rope = False
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    got_q, got_k = wrapper._apply_rotary_core(q, k, torch.ones(1), torch.zeros(1))
    assert got_q is q and got_k is k


def test_nope_and_force_rope_flags_from_config():
    """The attention attributes the CP wrapper's ``_use_rope`` reads: sliding layers carry the
    window (RoPE), full layers carry None (NoPE), and ``force_rope`` stays off for sparse layers."""
    model = _tiny_model()
    sliding, full = (layer.self_attn for layer in model.model.layers)
    assert sliding.sliding_window is not None
    assert full.sliding_window is None
    assert not sliding.force_rope and not full.force_rope


def test_tiny_save_load_roundtrip(tmp_path):
    """Pins the per-expert<->fused converter the base gather relies on: a save writes per-expert
    ``experts.{i}.*`` keys and a plain reload reproduces the logits exactly."""
    model = _tiny_model()
    model.save_pretrained(tmp_path)

    with safe_open(tmp_path / "model.safetensors", framework="pt") as shard:
        saved = set(shard.keys())
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in saved, (
        "the save-side per-expert revert is gone — hub_per_expert_keys and unfuse_moe_experts "
        "no longer describe this family's checkpoints"
    )
    assert not any(key.endswith("experts.gate_up_proj") for key in saved)

    reloaded = Cohere2MoeForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()

    input_ids = torch.randint(
        0, TINY_COHERE2_MOE_CONFIG["vocab_size"], (1, 12), generator=torch.Generator().manual_seed(2)
    )
    with torch.no_grad():
        before = model(input_ids).logits
        after = reloaded(input_ids).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_fix_rotary_inv_freq_recomputes_generic_rotary():
    model = _tiny_model()
    reference = model.model.rotary_emb.inv_freq.clone()
    model.model.rotary_emb.inv_freq = torch.zeros_like(reference)
    fix_rotary_inv_freq(model)
    assert torch.allclose(model.model.rotary_emb.inv_freq, reference)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32


class _SpyInterface:
    """Records whether the attention interface received ``position_ids``, then delegates to eager."""

    def __init__(self):
        self.saw_position_ids: list[bool] = []

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        self.saw_position_ids.append(kwargs.get("position_ids") is not None)
        kwargs.pop("position_ids", None)
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)


def _packed_inputs(doc_lens: tuple[int, ...], seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, TINY_COHERE2_MOE_CONFIG["vocab_size"], (1, sum(doc_lens)), generator=generator)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).unsqueeze(0)
    return input_ids, position_ids


def test_attention_interface_receives_position_ids_natively():
    """Cohere2 MoE needs NO position_ids patch: the model forward hands the tensor through layer
    kwargs to every attention interface. If this fails, transformers dropped that plumbing and the
    family needs the Mistral4/Zaya ``PositionIdsInjectingRegistry`` treatment."""
    model = _tiny_model()
    input_ids, position_ids = _packed_inputs((5, 4, 3))

    spy = _SpyInterface()
    ALL_ATTENTION_FUNCTIONS.register("cohere2_moe_spy", spy)
    model.config._attn_implementation = "cohere2_moe_spy"
    try:
        with torch.no_grad():
            model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    finally:
        model.config._attn_implementation = "eager"
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop("cohere2_moe_spy", None)

    assert spy.saw_position_ids and all(spy.saw_position_ids), (
        "the attention interface no longer receives position_ids — packed rows would run as one "
        "dense causal sequence; add a position_ids-injecting patch for this family"
    )


def test_dense_packed_isolation():
    """On the dense (eager) path a packed row with per-document position_ids must reproduce the
    second document's stand-alone logits; without the resets the same probe must leak, or the
    isolation check proves nothing."""
    model = _tiny_model()
    doc = 6
    input_ids, position_ids = _packed_inputs((doc, doc))

    with torch.no_grad():
        packed = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[:, doc:]
        alone = model(input_ids=input_ids[:, doc:], position_ids=position_ids[:, doc:], use_cache=False).logits
        drift = (packed - alone).abs().max().item()

        continuous = torch.arange(2 * doc).unsqueeze(0)
        leaked = model(input_ids=input_ids, position_ids=continuous, use_cache=False).logits[:, doc:]
        leak = (leaked - alone).abs().max().item()

    assert leak > 1e-3, "the leak probe reads isolated — the isolation assertion below is vacuous"
    assert drift < NOISE_BOUND, (
        f"packed documents attend across each other ({drift:.2e}) — the position_ids-driven packed "
        f"mask regressed for cohere2_moe"
    )


def test_pp_rejects_an_unapplied_logit_scale():
    """The PP stage head computes the bare matmul, so a non-unit Cohere ``logit_scale`` must be
    refused loudly (unit scale passes — Command A+ ships 1.0 and is instead refused by the generic
    tied-embeddings gate)."""
    scaled = _tiny_model(tie_word_embeddings=False)
    with pytest.raises(ValueError, match="logit_scale"):
        _reject_unapplied_logit_scale(scaled)

    unit = _tiny_model(tie_word_embeddings=False, logit_scale=1.0)
    _reject_unapplied_logit_scale(unit)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
