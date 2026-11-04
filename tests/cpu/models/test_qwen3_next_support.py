#!/usr/bin/env python
"""Qwen3-Next family contract. ``Qwen3NextSparseMoeBlock`` is Qwen3.5-MoE's block parameter for parameter, except its
router renormalizes under ``norm_topk_prob``, so the EP layer overrides the ``_biased_route`` hook."""

from __future__ import annotations

import copy

import pytest
import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3_5MoeTextConfig,
    Qwen3NextConfig,
)

import halo.model_loading.parallel.ep as ep_lazy_loader
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.tensor_parallel.parallelize_attention import reject_unshardable_tp_model
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.models.attention.selection import uses_fa4_dedicated_head_dim
from halo.models.moe.balancing import (
    accepts_native_balancing_bias,
    is_transient_balancing_router,
    resolve_balancing_mode,
)
from halo.models.moe.family_specs import QWEN3_NEXT, find_moe_family_spec, wraps_experts
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from tests.common.models import TINY_QWEN3_NEXT_CONFIG, TINY_QWEN35_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.weight_sync import StockModel

E = TINY_QWEN3_NEXT_CONFIG["num_experts"]
H = TINY_QWEN3_NEXT_CONFIG["hidden_size"]
TOKENS = 13


def _config(**overrides) -> Qwen3NextConfig:
    return Qwen3NextConfig(**{**TINY_QWEN3_NEXT_CONFIG, **overrides})


def _build(config: PretrainedConfig) -> PreTrainedModel:
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).eval()


def _tiny_model(**overrides) -> PreTrainedModel:
    return _build(_config(**overrides))


def _wrapped_block(
    config: PretrainedConfig, layer_class: type[EPQwen3_5MoELayer] = EPQwen3NextMoELayer
) -> tuple[torch.nn.Module, EPQwen3_5MoELayer]:
    """``(untouched reference block, EP layer around an identical copy)``."""
    block = _build(config).model.layers[0].mlp
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return copy.deepcopy(block), layer_class(block, single_process_ep_config(E)).cpu()


# The router renormalizes always (Qwen3.5) or per ``norm_topk_prob`` (Qwen3-Next); ``_biased_route`` must match it.
ROUTER_CASES = [
    pytest.param(lambda: Qwen3_5MoeTextConfig(**TINY_QWEN35_MOE_CONFIG), EPQwen3_5MoELayer, id="qwen3_5_moe"),
    pytest.param(lambda: _config(norm_topk_prob=True), EPQwen3NextMoELayer, id="qwen3_next-norm_topk_prob"),
    pytest.param(lambda: _config(norm_topk_prob=False), EPQwen3NextMoELayer, id="qwen3_next-no_norm_topk_prob"),
]


def _wrap_every_block(model: PreTrainedModel) -> list[EPQwen3NextMoELayer]:
    layers = []
    for decoder_layer in model.model.layers:
        decoder_layer.mlp = EPQwen3NextMoELayer(decoder_layer.mlp, single_process_ep_config(E)).cpu()
        layers.append(decoder_layer.mlp)
    return layers


def _tiny_model_with_ep_layers() -> PreTrainedModel:
    model = _tiny_model()
    _wrap_every_block(model)
    return model


def test_registrations():
    assert MOE_LAYER_MAP["Qwen3NextSparseMoeBlock"] is EPQwen3NextMoELayer
    assert dict(ep_layer_class_by_model_type())["qwen3_next"] is EPQwen3NextMoELayer
    assert find_moe_family_spec("qwen3_next") is QWEN3_NEXT
    assert wraps_experts(_config(), ep_layers_requested=True)


def test_the_fa4_head_dim_256_verdict_reads_the_head_widths():
    assert uses_fa4_dedicated_head_dim(Qwen3NextConfig())
    assert not uses_fa4_dedicated_head_dim(_config())


@pytest.mark.parametrize(("make_config", "layer_class"), ROUTER_CASES)
def test_a_zero_balancing_bias_reproduces_the_router(make_config, layer_class):
    """Qwen3.5's unconditional renormalization would mis-gate a ``norm_topk_prob: false`` router."""
    _block, layer = _wrapped_block(make_config(), layer_class)
    assert layer.enable_bias_balancing() is True
    assert is_transient_balancing_router(layer)
    hidden = torch.randn(TOKENS, H, generator=torch.Generator().manual_seed(3))

    _logits, router_weights, router_indices = layer.gate(hidden)
    indices, weights = layer._route_hidden_states(hidden)

    assert torch.equal(indices, router_indices)
    assert torch.allclose(weights, router_weights, atol=1e-7, rtol=0)


@pytest.mark.parametrize(("make_config", "layer_class"), ROUTER_CASES)
def test_the_biased_layer_forward_matches_the_block_at_zero_bias(make_config, layer_class):
    reference, layer = _wrapped_block(make_config(), layer_class)
    layer.enable_bias_balancing()
    hidden = torch.randn(2, TOKENS, H, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        expected = reference(hidden)
        actual = layer(hidden)
    assert expected.abs().max().item() > 1e-2, "reference output is ~zero; nothing was compared"
    assert torch.allclose(actual, expected, atol=2e-6 * expected.abs().max().item(), rtol=0)


def test_a_balancing_bias_moves_selection_only():
    _block, layer = _wrapped_block(_config(norm_topk_prob=False))
    layer.enable_bias_balancing()
    hidden = torch.randn(TOKENS, H, generator=torch.Generator().manual_seed(5))
    logits = layer.gate(hidden)[0]
    natural = logits.argsort(dim=-1)[:, 0]  # each token's least likely expert
    target = int(natural[0])
    layer.balancing_biases[target] = 1e3

    indices, weights = layer._route_hidden_states(hidden)
    assert (indices == target).any(dim=-1).all(), "the balancing bias never reached selection"
    unbiased = torch.softmax(logits, dim=-1, dtype=torch.float).gather(-1, indices)
    assert torch.allclose(weights, unbiased.to(weights.dtype), atol=1e-7, rtol=0), "the bias leaked into the weights"


def test_balancing_auto_resolves_aux_loss_and_bias_update_has_no_slot():
    """The forward honours ``output_router_logits``, so ``auto`` keeps the aux loss. No checkpoint slot exists for a
    routing bias, so only the explicit transient side-buffer can carry one."""
    model = _tiny_model()
    layers = _wrap_every_block(model)
    assert resolve_balancing_mode("auto", model, is_moe=True) == "aux_loss"
    assert not accepts_native_balancing_bias(model)
    assert not any(layer.can_adopt_native_balancing() for layer in layers)


def test_the_gathered_export_is_the_hub_checkpoint_layout(tmp_path):
    """The layout the Qwen3-Next hub checkpoints and both rollout engines' loaders read."""
    model = _tiny_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    saved = {}
    for shard in tmp_path.glob("*.safetensors"):
        saved.update(load_file(shard))

    for index, layer in enumerate(_wrap_every_block(model)):
        prefix = f"model.layers.{index}.mlp."
        hub = {key[len(prefix) :]: tensor for key, tensor in saved.items() if key.startswith(f"{prefix}experts.")}
        assert len(hub) == 3 * E, f"premise: the stock save wrote {sorted(hub)[:4]}..."
        gathered = layer.gather_expert_state_dict(merge_lora=True)
        assert set(gathered) == set(hub)
        for key, tensor in hub.items():
            assert torch.equal(gathered[key], tensor), f"layer {index}: {key} diverges from the hub layout"


@pytest.mark.parametrize("ep_size", [1, 2])
def test_the_lazy_ep_load_matches_from_pretrained(tmp_path, monkeypatch, ep_size):
    """EP patching is stubbed: it needs DeepEP and process groups."""
    _tiny_model().to(torch.bfloat16).save_pretrained(tmp_path)
    weight_map = resolve_checkpoint_weights(str(tmp_path)).weight_map
    assert any(".mlp.experts.0.gate_proj." in key for key in weight_map), "premise: per-expert hub keys"
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)
    reference = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).state_dict()

    ep_config = EPConfig(
        ep_size=ep_size, stage_world_size=ep_size, nvlink_domain_size=ep_size, experts_fsdp_managed=ep_size == 1
    )
    lazy = load_ep_model_lazy(
        str(tmp_path),
        ep_config,
        AutoConfig.from_pretrained(tmp_path),
        dtype=torch.bfloat16,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
    ).state_dict()

    assert lazy.keys() == reference.keys()
    local = slice(0, E // ep_size)
    fused = ("experts.gate_up_proj", "experts.down_proj")
    expected = {key: t[local] if key.endswith(fused) else t for key, t in reference.items()}
    stale = [key for key, tensor in lazy.items() if not torch.equal(tensor, expected[key])]
    assert not stale, f"{len(stale)} tensors differ from from_pretrained: {stale[:5]}"


def test_context_parallelism_is_refused_on_the_gated_deltanet_layers():
    with pytest.raises(CPConfigError, match="linear_attention"):
        reject_unsupported_ulysses_model(_tiny_model(), cp_size=2)


def test_tensor_parallelism_is_refused_on_the_unplanned_attention():
    """``Qwen3NextAttention`` is no class the selective attention-TP plan shards, so a TP run would shard nothing."""
    with pytest.raises(ValueError, match="shards ZERO attention layers"):
        reject_unshardable_tp_model(_tiny_model(), tp_size=2)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_weight_sync_is_refused_on_both_engines(backend):
    """No engine fact refuses the family, so the spec's refusal is what stops it, live EP layers or none."""
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(StockModel("qwen3_next"), backend)
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(_tiny_model_with_ep_layers(), backend)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
