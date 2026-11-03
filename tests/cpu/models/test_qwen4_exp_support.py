#!/usr/bin/env python
"""Qwen3.8-Flash-Next (``qwen4_exp``) family contract. ``Qwen4ExpTextSparseMoeBlock`` is Qwen3.5-MoE's block under
Qwen3-Next's router, which renormalizes the top-k weights only under ``norm_topk_prob``."""

from __future__ import annotations

import copy
import json
import logging
from types import MethodType

import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, PreTrainedModel, Qwen4ExpConfig, Qwen4ExpTextConfig

from halo.callbacks.balancing import apply_balancing_mode
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.tensor_parallel.parallelize_attention import reject_unshardable_tp_model
from halo.model_loading.liger.cross_entropy import build_lce_forward
from halo.model_loading.liger.family_specs import LIGER_FAMILY_SPECS
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import resolve_attn_implementation
from halo.models.attention.tree import reject_unsupported_tree_attention
from halo.models.moe.balancing import resolve_balancing_mode
from halo.models.moe.family_specs import QWEN4_EXP, find_moe_family_spec, wraps_experts
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from tests.common.models import QWEN3_8_FLASH_NEXT, TINY_QWEN4_EXP_CONFIG, TINY_QWEN4_EXP_VISION_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tokenizers import skip_uncached
from tests.common.weight_sync import StockModel

E = TINY_QWEN4_EXP_CONFIG["num_experts"]
H = TINY_QWEN4_EXP_CONFIG["hidden_size"]
TOKENS = 13


def _config(**overrides) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(**{**TINY_QWEN4_EXP_CONFIG, **overrides})


def _tiny_model(**overrides) -> PreTrainedModel:
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(_config(**overrides)).eval()


def _wrapped_block(norm_topk_prob: bool) -> tuple[torch.nn.Module, EPQwen4ExpMoELayer]:
    """``(untouched stock block, EP layer around an identical copy)``."""
    block = _tiny_model(norm_topk_prob=norm_topk_prob).model.layers[0].mlp
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return copy.deepcopy(block), EPQwen4ExpMoELayer(block, single_process_ep_config(E)).cpu()


def _wrap_every_block(model: PreTrainedModel) -> list[EPQwen4ExpMoELayer]:
    layers = []
    for decoder_layer in model.model.layers:
        decoder_layer.mlp = EPQwen4ExpMoELayer(decoder_layer.mlp, single_process_ep_config(E)).cpu()
        layers.append(decoder_layer.mlp)
    return layers


def test_both_model_types_resolve_the_family_and_its_layer():
    assert MOE_LAYER_MAP["Qwen4ExpTextSparseMoeBlock"] is EPQwen4ExpMoELayer
    claims = ep_layer_class_by_model_type()
    composite = Qwen4ExpConfig(text_config=TINY_QWEN4_EXP_CONFIG, vision_config=TINY_QWEN4_EXP_VISION_CONFIG)
    for model_type, config in (("qwen4_exp_text", _config()), ("qwen4_exp", composite)):
        assert claims[model_type] is EPQwen4ExpMoELayer
        assert find_moe_family_spec(model_type) is QWEN4_EXP
        assert wraps_experts(config, ep_layers_requested=True), f"{model_type}: the EP layers would not apply"


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_the_ep_layer_reproduces_the_block_and_its_input_gradient(norm_topk_prob):
    reference, layer = _wrapped_block(norm_topk_prob)
    hidden = torch.randn(2, TOKENS, H, generator=torch.Generator().manual_seed(4))
    expected_input, actual_input = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
    expected = reference(expected_input)
    actual = layer(actual_input)
    expected.square().sum().backward()
    actual.square().sum().backward()

    scale = expected.abs().max().item()
    assert scale > 1e-2, "reference output is ~zero; nothing was compared"
    assert torch.allclose(actual, expected, atol=2e-6 * scale, rtol=0)
    assert torch.allclose(actual_input.grad, expected_input.grad, atol=1e-5 * expected_input.grad.abs().max(), rtol=0)


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_a_zero_balancing_bias_reproduces_the_router(norm_topk_prob):
    """Qwen3.5-MoE's always-renormalizing route would mis-gate ``norm_topk_prob: false``."""
    _block, layer = _wrapped_block(norm_topk_prob)
    assert layer.enable_bias_balancing() is True
    hidden = torch.randn(TOKENS, H, generator=torch.Generator().manual_seed(3))

    _logits, router_weights, router_indices = layer.gate(hidden)
    indices, weights = layer._route_hidden_states(hidden)

    assert torch.equal(indices, router_indices)
    assert torch.allclose(weights, router_weights, atol=1e-7, rtol=0)


def test_the_gathered_export_is_the_release_fused_layout():
    """The layout and orientation the release ships (``[E, 2·I, H]`` and ``[E, H, I]``), with no per-expert tensors."""
    model = _tiny_model()
    stock = [
        {
            f"experts.{name}": getattr(layer.mlp.experts, name).detach().clone()
            for name in ("gate_up_proj", "down_proj")
        }
        for layer in model.model.layers
    ]
    for index, layer in enumerate(_wrap_every_block(model)):
        gathered = layer.gather_expert_state_dict(merge_lora=True)
        assert set(gathered) == set(stock[index]), f"layer {index}: the gather wrote {sorted(gathered)}"
        for key, tensor in stock[index].items():
            assert torch.equal(gathered[key], tensor), f"layer {index}: {key} diverges from the block's tensor"


def test_the_release_ships_its_experts_fused():
    """One fused bank per projection and layer under the block's names, so ``from_pretrained`` converts nothing."""
    try:
        index = hf_hub_download(QWEN3_8_FLASH_NEXT, SAFETENSORS_INDEX_FILE, local_files_only=True)
    except LocalEntryNotFoundError:
        skip_uncached(QWEN3_8_FLASH_NEXT, "checkpoint index")
    with open(index) as handle:
        weight_map = json.load(handle)["weight_map"]
    expert_keys = {key for key in weight_map if ".layers.0.mlp.experts." in key and not key.startswith("mtp.")}
    assert expert_keys == {
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    }


def test_auto_balancing_takes_the_aux_loss_on_the_family_head():
    model = _tiny_model()
    _wrap_every_block(model)
    assert resolve_balancing_mode("auto", model, is_moe=True) == "aux_loss"


def test_auto_balancing_trains_unbalanced_under_the_fused_head(caplog):
    """The fused head refuses ``output_router_logits`` and no checkpoint slot carries a routing bias. The recipes pick
    the mode instead."""
    (spec,) = [spec for spec in LIGER_FAMILY_SPECS if "qwen4_exp_text" in spec.model_types]
    model = _tiny_model()
    _wrap_every_block(model)
    model.forward = MethodType(build_lce_forward(router_aux_loss_in_head=spec.router_aux_loss_in_head), model)
    with caplog.at_level(logging.WARNING):
        assert resolve_balancing_mode("auto", model, is_moe=True) == "none"
    assert "TRAINS UNBALANCED" in caplog.text


def test_the_fused_head_refuses_router_logits():
    (spec,) = [spec for spec in LIGER_FAMILY_SPECS if "qwen4_exp_text" in spec.model_types]
    model = _tiny_model()
    fused = MethodType(build_lce_forward(router_aux_loss_in_head=spec.router_aux_loss_in_head), model)
    with pytest.raises(ValueError, match="output_router_logits=False"):
        fused(input_ids=torch.zeros(1, 4, dtype=torch.long), output_router_logits=True)


def test_bias_update_is_refused_without_a_checkpoint_slot():
    model = _tiny_model()
    _wrap_every_block(model)
    with pytest.raises(ValueError, match="bias_update_transient"):
        apply_balancing_mode(model, "bias_update", is_moe=True)


def test_context_parallelism_is_refused_on_the_gated_deltanet_layers():
    with pytest.raises(CPConfigError, match="linear_attention"):
        reject_unsupported_ulysses_model(_tiny_model(), cp_size=2)


def test_tensor_parallelism_is_refused_on_the_unplanned_attention():
    with pytest.raises(ValueError, match="shards ZERO attention layers"):
        reject_unshardable_tp_model(_tiny_model(), tp_size=2)


def test_a_flash_request_falls_back_to_sdpa_and_padding_free_is_refused():
    """The class supports no flash kernel. No backend serves a padding-free row on the M-RoPE decoder (SDPA reads no
    ``cu_seqlens``)."""
    config = _config()
    config._attn_implementation = resolve_attn_implementation(
        config, "flash_attention_2", torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
    )
    assert config._attn_implementation == "sdpa"
    with pytest.raises(ValueError, match="rotates by M-RoPE.*the dense backends take no cu_seqlens"):
        reject_unsupported_collator_options(
            None,
            padding_free=True,
            packing=False,
            train_on_completions_only=False,
            assistant_message_template=None,
            use_context_parallel=False,
            train_on_last_assistant_only=False,
            model_config=config,
            per_device_train_batch_size=1,
            keeps_packed_rows=False,
        )


def test_tree_packing_is_refused():
    with pytest.raises(ValueError, match="tree packing is not available for model_type='qwen4_exp_text'"):
        reject_unsupported_tree_attention(_config(), "sdpa", "gather")


@pytest.mark.parametrize("model_type", QWEN4_EXP.model_types)
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_weight_sync_is_refused_on_both_engines(backend, model_type):
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(StockModel(model_type), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_weight_sync_is_refused_with_live_ep_layers(backend):
    model = _tiny_model()
    _wrap_every_block(model)
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, backend)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
