"""Inkling 2's EP layer over frozen NVFP4/MXFP8 expert banks: the same function as the upstream block on the
dequantized weights, LoRA adapters that train exactly as over a bf16 base, and no expert weight left to train."""

import copy

import pytest
import torch
import torch.nn as nn
from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2SparseMoeBlock

from halo.distributed.expert_parallel.families.inkling2 import EPInkling2MoELayer
from halo.distributed.expert_parallel.moe_layer.frozen_banks import FROZEN_SLOTS, is_frozen_bank_container
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.inkling2 import NVFP4_OUTER_SCALE_DENOMINATOR, quantize_release_bank
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

E, H, M, K = 8, 64, 32, 2
TOKENS = (2, 6)
FORMATS = ("nvfp4", "mxfp8")


def _block(dtype: torch.dtype) -> Inkling2SparseMoeBlock:
    torch.manual_seed(0)
    block = Inkling2SparseMoeBlock(
        Inkling2TextConfig(
            hidden_size=H,
            moe_intermediate_size=M,
            n_routed_experts=E,
            num_experts_per_tok=K,
            route_scale=2.5,
            mlp_gate_softcap=0.5,
            mlp_up_softcap=0.3,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
        )
    ).to(dtype)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    nn.init.normal_(block.gate.e_score_correction_bias, std=0.3)
    # transformers pins the router to fp32 and routes in fp32.
    block.gate.float()
    return block


def _dequantized_and_frozen(quant_format: str, dtype: torch.dtype):
    """The block with its experts round-tripped through ``quant_format`` (the reference), and its twin whose
    container holds the packed slots the frozen loader fills."""
    reference = _block(dtype)
    frozen = copy.deepcopy(reference)
    experts = frozen.experts
    for slot in FROZEN_SLOTS:
        bank, stored = quantize_release_bank(getattr(reference.experts, slot.weight).data, quant_format)
        getattr(reference.experts, slot.weight).data = bank.dequantize_linear().to(dtype)
        del experts._parameters[slot.weight]
        experts.register_buffer(slot.weight, stored[""])
        experts.register_buffer(slot.scale, stored["_scale"])
        if quant_format == "nvfp4":
            experts.register_buffer(slot.amax, stored["_amax"])
            experts.register_buffer(slot.denominator, torch.tensor([NVFP4_OUTER_SCALE_DENOMINATOR]))
    assert is_frozen_bank_container(experts)
    return reference, frozen


def _tokens(dtype: torch.dtype) -> torch.Tensor:
    return torch.randn((*TOKENS, H), generator=torch.Generator().manual_seed(3)).to(dtype)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_frozen_layer_is_the_upstream_block_on_the_dequantized_weights(quant_format):
    reference, frozen = _dequantized_and_frozen(quant_format, torch.float32)
    layer = EPInkling2MoELayer(frozen, single_process_ep_config(E))
    assert layer.has_frozen_banks

    ref_tokens = _tokens(torch.float32).requires_grad_(True)
    layer_tokens = ref_tokens.detach().clone().requires_grad_(True)
    expected, actual = reference(ref_tokens), layer(layer_tokens)
    assert expected.abs().max() > 1e-2
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    cotangent = torch.randn(expected.shape, generator=torch.Generator().manual_seed(4))
    expected.backward(cotangent)
    actual.backward(cotangent)
    assert ref_tokens.grad.abs().max() > 1e-2
    torch.testing.assert_close(layer_tokens.grad, ref_tokens.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_frozen_layer_trains_no_expert_weight(quant_format):
    _reference, frozen = _dequantized_and_frozen(quant_format, torch.float32)
    layer = EPInkling2MoELayer(frozen, single_process_ep_config(E, expert_lora=ExpertLoraSpec(r=8, alpha=16)))
    trainable = {name for name, param in layer.named_parameters() if param.requires_grad}
    assert not {slot.weight for slot in FROZEN_SLOTS} & {name for name, _ in layer.named_parameters()}
    assert {f"{slot.weight}_lora_{side}" for slot in FROZEN_SLOTS for side in "AB"} <= trainable
    assert all(not name.startswith(("gate_up_proj.", "down_proj.")) for name in trainable)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_lora_over_a_frozen_bank_trains_as_over_the_bf16_base(quant_format):
    """The adapters see the same base function, so outputs and adapter gradients must match the bf16-param layer
    built from the dequantized weights with the same adapter init."""
    spec = ExpertLoraSpec(r=8, alpha=16)
    reference, frozen = _dequantized_and_frozen(quant_format, torch.bfloat16)
    torch.manual_seed(7)
    bf16_layer = EPInkling2MoELayer(reference, single_process_ep_config(E, expert_lora=spec))
    torch.manual_seed(7)
    frozen_layer = EPInkling2MoELayer(frozen, single_process_ep_config(E, expert_lora=spec))
    assert frozen_layer.has_frozen_banks and not bf16_layer.has_frozen_banks
    # Non-zero B so the delta enters the output; identical in both layers.
    for layer in (bf16_layer, frozen_layer):
        for slot in FROZEN_SLOTS:
            torch.manual_seed(11)
            nn.init.normal_(getattr(layer, f"{slot.weight}_lora_B"), std=0.05)
            assert getattr(layer, f"{slot.weight}_lora_A").dtype == torch.bfloat16

    tokens = _tokens(torch.bfloat16)
    out_bf16, out_frozen = bf16_layer(tokens), frozen_layer(tokens)
    torch.testing.assert_close(out_frozen, out_bf16, rtol=2e-2, atol=2e-2)
    out_bf16.float().pow(2).sum().backward()
    out_frozen.float().pow(2).sum().backward()
    for slot in FROZEN_SLOTS:
        for side in "AB":
            name = f"{slot.weight}_lora_{side}"
            expected, actual = getattr(bf16_layer, name).grad, getattr(frozen_layer, name).grad
            assert expected.abs().max() > 0, name
            torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("quant_format", FORMATS)
def test_frozen_layer_refuses_every_expert_gather_and_keeps_banks_out_of_checkpoints(quant_format):
    """A merged save, weight sync or gathered export of a frozen layer would write no experts at all."""
    _reference, frozen = _dequantized_and_frozen(quant_format, torch.float32)
    layer = EPInkling2MoELayer(frozen, single_process_ep_config(E, expert_lora=ExpertLoraSpec(r=8, alpha=16)))
    with pytest.raises(ValueError, match="frozen quantized expert banks"):
        layer.start_gather_expert_state_dict()
    with pytest.raises(ValueError, match="frozen quantized expert banks"):
        layer.gather_expert_state_dict()
    assert not [key for key in layer.state_dict() if "_frozen_" in key]
    assert {f"{slot.weight}_lora_{side}" for slot in FROZEN_SLOTS for side in "AB"} <= set(layer.state_dict())


_FROZEN_OK = {"ep_size": 2, "expert_lora": ExpertLoraSpec(r=8, alpha=16), "frozen_quantized_experts": True}


def test_frozen_quantized_experts_builds_beside_an_ep_group_and_expert_lora():
    assert make_parallelism_config(world_size=2, gpus_per_node=2, **_FROZEN_OK).frozen_quantized_experts


@pytest.mark.parametrize(
    ("override", "named"),
    [
        ({"ep_size": 1}, "expert_parallel_size >= 2"),
        ({"ep_lazy_loading": False}, "ep_lazy_loading: true"),
        ({"expert_lora": None}, "lora_target_modules"),
        ({"merge_expert_lora_on_save": True}, "merge_expert_lora_on_save: false"),
    ],
)
def test_frozen_quantized_experts_names_each_missing_requirement(override, named):
    with pytest.raises(ValueError, match=named):
        make_parallelism_config(world_size=2, gpus_per_node=2, **{**_FROZEN_OK, **override})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
