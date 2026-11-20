"""Inkling 2's EP layer over frozen NVFP4/MXFP8 expert banks: the same function as the upstream block on the
dequantized weights, LoRA adapters that train exactly as over a bf16 base, and no expert weight left to train."""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3MoeConfig
from transformers.models.inkling2.configuration_inkling2 import Inkling2Config, Inkling2TextConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2SparseMoeBlock

from halo.callbacks.model_flops import estimate_rank_flops
from halo.configs.parallelism_arguments import reject_frozen_experts_without_adapters
from halo.distributed.expert_parallel.families.inkling2 import EPInkling2MoELayer
from halo.distributed.expert_parallel.families.roster import ep_layer_classes, frozen_bank_families
from halo.entrypoints import script
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel import whole_model
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.moe.family_specs import INKLING2, frozen_expert_sidecar_attrs
from tests.common.inkling2 import NVFP4_OUTER_SCALE_DENOMINATOR, quantize_release_bank
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

E, H, M, K = 8, 64, 32, 2
TOKENS = (2, 6)
FORMATS = ("nvfp4", "mxfp8")
FROZEN_SLOTS = INKLING2.frozen_expert_slots


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


class _OneLayerModel(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.mlp = layer
        self.config = Inkling2TextConfig(
            hidden_size=H, n_routed_experts=E, num_experts_per_tok=K, num_hidden_layers=1, vocab_size=32
        )


@pytest.mark.parametrize("quant_format", FORMATS)
def test_frozen_banks_count_as_expert_weights_in_the_flops_model(quant_format):
    """The banks are buffers, so a count over parameters alone leaves every expert GEMM out of MFU."""
    spec = ExpertLoraSpec(r=8, alpha=16)
    reference, frozen = _dequantized_and_frozen(quant_format, torch.bfloat16)
    bf16_layer = EPInkling2MoELayer(reference, single_process_ep_config(E, expert_lora=spec))
    frozen_layer = EPInkling2MoELayer(frozen, single_process_ep_config(E, expert_lora=spec))
    assert frozen_layer.frozen_bank_weight_count() == E * (H * 2 * M + M * H)
    assert bf16_layer.frozen_bank_weight_count() == 0

    parallelism = make_parallelism_config(world_size=1, gpus_per_node=1)
    expected, actual = (
        estimate_rank_flops(_OneLayerModel(layer), seq_len=64, parallelism_config=parallelism, full_model_params=None)
        for layer in (bf16_layer, frozen_layer)
    )
    assert expected.local_expert_params > 0
    assert actual.local_expert_params == expected.local_expert_params
    assert actual.model_flops_per_token == expected.model_flops_per_token
    assert actual.active_model_flops_per_token == expected.active_model_flops_per_token


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


@pytest.mark.parametrize("quant_format", FORMATS)
def test_the_load_warms_every_frozen_bank_behind_the_fence(quant_format, monkeypatch):
    """The load's warm-up reads the banks of each EP layer in the model, frozen or not, inside the join's barrier."""
    _reference, frozen = _dequantized_and_frozen(quant_format, torch.float32)
    frozen_layer = EPInkling2MoELayer(frozen, single_process_ep_config(E))
    bf16_layer = EPInkling2MoELayer(_block(torch.float32), single_process_ep_config(E))
    assert bf16_layer.frozen_expert_banks() == ()
    events = []

    @contextlib.contextmanager
    def fence(_what):
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(whole_model, "get_global_world_size", lambda: 2)
    monkeypatch.setattr(whole_model, "barrier_on_exit", fence)
    for attention_warm_up in ("warm_fa4_kernels", "warm_flex_sliding_kernels", "warm_masked_matmul_kernels"):
        monkeypatch.setattr(whole_model, attention_warm_up, lambda *a, **k: None)
    monkeypatch.setattr(whole_model, "warm_fused_dequant_kernels", lambda banks: events.append(list(banks)))
    whole_model.warm_attention_kernels(nn.ModuleList([bf16_layer, frozen_layer]), dtype=torch.bfloat16)

    entered, banks, left = events
    assert (entered, left) == ("enter", "exit")
    assert [bank.quant_format for bank in banks] == [quant_format] * len(FROZEN_SLOTS)
    for bank, slot in zip(banks, FROZEN_SLOTS, strict=True):
        assert bank.data is getattr(frozen_layer, f"{slot.weight}_frozen_data")


_FROZEN_OK = {"ep_size": 2, "frozen_quantized_experts": True}


@pytest.mark.parametrize("expert_lora", [None, ExpertLoraSpec(r=8, alpha=16)], ids=["no_expert_lora", "expert_lora"])
def test_frozen_quantized_experts_builds_beside_an_ep_group(expert_lora):
    """Attention-only LoRA leaves the banks untrained, as expert LoRA leaves them frozen."""
    config = make_parallelism_config(world_size=2, gpus_per_node=2, expert_lora=expert_lora, **_FROZEN_OK)
    assert config.frozen_quantized_experts


@pytest.mark.parametrize(
    ("override", "named"),
    [
        ({"ep_size": 1}, "expert_parallel_size >= 2"),
        ({"expert_tp_size": 2}, "expert_tensor_parallel_size: 1"),
        ({"ep_lazy_loading": False}, "ep_lazy_loading: true"),
        ({"merge_expert_lora_on_save": True}, "merge_expert_lora_on_save: false"),
        ({"lowp": LowpSpec(precision="fp8")}, "lowp_precision: bf16 or lowp_apply_moe_experts: false"),
        ({"lowp": LowpSpec(precision="fp4")}, "lowp_precision: bf16 or lowp_apply_moe_experts: false"),
        ({"lowp": LowpSpec(precision="mxfp4")}, "lowp_precision: bf16 or lowp_apply_moe_experts: false"),
    ],
)
def test_frozen_quantized_experts_names_each_missing_requirement(override, named):
    with pytest.raises(ValueError, match=named):
        make_parallelism_config(world_size=2, gpus_per_node=2, **{**_FROZEN_OK, **override})


@pytest.mark.parametrize(
    ("frozen", "use_peft", "refused"),
    [(True, False, True), (True, True, False), (False, False, False)],
    ids=["full-fine-tune", "adapters", "bf16-experts"],
)
def test_frozen_quantized_experts_trains_adapters_alone(frozen, use_peft, refused):
    """No save exports the banks, so a full fine-tune over them would fail at its first checkpoint."""
    dist_args, model_config = SimpleNamespace(frozen_quantized_experts=frozen), SimpleNamespace(use_peft=use_peft)
    if not refused:
        reject_frozen_experts_without_adapters(dist_args, model_config)
        return
    with pytest.raises(ValueError, match="frozen_quantized_experts needs use_peft: true"):
        reject_frozen_experts_without_adapters(dist_args, model_config)


def test_a_training_script_refuses_a_full_fine_tune_over_frozen_banks_before_it_builds_its_config(monkeypatch):
    def reached(*args, **kwargs):
        raise AssertionError("the parallelism config was built")

    monkeypatch.setattr(script, "agree_rank_uniform_env", lambda: None)
    monkeypatch.setattr(script, "_pin_single_process_to_bound_gpu", lambda training_config: None)
    monkeypatch.setattr(script, "parallelism_config_from_args", reached)

    with pytest.raises(ValueError, match="frozen_quantized_experts needs use_peft: true"):
        script.init_training_script(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(use_peft=False),
            SimpleNamespace(frozen_quantized_experts=True),
            script_prefix="sft",
            trainer_cls=object,
        )


def test_frozen_quantized_experts_take_low_precision_on_the_dense_mlp_alone():
    lowp = LowpSpec(precision="fp8", apply_moe_experts=False)
    assert make_parallelism_config(world_size=2, gpus_per_node=2, **_FROZEN_OK, lowp=lowp).frozen_quantized_experts


def test_the_frozen_bank_families_are_the_ep_layers_that_declare_them():
    assert frozen_bank_families() == (INKLING2,)
    for cls in ep_layer_classes():
        assert not cls.SUPPORTS_FROZEN_BANKS or cls.FAMILY.frozen_expert_slots, cls.__name__


def test_the_per_expert_sidecars_come_from_the_declared_slots():
    assert set(frozen_expert_sidecar_attrs()) == {
        "gate_up_proj_scale_inv",
        "w13_weight_amax",
        "down_proj_scale_inv",
        "w2_weight_amax",
    }


def _reject_unsupported_model(model_config) -> None:
    config = make_parallelism_config(world_size=2, gpus_per_node=2, **_FROZEN_OK)
    reject_unsupported_model(config, model_config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=True)


def test_another_family_is_refused_from_its_config_before_the_load():
    with pytest.raises(ValueError, match=r"frozen_quantized_experts.*\['inkling2'\].*'qwen3_moe'"):
        _reject_unsupported_model(Qwen3MoeConfig(num_experts=8))


@pytest.mark.parametrize("config_class", [Inkling2TextConfig, Inkling2Config])
def test_inkling2_passes_the_config_time_family_gate(config_class):
    _reject_unsupported_model(config_class())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
