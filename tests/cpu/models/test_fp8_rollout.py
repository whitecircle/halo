#!/usr/bin/env python
"""Which tensors a block-FP8 rollout server stores quantized, per validated family, and which families are refused."""

import pytest
import torch
from transformers import InternVLConfig, LlamaConfig, MixtralConfig

from halo.kernels.lowp.block_fp8 import quantize_block_fp8
from halo.models import fp8_rollout
from halo.models.fp8_rollout import Fp8RolloutPlan, fp8_rollout_refusal
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, QWEN3_5
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model
from tests.common.weight_sync import RecordingSender

_FAMILIES = {
    "qwen3_5_moe": TINY_MOE_VLM_FAMILIES["qwen3_5_moe"],
    "qwen3_5_moe_text": TINY_MOE_FAMILIES["qwen3_5_moe_text"],
}
# Substrings of the names a push must keep in bf16 whatever the family, and the module leaves it quantizes.
_STAYS_BF16 = ("linear_attn", "visual", "shared_expert_gate", "mlp.gate.", "norm", "embed_tokens", "lm_head", "mtp")
_QUANTIZED_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _push(family_name: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = tiny_family_model(_FAMILIES[family_name]).to(torch.bfloat16)
    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender, precision="fp8_block")
    return {param.name: param.value for param in sender.params}


@pytest.mark.parametrize("family_name", list(_FAMILIES))
def test_a_push_stores_exactly_the_projections_as_fp8_with_a_scale_each(family_name):
    pushed = _push(family_name)
    quantized = {name for name, value in pushed.items() if value.dtype == torch.float8_e4m3fn}
    assert quantized, "nothing was quantized"
    assert all(name.endswith(".weight") for name in quantized)
    assert {name.rpartition(".")[0].rpartition(".")[2] for name in quantized} <= _QUANTIZED_LEAVES
    assert {f"{name}_scale_inv" for name in quantized} == {n for n in pushed if n.endswith("_scale_inv")}
    for name in quantized:
        scale = pushed[f"{name}_scale_inv"]
        rows, cols = pushed[name].shape
        assert scale.dtype == torch.float32 and scale.shape == (-(-rows // 128), -(-cols // 128))
    for name, value in pushed.items():
        if any(part in name for part in _STAYS_BF16) and not name.endswith("_scale_inv"):
            assert value.dtype != torch.float8_e4m3fn, f"{name} must stay bf16"
        if name.endswith("_scale_inv"):
            assert name.removesuffix("_scale_inv") in quantized


def test_every_projection_of_a_qwen35_moe_layer_quantizes_and_its_gdn_block_and_router_do_not():
    pushed = _push("qwen3_5_moe")
    layer = "model.language_model.layers.3"
    for leaf in (
        "mlp.experts.0.gate_proj",
        "mlp.experts.7.down_proj",
        "mlp.shared_expert.up_proj",
        "self_attn.q_proj",
    ):
        assert pushed[f"{layer}.{leaf}.weight"].dtype == torch.float8_e4m3fn, leaf
    for name in ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", f"{layer}.mlp.gate.weight"):
        assert pushed[name].dtype == torch.bfloat16, name
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in pushed, "the fused bank must be split"


def test_a_mtp_head_a_tower_and_a_layer_past_the_trunk_stay_bf16():
    plan = Fp8RolloutPlan(num_layers=2, expert_keys=("gate_proj", "up_proj", "down_proj"))
    weight = torch.randn(8, 8)
    kept = (
        "mtp.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.attn.proj.weight",
        "model.language_model.layers.2.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
    )
    for name in kept:
        assert [n for n, _ in plan.expand(name, weight)] == [name], name
    assert len(list(plan.expand("model.language_model.layers.1.self_attn.q_proj.weight", weight))) == 2


def test_a_one_dimensional_or_fused_tensor_without_expert_keys_passes_through():
    plan = Fp8RolloutPlan(num_layers=2)
    bank = torch.randn(2, 8, 8)
    assert [n for n, _ in plan.expand("model.layers.0.mlp.experts.gate_up_proj", bank)] == [
        "model.layers.0.mlp.experts.gate_up_proj"
    ]
    assert [n for n, _ in plan.expand("model.layers.0.self_attn.q_proj.bias", torch.randn(8))] == [
        "model.layers.0.self_attn.q_proj.bias"
    ]


def _reference_experts(bank: torch.Tensor, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """The per-expert quantization the batched path must reproduce, one expert and one projection at a time."""
    expected = {}
    for expert in range(bank.shape[0]):
        for name, part in zip(names, bank[expert].chunk(len(names)), strict=True):
            weights, scales = quantize_block_fp8(part)
            expected[f"m.layers.0.mlp.experts.{expert}.{name}.weight"] = weights
            expected[f"m.layers.0.mlp.experts.{expert}.{name}.weight_scale_inv"] = scales
    return expected


def _expand_banks(width, monkeypatch, budget=None):
    if budget is not None:
        monkeypatch.setattr(fp8_rollout, "_QUANTIZE_ELEMENTS", budget)
    torch.manual_seed(0)
    plan = Fp8RolloutPlan(num_layers=1, expert_keys=("gate_proj", "up_proj", "down_proj"))
    gate_up, down = torch.randn(7, 2 * width, 130), torch.randn(7, 130, width)
    calls = []
    real = fp8_rollout.quantize_block_fp8
    monkeypatch.setattr(fp8_rollout, "quantize_block_fp8", lambda w: calls.append(tuple(w.shape)) or real(w))
    got = dict(plan.expand("m.layers.0.mlp.experts.gate_up_proj", gate_up))
    got |= dict(plan.expand("m.layers.0.mlp.experts.down_proj", down))
    expected = _reference_experts(gate_up, ("gate_proj", "up_proj")) | _reference_experts(down, ("down_proj",))
    return got, expected, calls


@pytest.mark.parametrize("width", [128, 192], ids=["aligned", "off-the-block"])
@pytest.mark.parametrize("budget", [None, 3 * 130 * 128], ids=["one-call", "sliced"])
def test_a_fused_bank_equals_the_per_expert_quantization_whatever_the_batching(width, budget, monkeypatch):
    got, expected, _ = _expand_banks(width, monkeypatch, budget)
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        assert got[name].shape == value.shape and torch.equal(got[name].view(torch.uint8), value.view(torch.uint8)), (
            name
        )


def test_a_bank_is_one_quantize_call_not_one_per_expert(monkeypatch):
    """Aligned halves share one call over the gate/up bank; an off-block width quantizes each half apart."""
    _, _, aligned = _expand_banks(128, monkeypatch)
    assert aligned == [(7, 256, 130), (7, 130, 128)]
    _, _, off_block = _expand_banks(192, monkeypatch)
    assert off_block == [(7, 192, 130), (7, 192, 130), (7, 130, 192)]


def test_the_experts_are_views_of_the_batched_result_that_the_client_may_keep(monkeypatch):
    got, _, _ = _expand_banks(128, monkeypatch)
    weight = got["m.layers.0.mlp.experts.3.up_proj.weight"]
    assert weight.is_contiguous() and weight._base is not None
    assert got["m.layers.0.mlp.experts.3.up_proj.weight_scale_inv"].is_contiguous()


def test_only_the_families_a_gpu_row_validated_declare_block_fp8_rollouts():
    assert {spec for spec in MOE_FAMILY_SPECS if spec.fp8_rollout_refusal is None} == {QWEN3_5}


@pytest.mark.parametrize("family_name", list(_FAMILIES))
def test_a_validated_family_has_a_plan(family_name):
    config = tiny_family_model(_FAMILIES[family_name]).config
    assert fp8_rollout_refusal(config) is None
    Fp8RolloutPlan.for_config(config)


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (LlamaConfig(), "no GPU row"),
        (tiny_family_model(TINY_DENSE_FAMILIES["qwen3"]).config, "no GPU row"),
        (tiny_family_model(TINY_DENSE_FAMILIES["qwen3_5"]).config, "no GPU row"),
        (tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]).config, "no GPU row"),
        (MixtralConfig(), "no family spec"),
        (tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]).config, "no GPU row"),
        (tiny_family_model(TINY_MOE_FAMILIES["deepseek_v4"]).config, "no GPU row"),
        (
            InternVLConfig(text_config=tiny_family_model(TINY_MOE_FAMILIES["qwen3_5_moe_text"]).config.to_dict()),
            "the composite 'internvl' wraps",
        ),
    ],
    ids=[
        "dense-llama",
        "dense-qwen3",
        "dense-qwen3.5",
        "qwen3-moe",
        "moe-without-spec",
        "gpt-oss",
        "deepseek-v4",
        "composite-wrapping-qwen3.5-moe",
    ],
)
def test_every_other_family_is_refused_with_its_reason(config, reason):
    assert reason in fp8_rollout_refusal(config)
    with pytest.raises(ValueError, match=reason):
        Fp8RolloutPlan.for_config(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
