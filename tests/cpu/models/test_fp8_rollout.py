#!/usr/bin/env python
"""Which tensors a block-FP8 rollout server stores quantized, per validated family, and which families are refused."""

import pytest
import torch
from transformers import LlamaConfig, MixtralConfig

from halo.models.fp8_rollout import Fp8RolloutPlan, fp8_rollout_refusal
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, QWEN3, QWEN3_5
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model
from tests.common.weight_sync import RecordingSender

_FAMILIES = {
    "qwen3_5_moe": TINY_MOE_VLM_FAMILIES["qwen3_5_moe"],
    "qwen3_5_moe_text": TINY_MOE_FAMILIES["qwen3_5_moe_text"],
    "qwen3_moe": TINY_MOE_FAMILIES["qwen3_moe"],
    "qwen3": TINY_DENSE_FAMILIES["qwen3"],
    "qwen3_5": TINY_DENSE_FAMILIES["qwen3_5"],
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


def test_only_the_families_a_gpu_row_validated_declare_block_fp8_rollouts():
    assert {spec for spec in MOE_FAMILY_SPECS if spec.fp8_rollout_refusal is None} == {QWEN3, QWEN3_5}


@pytest.mark.parametrize("family_name", list(_FAMILIES))
def test_a_validated_family_has_a_plan(family_name):
    config = tiny_family_model(_FAMILIES[family_name]).config
    assert fp8_rollout_refusal(config) is None
    Fp8RolloutPlan.for_config(config)


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (LlamaConfig(), "no GPU row"),
        (MixtralConfig(), "no family spec"),
        (tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]).config, "no GPU row"),
        (tiny_family_model(TINY_MOE_FAMILIES["deepseek_v4"]).config, "no GPU row"),
    ],
    ids=["dense-llama", "moe-without-spec", "gpt-oss", "deepseek-v4"],
)
def test_every_other_family_is_refused_with_its_reason(config, reason):
    assert reason in fp8_rollout_refusal(config)
    with pytest.raises(ValueError, match=reason):
        Fp8RolloutPlan.for_config(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
