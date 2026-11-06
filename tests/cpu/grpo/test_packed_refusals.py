#!/usr/bin/env python
"""Everything ``packed_microbatch_tokens`` refuses at construction, each with the mechanism and a remedy in its message:
options that change the objective, models and backends that would cross documents, and budgets no dispatch carries."""

import types

import pytest

import halo.models.segment_markers as segment_markers
from halo.models.attention.gpt_oss_sinks import SINKS_POLICY_ATTR, SinksPolicy
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.packed_rules import (
    reject_undispatchable_budget,
    reject_unpackable_model,
    reject_unpackable_options,
    reject_unpackable_trainer,
    resolve_dispatch_ceiling,
)
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    tiny_family_model,
)


def _args(**overrides) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        **{
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "loss_type": "dapo",
            "top_entropy_quantile": 1.0,
            "steps_per_generation": 8,
            "num_iterations": 1,
            "gradient_accumulation_steps": 8,
            "eval_strategy": "no",
            "eval_on_start": False,
        }
        | overrides
    )


def _options(**overrides) -> dict:
    return {"use_chunked_logprobs": True, "length_ordered": None, "misaligned_accumulation": False} | overrides


def test_the_admitted_options_pass():
    reject_unpackable_options(_args(), **_options())
    reject_unpackable_options(_args(loss_type="grpo", eval_strategy="steps"), **_options())


_OPTION_REFUSALS = {
    "full logits": (_args(), {"use_chunked_logprobs": False}, "use_chunked_grpo_logprobs: true"),
    "batch size": (_args(per_device_train_batch_size=2), {}, "per_device_train_batch_size: 1"),
    "bnpo": (_args(loss_type="bnpo"), {}, "depend on which rows share one"),
    "vespo": (_args(loss_type="vespo"), {}, "loss_type='vespo'"),
    "entropy quantile": (_args(top_entropy_quantile=0.2), {}, "top_entropy_quantile"),
    "length deal": (_args(), {"length_ordered": True}, "Unset length_ordered_microbatches"),
    "misaligned": (_args(gradient_accumulation_steps=12), {"misaligned_accumulation": True}, "Align them"),
    "eval batch": (
        _args(eval_strategy="steps", per_device_eval_batch_size=4),
        {},
        "per_device_eval_batch_size: 1",
    ),
    "eval on start": (
        _args(eval_on_start=True, per_device_eval_batch_size=2),
        {},
        "per_device_eval_batch_size: 1",
    ),
}


@pytest.mark.parametrize("case", _OPTION_REFUSALS, ids=list(_OPTION_REFUSALS))
def test_each_option_refusal_names_its_mechanism_and_remedy(case):
    args, options, expected = _OPTION_REFUSALS[case]
    with pytest.raises(ValueError, match=expected):
        reject_unpackable_options(args, **_options(**options))


def test_an_eval_batch_other_than_one_is_ignored_when_the_run_never_evaluates():
    reject_unpackable_options(_args(per_device_eval_batch_size=8), **_options())


def test_a_trainer_declaring_a_refusal_is_refused_with_it_and_the_others_pass():
    assert DistributedOnlineGRPOTrainer.PACKED_ROWS_REFUSAL is None
    reject_unpackable_trainer(DistributedOnlineGRPOTrainer, DistributedOnlineGRPOTrainer.PACKED_ROWS_REFUSAL)
    with pytest.raises(ValueError, match=r"DistributedSDPGTrainer.*teacher pass.*Unset packed_microbatch_tokens"):
        reject_unpackable_trainer(DistributedSDPGTrainer, DistributedSDPGTrainer.PACKED_ROWS_REFUSAL)


@pytest.fixture
def kernels_present(monkeypatch):
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("causal_conv1d", lambda: True),))


def _model(family, overrides=None):
    return tiny_family_model(family, overrides=overrides)


_ADMITTED = {
    "llama": (TINY_DENSE_ATTENTION_FAMILIES["llama"], "flash_attention_2"),
    "qwen3_5 GatedDeltaNet M-RoPE": (TINY_DENSE_FAMILIES["qwen3_5"], "flash_attention_2"),
    "qwen3_next": (TINY_MOE_FAMILIES["qwen3_next"], "flash_attention_2"),
    "qwen3 moe": (TINY_MOE_FAMILIES["qwen3_moe"], "flash_attention_4"),
    "nemotron_h": (TINY_MOE_FAMILIES["nemotron_h"], "flash_attention_2"),
    "lfm2_moe": (TINY_MOE_FAMILIES["lfm2_moe"], "flash_attention_2"),
    "gpt_oss on FA4": (TINY_MOE_FAMILIES["gpt_oss"], "flash_attention_4"),
}


@pytest.mark.parametrize("case", _ADMITTED, ids=list(_ADMITTED))
def test_the_families_whose_rows_isolate_are_admitted_on_a_varlen_backend(case, kernels_present):
    family, attn = _ADMITTED[case]
    reject_unpackable_model(_model(family), attn)


_MODEL_REFUSALS = {
    "deepseek_v4": ("deepseek_v4", "flash_attention_2", "row index"),
    "mistral4": ("mistral4", "flash_attention_2", "row index"),
    "glm5_next": ("glm5_next", "flash_attention_2", "receive no document boundary"),
    "inkling": ("inkling_text", "flash_attention_2", "receive no document boundary"),
}


def test_a_linear_attention_family_no_marker_reaches_is_refused(kernels_present):
    with pytest.raises(ValueError, match="receive no document boundary"):
        reject_unpackable_model(tiny_family_model(TINY_BAILING_RING_FAMILY), "flash_attention_2")


@pytest.mark.parametrize("case", _MODEL_REFUSALS, ids=list(_MODEL_REFUSALS))
def test_a_family_a_row_would_cross_or_misread_is_refused(case, kernels_present):
    name, attn, why = _MODEL_REFUSALS[case]
    with pytest.raises(ValueError, match=why):
        reject_unpackable_model(_model(TINY_MOE_FAMILIES[name]), attn)


@pytest.mark.parametrize("attn", ["sdpa", "eager", "flex_attention", None])
def test_a_backend_that_is_not_varlen_flash_is_refused_for_every_family(attn, kernels_present):
    with pytest.raises(ValueError, match="needs a varlen flash attention backend"):
        reject_unpackable_model(_model(TINY_DENSE_ATTENTION_FAMILIES["llama"]), attn)


def test_a_gated_delta_net_family_without_its_kernels_is_refused(monkeypatch):
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("causal_conv1d", lambda: False),))
    with pytest.raises(ValueError, match="causal_conv1d unavailable"):
        reject_unpackable_model(_model(TINY_DENSE_FAMILIES["qwen3_5"]), "flash_attention_2")


def test_live_sinks_need_fa4_and_neutralized_ones_take_any_varlen_backend(kernels_present):
    model = _model(TINY_MOE_FAMILIES["gpt_oss"])
    with pytest.raises(ValueError, match="live attention sinks needs flash_attention_4"):
        reject_unpackable_model(model, "flash_attention_2")
    setattr(model, SINKS_POLICY_ATTR, SinksPolicy.NEUTRALIZED)
    reject_unpackable_model(model, "flash_attention_2")


def test_the_mrope_decoders_are_admitted_with_their_markers(kernels_present):
    for family in (TINY_DENSE_FAMILIES["qwen3_5"], TINY_MOE_FAMILIES["qwen3_5_moe_text"]):
        reject_unpackable_model(_model(family), "flash_attention_2")


def _parallelism(ep_size: int, requires_rdma: bool = False):
    return types.SimpleNamespace(ep_size=ep_size, requires_rdma=requires_rdma)


def _moe_config():
    return _model(TINY_MOE_FAMILIES["qwen3_moe"]).config


def test_a_budget_the_dispatch_cannot_carry_is_refused_before_the_first_rollout():
    config = _moe_config()
    reject_undispatchable_budget(_parallelism(1), config, 10**9)
    reject_undispatchable_budget(_parallelism(8), config, 8192)
    with pytest.raises(ValueError, match="packed_microbatch_tokens=10000000 cannot be dispatched"):
        reject_undispatchable_budget(_parallelism(8), config, 10_000_000)
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        reject_undispatchable_budget(_parallelism(16, requires_rdma=True), config, 9000)


def test_the_planner_ceiling_is_the_tighter_of_the_wire_limit_and_the_gin_ceiling():
    config = _moe_config()
    assert resolve_dispatch_ceiling(_parallelism(1), config) is None
    node_local = resolve_dispatch_ceiling(_parallelism(8), config)
    cross_node = resolve_dispatch_ceiling(_parallelism(16, requires_rdma=True), config)
    assert cross_node == 8192 and node_local >= cross_node
    assert node_local % 256 == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
