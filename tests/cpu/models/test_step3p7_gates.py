#!/usr/bin/env python
"""CPU gates for Step-3.7-Flash (``step3p7``): parallelism, attention, loading and balancing verdicts.

Step-3.7-Flash is a composite VLM (``Step3p7ForConditionalGeneration``, no CausalLM sibling) whose
45-layer text tower interleaves full and sliding attention on a period-4 pattern with PER-LAYER
head counts (64 full / 96 sliding via ``per_layer_config``), gates every head's attention output
through a sigmoid ``g_proj``, declares ``_supports_flash_attn = False``, and strips the aux-loss
fields (``router_aux_loss_coef`` / ``output_router_logits``) from its config entirely. Each fact
closes a toolkit path that must refuse LOUDLY at config time — and the per-layer head counts make
the bare ``config.num_attention_heads`` read RAISE, so every reachable consumer must resolve heads
through the per-layer-aware seam instead.

    python tests/cpu/models/test_step3p7_gates.py
"""

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config, Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel

from src.distributed.context_parallel.layers.registry import WRAPPER_CLASS_MAP
from src.distributed.context_parallel.validation import UlyssesConfigError, validate_model_for_ulysses
from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.distributed.pipeline_parallel.split import (
    MTP_LAYER_COUNT_FIELDS,
    PP_SPEC_MAP,
    compute_layer_partition,
    layer_types_period,
    validate_model_supports_pp,
)
from src.distributed.pipeline_parallel.stage import reject_layer_type_rebase
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES
from src.distributed.tensor_parallel.parallelize_attention import (
    apply_tp_to_attention_only,
    validate_tp_head_divisibility,
)
from src.models.attention_geometry import (
    resolve_head_dim,
    resolve_num_key_value_heads,
)
from src.models.loading.config_levels import get_config_field
from src.models.modality import config_declares_multimodality, is_vlm_model
from src.models.moe_balancing import (
    ROUTER_EXPERT_COUNT_FIELDS,
    ROUTER_LOGITS_FORCED_OFF_ATTR,
    get_first_router_field,
    resolve_balancing_mode,
    resolve_expert_ffn_shard_width,
    resolve_router_topk,
)
from src.models.patches.attention import model_fa4_backward_nan_prone, resolve_attn_implementation
from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks
from tests.common.distributed import fake_process_group_mesh

PartialState()  # the attention resolver and balancing strategy log through accelerate's logger

# The real checkpoint's shape at divisibility floors: period-4 layer_types (full, sliding x3),
# HETEROGENEOUS head counts (4 full / 6 sliding, mirroring the hub's 64/96), dense + sparse MLPs.
_PERIOD = ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]
TINY_STEP3P7_TEXT_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_sliding_attention_heads": 6,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "share_expert_dim": 16,
    "sliding_window": 8,
    "layer_types": _PERIOD * 2,
    "mlp_layer_types": ["dense", "sparse"] * 4,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
}


def _tiny_text_config() -> Step3p7TextConfig:
    return Step3p7TextConfig(**TINY_STEP3P7_TEXT_CONFIG)


def _tiny_text_model() -> Step3p7TextModel:
    torch.manual_seed(0)
    return Step3p7TextModel(_tiny_text_config())


def _composite_config() -> Step3p7Config:
    return Step3p7Config(text_config=dict(TINY_STEP3P7_TEXT_CONFIG))


class _ConfigOnlyShell(nn.Module):
    """A module tree with no recognisable layers — the meta-init shape CP validates config-first."""

    def __init__(self, config):
        super().__init__()
        self.config = config


# Attention resolution


def test_attention_resolution_lands_on_sdpa():
    """``Step3p7ForConditionalGeneration._supports_flash_attn = False`` — resolved through the
    ImageTextToText mapping, since no CausalLM sibling exists — must divert every flash request to
    sdpa at resolve time; returning flash defers the failure to transformers' model build, after
    the whole checkpoint has been fetched. Not an FA4-NaN entry: the refusal derives from the class
    flag and covers FA2/FA3 too."""
    assert model_fa4_backward_nan_prone(_composite_config()) is False
    for requested in (None, "flash_attention_4", "flash_attention_2"):
        assert resolve_attn_implementation(_composite_config(), requested, torch.bfloat16) == "sdpa", requested
    assert resolve_attn_implementation(_composite_config(), "sdpa", torch.bfloat16) == "sdpa"
    assert resolve_attn_implementation(_composite_config(), "eager", torch.bfloat16) == "eager"


# Context parallelism


def test_cp_wrapper_lookup_refuses_step3p7():
    """No Ulysses wrapper claims ``Step3p7Attention`` (per-layer head counts, per-head sigmoid
    ``g_proj`` gate applied after the attention output, sliding masks), and the family has no
    linear-attention layers so the ``layer_types`` gate stays silent — the wrapper lookup itself is
    the refusal, and it must fire loudly instead of leaving layers unwrapped."""
    assert "Step3p7Attention" not in WRAPPER_CLASS_MAP
    with pytest.raises(UlyssesConfigError, match="does not appear to use a supported attention module"):
        validate_model_for_ulysses(_tiny_text_model(), 2)


def test_cp_config_only_shell_hits_the_same_refusal():
    """A meta-init shell (no instantiated modules) must be refused by the same lookup — the
    config-only ``layer_types`` gate matches nothing here (no ``linear_attention`` entries), so the
    wrapper scan is what stands between step3p7 and a silent unwrapped CP run."""
    with pytest.raises(UlyssesConfigError, match="does not appear to use a supported attention module"):
        validate_model_for_ulysses(_ConfigOnlyShell(_composite_config()), 2)


# Tensor parallelism


def test_tp_shards_zero_layers_and_raises():
    """``Step3p7Attention`` is not in TP_SHARDABLE_ATTENTION_CLASSES (per-layer head counts break
    the uniform q/k/v plan), so selective TP must reach the zero-sharded-layers raise — which also
    requires the head-divisibility preamble to survive the per-layer-heterogeneous config instead
    of dying on the ambiguous ``num_attention_heads`` read."""
    assert "Step3p7Attention" not in TP_SHARDABLE_ATTENTION_CLASSES
    model = _tiny_text_model()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        with pytest.raises(ValueError, match="sharded ZERO attention layers"):
            apply_tp_to_attention_only(model, mesh)


def test_tp_head_divisibility_checks_every_per_layer_head_count():
    """The gate reads heads through the per-layer seam (the bare attribute raises transformers'
    AmbiguousGlobalPerLayerAttributeError) and must check EVERY declared value: 4 and 6 both split
    2-ways, but tp4 must be rejected on the sliding layers' 6 even though the full layers' 4
    divides."""
    config = _tiny_text_config()
    validate_tp_head_divisibility(config, 1)
    validate_tp_head_divisibility(config, 2)
    with pytest.raises(ValueError, match=r"num_attention_heads \(6\)"):
        validate_tp_head_divisibility(config, 4)


# Pipeline parallelism


def test_pp_accepts_a_loaded_step3p7_text_model():
    """No refusal spec: the decoder threads hidden states only (no cross-layer streams), the
    embeddings are untied, and a freshly constructed model declares no MTP tail
    (``num_nextn_predict_layers`` stays 0 — the registered spelling — because transformers drops
    the checkpoint's MTP layers at load), so the generic contiguous split must accept it."""
    assert "Step3p7TextModel" not in PP_SPEC_MAP
    assert "num_nextn_predict_layers" in MTP_LAYER_COUNT_FIELDS
    model = _tiny_text_model()
    assert model.config.num_nextn_predict_layers == 0
    validate_model_supports_pp(model)


def test_pp_partition_respects_the_layer_types_period():
    """``layer_types_period`` must read the period-4 pattern off the text config, and the
    head-weighted rebalance must move whole periods only — the same weights cut mid-period when the
    period is not enforced."""
    assert layer_types_period(_tiny_text_config()) == 4
    assert compute_layer_partition(8, 2, head_layer_equivalents=2.0, boundary_period=1) == [(0, 5), (5, 8)]
    assert compute_layer_partition(8, 2, head_layer_equivalents=2.0, boundary_period=4) == [(0, 4), (4, 8)]


def test_pp_rebase_gate_judges_shift_invariance_on_the_real_truncated_pattern():
    """The real checkpoint holds 45 layers: 11 whole periods plus one trailing full-attention layer
    (the 48-entry hub list minus the 3-layer MTP pad). The stage gate compares suffix to prefix, so
    a whole-period offset passes even though 45 is not a period multiple — the manual ``pp_split``
    path — while any other offset would swap sliding and full masks and must refuse."""
    config = Step3p7TextConfig(
        **{
            **TINY_STEP3P7_TEXT_CONFIG,
            "num_hidden_layers": 45,
            "layer_types": (_PERIOD * 12)[:45],
            "mlp_layer_types": None,
        }
    )
    shell = _ConfigOnlyShell(config)
    reject_layer_type_rebase(shell, 0, 0)
    reject_layer_type_rebase(shell, 24, 1)
    reject_layer_type_rebase(shell, 44, 1)
    with pytest.raises(ValueError, match="not a whole number of periods"):
        reject_layer_type_rebase(shell, 23, 1)


# Balancing config-field registries


def test_balancing_registries_resolve_on_the_real_config():
    """The router registries must resolve step3p7's spellings on the text config AND through the
    composite wrapper — ``num_experts_per_tok`` directly, the expert count through the
    ``num_local_experts``/``moe_num_experts`` attribute_map entries onto ``n_routed_experts``,
    ``moe_intermediate_size`` directly. A miss silently classifies the model as dense."""
    assert Step3p7TextConfig.attribute_map.get("moe_num_experts") == "n_routed_experts"
    assert Step3p7TextConfig.attribute_map.get("moe_top_k") == "num_experts_per_tok"
    for config in (Step3p7TextConfig(), Step3p7Config()):
        assert resolve_router_topk(config) == 8
        assert get_first_router_field(config, ROUTER_EXPERT_COUNT_FIELDS) == 288
        assert resolve_expert_ffn_shard_width(config) == 1280
    for config in (_tiny_text_config(), _composite_config()):
        assert resolve_router_topk(config) == 2
        assert get_first_router_field(config, ROUTER_EXPERT_COUNT_FIELDS) == 4
        assert resolve_expert_ffn_shard_width(config) == 16


def test_balancing_auto_resolves_none_and_an_explicit_aux_loss_warns_and_stays_off():
    """Plain HF tree: no native ``balancing_biases`` buffer, no EP wrappers, and a forward that never
    takes ``output_router_logits`` — so ``auto`` resolves to ``none``; like Gemma 4, plain-tree
    step3p7 has no balancing route. An EXPLICIT ``aux_loss`` still has to survive the family
    stripping ``router_aux_loss_coef`` and ``output_router_logits`` from its config entirely: the
    no-usable-coefficient branch stamps router logits forced-off and leaves them off, instead of
    crashing on the absent field."""
    model = _tiny_text_model()
    with pytest.raises(AttributeError):
        _ = model.config.router_aux_loss_coef
    assert resolve_balancing_mode("auto", model, is_moe=True) == "none"
    apply_balancing_strategy(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    assert getattr(model.config, ROUTER_LOGITS_FORCED_OFF_ATTR, False) is True
    assert getattr(model.config, "output_router_logits", None) is None


# Per-layer head heterogeneity


def test_per_layer_head_reads_resolve_through_the_registry_seam():
    """``config.num_attention_heads`` RAISES on this family (a RuntimeError the getattr default
    does not swallow), so every reachable consumer must read heads through ``get_config_field``'s
    per-layer resolution; the homogeneous fields keep resolving directly."""
    config = _tiny_text_config()
    with pytest.raises(RuntimeError):
        _ = config.num_attention_heads
    assert get_config_field(config, "num_attention_heads", per_layer_reduce=max) == 6
    assert resolve_head_dim(config) == 8
    assert resolve_num_key_value_heads(config) == 2


def test_sink_reconstruction_is_family_gated_before_the_head_read():
    """``neutralized_gpt_oss_sinks`` runs on every gathered save and RL weight sync; reading the
    head count before the family gate raised on step3p7's ambiguous ``num_attention_heads``. A
    non-sinks model must come back empty without touching that field."""
    assert neutralized_gpt_oss_sinks(_tiny_text_model()) == {}


# VLM classification


def test_vlm_probe_classifies_step3p7():
    """The composite config declares multimodality (ImageTextToText mapping + ``vision_config``
    sub-config) and the bare text config does not; the checkpoint-level probe must answer from the
    config, not the name heuristic."""
    assert config_declares_multimodality(Step3p7Config()) is True
    assert config_declares_multimodality(Step3p7TextConfig()) is False
    assert is_vlm_model("stepfun-ai/Step-3.7-Flash", config=Step3p7Config()) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
