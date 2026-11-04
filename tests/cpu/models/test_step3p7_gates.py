#!/usr/bin/env python
"""Step-3.7-Flash (``step3p7``) parallelism, attention, loading and balancing verdicts. Its per-layer head counts make
the bare ``config.num_attention_heads`` read raise, so every consumer resolves heads through the per-layer seam."""

import pytest
import torch
import torch.nn as nn
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config, Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel

from halo.callbacks.balancing import apply_balancing_mode
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.partition import (
    compute_layer_partition,
    layer_types_period,
    validate_partition,
)
from halo.distributed.tensor_parallel.parallelize_attention import (
    TP_SHARDABLE_ATTENTION_CLASSES,
    apply_tp_to_attention_only,
    validate_tp_head_divisibility,
)
from halo.models.attention.gpt_oss_sinks import SinksPolicy, neutralized_gpt_oss_sinks
from halo.models.attention.layout import resolve_head_dim, resolve_num_key_value_heads
from halo.models.attention.selection import resolve_attn_implementation, supports_fa4_backward
from halo.models.config_access import get_config_field
from halo.models.modality import config_declares_multimodality, is_vlm_model
from halo.models.moe.balancing import resolve_balancing_mode
from halo.models.moe.config_fields import resolve_expert_count, resolve_expert_ffn_width, resolve_router_topk
from halo.models.moe.router_logits import ROUTER_LOGITS_FORCED_OFF_ATTR
from halo.models.mtp.spec import MTP_LAYER_COUNT_FIELDS
from tests.common.distributed import fake_process_group_mesh

# The real checkpoint's shape at divisibility floors: head counts 4 full / 6 sliding mirror the hub's 64/96.
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
    """No recognisable layers: the meta-init shape CP validates config-first."""

    def __init__(self, config):
        super().__init__()
        self.config = config


def test_attention_resolution_lands_on_sdpa():
    """``_supports_flash_attn = False`` diverts every flash request to sdpa at resolve time. Returning flash would
    defer the failure to transformers' model build, after the whole checkpoint has been fetched."""
    assert supports_fa4_backward(_composite_config()) is True
    expected = {
        None: "sdpa",
        "flash_attention_4": "sdpa",
        "flash_attention_2": "sdpa",
        "sdpa": "sdpa",
        "eager": "eager",
    }
    for requested, backend in expected.items():
        resolved = resolve_attn_implementation(
            _composite_config(), requested, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
        assert resolved == backend, requested


def test_cp_wrapper_lookup_refuses_step3p7():
    """No Ulysses wrapper claims ``Step3p7Attention`` (per-layer head counts, ``g_proj`` gate, sliding masks). The
    family has no linear-attention layers, so the ``layer_types`` gate stays silent and the wrapper lookup refuses."""
    assert "Step3p7Attention" not in ULYSSES_ATTENTION_MAP
    with pytest.raises(CPConfigError, match="does not appear to use a supported attention module"):
        reject_unsupported_ulysses_model(_tiny_text_model(), 2)


def test_cp_config_only_shell_hits_the_same_refusal():
    """The config-only ``layer_types`` gate matches nothing here, so the wrapper scan stops a silent CP run."""
    with pytest.raises(CPConfigError, match="does not appear to use a supported attention module"):
        reject_unsupported_ulysses_model(_ConfigOnlyShell(_composite_config()), 2)


def test_tp_shards_zero_layers_and_raises():
    """Per-layer head counts break the uniform q/k/v plan. The head-divisibility preamble must also survive the
    per-layer config to reach this raise."""
    assert "Step3p7Attention" not in TP_SHARDABLE_ATTENTION_CLASSES
    model = _tiny_text_model()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        with pytest.raises(ValueError, match="ZERO attention layers"):
            apply_tp_to_attention_only(model, mesh)


def test_tp_head_divisibility_checks_every_per_layer_head_count():
    """The bare attribute raises transformers' AmbiguousGlobalPerLayerAttributeError. 4 and 6 both split 2-ways, but
    tp4 must fail on the sliding layers' 6."""
    config = _tiny_text_config()
    validate_tp_head_divisibility(config, 1)
    validate_tp_head_divisibility(config, 2)
    with pytest.raises(ValueError, match=r"num_attention_heads \(6\)"):
        validate_tp_head_divisibility(config, 4)


def test_pp_accepts_a_loaded_step3p7_text_model():
    """The decoder threads hidden states only, the embeddings are untied, and transformers drops the checkpoint's MTP
    layers at load, so the contiguous split must accept it."""
    spec = PP_SPEC_MAP["Step3p7TextModel"]
    assert spec.validated and spec.refusal is None and spec.stream_forward is None
    assert "num_nextn_predict_layers" in MTP_LAYER_COUNT_FIELDS
    model = _tiny_text_model()
    assert model.config.num_nextn_predict_layers == 0
    reject_unsupported_pp_model(model)


def test_pp_partition_respects_the_layer_types_period():
    """The same weights cut mid-period when the period is not enforced."""
    assert layer_types_period(_tiny_text_config()) == 4
    assert compute_layer_partition(8, 2, head_layer_equivalents=2.0, boundary_period=1) == [(0, 5), (5, 8)]
    assert compute_layer_partition(8, 2, head_layer_equivalents=2.0, boundary_period=4) == [(0, 4), (4, 8)]


def test_pp_rebase_gate_judges_shift_invariance_on_the_real_truncated_pattern():
    """The real 45 layers are 11 whole periods plus one trailing full-attention layer (the 48-entry hub list minus the
    3-layer MTP pad). A whole-period offset passes even so; any other would swap sliding and full masks."""
    config = Step3p7TextConfig(
        **{
            **TINY_STEP3P7_TEXT_CONFIG,
            "num_hidden_layers": 45,
            "layer_types": (_PERIOD * 12)[:45],
            "mlp_layer_types": None,
        }
    )
    shell = _ConfigOnlyShell(config)
    validate_partition(shell, [(0, 24), (24, 45)])
    validate_partition(shell, [(0, 44), (44, 45)])
    with pytest.raises(ValueError, match="not a whole number of periods"):
        validate_partition(shell, [(0, 23), (23, 45)])


def test_balancing_registries_resolve_on_the_real_config():
    """A miss silently classifies the model as dense."""
    assert Step3p7TextConfig.attribute_map.get("moe_num_experts") == "n_routed_experts"
    assert Step3p7TextConfig.attribute_map.get("moe_top_k") == "num_experts_per_tok"
    for config in (Step3p7TextConfig(), Step3p7Config()):
        assert resolve_router_topk(config) == 8
        assert resolve_expert_count(config) == 288
        assert resolve_expert_ffn_width(config) == 1280
    for config in (_tiny_text_config(), _composite_config()):
        assert resolve_router_topk(config) == 2
        assert resolve_expert_count(config) == 4
        assert resolve_expert_ffn_width(config) == 16


def test_balancing_auto_resolves_none_and_an_explicit_aux_loss_warns_and_stays_off():
    """No native bias buffer, no EP layers and a forward that never takes ``output_router_logits``, so like Gemma 4 the
    plain tree has no route. An explicit ``aux_loss`` must survive the config having no ``router_aux_loss_coef``."""
    model = _tiny_text_model()
    with pytest.raises(AttributeError):
        _ = model.config.router_aux_loss_coef
    assert resolve_balancing_mode("auto", model, is_moe=True) == "none"
    apply_balancing_mode(model, "aux_loss", policy_gradient_loss=False, is_moe=True)
    assert getattr(model.config, ROUTER_LOGITS_FORCED_OFF_ATTR, False) is True
    assert not getattr(model.config, "output_router_logits", False)


def test_per_layer_head_reads_resolve_through_the_registry_seam():
    """A RuntimeError the getattr default does not swallow, so consumers read heads through ``get_config_field``."""
    config = _tiny_text_config()
    with pytest.raises(RuntimeError):
        _ = config.num_attention_heads
    assert get_config_field(config, "num_attention_heads", per_layer_reduce=max) == 6
    assert resolve_head_dim(config) == 8
    assert resolve_num_key_value_heads(config) == 2


def test_sink_reconstruction_is_family_gated_before_the_head_read():
    """``neutralized_gpt_oss_sinks`` runs on every gathered save and RL weight sync; a head read before the family gate
    would raise here."""
    assert neutralized_gpt_oss_sinks(_tiny_text_model()) == {}


def test_vlm_probe_classifies_step3p7():
    """The checkpoint-level probe answers from the config over the name heuristic."""
    assert config_declares_multimodality(Step3p7Config()) is True
    assert config_declares_multimodality(Step3p7TextConfig()) is False
    assert is_vlm_model("stepfun-ai/Step-3.7-Flash", config=Step3p7Config()) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
