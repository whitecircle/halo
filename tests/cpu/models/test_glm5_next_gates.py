#!/usr/bin/env python
"""CPU gates for GLM-5 (``glm5_next``): parallelism, attention, collator and loading refusals.

GLM-5.3-Flash interleaves 34 KDA linear-attention layers (causal Conv1d + recurrent delta-rule
scan) with 11 DeepSeek-sparse-attention layers (MLA + indexer), threads a 4x-widened
hyper-connection residual stream through every layer, declares ``_supports_flash_attn = False``,
and ships only the composite ``Glm5NextForConditionalGeneration`` — no CausalLM sibling. Each of
those facts closes a toolkit path, and each closure must refuse LOUDLY at config time; the silent
alternative is cross-rank state corruption (CP), a replicated model on a sharded mesh (TP), or a
crash deep inside transformers (flash, text_only). PP is the exception: the widened stream IS the
stage boundary under ``Glm5NextPPSpec`` (tests/cpu/parallelism/test_pp_hyper_connection_stages.py),
and only the generic gates (MTP metadata, tie) still apply here.

    python tests/cpu/models/test_glm5_next_gates.py
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

from src.data.collators.factory import select_data_collator
from src.distributed.context_parallel.validation import (
    _UNSUPPORTED_SEQUENCE_AXIS_LAYERS,
    UlyssesConfigError,
    validate_model_for_ulysses,
)
from src.distributed.pipeline_parallel.split import MTP_LAYER_COUNT_FIELDS, PP_SPEC_MAP, validate_model_supports_pp
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES
from src.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from src.models.moe_balancing import (
    ROUTER_EXPERT_COUNT_FIELDS,
    get_first_router_field,
    resolve_expert_ffn_shard_width,
    resolve_router_topk,
)
from src.models.patches.attention import (
    model_fa4_backward_nan_prone,
    resolve_attn_implementation,
    validate_attn_implementation,
)
from tests.common.distributed import fake_process_group_mesh

PartialState()  # the collator factory logs through accelerate's logger, which needs the state

# One layer of each block type, MLA/indexer/KDA widths shrunk to the divisibility floors
# (index_topk must divide by index_kpool).
TINY_GLM5_NEXT_TEXT_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "moe_intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "kv_lora_rank": 16,
    "q_lora_rank": 16,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    "index_topk": 16,
    "index_head_dim": 8,
    "index_n_heads": 2,
    "index_kpool": 4,
    "linear_head_dim": 8,
    "linear_num_heads": 4,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "max_position_embeddings": 128,
    "pad_token_id": 0,
}


def _tiny_text_model() -> Glm5NextTextModel:
    torch.manual_seed(0)
    return Glm5NextTextModel(Glm5NextTextConfig(**TINY_GLM5_NEXT_TEXT_CONFIG))


class _ConfigOnlyShell(nn.Module):
    """A module tree with no recognisable layers — the meta-init shape CP validates config-first."""

    def __init__(self, config):
        super().__init__()
        self.config = config


# Context parallelism — both refusal paths


def test_cp_module_scan_refuses_the_kda_layer():
    """The instantiated tree must be refused by the module scan itself, naming the KDA class —
    the scan runs before the config gate and before any attention-class probing."""
    assert "Glm5NextTextLinearAttention" in _UNSUPPORTED_SEQUENCE_AXIS_LAYERS
    with pytest.raises(UlyssesConfigError, match="instantiates 'Glm5NextTextLinearAttention'"):
        validate_model_for_ulysses(_tiny_text_model(), 2)


def test_cp_config_only_gate_refuses_a_meta_init_shell():
    """Without instantiated modules the ``layer_types`` gate must still refuse, counting the
    linear-attention layers and naming the sequence-axis mechanism."""
    config = Glm5NextConfig(text_config=dict(TINY_GLM5_NEXT_TEXT_CONFIG))
    with pytest.raises(UlyssesConfigError) as excinfo:
        validate_model_for_ulysses(_ConfigOnlyShell(config), 2)
    message = str(excinfo.value)
    assert "1 of 2 layers are ``linear_attention``" in message, message
    assert "recurrent scan over the sequence axis" in message, message


def test_cp_config_only_gate_is_family_agnostic():
    """Any family declaring ``linear_attention`` layer types is refused by the same config-only gate."""

    class _Config:
        model_type = "qwen3_5_moe"
        layer_types = ["linear_attention", "full_attention"]

    with pytest.raises(UlyssesConfigError, match=r"1 of 2 layers are ``linear_attention``"):
        validate_model_for_ulysses(_ConfigOnlyShell(_Config()), 2)


# Pipeline parallelism


def test_pp_accepts_the_family_including_the_hub_mtp_metadata_field():
    """``Glm5NextPPSpec`` carries the widened-stream boundary (its own suite:
    tests/cpu/parallelism/test_pp_hyper_connection_stages.py), so the family passes structural
    validation — including with the hub config's metadata-only ``num_nextn_predict_layers: 1``
    (layer 45, dropped at load via ``_keys_to_ignore_on_load_unexpected``): the MTP gate judges the
    BUILT tree, and this one holds exactly ``num_hidden_layers`` layers."""
    spec = PP_SPEC_MAP["Glm5NextTextModel"]
    assert spec.SUPPORTS_PP is True
    assert "num_nextn_predict_layers" in MTP_LAYER_COUNT_FIELDS
    model = _tiny_text_model()
    model.config.num_nextn_predict_layers = 1
    assert len(model.layers) == model.config.num_hidden_layers, "premise: no MTP tail was built"
    validate_model_supports_pp(model, "none")


# Tensor parallelism


def test_tp_shards_zero_layers_and_raises():
    """Neither attention class may enter TP_SHARDABLE_ATTENTION_CLASSES (the indexer and KDA make
    the q/k/v plan unsound), so selective TP must hit the zero-sharded-layers raise instead of
    leaving every weight replicated on a mesh that assumes sharding."""
    assert "Glm5NextTextAttention" not in TP_SHARDABLE_ATTENTION_CLASSES
    assert "Glm5NextTextLinearAttention" not in TP_SHARDABLE_ATTENTION_CLASSES
    model = _tiny_text_model()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        with pytest.raises(ValueError, match="sharded ZERO attention layers"):
            apply_tp_to_attention_only(model, mesh)


# Attention resolution


def test_attention_resolution_lands_on_sdpa():
    """``Glm5NextForConditionalGeneration._supports_flash_attn = False`` — the resolver must divert
    every flash request to sdpa itself; returning flash defers the failure to transformers' model
    build, after the whole checkpoint has been fetched. Not an FA4-NaN entry: the refusal derives
    from the class flag and covers FA2/FA3 too."""
    assert model_fa4_backward_nan_prone(Glm5NextConfig()) is False
    for requested in (None, "flash_attention_4", "flash_attention_2"):
        assert resolve_attn_implementation(Glm5NextConfig(), requested, torch.bfloat16) == "sdpa", requested
    assert resolve_attn_implementation(Glm5NextConfig(), "sdpa", torch.bfloat16) == "sdpa"
    assert resolve_attn_implementation(Glm5NextConfig(), "eager", torch.bfloat16) == "eager"


def test_attention_flash_gate_reads_the_class_flag_not_a_family_list():
    """The same derivation must refuse flash for every family whose class declares it unsupported
    (DeepSeek-V4: no sdpa either, so it lands on eager) and keep it for one that does not."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    assert validate_attn_implementation(DeepseekV4Config(), "flash_attention_2") == "eager"
    assert validate_attn_implementation(Qwen3Config(), "flash_attention_2") == "flash_attention_2"


# Packing collator — no GDN kwargs, no crash (KDA shares the Zaya/Ling mixer-crossing semantics)


def _tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            out["input_ids"].append(ids + [tok.pad_token_id] * pad)
            out["attention_mask"].append([1] * len(ids) + [0] * pad)
        return {key: torch.tensor(value) for key, value in out.items()}

    tok.pad.side_effect = _pad
    return tok


def test_packing_collator_round_trips_without_gdn_kwargs():
    """glm5_next's modeling reads no ``seq_idx``/``cu_seq_lens`` (the KDA conv and scan cross packed
    document boundaries by construction — the Zaya/Ling semantics), so the factory must neither
    emit the markers nor trip the GDN wheel refusal, and the packed batch must build cleanly."""
    config = Glm5NextConfig(text_config=dict(TINY_GLM5_NEXT_TEXT_CONFIG))
    config._attn_implementation = "sdpa"
    collator = select_data_collator(_tokenizer(), packing=True, model_config=config)
    batch = collator.torch_call([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "seq_lengths": [2, 1]}])
    assert "seq_idx" not in batch and "cu_seq_lens_q" not in batch
    assert batch["input_ids"].tolist() == [[5, 6, 7]]


# Balancing config-field registries


def test_balancing_registries_resolve_on_the_real_config():
    """The router registries must resolve GLM-5's spellings on the text config AND through the
    composite wrapper: ``num_experts_per_tok`` directly, the expert count via the
    ``num_local_experts``→``n_routed_experts`` attribute_map, ``moe_intermediate_size`` directly.
    A miss silently classifies the model as dense (no EP wrapping, no balancing, no MoE metrics)."""
    assert Glm5NextTextConfig.attribute_map.get("num_local_experts") == "n_routed_experts"
    for config in (Glm5NextTextConfig(), Glm5NextConfig()):
        assert resolve_router_topk(config) == 8
        assert get_first_router_field(config, ROUTER_EXPERT_COUNT_FIELDS) == 288
        assert resolve_expert_ffn_shard_width(config) == 2048


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
