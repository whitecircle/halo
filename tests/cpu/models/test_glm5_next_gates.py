#!/usr/bin/env python
"""GLM-5 (``glm5_next``) parallelism, attention, collator and loading gates.
Each refuses at config time; otherwise CP corrupts state, TP replicates on a sharded mesh and flash crashes late."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from halo.data.collators.factory import select_data_collator
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import (
    _UNSUPPORTED_SEQUENCE_AXIS_LAYERS,
    reject_unsupported_ulysses_model,
)
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import MTP_LAYER_COUNT_FIELDS, reject_unsupported_pp_model
from halo.distributed.tensor_parallel.parallelize_attention import (
    TP_SHARDABLE_ATTENTION_CLASSES,
    apply_tp_to_attention_only,
)
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import (
    declares_flash_attention,
    resolve_attn_implementation,
    validate_attn_implementation,
)
from halo.models.moe.config_fields import resolve_expert_count, resolve_expert_ffn_width, resolve_router_topk
from tests.common.distributed import fake_process_group_mesh

# One layer of each block type, widths shrunk to the divisibility floors (index_topk must divide by index_kpool).
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
    "layer_types": ["linear_attention", "indexed_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "max_position_embeddings": 128,
    "pad_token_id": 0,
}


def _tiny_text_model() -> Glm5NextTextModel:
    torch.manual_seed(0)
    return Glm5NextTextModel(Glm5NextTextConfig(**TINY_GLM5_NEXT_TEXT_CONFIG))


class _ConfigOnlyShell(nn.Module):
    """The meta-init shape CP validates config-first."""

    def __init__(self, config):
        super().__init__()
        self.config = config


def test_cp_module_scan_refuses_the_kda_layer():
    """The scan runs before the config gate and before any attention-class probing."""
    assert "Glm5NextTextLinearAttention" in _UNSUPPORTED_SEQUENCE_AXIS_LAYERS
    with pytest.raises(CPConfigError, match="instantiates 'Glm5NextTextLinearAttention'"):
        reject_unsupported_ulysses_model(_tiny_text_model(), 2)


def test_cp_config_only_gate_refuses_a_meta_init_shell():
    config = Glm5NextConfig(text_config=dict(TINY_GLM5_NEXT_TEXT_CONFIG))
    with pytest.raises(CPConfigError) as excinfo:
        reject_unsupported_ulysses_model(_ConfigOnlyShell(config), 2)
    message = str(excinfo.value)
    assert "1 of 2 layers are ``linear_attention``" in message, message
    assert "recurrent scan over the sequence axis" in message, message


def test_cp_config_only_gate_is_family_agnostic():
    class _Config:
        model_type = "qwen3_5_moe"
        layer_types = ["linear_attention", "full_attention"]

    with pytest.raises(CPConfigError, match=r"1 of 2 layers are ``linear_attention``"):
        reject_unsupported_ulysses_model(_ConfigOnlyShell(_Config()), 2)


def test_pp_accepts_the_family_including_the_hub_mtp_metadata_field():
    """The MTP gate judges the built tree, which holds exactly ``num_hidden_layers`` layers here.
    The hub's ``num_nextn_predict_layers: 1`` is metadata: ``_keys_to_ignore_on_load_unexpected`` drops layer 45."""
    spec = PP_SPEC_MAP["Glm5NextTextModel"]
    assert spec.refusal is None
    assert "num_nextn_predict_layers" in MTP_LAYER_COUNT_FIELDS
    model = _tiny_text_model()
    model.config.num_nextn_predict_layers = 1
    assert len(model.layers) == model.config.num_hidden_layers, "premise: no MTP tail was built"
    reject_unsupported_pp_model(model, "none")


def test_tp_shards_zero_layers_and_raises():
    """The indexer and KDA make the q/k/v plan unsound, so selective TP must raise instead of replicating."""
    assert "Glm5NextTextAttention" not in TP_SHARDABLE_ATTENTION_CLASSES
    assert "Glm5NextTextLinearAttention" not in TP_SHARDABLE_ATTENTION_CLASSES
    model = _tiny_text_model()
    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        with pytest.raises(ValueError, match="ZERO attention layers"):
            apply_tp_to_attention_only(model, mesh)


def test_attention_resolution_lands_on_sdpa():
    """Returning flash would defer the failure to transformers' model build, after the whole checkpoint is fetched.
    FA4's own refusals bind only a class that runs a flash label, so an FA4 request reaches the class flag too."""
    assert not declares_flash_attention(Glm5NextConfig())
    expected = {
        None: "sdpa",
        "flash_attention_4": "sdpa",
        "flash_attention_2": "sdpa",
        "sdpa": "sdpa",
        "eager": "eager",
    }
    for requested, backend in expected.items():
        resolved = resolve_attn_implementation(
            Glm5NextConfig(), requested, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
        )
        assert resolved == backend, requested


def test_attention_flash_gate_reads_the_class_flag_not_a_family_list():
    """DeepSeek-V4 has no sdpa either, so it lands on eager."""
    assert validate_attn_implementation(DeepseekV4Config(), "flash_attention_2") == "eager"
    assert validate_attn_implementation(Qwen3Config(), "flash_attention_2") == "flash_attention_2"


def _tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "right"
    return tok


@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
def test_dsa_layers_leak_across_a_packed_row_on_a_dense_backend(attn_impl):
    """The forward hands its DSA layers only the 2D padding mask, so a packed row runs as one causal sequence.
    A DSA-only stack keeps the KDA layers' own crossing out of the reading.
    At drift 0 upstream fixed the mask: clear ``GLM5_NEXT.dense_packing_leaks`` and retire this pin."""
    torch.manual_seed(0)
    config = Glm5NextTextConfig(
        **{
            **TINY_GLM5_NEXT_TEXT_CONFIG,
            "layer_types": ["indexed_attention", "indexed_attention"],
            "mlp_layer_types": ["dense", "dense"],
        }
    )
    config._attn_implementation = attn_impl
    model = Glm5NextTextModel(config).eval()
    lens = (20, 20)
    input_ids = torch.randint(2, config.vocab_size, (1, sum(lens)), generator=torch.Generator().manual_seed(7))
    position_ids = torch.cat([torch.arange(n) for n in lens])[None]
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % (config.vocab_size - 2) + 2
    with torch.no_grad():
        base, swapped = (
            model(input_ids=ids, position_ids=position_ids, use_cache=False).last_hidden_state[0, lens[0] :]
            for ids in (input_ids, variant)
        )
    drift = (base - swapped).abs().max().item()
    assert drift > 1e-3 * base.abs().max().item(), (
        f"GLM-5 Next {attn_impl} isolated packed documents in its DSA layers: upstream fixed the mask plumbing; "
        "clear GLM5_NEXT.dense_packing_leaks and retire this defect pin"
    )


@pytest.mark.parametrize("allow_crossing", [False, True], ids=["default", "opted-in"])
@pytest.mark.parametrize("attn_impl", ["sdpa", "eager"])
@pytest.mark.parametrize(
    "build",
    [
        lambda: Glm5NextConfig(text_config=dict(TINY_GLM5_NEXT_TEXT_CONFIG)),
        lambda: Glm5NextTextConfig(**TINY_GLM5_NEXT_TEXT_CONFIG),
    ],
    ids=["composite", "text"],
)
def test_flat_packing_is_refused_on_every_backend_the_family_runs(build, attn_impl, allow_crossing):
    """No flash support upstream, so every backend is dense, where the DSA layers leak whatever the opt-in admits."""
    config = build()
    config._attn_implementation = attn_impl
    with pytest.raises(
        ValueError, match="packing=True on attn_implementation=.* is refused for model_type='glm5_next"
    ):
        select_data_collator(
            _tokenizer(),
            train_on_completions_only=False,
            packing=True,
            model_config=config,
            allow_cross_document_linear_attention=allow_crossing,
        )


def test_balancing_registries_resolve_on_the_real_config():
    """A miss silently classifies the model as dense: no EP wrapping, no balancing, no MoE metrics."""
    assert Glm5NextTextConfig.attribute_map.get("num_local_experts") == "n_routed_experts"
    for config in (Glm5NextTextConfig(), Glm5NextConfig()):
        assert resolve_router_topk(config) == 8
        assert resolve_expert_count(config) == 288
        assert resolve_expert_ffn_width(config) == 2048


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
