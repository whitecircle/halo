#!/usr/bin/env python
"""GatedDeltaNet detection reads the layers a config declares, not its ``model_type``.

A GatedDeltaNet layer carries conv and recurrent state across a packed row's documents unless the collator hands
it ``seq_idx`` and ``cu_seq_lens``, and the miss is silent. The verdict is ``linear_attention`` in ``layer_types``
on a config declaring GatedDeltaNet's key/value head split; it must keep every roster family's former
name-keyed verdict and reach a GatedDeltaNet family under any other name. The transformers-wide sweep pins the
configs whose verdict moved at the pinned transformers; a bump that ships a new hybrid family fails it until the
family's layers are checked and its verdict is added to ``MOVED``.

    python tests/cpu/models/test_gated_delta_net_detection.py
"""

import contextlib
import functools

import pytest
import transformers
from transformers.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig

from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.models import segment_markers
from halo.models.config_access import config_model_types, model_type_matches
from halo.models.segment_markers import (
    SegmentMarkers,
    reject_unmarked_linear_attention_rows,
    require_segment_aware_kernels,
    segment_markers_for,
)
from tests.common.packed_flash_varlen import PACKING_FAMILIES
from tests.common.segment_isolation import FAMILIES as SEGMENT_ISOLATION_FAMILIES
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    UnknownHybridConfig,
    tiny_family_model,
)

# The name list the derivation replaced, kept as the oracle every roster family's verdict is held to.
NAMED_GDN_PREFIXES = ("qwen3_5", "qwen3_next")
SHORT_CONV_PREFIXES = ("lfm2",)
ROSTER = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# The transformers configs whose verdict moves: GatedDeltaNet under a name the list never held, and the Qwen3.5
# vision towers the prefix matched, which hold no GatedDeltaNet layer.
MOVED = {"olmo_hybrid", "qwen4_exp", "qwen4_exp_text", "qwen3_5_vision", "qwen3_5_moe_vision"}
# The configs that refuse default construction, so the sweep cannot judge them.
UNCONSTRUCTIBLE = {"gemma4_assistant", "nougat"}
GDN_HEADS = {"linear_num_key_heads": 2, "linear_num_value_heads": 4}


def _named_verdict(config) -> SegmentMarkers:
    """The markers the name list gave ``config``."""
    named = model_type_matches(config, *NAMED_GDN_PREFIXES)
    return SegmentMarkers(seq_idx=named or model_type_matches(config, *SHORT_CONV_PREFIXES), cu_seq_lens=named)


@functools.cache
def _roster_config(family: str):
    return tiny_family_model(ROSTER[family]).config


def _outcome(verdict, config):
    """``verdict(config)``, or the error it raises: a composite with several text towers has no single text
    config, and both verdicts read through one."""
    try:
        return verdict(config)
    except ValueError as error:
        return type(error)


@pytest.mark.parametrize("family", sorted(ROSTER))
def test_every_roster_family_keeps_its_verdict(family):
    """The name list's verdict, or for a family it never held (``MOVED``) the GatedDeltaNet markers."""
    config = _roster_config(family)
    if set(config_model_types(config)) & MOVED:
        assert segment_markers_for(config) == SegmentMarkers(seq_idx=True, cu_seq_lens=True)
    else:
        assert segment_markers_for(config) == _named_verdict(config)


def test_the_roster_spans_both_verdicts():
    """A sweep where nothing is detected, or everything is, would pass the per-family pin vacuously."""
    detected = {family for family in ROSTER if segment_markers_for(_roster_config(family)).cu_seq_lens}
    assert detected == {"qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "qwen4_exp", "qwen4_exp_text"}


def test_every_qwen3_next_fixture_builds_qwen3_next():
    """The fixture tables that key a family ``qwen3_next`` build Qwen3-Next: an entry building another MoE family
    under that key would hold it to Qwen3-Next's verdicts."""
    fixtures = {
        "roster": ROSTER["qwen3_next"],
        "segment isolation": SEGMENT_ISOLATION_FAMILIES["qwen3_next"],
        "packing sweep": PACKING_FAMILIES["qwen3_next"],
    }
    built = {table: tiny_family_model(family).config.model_type for table, family in fixtures.items()}
    assert built == dict.fromkeys(fixtures, "qwen3_next")


def test_across_transformers_only_gated_delta_net_layers_decide():
    """Every config transformers ships at its defaults: Kimi Delta Attention (GLM-5 Next, Kimi Linear), Lightning
    attention (MiniMax) and the Mamba hybrids keep no markers; only GatedDeltaNet families gain them, and only
    the Qwen3.5 vision towers lose them."""
    moved, unconstructible, compared = set(), set(), 0
    for model_type, config_cls in transformers.CONFIG_MAPPING.items():
        try:
            config = config_cls()
        except Exception:
            unconstructible.add(model_type)
            continue
        compared += 1
        if _outcome(segment_markers_for, config) != _outcome(_named_verdict, config):
            moved.add(model_type)
    assert compared > 300, compared
    assert unconstructible == UNCONSTRUCTIBLE
    assert moved == MOVED


@pytest.mark.parametrize("config", [Qwen4ExpConfig(), Qwen4ExpTextConfig()], ids=["wrapper", "text"])
def test_qwen4_exp_packed_rows_get_the_markers(config):
    assert segment_markers_for(config) == SegmentMarkers(seq_idx=True, cu_seq_lens=True)


def test_an_unknown_model_type_with_gated_delta_net_layers_is_detected(monkeypatch):
    config = UnknownHybridConfig(layer_types=["linear_attention", "linear_attention", "full_attention"], **GDN_HEADS)

    assert segment_markers_for(config) == SegmentMarkers(seq_idx=True, cu_seq_lens=True)
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("fla>=0.2.2", lambda: False),))
    with pytest.raises(ValueError, match="halo_unknown_hybrid.*fla>=0.2.2 unavailable"):
        require_segment_aware_kernels(config, "packing")


def test_an_unknown_model_type_with_gated_delta_net_layers_refuses_packing_under_pp(monkeypatch):
    """Pipeline parallelism keeps packed rows, which carry no per-row ``cu_seq_lens``."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", (("fla>=0.2.2", lambda: True),))
    config = UnknownHybridConfig(layer_types=["linear_attention", "full_attention"], **GDN_HEADS)
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="packing under pipeline parallelism.*halo_unknown_hybrid"):
        reject_unsupported_collator_options(
            None,
            padding_free=False,
            packing=True,
            train_on_completions_only=False,
            assistant_message_template=None,
            use_context_parallel=False,
            train_on_last_assistant_only=False,
            model_config=config,
            per_device_train_batch_size=1,
            keeps_packed_rows=True,
        )


@pytest.mark.parametrize(
    ("fields", "refused"), [(GDN_HEADS, False), ({}, True)], ids=["gated_delta_net_heads", "no_head_split"]
)
def test_the_linear_attention_gate_reads_the_same_verdict(fields, refused, monkeypatch):
    """Under a name no list holds, linear-attention layers with GatedDeltaNet's head split take their markers and
    pack; the same layers without it read no marker, so their multi-document rows are refused."""
    monkeypatch.setattr(segment_markers, "GDN_SEGMENT_AWARE_BACKENDS", ())
    config = UnknownHybridConfig(layer_types=["linear_attention", "full_attention"], **fields)
    expectation = pytest.raises(ValueError, match="'halo_unknown_hybrid': its linear-attention layers")
    with expectation if refused else contextlib.nullcontext():
        reject_unmarked_linear_attention_rows(config, "packing", allow_cross_document_linear_attention=False)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"layer_types": ["full_attention"] * 2, **GDN_HEADS}, "no linear layer"),
        ({"layer_types": ["linear_attention", "full_attention"], "linear_num_heads": 4}, "Kimi Delta Attention"),
        ({"layer_types": ["linear_attention", "full_attention"]}, "no GatedDeltaNet geometry"),
        ({"layer_types": ["linear_attention", "full_attention"], "linear_num_key_heads": 2}, "no value heads"),
        ({"layer_types": ["linear_attention", "full_attention"], "linear_num_value_heads": 4}, "no key heads"),
    ],
)
def test_linear_layers_without_gated_delta_net_are_not_detected(fields, reason):
    assert segment_markers_for(UnknownHybridConfig(**fields)) == SegmentMarkers(), reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
