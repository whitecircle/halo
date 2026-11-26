#!/usr/bin/env python
"""What a pipeline stage hands its backbone beside the activation, across every registered family.
Everywhere but a marker-reading family's packed microbatch, it hands what :func:`_previous_kwargs` gives."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.runner import PipelineRunner
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from halo.models.attention.selection import effective_attn_implementation, is_flash_attn_implementation
from halo.models.introspection.module_tree import backbone_with_layers
from halo.models.segment_markers import packed_row_varlen_kwargs, segment_marker_kwargs, segment_markers_for
from tests.common.packed_flash_varlen import SWEEP_ATTN_IMPLEMENTATION, set_attn_implementation
from tests.common.pp_fakes import fake_runner
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model

# A composite's text tower for the shipped wrappers.
ROSTER = {
    "Qwen3Model": TINY_DENSE_FAMILIES["qwen3"],
    "GptOssModel": TINY_MOE_FAMILIES["gpt_oss"],
    "Qwen3MoeModel": TINY_MOE_FAMILIES["qwen3_moe"],
    "Glm4MoeLiteModel": TINY_MOE_FAMILIES["glm4_moe_lite"],
    "LagunaModel": TINY_MOE_FAMILIES["laguna"],
    "InklingTextModel": TINY_MOE_FAMILIES["inkling_text"],
    "Qwen3_5TextModel": TINY_DENSE_FAMILIES["qwen3_5"],
    "Qwen3_5MoeTextModel": TINY_MOE_FAMILIES["qwen3_5_moe_text"],
    "Step3p7TextModel": TINY_MOE_FAMILIES["step3p7"],
    "DeepseekV4Model": TINY_MOE_FAMILIES["deepseek_v4"],
    "Glm5NextTextModel": TINY_MOE_FAMILIES["glm5_next"],
    "Qwen4ExpTextModel": TINY_MOE_FAMILIES["qwen4_exp_text"],
    "NemotronHModel": TINY_MOE_FAMILIES["nemotron_h"],
}
# The kwarg later stages take the token ids under.
TOKEN_ID_READERS = {"Qwen4ExpTextModel": "ple_input_ids"}
LENGTH = 32
BACKENDS = (SWEEP_ATTN_IMPLEMENTATION, "flash_attention_2", "sdpa", "eager")


def _config(backbone: str, attn_implementation: str):
    config = tiny_family_model(ROSTER[backbone]).config
    set_attn_implementation(config, attn_implementation)
    return config


def _microbatches() -> dict[str, dict]:
    """``(hidden rows, attention_mask, position_ids)`` per shape."""
    packed = torch.cat([torch.arange(9), torch.arange(14), torch.arange(LENGTH - 23) % 256]).unsqueeze(0)
    one_document = torch.arange(LENGTH).unsqueeze(0)
    return {
        "packed": (torch.zeros(1, LENGTH), None, packed),
        "padded": (torch.zeros(1, LENGTH), torch.ones(1, LENGTH, dtype=torch.long), one_document),
        "one_document": (torch.zeros(1, LENGTH), None, one_document),
        "two_rows": (torch.zeros(2, LENGTH), None, packed.expand(2, -1)),
        "no_positions": (torch.zeros(1, LENGTH), None, None),
    }


def _previous_kwargs(config, hidden, attention_mask, position_ids) -> dict:
    """The flash varlen set alone, on a one-row unmasked microbatch, for a family whose mixers read no
    ``cu_seq_lens``."""
    if (
        attention_mask is not None
        or position_ids is None
        or hidden.shape[0] != 1
        or not is_flash_attn_implementation(effective_attn_implementation(config))
        or segment_markers_for(config).cu_seq_lens
    ):
        return {}
    return packed_row_varlen_kwargs(position_ids)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        torch.equal(a[key], b[key]) if torch.is_tensor(a[key]) else a[key] == b[key] for key in a
    )


def test_the_roster_is_every_registered_backbone():
    assert set(ROSTER) == {name for name, spec in PP_SPEC_MAP.items() if spec.validated}
    for name, family in ROSTER.items():
        assert type(backbone_with_layers(tiny_family_model(family))).__name__ == name


def test_only_the_declared_families_take_token_ids_on_a_later_stage():
    declared = {name: spec.token_ids_kwarg for name, spec in PP_SPEC_MAP.items() if spec.token_ids_kwarg is not None}
    assert declared == TOKEN_ID_READERS


@pytest.mark.parametrize("token_ids_kwarg", [None, "ple_input_ids"])
def test_the_runner_hands_token_ids_exactly_where_the_stage_reads_them(token_ids_kwarg):
    runner = fake_runner()
    runner.token_ids_kwarg = token_ids_kwarg
    input_ids = torch.arange(8).view(2, 4)
    mask, positions = torch.ones_like(input_ids), torch.arange(4).expand(2, -1)
    args, kwargs = PipelineRunner._schedule_inputs(runner, input_ids, mask, positions)
    expected = {"attention_mask", "position_ids"} | ({"token_ids"} if token_ids_kwarg else set())
    assert args == (input_ids,) and set(kwargs) == expected
    if token_ids_kwarg:
        assert kwargs["token_ids"] is input_ids


@pytest.mark.parametrize("attn_implementation", BACKENDS)
@pytest.mark.parametrize("backbone", sorted(ROSTER))
def test_segment_kwargs_change_only_on_a_marker_familys_packed_row(backbone, attn_implementation):
    config = _config(backbone, attn_implementation)
    stage = SimpleNamespace(config=config)
    markers = segment_markers_for(config)
    reads_markers = markers.seq_idx or markers.cu_seq_lens
    for shape, (hidden, attention_mask, position_ids) in _microbatches().items():
        handed = PipelineStageModule._segment_kwargs(stage, hidden, attention_mask, position_ids)
        if reads_markers and shape == "packed":
            if is_flash_attn_implementation(effective_attn_implementation(config)):
                markers = replace(markers, cu_seq_lens=True)
            assert _same(handed, segment_marker_kwargs(position_ids, markers)), (backbone, shape)
            assert "seq_idx" in handed, (backbone, shape)
        else:
            assert _same(handed, _previous_kwargs(config, hidden, attention_mask, position_ids)), (backbone, shape)


def test_the_sweep_reaches_both_sides_of_the_change():
    """Anti-vacuity: the roster holds both kinds of family."""
    configs = {name: _config(name, SWEEP_ATTN_IMPLEMENTATION) for name in ROSTER}
    readers = {name for name, config in configs.items() if segment_markers_for(config).seq_idx}
    assert {"Qwen3_5TextModel", "Qwen3_5MoeTextModel", "Qwen4ExpTextModel"} <= readers
    hidden, mask, positions = _microbatches()["packed"]
    assert _previous_kwargs(configs["Qwen3Model"], hidden, mask, positions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
