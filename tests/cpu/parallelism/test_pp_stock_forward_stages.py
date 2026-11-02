#!/usr/bin/env python
"""The stock-forward pipeline families (``STOCK_FORWARD_PP``): two chained stages reproduce the unsplit backbone
bitwise.

Every stage runs the family's upstream forward, so a split is exact only while the decoder loop hands nothing but
the hidden state from one layer to the next and reads list position only as ``config.layer_types[i]``, which
``validate_partition`` holds to whole periods. Each stage chain is scored against the unsplit backbone through the
last stage's own head transform, so a family's head spelling (Inkling's μP division) stays out of the split's
verdict.

    python tests/cpu/parallelism/test_pp_stock_forward_stages.py
"""

import functools

import pytest
import torch

from halo.distributed.pipeline_parallel import stage as stage_module
from halo.distributed.pipeline_parallel.families import STOCK_FORWARD_PP
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.models.module_tree import backbone_with_layers
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_PP_SPLIT_OVERRIDES, tiny_family_model

PP_SIZE = 2
BATCH = 2
# Longer than every tiny sliding window, so a sliding layer handed the full mask changes its output.
SEQ = 96
# One tiny family per STOCK_FORWARD_PP backbone, split by the default two-stage partition.
CHAINED_FAMILIES = (
    "glm4_moe_lite",
    "gpt_oss",
    "inkling_text",
    "laguna",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe_text",
    "qwen3_moe",
    "step3p7",
)
FAMILIES = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
# Two families with heterogeneous layer_types, each with a cut off its period that re-bases the loop's
# config.layer_types[i] read.
LAYER_TYPES_FAMILIES = {"laguna": [1, 7], "inkling_text": [1, 3]}


def _model(family: str):
    torch.manual_seed(0)
    return tiny_family_model(FAMILIES[family], overrides=TINY_PP_SPLIT_OVERRIDES.get(family)).eval()


@functools.cache
def _backbone_name(family: str) -> str:
    return type(backbone_with_layers(_model(family))).__name__


def _input_ids(model) -> torch.Tensor:
    vocab_size = model.config.get_text_config().vocab_size
    return torch.randint(0, vocab_size, (BATCH, SEQ), generator=torch.Generator().manual_seed(5))


def _chain_vs_unsplit(family: str, pp_split: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(logits through the stage chain, the unsplit backbone's hidden state through the last stage's head)``."""
    model = _model(family)
    input_ids = _input_ids(model)
    stages = [
        build_pipeline_stage(_model(family), rank, PP_SIZE, moe_balancing="none", pp_split=pp_split)
        for rank in range(PP_SIZE)
    ]
    with torch.no_grad():
        hidden = backbone_with_layers(model)(input_ids=input_ids, use_cache=False).last_hidden_state
        reference = stages[-1].head_transform.project(model.get_output_embeddings(), hidden)
        activation = input_ids
        for stage in stages:
            activation = stage(activation)
    return activation, reference


def test_every_stock_forward_family_has_a_chained_case():
    assert {_backbone_name(family) for family in CHAINED_FAMILIES} == set(STOCK_FORWARD_PP.hf_module_names)


@pytest.mark.parametrize("family", CHAINED_FAMILIES)
def test_chained_stages_reproduce_the_unsplit_backbone_bitwise(family):
    """The split reorders no arithmetic, so any drift means a cut severed state the loop carries across layers or
    re-based a positional read."""
    logits, reference = _chain_vs_unsplit(family)
    assert logits.shape == reference.shape
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


@pytest.mark.parametrize(("family", "off_period_split"), sorted(LAYER_TYPES_FAMILIES.items()))
def test_a_cut_off_the_layer_types_period_is_refused(family, off_period_split):
    with pytest.raises(ValueError, match=r"config\.layer_types"):
        build_pipeline_stage(_model(family), 0, PP_SIZE, moe_balancing="none", pp_split=off_period_split)


@pytest.mark.parametrize(("family", "off_period_split"), sorted(LAYER_TYPES_FAMILIES.items()))
def test_a_cut_off_the_layer_types_period_would_change_the_model(family, off_period_split, monkeypatch):
    """Anti-vacuity for the gate above and for the bitwise check: past the gate, the off-period cut hands the later
    stage's layers another layer's mask, and the chain no longer matches."""
    monkeypatch.setattr(stage_module, "validate_partition", lambda *_args: None)
    logits, reference = _chain_vs_unsplit(family, off_period_split)
    assert not torch.equal(logits, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
