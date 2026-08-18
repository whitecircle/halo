#!/usr/bin/env python
"""Tests for native EP expert-LoRA config plumbing: ExpertLoraSpec, the projection-coverage
maps, and split_expert_lora_targets (which peels expert-FFN entries out of lora_target_modules).

These are the pure-logic pieces that gate whether grouped-LoRA reaches the EP layers at all; the
forward/grad-sync/save behaviour is covered by the GPU test (tests/gpu/trainers/lora/).

Run: python tests/cpu/peft/test_expert_lora_targets.py
"""

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from src.distributed.expert_parallel.config import (
    LORA_PROJECTION_COVERAGE,
    ExpertLoraSpec,
    expert_target_projections,
)
from src.distributed.loading import peft_setup as model_info
from src.distributed.loading.peft_setup import split_expert_lora_targets


@dataclass
class _MockModelConfig:
    """Minimal stand-in for trl ModelConfig (only the fields split_expert_lora_targets reads)."""

    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: object = None
    model_name_or_path: str = "dummy/moe"


@contextmanager
def _as_moe(is_moe: bool = True):
    """split_expert_lora_targets loads the model's AutoConfig to gate the peel on MoE — stub both
    the config load and the MoE check so these pure-logic tests need no real model."""
    with (
        patch.object(model_info.AutoConfig, "from_pretrained", return_value=object()),
        patch.object(model_info, "config_has_experts", return_value=is_moe),
    ):
        yield


# ExpertLoraSpec


def test_spec_scaling_is_alpha_over_r():
    assert ExpertLoraSpec(r=8, alpha=16).scaling == 2.0
    assert ExpertLoraSpec(r=32, alpha=16).scaling == 0.5


def test_spec_default_projections_are_all_three():
    assert ExpertLoraSpec(r=8, alpha=16).projections == frozenset({"gate", "up", "down"})


def test_spec_adapts_fused_and_separate_and_gmm_attrs():
    spec = ExpertLoraSpec(r=8, alpha=16)  # all projections
    # fused gate_up covers gate+up; separate and de-interleaved variants each covered
    for attr in ("gate_up_proj", "gate_proj", "up_proj", "down_proj", "gate_proj_gmm", "up_proj_gmm"):
        assert spec.adapts(attr), attr


def test_spec_never_adapts_bias_or_router():
    spec = ExpertLoraSpec(r=8, alpha=16)
    for attr in ("gate_up_proj_bias", "down_proj_bias", "router", "gate", "weight"):
        assert not spec.adapts(attr), attr


def test_spec_subset_projection_only_adapts_matching_attrs():
    down_only = ExpertLoraSpec(r=8, alpha=16, projections=frozenset({"down"}))
    assert down_only.adapts("down_proj")
    assert not down_only.adapts("gate_proj")
    # fused gate_up carries gate+up, so a 'down'-only spec must NOT touch it
    assert not down_only.adapts("gate_up_proj")

    gate_only = ExpertLoraSpec(r=8, alpha=16, projections=frozenset({"gate"}))
    # fused gate_up carries the gate half → adapting gate means adapting the fused tensor
    assert gate_only.adapts("gate_up_proj")
    assert gate_only.adapts("gate_proj")
    assert not gate_only.adapts("up_proj")
    assert not gate_only.adapts("down_proj")


def test_spec_is_frozen():
    spec = ExpertLoraSpec(r=8, alpha=16)
    raised = False
    try:
        spec.r = 99  # frozen dataclass → should raise
    except Exception:
        raised = True
    assert raised, "ExpertLoraSpec should be immutable (frozen)"


# Coverage-map consistency (guards drift between the two maps)


def test_every_stored_weight_root_is_a_usable_target_name():
    """The user-facing target vocabulary is DERIVED from the stored-attr coverage, not restated.

    A family adding an expert-weight root gets a coverage entry (enforced by
    test_expert_lora_projection_coverage) and must get the matching ``lora_target_modules`` name for
    free — a second hand-maintained map would silently leave the new root unnameable.
    """
    for root, coverage in LORA_PROJECTION_COVERAGE.items():
        assert expert_target_projections(root) == coverage, f"{root} is not resolvable as a target name"


def test_container_targets_request_every_stored_projection():
    """``experts`` means the whole FFN, however many projections the roster stores."""
    logical_from_storage = set().union(*LORA_PROJECTION_COVERAGE.values())
    for alias in ("experts", "mlp.experts"):
        assert expert_target_projections(alias) == logical_from_storage == {"gate", "up", "down"}


def test_non_expert_names_resolve_to_nothing():
    for name in ("q_proj", "all-linear", ".*gate_proj", "lm_head"):
        assert expert_target_projections(name) is None


# split_expert_lora_targets


def test_split_mixed_targets_peels_experts_keeps_attention():
    cfg = _MockModelConfig(lora_target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    with _as_moe():
        spec = split_expert_lora_targets(cfg)
    assert spec is not None
    assert spec.r == 16 and spec.alpha == 32 and abs(spec.dropout - 0.05) < 1e-9
    assert spec.projections == frozenset({"gate", "up", "down"})
    # attention targets remain so PEFT still wraps them
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]


def test_split_experts_alias_means_all_projections_and_strips_to_empty():
    cfg = _MockModelConfig(lora_target_modules=["experts"])
    with _as_moe():
        spec = split_expert_lora_targets(cfg)
    assert spec.projections == frozenset({"gate", "up", "down"})
    assert cfg.lora_target_modules == []  # expert-only → no attention targets left


def test_split_fused_gate_up_target_covers_gate_and_up():
    cfg = _MockModelConfig(lora_target_modules=["gate_up_proj"])
    with _as_moe():
        spec = split_expert_lora_targets(cfg)
    assert spec.projections == frozenset({"gate", "up"})


def test_split_dense_model_returns_none_and_keeps_expert_targets():
    # On a DENSE model gate/up/down_proj are ordinary nn.Linear MLPs stock PEFT wraps — peeling them
    # would silently drop the MLP adapters, so the gate must return None and leave targets intact.
    cfg = _MockModelConfig(lora_target_modules=["q_proj", "gate_proj", "up_proj", "down_proj"])
    with _as_moe(is_moe=False):
        assert split_expert_lora_targets(cfg) is None
    assert cfg.lora_target_modules == ["q_proj", "gate_proj", "up_proj", "down_proj"]


def test_split_attention_only_returns_none_and_leaves_targets():
    cfg = _MockModelConfig(lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    with _as_moe():
        assert split_expert_lora_targets(cfg) is None
    assert cfg.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_split_noop_when_peft_disabled():
    cfg = _MockModelConfig(use_peft=False, lora_target_modules=["gate_proj"])
    # Returns before any config load, so no MoE stub needed.
    assert split_expert_lora_targets(cfg) is None
    assert cfg.lora_target_modules == ["gate_proj"]  # untouched


def test_split_passthrough_for_regex_string_targets():
    cfg = _MockModelConfig(lora_target_modules=".*proj")
    assert split_expert_lora_targets(cfg) is None
    assert cfg.lora_target_modules == ".*proj"  # a regex can't be safely split → left as-is


def test_split_inherits_r_alpha_dropout_from_model_config():
    cfg = _MockModelConfig(lora_r=4, lora_alpha=64, lora_dropout=0.0, lora_target_modules=["down_proj"])
    with _as_moe():
        spec = split_expert_lora_targets(cfg)
    assert spec.r == 4 and spec.alpha == 64 and spec.dropout == 0.0
    assert spec.scaling == 16.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
