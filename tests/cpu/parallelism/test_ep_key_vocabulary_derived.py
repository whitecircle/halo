#!/usr/bin/env python
"""The EP checkpoint-key vocabulary must be DERIVED from the layer classes, not restated.

Hand-written expert-key regexes in ``lazy_plan.py`` drift from the expert-container attributes the
layer classes declare: most families declare the base's single ``experts`` while GLM-4/Laguna also
serve ``routed_experts``, and an alternation naming only one of them plans every fused expert key
``REPLICATE`` for the other — each rank reading the whole ``[E, 2M, H]`` tensor onto its own GPU,
OOM-ing against CUDA rather than the loader — while the per-expert variant falls through
``ExpertKeyLayout.detect`` to ``FUSED``, so the fuser never runs and the params die on meta. Both
are silent.

Run: pytest tests/cpu/parallelism/test_ep_key_vocabulary_derived.py
"""

import dataclasses
import json
from unittest.mock import patch

import pytest

from src.distributed.expert_parallel import lazy_plan
from src.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from src.distributed.expert_parallel.lazy_plan import (
    FUSED_EXPERT_KEY_PATTERN,
    PER_EXPERT_KEY_PATTERN,
    ExpertKeyLayout,
    is_lazy_loadable,
)
from src.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS, to_hub_expert_tensors
from src.models.moe.family_specs import GLU_PROJECTIONS, MOE_FAMILY_SPECS, ZAYA, MoEFamilySpec, experts_container_attrs


def test_every_declared_container_is_matched_by_both_patterns():
    """The regexes must cover the WHOLE declared vocabulary — including ``routed_experts``, which a
    hand-written alternation omits while GLM-4/Laguna's ``_find_experts_container`` accepts it."""
    for container in experts_container_attrs():
        fused = f"model.layers.0.mlp.{container}.gate_up_proj"
        individual = f"model.layers.0.mlp.{container}.3.gate_proj.weight"
        assert FUSED_EXPERT_KEY_PATTERN.search(fused), container
        assert ExpertKeyLayout.detect({fused: "model.safetensors"}) is ExpertKeyLayout.FUSED
        m = PER_EXPERT_KEY_PATTERN.search(individual)
        assert m is not None and m.group(1) == container and int(m.group(2)) == 3
        assert ExpertKeyLayout.detect({individual: "model.safetensors"}) is ExpertKeyLayout.PER_EXPERT


def test_container_vocabulary_is_the_union_of_the_class_declarations():
    assert set(experts_container_attrs()) == {
        attr for spec in MOE_FAMILY_SPECS for attr in spec.experts_container_attrs
    }
    # Concrete membership, not derived == derived: dropping a spelling would shrink both sides.
    # The vocabulary is per-family: the base declares the one name every current HF block uses, and the
    # second spelling reaches the union only because GLM-4 (hence Laguna) declares it.
    assert {"experts", "routed_experts"} <= set(experts_container_attrs())
    assert MoEFamilySpec(model_types=()).experts_container_attrs == ("experts",)
    assert "routed_experts" in EPGlm4MoELayer.FAMILY.experts_container_attrs


def test_fused_key_vocabulary_covers_the_expert_biases():
    """The fused vocabulary must carry the expert BIASES too, else a GptOss bias tensor is planned
    REPLICATE and every rank loads all experts' biases instead of its own slice."""
    assert {"gate_up_proj_bias", "down_proj_bias"} <= set(HF_FUSED_EXPERT_KEYS)
    for key in ("gate_up_proj_bias", "down_proj_bias"):
        assert FUSED_EXPERT_KEY_PATTERN.search(f"model.layers.0.mlp.experts.{key}")


def test_gptoss_expert_roots_extend_the_base_rather_than_restate_them():
    """A GLU root or a grouped-GEMM key missing from the GptOss roots drops that weight from grad-sync
    registration — DP divergence with no error."""
    assert set(GLU_PROJECTIONS) <= set(EPGptOssMoELayer.FAMILY.expert_weight_roots)
    gmm_keys = {f"{half}_proj_gmm{suffix}" for half in ("gate", "up") for suffix in ("", "_bias")}
    assert gmm_keys <= set(EPGptOssMoELayer.FAMILY.expert_weight_roots)


def test_gptoss_layout_refuses_a_key_it_has_no_branch_for():
    """An expert param outside the layouts GptOss re-interleaves fails loud rather than being silently
    dropped from the export."""
    with pytest.raises(ValueError, match="unexpected expert params"):
        to_hub_expert_tensors(EPGptOssMoELayer.FAMILY.expert_layout, {"router_bias": None}, device="cpu")


def _write_index(tmp_path, keys: list[str]) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in keys}})
    )


def test_lazy_support_is_read_from_the_class_declaration(tmp_path):
    """A family whose spec declares ``supports_lazy_loading=False`` must be rejected from config.json
    alone, without scanning the weight map (an empty dir must still be rejected)."""
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    refusing = dataclasses.replace(ZAYA, supports_lazy_loading=False)
    with patch.object(lazy_plan, "find_moe_family_spec", lambda _model_type: refusing):
        assert is_lazy_loadable(str(tmp_path)) is False

    (tmp_path / "config.json").write_text('{"model_type": "gpt_oss"}')
    assert is_lazy_loadable(str(tmp_path)) is True


def test_zaya_supports_lazy_loading_but_the_legacy_nested_layout_is_still_refused(tmp_path):
    """Zaya's checkpoint is the fused ``experts.gate_up_proj``/``down_proj`` layout the lazy loader
    slices directly, so the class declares support — but the structural backstop must still refuse a
    layout that nests a module per expert (pre-5.14 Zaya checkpoints,
    ``experts.local_experts.N.linear_fc{1,2}``): those keys match neither expert pattern, so
    ``detect()`` calls the checkpoint FUSED and every expert plan is silently dropped."""
    assert EPZayaMoELayer.FAMILY.supports_lazy_loading is True
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    _write_index(tmp_path, ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"])
    assert is_lazy_loadable(str(tmp_path)) is True

    _write_index(tmp_path, ["model.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"])
    assert is_lazy_loadable(str(tmp_path)) is False


def test_unclaimed_model_type_still_falls_back_to_the_structural_probe(tmp_path):
    """No class claims the model_type, so the nested-per-expert-module layout must still be caught."""
    (tmp_path / "config.json").write_text('{"model_type": "some_unregistered_moe"}')
    _write_index(tmp_path, ["model.layers.0.blk.experts.local_experts.0.linear_fc1.weight"])
    assert is_lazy_loadable(str(tmp_path)) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
