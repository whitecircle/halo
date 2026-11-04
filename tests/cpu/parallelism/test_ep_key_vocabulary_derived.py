#!/usr/bin/env python
"""The EP checkpoint-key vocabulary derives from the layer classes. A hand regex omitting ``routed_experts`` plans
fused keys ``REPLICATE``, OOM-ing every rank, and leaves per-expert params dead on meta."""

import dataclasses
import json
from unittest.mock import patch

import pytest

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.model_loading.parallel import lazy_plan
from halo.model_loading.parallel.lazy_plan import (
    FUSED_EXPERT_KEY_PATTERN,
    PER_EXPERT_KEY_PATTERN,
    ExpertKeyLayout,
    is_lazy_loadable,
)
from halo.models.moe.expert_layout import GLU_PROJECTIONS, HF_FUSED_EXPERT_KEYS, to_hub_expert_tensors
from halo.models.moe.family_specs import (
    MOE_FAMILY_SPECS,
    ZAYA,
    MoEFamilySpec,
    experts_container_attrs,
)


def test_every_declared_container_is_matched_by_both_patterns():
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
    # Concrete membership, since dropping a spelling would shrink both derived sides; the second reaches the union only
    # through GLM-4.
    assert {"experts", "routed_experts"} <= set(experts_container_attrs())
    assert MoEFamilySpec(model_types=()).experts_container_attrs == ("experts",)
    assert "routed_experts" in EPGlm4MoELayer.FAMILY.experts_container_attrs


def test_fused_key_vocabulary_covers_the_expert_biases():
    """Otherwise a GptOss bias is planned REPLICATE and every rank loads all experts' biases."""
    assert {"gate_up_proj_bias", "down_proj_bias"} <= set(HF_FUSED_EXPERT_KEYS)
    for key in ("gate_up_proj_bias", "down_proj_bias"):
        assert FUSED_EXPERT_KEY_PATTERN.search(f"model.layers.0.mlp.experts.{key}")


def test_gptoss_expert_roots_extend_the_base_rather_than_restate_them():
    """A missing root drops that weight from grad-sync registration: DP divergence with no error."""
    assert set(GLU_PROJECTIONS) <= set(EPGptOssMoELayer.FAMILY.expert_weight_roots)
    gmm_keys = {f"{half}_proj_gmm{suffix}" for half in ("gate", "up") for suffix in ("", "_bias")}
    assert gmm_keys <= set(EPGptOssMoELayer.FAMILY.expert_weight_roots)


def test_gptoss_layout_refuses_a_key_it_has_no_branch_for():
    """Otherwise it is silently dropped from the export."""
    with pytest.raises(ValueError, match="unexpected expert params"):
        to_hub_expert_tensors(EPGptOssMoELayer.FAMILY.expert_layout, {"router_bias": None}, device="cpu")


def _write_index(tmp_path, keys: list[str]) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in keys}})
    )


def test_lazy_support_is_read_from_the_class_declaration(tmp_path):
    """An empty dir must still be rejected."""
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    refusing = dataclasses.replace(ZAYA, supports_lazy_loading=False)
    with patch.object(lazy_plan, "find_moe_family_spec", lambda _model_type: refusing):
        assert is_lazy_loadable(str(tmp_path)) is False

    (tmp_path / "config.json").write_text('{"model_type": "gpt_oss"}')
    assert is_lazy_loadable(str(tmp_path)) is True


def test_zaya_supports_lazy_loading_but_the_legacy_nested_layout_is_still_refused(tmp_path):
    """Nested pre-5.14 keys match neither pattern, so ``detect()`` calls them FUSED and every plan is silently
    dropped."""
    assert EPZayaMoELayer.FAMILY.supports_lazy_loading is True
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    _write_index(tmp_path, ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"])
    assert is_lazy_loadable(str(tmp_path)) is True

    _write_index(tmp_path, ["model.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"])
    assert is_lazy_loadable(str(tmp_path)) is False


def test_unclaimed_model_type_still_falls_back_to_the_structural_probe(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "some_unregistered_moe"}')
    _write_index(tmp_path, ["model.layers.0.blk.experts.local_experts.0.linear_fc1.weight"])
    assert is_lazy_loadable(str(tmp_path)) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
