#!/usr/bin/env python
"""The EP checkpoint-key vocabulary must be DERIVED from the layer classes, not restated.

Hand-written expert-key regexes in ``lazy_loader.py`` drift from the expert-container attributes the
layer classes declare: most families declare the base's single ``experts`` while GLM-4/Laguna also
serve ``routed_experts``, and an alternation naming only one of them plans every fused expert key
``REPLICATE`` for the other — each rank reading the whole ``[E, 2M, H]`` tensor onto its own GPU,
OOM-ing against CUDA rather than the loader — while the per-expert variant falls through
``CheckpointFormat.detect`` to ``FUSED``, so the fuser never runs and the params die on meta. Both
are silent.

Run: pytest tests/cpu/parallelism/test_ep_key_vocabulary_derived.py
"""

import json
import sys
from unittest.mock import patch

import pytest

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    ep_layer_classes,
    experts_container_attrs,
    hf_fused_expert_keys,
)
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.expert_parallel.lazy_loader import (
    _FUSED_EXPERT_PATTERN,
    _INDIVIDUAL_EXPERT_PATTERN,
    CheckpointFormat,
    lazy_loader_supports_checkpoint,
)


def test_every_declared_container_is_matched_by_both_patterns():
    """The regexes must cover the WHOLE declared vocabulary — including ``routed_experts``, which a
    hand-written alternation omits while GLM-4/Laguna's ``_find_experts_container`` accepts it."""
    for container in experts_container_attrs():
        fused = f"model.layers.0.mlp.{container}.gate_up_proj"
        individual = f"model.layers.0.mlp.{container}.3.gate_proj.weight"
        assert _FUSED_EXPERT_PATTERN.search(fused), container
        assert CheckpointFormat.detect({fused: "model.safetensors"}) is CheckpointFormat.FUSED
        m = _INDIVIDUAL_EXPERT_PATTERN.search(individual)
        assert m is not None and m.group(1) == container and int(m.group(2)) == 3
        assert CheckpointFormat.detect({individual: "model.safetensors"}) is CheckpointFormat.INDIVIDUAL


def test_container_vocabulary_is_the_union_of_the_class_declarations():
    assert set(experts_container_attrs()) == {
        attr for cls in ep_layer_classes() for attr in cls._EXPERTS_CONTAINER_ATTRS
    }
    # Concrete membership, not derived == derived: dropping a spelling would shrink both sides.
    # The vocabulary is per-family: the base declares the one name every current HF block uses, and the
    # second spelling reaches the union only because GLM-4 (hence Laguna) declares it.
    assert {"experts", "routed_experts"} <= set(experts_container_attrs())
    assert EPMoELayerBase._EXPERTS_CONTAINER_ATTRS == ("experts",)
    assert "routed_experts" in EPGlm4MoELayer._EXPERTS_CONTAINER_ATTRS


def test_fused_key_vocabulary_covers_the_expert_biases():
    """The fused vocabulary must carry the expert BIASES too, else a GptOss bias tensor is planned
    REPLICATE and every rank loads all experts' biases instead of its own slice."""
    assert set(hf_fused_expert_keys()) == {key for cls in ep_layer_classes() for key in cls._HF_FUSED_EXPERT_KEYS}
    assert {"gate_up_proj_bias", "down_proj_bias"} <= set(hf_fused_expert_keys())
    for key in ("gate_up_proj_bias", "down_proj_bias"):
        assert _FUSED_EXPERT_PATTERN.search(f"model.layers.0.mlp.experts.{key}")


def test_gptoss_expert_roots_extend_the_base_rather_than_restate_them():
    """A base root missed by a restating subclass drops that weight from grad-sync registration — DP
    divergence with no error."""
    assert set(EPMoELayerBase._EXPERT_WEIGHT_ATTR_ROOTS) <= set(EPGptOssMoELayer._EXPERT_WEIGHT_ATTR_ROOTS)
    # The grouped-GEMM layout is declared once and reused by the roots, so they cannot disagree.
    assert set(EPGptOssMoELayer._GMM_EXPERT_KEYS) <= set(EPGptOssMoELayer._EXPERT_WEIGHT_ATTR_ROOTS)


def test_gptoss_merge_accepts_a_fused_key_added_to_the_base(monkeypatch):
    """The GptOss merge's accepted-key set must DERIVE from ``_HF_FUSED_EXPERT_KEYS``.

    Restated, a fused key added to the base reaches the base merge but not this one, and GptOss
    shards carrying it die on a bogus "unexpected expert params" — the family's own checkpoint
    rejected by its own merge. Adding the key to the base must be enough.
    """
    extended = (*EPMoELayerBase._HF_FUSED_EXPERT_KEYS, "sink_proj")
    monkeypatch.setattr(EPMoELayerBase, "_HF_FUSED_EXPERT_KEYS", extended)
    assert extended == EPGptOssMoELayer._HF_FUSED_EXPERT_KEYS  # inherited, not shadowed
    # Accepted past the vocabulary check (a distinct error), then refused rather than dropped.
    with pytest.raises(ValueError, match="no merge branch"):
        EPGptOssMoELayer.merge_shards_to_hf("model.layers.0.mlp", {"sink_proj": None})


def test_gptoss_merge_still_refuses_a_key_no_layout_declares():
    """Anti-vacuity: the derived set is not "accept everything" — an undeclared param still fails
    loud rather than being silently dropped from the merged checkpoint."""
    with pytest.raises(ValueError, match="unexpected expert params"):
        EPGptOssMoELayer.merge_shards_to_hf("model.layers.0.mlp", {"router_bias": None})


def _write_index(tmp_path, keys: list[str]) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in keys}})
    )


def test_lazy_support_is_read_from_the_class_declaration(tmp_path):
    """A family that declares ``_supports_lazy_loading = False`` must be rejected from config.json
    alone, without scanning the weight map (an empty dir must still be rejected)."""
    assert ep_layer_class_by_model_type()["zaya"] is EPZayaMoELayer
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    with patch.object(EPZayaMoELayer, "_supports_lazy_loading", False):
        assert lazy_loader_supports_checkpoint(str(tmp_path)) is False

    (tmp_path / "config.json").write_text('{"model_type": "gpt_oss"}')
    assert lazy_loader_supports_checkpoint(str(tmp_path)) is True


def test_zaya_supports_lazy_loading_but_the_legacy_nested_layout_is_still_refused(tmp_path):
    """Zaya's checkpoint is the fused ``experts.gate_up_proj``/``down_proj`` layout the lazy loader
    slices directly, so the class declares support — but the structural backstop must still refuse a
    layout that nests a module per expert (pre-5.14 Zaya checkpoints,
    ``experts.local_experts.N.linear_fc{1,2}``): those keys match neither expert pattern, so
    ``detect()`` calls the checkpoint FUSED and every expert plan is silently dropped."""
    assert EPZayaMoELayer._supports_lazy_loading is True
    (tmp_path / "config.json").write_text('{"model_type": "zaya"}')

    _write_index(tmp_path, ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"])
    assert lazy_loader_supports_checkpoint(str(tmp_path)) is True

    _write_index(tmp_path, ["model.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"])
    assert lazy_loader_supports_checkpoint(str(tmp_path)) is False


def test_unclaimed_model_type_still_falls_back_to_the_structural_probe(tmp_path):
    """No class claims the model_type, so the nested-per-expert-module layout must still be caught."""
    (tmp_path / "config.json").write_text('{"model_type": "some_unregistered_moe"}')
    _write_index(tmp_path, ["model.layers.0.blk.experts.local_experts.0.linear_fc1.weight"])
    assert lazy_loader_supports_checkpoint(str(tmp_path)) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
