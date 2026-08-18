#!/usr/bin/env python
"""The sharded-EP merge transform is owned by the EP layer class, not by a shared registry.

A module-level ``{layer class: transform}`` table would be a SECOND place every family has to be
registered: a family could declare its expert layout (``_PER_EXPERT_UNFUSED_KEYS``, or its own
gather override) and still be rejected by ``save_sharded_ep=True`` despite a perfectly mergeable
layout.

The property pinned here: a family that declares its layout in its OWN layer module is merge-supported
immediately — no edit to ``expert_weights.py``, ``saving.py`` or ``merge_ep_shards.py``. The
synthetic family below is registered at test time and must be supported end to end.

Run: pytest tests/cpu/checkpoint/test_ep_merge_is_class_owned.py
"""

import gc
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel import expert_weights as ew
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    resolve_ep_merge_layer_class,
    supported_ep_merge_model_types,
)
from src.distributed.expert_parallel.saving import _check_ep_merge_family_supported

E, H, M = 3, 8, 5

# Every "union what the families declare" cache the synthetic subclass perturbs.
_DERIVED_CACHES = (
    ew.ep_layer_class_by_model_type,
    ew.experts_container_attrs,
    ew.hf_fused_expert_keys,
    ew.per_expert_layouts,
    ew.per_expert_hub_model_types,
    ew.per_expert_fusion_map,
    ew.expert_weight_roots,
)


def _clear_derived_caches():
    for cache in _DERIVED_CACHES:
        cache.cache_clear()


@pytest.fixture
def new_family():
    """An EP family defined entirely in "its own module": one class, no shared-file edits.

    Torn down by dropping the class — ``__subclasses__`` holds weak references, so a collection
    removes it from the registry — and clearing every derived cache on both sides.
    """

    class _NewFamilyMoELayer(EPMoELayerBase):
        HF_MODULE_NAMES = ("SomeNewSparseMoeBlock",)
        HF_MODEL_TYPES = ("some_new_moe",)
        _PER_EXPERT_UNFUSED_KEYS = ("w_gate", "w_up", "w_down")

        def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
            return hidden_states

    _clear_derived_caches()
    try:
        yield _NewFamilyMoELayer
    finally:
        del _NewFamilyMoELayer
        gc.collect()
        _clear_derived_caches()


@pytest.fixture
def individual_glu_family():
    """A family whose own gather emits per-expert, declared through the second attribute.

    Qwen3 MoE and Bailing spell theirs exactly as GLM-4 does, so only a family with DISTINCT names
    can tell whether the derived caches read one attribute or both.
    """

    class _IndividualGluMoELayer(EPMoELayerBase):
        HF_MODULE_NAMES = ("SomeIndividualGluSparseMoeBlock",)
        HF_MODEL_TYPES = ("some_individual_glu_moe",)
        _HUB_PER_EXPERT_KEYS = ("p_gate", "p_up", "p_down")

        def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
            return hidden_states

    _clear_derived_caches()
    try:
        yield _IndividualGluMoELayer
    finally:
        del _IndividualGluMoELayer
        gc.collect()
        _clear_derived_caches()


def test_an_individual_glu_familys_names_reach_the_lazy_loader_vocabulary(individual_glu_family):
    """``per_expert_layouts`` / ``per_expert_fusion_map`` are the lazy loader's only vocabulary of
    per-expert projection names, so both must read the same declaration the convertible set does.

    Reading only the base-split attribute would advertise this family as convertible while the loader
    skipped every key the conversion wrote — a checkpoint the tool says it can produce and the loader
    then leaves on meta.
    """
    assert "some_individual_glu_moe" in ew.per_expert_hub_model_types()
    assert ("p_gate", "p_up", "p_down") in ew.per_expert_layouts()

    fusion = ew.per_expert_fusion_map()
    assert fusion["p_gate.weight"] == ("gate_up", 0)
    assert fusion["p_up.weight"] == ("gate_up", 1)
    assert fusion["p_down.weight"] == ("down", 0)


def test_declaring_both_per_expert_attributes_is_rejected_at_class_creation():
    """They name the same thing for two different gather paths, so carrying both lets them drift and
    leaves ``hub_per_expert_keys`` answering from whichever wins."""
    with pytest.raises(TypeError, match="_HUB_PER_EXPERT_KEYS"):

        class _BothAttributes(EPMoELayerBase):
            HF_MODULE_NAMES = ("SomeBothSparseMoeBlock",)
            HF_MODEL_TYPES = ("some_both_moe",)
            _PER_EXPERT_UNFUSED_KEYS = ("a", "b", "c")
            _HUB_PER_EXPERT_KEYS = ("d", "e", "f")

            def forward(self, hidden_states, **kwargs):  # pragma: no cover - never called
                return hidden_states


def _fake_model(model_type):
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type, auto_map=None)
    return model


def test_new_family_is_merge_supported_with_no_shared_file_edit(new_family):
    """Declaring the layout on the class is the whole registration: the resolver finds it, the
    construction-time ``save_sharded_ep`` gate accepts it, and it lists as supported."""
    assert resolve_ep_merge_layer_class("some_new_moe") is new_family
    assert "some_new_moe" in supported_ep_merge_model_types()
    _check_ep_merge_family_supported(_fake_model("some_new_moe"))  # must not raise


def test_new_family_merge_output_equals_its_own_gathered_save(new_family):
    """The invariant the class-owned transform buys: merged-from-sharded == gathered, key and tensor,
    for a family nobody wired up by hand."""
    gate_up = torch.randn(E, H, 2 * M)  # runtime matmul convention, as the shards store it
    down = torch.randn(E, M, H)

    merged = new_family.merge_shards_to_hf("model.layers.0.mlp", {"gate_up_proj": gate_up, "down_proj": down})

    fused_flinear = {
        "experts.gate_up_proj": gate_up.transpose(1, 2).contiguous(),
        "experts.down_proj": down.transpose(1, 2).contiguous(),
    }
    with patch.object(EPMoELayerBase, "_gather_fused_expert_state_dict", return_value=dict(fused_flinear)):
        gathered = new_family.gather_expert_state_dict(object.__new__(new_family), device="cpu")

    assert set(merged) == {f"model.layers.0.mlp.{key}" for key in gathered}
    assert set(gathered) == {f"experts.{i}.{proj}.weight" for i in range(E) for proj in ("w_gate", "w_up", "w_down")}
    for key, tensor in gathered.items():
        assert torch.equal(merged[f"model.layers.0.mlp.{key}"], tensor), key


def test_gather_override_without_merge_override_is_rejected_at_class_creation():
    """The two are inverses, so overriding one alone would make the sharded merge emit a layout that
    silently differs from the same family's gathered save. ``__init_subclass__`` must refuse the class
    outright — a runtime check would fire only after training wrote unmergeable shards."""
    with pytest.raises(TypeError, match="overrides exactly one of"):

        class _GatherOnly(EPMoELayerBase):
            def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False) -> dict:
                return {}

    with pytest.raises(TypeError, match="overrides exactly one of"):

        class _MergeOnly(EPMoELayerBase):
            @classmethod
            def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
                return {}

    gc.collect()
    _clear_derived_caches()


def test_declaring_per_expert_keys_and_overriding_the_gather_is_rejected():
    """The per-expert split is applied BY the base gather, so an override silently ignores the
    declaration and the checkpoint/vLLM layout drifts from the declared one. Zaya is the legitimate
    shape: override both and set ``_PER_EXPERT_UNFUSED_KEYS = None`` explicitly."""
    with pytest.raises(TypeError, match="_PER_EXPERT_UNFUSED_KEYS"):

        class _Contradictory(EPMoELayerBase):
            _PER_EXPERT_UNFUSED_KEYS = ("w_gate", "w_up", "w_down")

            def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False) -> dict:
                return {}

            @classmethod
            def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
                return {}

    # Anti-vacuity: the same overrides WITH the opt-out are accepted.
    class _Explicit(EPMoELayerBase):
        _PER_EXPERT_UNFUSED_KEYS = None

        def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False) -> dict:
            return {}

        @classmethod
        def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:
            return {}

    del _Explicit
    gc.collect()
    _clear_derived_caches()


def test_new_family_leaves_the_registry_untouched_after_teardown():
    """Guard the fixture itself: a leaked synthetic family would silently widen every derived union
    for the rest of the session. Collect first — the runner can still hold the previous test's
    fixture value when this one starts."""
    gc.collect()
    _clear_derived_caches()
    assert resolve_ep_merge_layer_class("some_new_moe") is None
    assert "some_new_moe" not in supported_ep_merge_model_types()


def test_gptoss_merge_refuses_unexpected_expert_params():
    """GptOss's own merge mirrors the base guard: an expert param outside the two layouts it
    re-interleaves (an ETP-only shard name, a future root) raises instead of being silently
    dropped into a checkpoint that loads and is wrong."""
    from src.distributed.expert_parallel.layers.gpt_oss import (
        EPGptOssMoELayer,
    )

    with pytest.raises(ValueError, match="unexpected expert params"):
        EPGptOssMoELayer.merge_shards_to_hf("model.layers.0.mlp", {"gate_proj": torch.zeros(E, H, M)})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
