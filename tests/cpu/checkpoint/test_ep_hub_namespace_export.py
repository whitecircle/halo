#!/usr/bin/env python
"""A hub-namespace-exporting family's gathered save must equal transformers' own hub layout, chunk by chunk.

``_EXPORTS_HUB_NAMESPACE`` makes the gathered EP save run transformers' save-side conversion revert
(the one ``save_pretrained`` runs) on every streamed chunk — the non-expert params as one chunk, each
gathered EP layer as its own — instead of writing the module-tree spelling a serving engine never
reads. Two properties carry that. The union of the per-chunk reverts must equal the whole-state-dict
revert: no reverse entry may fuse tensors across a chunk boundary, or a chunk would revert to a
partial fusion. And that result must be the layout a plain ``save_pretrained`` writes for the same
config — the hub layout is transformers' definition, not this file's. Both are pinned per declaring
family on a tiny CPU model; the roster is derived from the class hierarchy, so a family cannot start
declaring the flag without its own chunk-safety fixture landing here.

Run: ``python tests/cpu/checkpoint/test_ep_hub_namespace_export.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.distributed.expert_parallel.saving import _hub_namespace_export
from tests.common.models import TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
from tests.common.utils import safetensors_state_dict

_STEP3P7_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]


def _tiny_step3p7() -> tuple[torch.nn.Module, list[str]]:
    """The composite model plus the live-tree prefixes of the blocks the EP wrapper would replace.

    Unpatched on purpose: the gathered save emits ``F.linear``-convention expert tensors under the
    module-tree names, which is exactly the stock block's own ``state_dict`` spelling and layout.
    """
    torch.manual_seed(0)
    config = Step3p7Config(
        text_config=dict(TINY_STEP3P7_CONFIG),
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=2000,
    )
    model = Step3p7ForConditionalGeneration(config).to(torch.bfloat16)
    return model, [f"model.language_model.layers.{i}.mlp" for i in _STEP3P7_SPARSE_LAYERS]


def _step3p7_hub_keys(state: dict) -> None:
    """The hub spellings the serving engines read (pinned against the real ``Step-3.7-Flash`` index)
    must be present for every sparse layer, and no module-tree spelling may survive."""
    for i in _STEP3P7_SPARSE_LAYERS:
        for name in ("gate.weight", "router_bias", "gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            assert f"model.layers.{i}.moe.{name}" in state, f"layer {i}: hub key moe.{name} missing"
        for proj in ("gate", "up", "down"):
            assert f"model.layers.{i}.share_expert.{proj}_proj.weight" in state
    module_spellings = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector")
    leaked = sorted(key for key in state if any(spelling in key for spelling in module_spellings))
    assert not leaked, f"module-tree spellings survived the hub export: {leaked[:5]}"


# wrapper class -> (tiny-model factory, hub-key assertion)
FIXTURES = {
    EPStep3p7MoELayer: (_tiny_step3p7, _step3p7_hub_keys),
}
_IDS = [cls.__name__ for cls in FIXTURES]


def _bare(cls: type[EPMoELayerBase]) -> EPMoELayerBase:
    """A wrapper instance carrying only its class (the export seam reads nothing else)."""
    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    return layer


def _chunks(state_dict: dict, ep_prefixes: list[str]) -> list[dict]:
    """The save's streaming units: the non-expert params, then one chunk per EP layer."""
    non_expert = {k: v for k, v in state_dict.items() if not any(k.startswith(f"{p}.") for p in ep_prefixes)}
    per_layer = [{k: v for k, v in state_dict.items() if k.startswith(f"{p}.")} for p in ep_prefixes]
    assert all(per_layer), "every EP layer prefix must select tensors, or a chunk is vacuous"
    return [non_expert, *per_layer]


def _chunked_export(cls, model, ep_prefixes: list[str]) -> dict:
    export = _hub_namespace_export(model, [(prefix, _bare(cls)) for prefix in ep_prefixes])
    merged: dict = {}
    for chunk in _chunks(model.state_dict(), ep_prefixes):
        exported = export(chunk)
        collisions = set(exported) & set(merged)
        assert not collisions, f"two chunks exported the same key: {sorted(collisions)[:5]}"
        merged.update(exported)
    return merged


def test_every_declaring_family_has_a_chunk_safety_fixture():
    declaring = {cls for cls in ep_layer_classes() if cls._EXPORTS_HUB_NAMESPACE}
    assert declaring == set(FIXTURES), (
        f"{sorted(c.__name__ for c in declaring ^ set(FIXTURES))}: a family declaring "
        f"_EXPORTS_HUB_NAMESPACE must pin its per-chunk revert here"
    )


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_per_chunk_revert_equals_the_whole_revert(cls):
    factory, _hub_keys = FIXTURES[cls]
    model, ep_prefixes = factory()
    whole = revert_weight_conversion(model, dict(model.state_dict()))
    chunked = _chunked_export(cls, model, ep_prefixes)

    assert set(chunked) == set(whole), (
        f"per-chunk revert diverges from the whole-state-dict revert: "
        f"only chunked={sorted(set(chunked) - set(whole))[:5]}, only whole={sorted(set(whole) - set(chunked))[:5]}"
    )
    for key, tensor in whole.items():
        assert torch.equal(chunked[key], tensor), f"{key}: per-chunk revert changed the tensor"


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_export_is_the_layout_save_pretrained_writes(cls, tmp_path):
    factory, hub_keys = FIXTURES[cls]
    model, ep_prefixes = factory()
    model.save_pretrained(tmp_path)
    on_disk = safetensors_state_dict(tmp_path)
    exported = _chunked_export(cls, model, ep_prefixes)

    assert set(exported) == set(on_disk), (
        f"gathered export keys differ from save_pretrained's: "
        f"only export={sorted(set(exported) - set(on_disk))[:5]}, only disk={sorted(set(on_disk) - set(exported))[:5]}"
    )
    for key, tensor in on_disk.items():
        assert torch.equal(exported[key], tensor), f"{key}: export differs from save_pretrained"
    hub_keys(exported)


def test_non_declaring_families_export_the_module_tree_unchanged():
    """The seam is family-gated: everyone else keeps the spelling their artifacts and tools read."""
    model, ep_prefixes = _tiny_step3p7()
    non_declaring = sorted(
        (cls for cls in ep_layer_classes() if not cls._EXPORTS_HUB_NAMESPACE and vars(cls).get("HF_MODULE_NAMES")),
        key=lambda cls: cls.__name__,
    )
    assert non_declaring, "premise: at least one registered family keeps the module-tree spelling"
    export = _hub_namespace_export(model, [(prefix, _bare(non_declaring[0])) for prefix in ep_prefixes])
    state = model.state_dict()
    assert export(state) is state


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
