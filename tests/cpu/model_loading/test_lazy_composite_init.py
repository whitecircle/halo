#!/usr/bin/env python
"""A lazy load draws a checkpoint-absent tensor with the ``_init_weights`` of the model that owns it, as
``from_pretrained`` dispatches them, and a composite's coverage gate reads what its towers declare."""

import pytest
import torch
from transformers import PreTrainedModel

from halo.model_loading.checkpoint_coverage import ALLOW_MISSING_CHECKPOINT_KEYS_ENV, unexpected_missing_keys
from halo.model_loading.lazy_safetensors.materialization import init_checkpoint_absent_modules
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    tiny_family_model,
)

_ROSTER = {
    **TINY_DENSE_FAMILIES,
    **TINY_MOE_FAMILIES,
    **{f"vlm/{name}": family for name, family in TINY_MOE_VLM_FAMILIES.items()},
}
_COMPOSITES = {
    **{name: family for name, family in _ROSTER.items() if name.startswith("vlm/")},
    **TINY_UNWRAPPED_COMPOSITE_FAMILIES,
}
_LAYERSCALE = "lambda1"


def _nemotron_on_meta():
    with torch.device("meta"):
        return tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])


def _nested_models(model) -> list[tuple[str, PreTrainedModel]]:
    return [(name, module) for name, module in model.named_modules() if name and isinstance(module, PreTrainedModel)]


def _record_init_weights(model) -> dict[str, list]:
    """Every model's ``_init_weights`` replaced by a recorder of the modules it was handed, keyed by model path."""
    calls: dict[str, list] = {}
    for name, module in model.named_modules():
        if isinstance(module, PreTrainedModel):
            calls[name] = []
            module._init_weights = calls[name].append
    return calls


def _leaf_tensors(model, prefix: str) -> tuple[str, list[str]]:
    """A module under ``prefix`` holding direct tensors and no children, and their state-dict keys."""
    owner = next(
        name
        for name, module in model.named_modules()
        if name.startswith(f"{prefix}.") and not list(module.children()) and list(module.state_dict())
    )
    return owner, [f"{owner}.{key}" for key in model.get_submodule(owner).state_dict()]


def test_a_towers_own_parameter_is_drawn_by_the_towers_init():
    """The composite's inherited ``_init_weights`` does not know ``RadioLayerScale`` and leaves it at raw memory."""
    model = _nemotron_on_meta()
    state = model.state_dict()
    absent = {key for key in state if key.endswith(_LAYERSCALE)}
    assert absent, "the fixture carries no layer scale, so nothing is under test"
    init_checkpoint_absent_modules(model, set(state) - absent, "cpu", "unit", dtype=torch.bfloat16)
    value = model.config.vision_config.layerscale_value
    for key in absent:
        tensor = model.state_dict(keep_vars=True)[key]
        assert torch.equal(tensor.float(), torch.full_like(tensor, value).float()), key


@pytest.mark.parametrize("name", sorted(_COMPOSITES))
def test_every_composite_dispatches_to_the_model_owning_the_tensor(name, monkeypatch):
    """Only the nearest model is handed the module, as ``smart_apply`` would; the root sees none of it."""
    monkeypatch.setenv(ALLOW_MISSING_CHECKPOINT_KEYS_ENV, "1")
    torch.manual_seed(0)
    model = tiny_family_model(_COMPOSITES[name])
    state = model.state_dict()
    # The deepest model is one nothing nests below, so it is also the nearest of its own leaves.
    nested, _ = max(_nested_models(model), key=lambda item: item[0].count("."))
    owner, absent = _leaf_tensors(model, nested)
    calls = _record_init_weights(model)

    init_checkpoint_absent_modules(model, set(state) - set(absent), "cpu", "unit", dtype=torch.float32)

    assert model.get_submodule(owner) in calls[nested], (nested, owner)
    assert not [path for path, seen in calls.items() if path != nested and seen], calls


@pytest.mark.parametrize("name", sorted(_ROSTER))
def test_a_tensor_outside_every_nested_model_stays_on_the_root_init(name):
    """The head a task class adds: what every family's lazy load draws today."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[name])
    head = model.get_output_embeddings()
    owner = next((path for path, module in model.named_modules() if module is head), None)
    if owner is None or head.weight is model.get_input_embeddings().weight:
        pytest.skip("a tie supplies the head, so nothing is drawn")
    if any(owner.startswith(f"{path}.") for path, _ in _nested_models(model)):
        pytest.skip("the head sits inside a nested model")
    absent = {f"{owner}.{key}" for key in head.state_dict()}
    calls = _record_init_weights(model)
    init_checkpoint_absent_modules(model, set(model.state_dict()) - absent, "cpu", "unit", dtype=torch.float32)
    assert head in calls[""]
    assert not [path for path, seen in calls.items() if path and seen]


def test_the_gate_reads_what_a_tower_declares():
    """``RadioPreTrainedModel`` declares the layer scale the checkpoint never stores; the composite class does not."""
    model = _nemotron_on_meta()
    layerscale = [key for key in model.state_dict() if key.endswith(_LAYERSCALE)]
    assert layerscale, "the fixture carries no layer scale, so nothing is under test"
    assert not [p for p in type(model)._keys_to_ignore_on_load_missing if _LAYERSCALE in p], "composite class declares"
    assert unexpected_missing_keys(model, layerscale) == []
    undeclared = next(key for key in model.state_dict() if key.startswith("language_model.model.layers.0"))
    assert unexpected_missing_keys(model, [undeclared]) == [undeclared]


@pytest.mark.parametrize("name", sorted(_ROSTER))
def test_a_nested_declaration_excuses_only_its_pattern_on_the_whole_roster(name, monkeypatch):
    """Rebuilt with a declaration on its nested model's class, the root's aggregate carries it into the gate."""
    torch.manual_seed(0)
    family = _ROSTER[name]
    nested, child = _nested_models(tiny_family_model(family))[0]
    _, (declared, *_) = _leaf_tensors(tiny_family_model(family), nested)
    monkeypatch.setattr(type(child), "_keys_to_ignore_on_load_missing", [rf"^{declared}$"])
    torch.manual_seed(0)
    model = tiny_family_model(family)
    assert unexpected_missing_keys(model, [declared]) == []
    bystander = next(key for key in model.state_dict() if key != declared and key.startswith(f"{nested}."))
    assert unexpected_missing_keys(model, [bystander]) == [bystander]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
