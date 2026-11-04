import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from halo.models.module_tree import (
    DECODER_LAYER_LIST_ATTRS,
    decoder_layers,
    decoder_layers_attr,
    embedding_module,
    find_attr_name,
    find_module_path,
    hidden_state_backbone,
    module_path,
    split_layer_path,
)
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_LLAMA4_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    tiny_family_model,
)

# A composite holding ``language_model`` at its root, whose ``base_model`` is itself.
_ROOT_COMPOSITES = {"llama4": TINY_LLAMA4_FAMILY, "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}
_ROSTER = {
    name: family
    for name, family in {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}.items()
    if name not in _ROOT_COMPOSITES
}


class _Backbone(nn.Module):
    def __init__(self, attr: str):
        super().__init__()
        self.norm = None
        setattr(self, attr, nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone("h")
        self.lm_head = nn.Linear(2, 3)

    def get_output_embeddings(self):
        return self.lm_head


def test_find_attr_name_takes_the_first_set_candidate_and_misses_with_none():
    backbone = _Backbone("layers")
    assert find_attr_name(backbone, ("norm", "layers", "h")) == "layers"
    assert find_attr_name(backbone, ("norm", "missing")) is None


@pytest.mark.parametrize("attr", DECODER_LAYER_LIST_ATTRS)
def test_the_layer_list_resolves_under_every_spelling(attr):
    backbone = _Backbone(attr)
    assert decoder_layers_attr(backbone) == attr
    assert decoder_layers(backbone) is getattr(backbone, attr)


def test_a_backbone_without_a_layer_list_raises_naming_the_candidates_and_the_lenient_probe_returns_none():
    with pytest.raises(
        ValueError, match=r"Linear exposes no decoder-layer list under any of \('layers', 'h', 'layer'\)"
    ):
        decoder_layers_attr(nn.Linear(2, 2))
    assert decoder_layers(nn.Linear(2, 2)) is None


def test_module_path_is_strict_and_find_module_path_lenient():
    model = _Model()
    assert module_path(model, model.model.h[1]) == "model.h.1"
    assert find_module_path(model, model.model.h[1]) == "model.h.1"
    stranger = nn.Linear(2, 2)
    assert find_module_path(model, stranger) is None
    with pytest.raises(ValueError, match="Linear is not a module of _Model"):
        module_path(model, stranger)


def test_split_layer_path_reads_the_index_and_remainder_under_its_root():
    assert split_layer_path("model.h.3.mlp.weight", "model.h.") == (3, "mlp.weight")
    assert split_layer_path("model.h.3", "model.h.") == (3, "")
    assert split_layer_path("model.h.x.mlp", "model.h.") is None
    assert split_layer_path("lm_head.weight", "model.h.") is None


@pytest.mark.parametrize("name", sorted(_ROSTER))
def test_the_hidden_state_backbone_is_base_model_on_the_roster(name):
    """Every shipped family's head path enters where it always did, a composite's ``.model`` included."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[name])
    assert model.base_model is not model
    assert hidden_state_backbone(model) is model.base_model


@pytest.mark.parametrize("name", sorted(_ROOT_COMPOSITES))
def test_a_root_composite_answers_its_towers_backbone(name):
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_COMPOSITES[name])
    assert model.base_model is model, "premise: base_model names no module below this composite"
    assert hidden_state_backbone(model) is model.language_model.model


def test_a_wrapper_is_peeled_to_the_same_backbone():
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_COMPOSITES["nemotron_h_omni"])
    backbone = model.language_model.model
    assert hidden_state_backbone(get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"]))) is backbone


def test_embedding_module_reads_the_getter_and_misses_with_none():
    model = _Model()
    assert embedding_module(model, "output") is model.lm_head
    assert embedding_module(model, "input") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
