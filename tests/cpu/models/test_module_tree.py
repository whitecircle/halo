"""The module-tree lookups: the attribute probe misses with None, the layer-list lookup raises naming
its candidates, and a module path is strict or lenient by name."""

import pytest
import torch.nn as nn

from src.models.module_tree import (
    DECODER_LAYER_LIST_ATTRS,
    decoder_layers,
    decoder_layers_attr,
    embedding_module,
    find_attr_name,
    find_module_path,
    module_path,
    split_layer_path,
)


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
    with pytest.raises(ValueError, match=r"Linear exposes no decoder-layer list under any of \('layers', 'h'\)"):
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


def test_embedding_module_reads_the_getter_and_misses_with_none():
    model = _Model()
    assert embedding_module(model, "output") is model.lm_head
    assert embedding_module(model, "input") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
