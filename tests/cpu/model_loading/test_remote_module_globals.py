"""The names the compat shims bind through ``get_class_in_module``. ``Ring-mini-linear-2.0`` calls
``DynamicLayer()`` without importing it, a ``NameError`` only at ``generate``, after training."""

import pathlib
import sys

import pytest
import torch
import transformers.dynamic_module_utils
from transformers.cache_utils import DynamicCache, DynamicLayer

from halo.model_loading.remote_code import (
    _UNIMPORTED_REMOTE_GLOBALS,
    apply_remote_code_compat_shims,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

apply_remote_code_compat_shims()

# DynamicCache imported, DynamicLayer called but not imported.
REMOTE_SOURCE = '''
from transformers.cache_utils import Cache, DynamicCache


class RingLikeModel:
    """Stands in for the published modeling file's cached-attention path."""

    def append_layer(self, cache):
        cache.layers.append(DynamicLayer())
        return len(cache.layers)
'''

# A module binding the name itself stays exactly as written.
SELF_BINDING_SOURCE = """
class DynamicLayer:  # noqa: D101 — a deliberate homonym, not the transformers class
    pass


class OwnLayerModel:
    def layer_type(self):
        return DynamicLayer
"""


# What the release's file holds of its composite: the tower under ``language_model``, no head accessor of its own.
REMOTE_COMPOSITE_SOURCE = """
import torch


class NemotronH_Omni_Reasoning_V3(torch.nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
"""


PROBE_PREFIX = "remote_probe_"


@pytest.fixture(autouse=True)
def _drop_probe_modules():
    """One case reads the loaded module's namespace, so it is cleared afterwards."""
    yield
    for name in [n for n in sys.modules if n.startswith(PROBE_PREFIX)]:
        del sys.modules[name]


def _load(tmp_path, monkeypatch, source: str, class_name: str):
    module_name = f"{PROBE_PREFIX}{abs(hash(source)) % 10**8}"
    module_path = f"{module_name}.py"
    monkeypatch.setattr(transformers.dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))
    (pathlib.Path(tmp_path) / module_path).write_text(source, encoding="utf-8")
    sys.modules.pop(module_name, None)
    return transformers.dynamic_module_utils.get_class_in_module(class_name, module_path)


def test_a_called_but_unimported_name_is_bound(tmp_path, monkeypatch):
    """The call must succeed, not merely the attribute exist."""
    cls = _load(tmp_path, monkeypatch, REMOTE_SOURCE, "RingLikeModel")
    cache = DynamicCache()
    depth_before = len(cache.layers)
    # Without the shim this raises NameError: name 'DynamicLayer' is not defined.
    assert cls().append_layer(cache) == depth_before + 1
    assert isinstance(cache.layers[-1], DynamicLayer)


def test_the_bound_name_is_the_transformers_class(tmp_path, monkeypatch):
    """A stand-in would let the append succeed and the cache misbehave later."""
    cls = _load(tmp_path, monkeypatch, REMOTE_SOURCE, "RingLikeModel")
    assert sys.modules[cls.__module__].DynamicLayer is DynamicLayer


def test_a_module_that_binds_the_name_itself_is_untouched(tmp_path, monkeypatch):
    cls = _load(tmp_path, monkeypatch, SELF_BINDING_SOURCE, "OwnLayerModel")
    assert cls().layer_type() is not DynamicLayer, "the shim overwrote a name the module defined"


def test_every_bound_name_is_absent_from_the_published_import_list():
    """A name transformers stops exporting would bind None-ish and hide."""
    for name, symbol in _UNIMPORTED_REMOTE_GLOBALS.items():
        assert symbol is not None, f"{name} resolved to None — the shim would bind a dead name"
        assert isinstance(symbol, type), f"{name} must be a class, got {type(symbol)}"


def test_the_releases_own_composite_answers_with_its_towers_head(tmp_path, monkeypatch):
    """An export carries that file, and a tool on a local directory trusts it by default."""
    composite_class = _load(tmp_path, monkeypatch, REMOTE_COMPOSITE_SOURCE, "NemotronH_Omni_Reasoning_V3")
    tower = tiny_family_model(TINY_MOE_FAMILIES["nemotron_h"])
    composite = composite_class(tower)
    assert composite.get_output_embeddings() is tower.lm_head
    replacement = torch.nn.Linear(tower.lm_head.in_features, tower.lm_head.out_features + 8, bias=False)
    composite.set_output_embeddings(replacement)
    assert tower.lm_head is replacement


def test_another_remote_file_is_left_alone(tmp_path, monkeypatch):
    composite_class = _load(tmp_path, monkeypatch, REMOTE_SOURCE, "RingLikeModel")
    assert not hasattr(composite_class, "get_output_embeddings")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
