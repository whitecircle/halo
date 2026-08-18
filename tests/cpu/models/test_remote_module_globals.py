"""CPU tests for the names the compat shims bind into remote modeling modules.

``inclusionAI/Ring-mini-linear-2.0``'s published modeling file calls ``DynamicLayer()`` while
importing only ``Cache`` and ``DynamicCache``, so the bare global raises ``NameError`` on every
CACHED forward. Training runs with ``use_cache=False`` and never reaches it — the break surfaces at
``generate`` and at any check on a reloaded model, i.e. after the training everyone waits for.

The sibling shims restore names on ``transformers`` itself, which works because the remote file
imports those and the IMPORT is what fails. That cannot work here: the name is never imported, so
only the module whose global lookup misses can be given it. The shim therefore binds through
``get_class_in_module`` — the single funnel every remote-code class loads through.

These tests drive that funnel with a module reproducing the published file's shape, so they fail if
the binding is removed rather than merely asserting the dict is populated.

    python tests/cpu/models/test_remote_module_globals.py
"""

import pathlib
import sys

import pytest
import transformers.dynamic_module_utils
from transformers.cache_utils import DynamicCache, DynamicLayer

from src.models.patches.remote_code_compat import (
    _UNIMPORTED_REMOTE_GLOBALS,
    apply_remote_code_compat_shims,
)

apply_remote_code_compat_shims()

# The published file's shape: DynamicCache imported, DynamicLayer called but NOT imported.
REMOTE_SOURCE = '''
from transformers.cache_utils import Cache, DynamicCache


class RingLikeModel:
    """Stands in for the published modeling file's cached-attention path."""

    def append_layer(self, cache):
        cache.layers.append(DynamicLayer())
        return len(cache.layers)
'''

# A module that binds the name itself must be left exactly as written.
SELF_BINDING_SOURCE = """
class DynamicLayer:  # noqa: D101 — a deliberate homonym, not the transformers class
    pass


class OwnLayerModel:
    def layer_type(self):
        return DynamicLayer
"""


PROBE_PREFIX = "remote_probe_"


@pytest.fixture(autouse=True)
def _drop_probe_modules():
    """The loaded module must SURVIVE the test — one case reads its namespace — so it is cleared
    afterwards instead of inside the loader."""
    yield
    for name in [n for n in sys.modules if n.startswith(PROBE_PREFIX)]:
        del sys.modules[name]


def _load(tmp_path, monkeypatch, source: str, class_name: str):
    """Load ``source`` through the real ``get_class_in_module`` funnel and return the class."""
    module_name = f"{PROBE_PREFIX}{abs(hash(source)) % 10**8}"
    module_path = f"{module_name}.py"
    monkeypatch.setattr(transformers.dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path))
    (pathlib.Path(tmp_path) / module_path).write_text(source, encoding="utf-8")
    sys.modules.pop(module_name, None)
    return transformers.dynamic_module_utils.get_class_in_module(class_name, module_path)


def test_a_called_but_unimported_name_is_bound(tmp_path, monkeypatch):
    """The real failure: the call must succeed, not merely the attribute exist."""
    cls = _load(tmp_path, monkeypatch, REMOTE_SOURCE, "RingLikeModel")
    cache = DynamicCache()
    depth_before = len(cache.layers)
    # Without the shim this raises NameError: name 'DynamicLayer' is not defined.
    assert cls().append_layer(cache) == depth_before + 1
    assert isinstance(cache.layers[-1], DynamicLayer)


def test_the_bound_name_is_the_transformers_class(tmp_path, monkeypatch):
    """A stand-in object would let the append succeed and the cache misbehave later."""
    cls = _load(tmp_path, monkeypatch, REMOTE_SOURCE, "RingLikeModel")
    assert sys.modules[cls.__module__].DynamicLayer is DynamicLayer


def test_a_module_that_binds_the_name_itself_is_untouched(tmp_path, monkeypatch):
    """Absent names only — overwriting a module's own definition would change its behaviour."""
    cls = _load(tmp_path, monkeypatch, SELF_BINDING_SOURCE, "OwnLayerModel")
    assert cls().layer_type() is not DynamicLayer, "the shim overwrote a name the module defined"


def test_every_bound_name_is_absent_from_the_published_import_list():
    """Guards the premise: a name transformers no longer exports would bind None-ish and hide."""
    for name, symbol in _UNIMPORTED_REMOTE_GLOBALS.items():
        assert symbol is not None, f"{name} resolved to None — the shim would bind a dead name"
        assert isinstance(symbol, type), f"{name} must be a class, got {type(symbol)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
