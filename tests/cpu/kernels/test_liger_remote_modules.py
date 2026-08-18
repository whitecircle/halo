#!/usr/bin/env python
"""The deferred remote-code patch fires on the module transformers loads, and says so when it can't.

A ``trust_remote_code`` family's modeling module does not exist when Liger is applied, so its patch
is armed on the shared ``get_class_in_module`` funnel hook and fires later
(:mod:`src.kernels.liger.remote_modules`). This is the ONLY path by which Ling/Ring get any fused
kernel, and it has two silent failure modes worth pinning: the hook never firing, and a revision that
renamed one of the spec's classes being skipped without a word.

The funnel is driven for real — a modeling file written to a temp dir, then loaded through
transformers' own ``get_class_in_module`` — because a hand-called ``_fire`` would not prove the hook
is installed where transformers looks.

    pytest -m cpu tests/cpu/kernels/test_liger_remote_modules.py
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
import transformers.dynamic_module_utils
from accelerate import PartialState

from src.kernels.liger import remote_modules
from tests.common.utils import probe_findings

PartialState()  # the module logs through accelerate's rank-aware logger

_LOGGER_NAME = "src.kernels.liger.remote_modules"

_MODULE_SOURCE = textwrap.dedent(
    """
    class ProbeRemoteAlpha:
        pass

    class ProbeRemoteBeta:
        pass
    """
)


@pytest.fixture
def armed():
    """Restore the module-global arm list so a probe cannot leak into another test."""
    original = list(remote_modules._ARMED)
    yield remote_modules._ARMED
    remote_modules._ARMED.clear()
    remote_modules._ARMED.extend(original)


def _load_through_transformers(tmp_path: Path, class_name: str, suffix: str):
    """Write a modeling file and load a class from it the way transformers loads remote code.

    No import-path setup: ``get_class_in_module`` resolves the file by location, which is exactly
    why the hook has to sit on that function rather than on an import.
    """
    module_file = tmp_path / f"modeling_probe_{suffix}.py"
    module_file.write_text(_MODULE_SOURCE, encoding="utf-8")
    return transformers.dynamic_module_utils.get_class_in_module(class_name, module_file)


def test_the_patch_fires_when_transformers_loads_the_module(armed, tmp_path):
    """The whole point: arming before the file exists still patches it once it is loaded."""
    seen = []
    remote_modules.patch_remote_modules(("ProbeRemoteAlpha", "ProbeRemoteBeta"), seen.append)
    assert seen == [], "nothing is loaded yet, so nothing may have been patched"

    cls = _load_through_transformers(tmp_path, "ProbeRemoteAlpha", "fires")
    assert cls.__name__ == "ProbeRemoteAlpha"
    assert [module.__name__ for module in seen] == [cls.__module__], (
        "the armed patch did not reach the module transformers just loaded"
    )


def test_a_renamed_class_warns_instead_of_skipping_silently(armed, tmp_path, caplog):
    """A revision that renames one declared class must not leave the run quietly unfused.

    The applier has already logged that it armed the patch, so silence here reads as success.
    """
    seen = []
    remote_modules.patch_remote_modules(("ProbeRemoteAlpha", "ProbeRemoteGone"), seen.append)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _load_through_transformers(tmp_path, "ProbeRemoteAlpha", "renamed")

    assert seen == [], "a module missing a declared class must not be patched"
    assert any("ProbeRemoteGone" in record.getMessage() for record in caplog.records), (
        f"renamed class skipped silently: {[r.getMessage() for r in caplog.records]}"
    )


def test_an_unrelated_module_is_left_alone(armed, tmp_path, caplog):
    """Anti-vacuity: sharing NONE of the declared names must be silent, not a warning.

    Every remote class in the process funnels through this hook, so warning on modules that were
    never a candidate would bury the real drift signal.
    """
    seen = []
    remote_modules.patch_remote_modules(("ProbeRemoteMissingOne", "ProbeRemoteMissingTwo"), seen.append)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _load_through_transformers(tmp_path, "ProbeRemoteBeta", "unrelated")
    assert seen == []
    assert not [record for record in caplog.records if record.name == _LOGGER_NAME]


def test_arming_twice_does_not_stack(armed, tmp_path):
    """Liger is applied at model load and may be re-applied; the patch must run once per module."""
    seen = []
    for _ in range(3):
        remote_modules.patch_remote_modules(("ProbeRemoteAlpha", "ProbeRemoteBeta"), seen.append)
    _load_through_transformers(tmp_path, "ProbeRemoteAlpha", "idempotent")
    assert len(seen) == 1, f"the patch ran {len(seen)} times; arming is not idempotent"


@pytest.mark.parametrize("liger_first", [False, True])
def test_the_hook_composes_with_the_remote_code_compat_shims(liger_first):
    """Both register on ONE shared funnel wrapper, and every registrant runs whatever the order.

    ``remote_code_compat`` binds names a remote file uses without importing; dropping that callback
    would turn a Ling forward into a ``NameError`` the moment Liger armed a patch, and conversely.
    Two independent wrappers used to nest, so a re-application stacked another — the funnel must be
    wrapped exactly once and re-registering must not change it.

    Subprocess per order: both registrars are process-global and guarded against re-installing, so a
    second order cannot be exercised in a process that already ran the first.
    """
    script = f"""
import transformers.dynamic_module_utils as dmu
from accelerate import PartialState

PartialState()
from src.kernels.liger.remote_modules import _fire, patch_remote_modules
from src.models.patches import remote_code_compat
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.models.patches.remote_code_hooks import _HOOKS

stock = dmu.get_class_in_module
installers = {{
    "liger": lambda: patch_remote_modules(("ProbeAlpha", "ProbeBeta"), lambda module: None),
    "compat": apply_remote_code_compat_shims,
}}
order = ["liger", "compat"] if {liger_first} else ["compat", "liger"]
installers[order[0]]()
first = dmu.get_class_in_module
installers[order[1]]()
second = dmu.get_class_in_module

problems = []
if first is stock:
    problems.append(f"{{order[0]}} did not install")
if second is not first:
    problems.append(f"{{order[1]}} wrapped the funnel a second time instead of registering on it")
for name, hook in (("liger", _fire), ("compat", remote_code_compat._repair_remote_module)):
    if hook not in _HOOKS:
        problems.append(f"{{name}}'s callback is not registered on the funnel")
# Re-installing either must not stack another wrapper or duplicate a callback.
installers[order[0]]()
installers[order[1]]()
if dmu.get_class_in_module is not second:
    problems.append("a re-install stacked another wrapper")
if len(_HOOKS) != len(set(map(id, _HOOKS))):
    problems.append("a re-install duplicated a callback")
print("PROBLEMS:" + "|".join(problems))
"""
    problems = probe_findings(script, "PROBLEMS:")
    assert not problems, "the two funnel registrants do not compose:\n" + "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
