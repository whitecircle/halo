#!/usr/bin/env python
"""The deferred remote-code patch fires on the module transformers loads, and says so when it can't.

A ``trust_remote_code`` family's modeling module does not exist when Liger is applied, so its patch
is armed on the shared ``get_class_in_module`` funnel hook and fires later
(:mod:`src.kernels.liger.remote_modules`). It is the ONLY path by which Ling/Ring get any fused kernel
and the one Laguna's released repos load through. Three silent failure modes are worth pinning: the
hook never firing, a revision that renamed one of the spec's classes being skipped without a word, and
a native family's spec patching only the in-library module its hub code bypasses.

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
from src.kernels.liger.builder import _PATCHED_MARKER, LigerFamilySpec
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

# The classes Laguna's spec patches, as the released repos' own `modeling_laguna.py` declares them
# (constructor signatures and the MLP's `act_fn` included); everything else in that file is omitted.
_LAGUNA_HUB_MODULE_SOURCE = textwrap.dedent(
    """
    import torch
    from torch import nn
    from transformers.activations import ACT2FN


    class LagunaRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps


    class LagunaMLP(nn.Module):
        def __init__(self, config, intermediate_size=None):
            super().__init__()
            size = config.intermediate_size if intermediate_size is None else intermediate_size
            self.gate_proj = nn.Linear(config.hidden_size, size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, size, bias=False)
            self.down_proj = nn.Linear(size, config.hidden_size, bias=False)
            self.act_fn = ACT2FN[config.hidden_act]


    class LagunaForCausalLM(nn.Module):
        def forward(self, input_ids=None, labels=None, **kwargs):
            raise NotImplementedError
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


def test_lagunas_hub_modeling_code_takes_the_library_swaps(tmp_path):
    """Laguna's released repos load their own modeling file under ``trust_remote_code``.

    Its norm, MLP and head are the in-library classes verbatim, so the family spec has to reach both
    modules: one naming only ``transformers.models.laguna`` logs every kernel as applied while the
    classes the run actually builds stay eager. Subprocess: the applier swaps classes process-wide.
    """
    module_file = tmp_path / "modeling_laguna.py"
    module_file.write_text(_LAGUNA_HUB_MODULE_SOURCE, encoding="utf-8")
    script = f"""
import inspect, sys, types
import transformers.dynamic_module_utils as dmu
from accelerate import PartialState

PartialState()
from transformers.models.laguna import modeling_laguna
from src.kernels.liger.orchestrator import resolve_liger_applier

applier = resolve_liger_applier("laguna")
requested = {{"rms_norm", "swiglu", "fused_linear_cross_entropy"}}
applier(**{{name: name in requested for name in set(inspect.signature(applier).parameters) - {{"model"}}}})
hub = sys.modules[dmu.get_class_in_module("LagunaForCausalLM", {str(module_file)!r}).__module__]

problems = []
for origin, module in (("library", modeling_laguna), ("hub", hub)):
    for name, role in (("LagunaRMSNorm", "rms_norm"), ("LagunaMLP", "glu_mlp")):
        if getattr(getattr(module, name), {_PATCHED_MARKER!r}, None) != role:
            problems.append(f"{{origin}} {{name}} kept its eager forward")
    if module.LagunaForCausalLM.forward.__module__ != "src.kernels.liger.lce_forward":
        problems.append(f"{{origin}} LagunaForCausalLM kept its unfused head")
mlp = hub.LagunaMLP(types.SimpleNamespace(hidden_size=8, intermediate_size=16, hidden_act="silu"))
if getattr(mlp, "_halo_glu_mul", None) is None:
    problems.append("the hub LagunaMLP's SiLU was not recognized as fusable")
print("PROBLEMS:" + "|".join(problems))
"""
    problems = probe_findings(script, "PROBLEMS:")
    assert not problems, "Laguna's two modeling modules are not patched alike:\n" + "\n".join(problems)


def test_a_remote_spec_identifies_its_module_by_every_class_it_patches():
    """A module missing an identifying class is reported and left unpatched; one missing a class the
    roles swap but the identifying set omits would fail its model load instead."""
    with pytest.raises(ValueError, match="without listing them in remote_classes"):
        LigerFamilySpec(
            model_types=("probe",),
            remote_classes=("ProbeRMSNorm",),
            rms_norm=("ProbeRMSNorm",),
            glu_mlp=("ProbeMLP",),
        )
    with pytest.raises(ValueError, match="no module to patch"):
        LigerFamilySpec(model_types=("probe",), rms_norm=("ProbeRMSNorm",))


@pytest.mark.parametrize("liger_first", [False, True])
def test_the_hook_composes_with_the_remote_code_compat_shims(liger_first):
    """Both register on ONE shared funnel wrapper, and every registrant runs whatever the order.

    ``remote_code_compat`` binds names a remote file uses without importing; dropping that callback
    would turn a Ling forward into a ``NameError`` the moment Liger armed a patch, and conversely.
    Two independent wrappers would nest, so a re-application stacks another — the funnel must be
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
