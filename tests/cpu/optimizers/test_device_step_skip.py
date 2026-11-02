#!/usr/bin/env python
"""CPU test: a device skip flag (``halo/optimizers/step_skip.py``) leaves every toolkit optimizer's step exact.

Under a False flag a step writes what an unflagged step does, bit for bit; under a True flag it writes nothing, and
settling it as skipped returns the state dict to what it was (:func:`tests.common.device_step_skip.
device_skip_mismatches`). AdamWBF16's eager paths are held to the implementation before the flag
(``PerParameterAdamWBF16``); its fused kernel and Muon's run under Triton's interpreter in a fresh process
(``TRITON_INTERPRET`` is read when a kernel is defined), against their own unflagged steps. The CUDA legs are
``tests/gpu/optimizers/test_adamw_bf16_fused.py`` and ``tests/gpu/optimizers/test_muon.py``.

Run: python tests/cpu/optimizers/test_device_step_skip.py
"""

import weakref

import pytest
import torch

from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.common.adamw_bf16_reference import PerParameterAdamWBF16
from tests.common.device_step_skip import adamw_bf16_builder, device_skip_mismatches
from tests.common.repo import probe_findings

_MARKER = "DEVICE_SKIP_MISMATCHES:"
_FUSED_PROBE = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch
import torch.nn as nn
from halo.optimizers import adamw_bf16
from tests.common.device_step_skip import adamw_bf16_builder, device_skip_mismatches


class CudaParameter(nn.Parameter):
    @property
    def is_cuda(self):
        return True


torch.Tensor.pin_memory = lambda self: self
findings = []
for per_launch in (adamw_bf16._MAX_TENSORS_PER_LAUNCH, 1):
    adamw_bf16._tensors_per_launch = lambda device, n=per_launch: n
    build = adamw_bf16_builder("cpu", parameter_cls=CudaParameter)
    findings += [f"{{per_launch}} per launch: {{f}}" for f in device_skip_mismatches(build, build)]
print({_MARKER!r} + "|".join(findings))
"""
_MUON_PROBE = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
from tests.common.device_step_skip import device_skip_mismatches, muon_builder
build = muon_builder("cpu")
print({_MARKER!r} + "|".join(device_skip_mismatches(build, build)))
"""


class _Ignores(PerParameterAdamWBF16):
    """Takes the flag and steps as if it were never set."""

    def step(self, closure=None):
        self._take_step_skip(self.param_groups)
        return super().step(closure)


class _KeepsBookkeeping(AdamWBF16):
    """Rolls nothing back."""

    def settle_step(self, skipped):
        super().settle_step(skipped=False)


def test_adamw_bf16_eager_paths_skip_on_the_flag_and_step_as_before():
    """The eager bf16 path and the fp32 path: a False flag writes what the step before the flag wrote."""
    assert (
        device_skip_mismatches(
            adamw_bf16_builder("cpu", optimizer_cls=PerParameterAdamWBF16),
            adamw_bf16_builder("cpu", optimizer_cls=AdamWBF16),
        )
        == []
    )


def test_the_check_sees_a_flag_the_optimizer_ignores():
    """Negative control: an optimizer stepping through a True flag, as one without the skip would, is caught on the
    skipped steps, its parameters and moments written."""
    findings = device_skip_mismatches(
        adamw_bf16_builder("cpu", optimizer_cls=PerParameterAdamWBF16),
        adamw_bf16_builder("cpu", optimizer_cls=_Ignores),
    )
    assert any(f.startswith("skipped step 3: param") for f in findings), findings
    assert any("'exp_avg' written" in f for f in findings), findings


def test_the_check_sees_bookkeeping_left_behind():
    """Negative control: a settle that keeps the step counts and the first step's state is caught."""
    findings = device_skip_mismatches(
        adamw_bf16_builder("cpu", optimizer_cls=AdamWBF16), adamw_bf16_builder("cpu", optimizer_cls=_KeepsBookkeeping)
    )
    assert any(f.startswith("skipped step 1, settled: state covers params") for f in findings), findings
    assert any(f.startswith("skipped step 3, settled: param 0 'step' differs") for f in findings), findings


def test_fused_kernel_skips_on_the_flag_and_steps_as_before_interpreted():
    """The fused launch at the multi-tensor width and at one tensor per launch."""
    assert probe_findings(_FUSED_PROBE, _MARKER) == []


def test_muon_skips_on_the_flag_and_steps_as_before_interpreted():
    """Both legs: the momentum and weight kernels of the matrices, and the scalar AdamWBF16 of the norm."""
    assert probe_findings(_MUON_PROBE, _MARKER) == []


def test_settling_an_unflagged_step_as_skipped_raises():
    """The restore step and any step without a flag record nothing to roll back."""
    params, optimizer = adamw_bf16_builder("cpu")()
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.skip_next_step_if(torch.tensor(True))
    optimizer.step()
    optimizer.step()

    with pytest.raises(RuntimeError, match="ran without a skip flag"):
        optimizer.settle_step(skipped=True)


@pytest.mark.parametrize("skipped", [False, True], ids=["finite", "skipped"])
def test_a_settled_step_keeps_no_state_alive(skipped):
    """Settled either way, a flagged step lets go of the state it recorded: state a resumed run's load replaces is
    freed at once, not held until the next step."""
    params, optimizer = adamw_bf16_builder("cpu")()
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    old = weakref.ref(optimizer.state[params[0]]["exp_avg"])
    optimizer.skip_next_step_if(torch.tensor(skipped))
    optimizer.step()

    optimizer.settle_step(skipped=skipped)
    optimizer.state.clear()

    assert old() is None, "the settled step's record keeps the replaced optimizer state alive"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
