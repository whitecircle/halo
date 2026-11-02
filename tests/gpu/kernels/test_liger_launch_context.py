#!/usr/bin/env python
"""Liger's kernels under the toolkit's same-device launch context: skipped switch, same results, right device.

``pin_liger_device_context`` rebinds ``device_context`` in Liger's RMSNorm, SwiGLU, GeGLU and fused CE+TVD modules
to a fast path that skips ``torch.cuda.device`` when the tensor is already on the current device. Each op runs forward
and backward through its autograd function while two spies watch it: one counts the ``torch.cuda.device`` switches
Liger's own context enters, the other records the device Triton reads at each kernel launch.

- On the current device the fast path takes no switch, launches there, and matches Liger's own context bit for bit.
- On the other GPU (inputs on the peer while the current device is this rank's own) Liger's context must still
  switch, so every launch lands on the tensor's device, the current device is unchanged afterwards, and the results
  match the same op run with the peer current, bit for bit. A fast path that skipped the switch here would launch on
  this rank's device, the multi-GPU fault Liger's context exists for.

Every rank needs both GPUs visible.

    torchrun --nproc_per_node=2 tests/gpu/kernels/test_liger_launch_context.py
"""

import contextlib
import sys

import torch
from liger_kernel.ops.fused_ce_tvd import LigerFusedCETVDFunction
from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.utils import device_context as liger_device_context
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from triton.runtime import driver

from halo.model_loading.liger.dispatch import sanitize_liger_config
from halo.model_loading.liger.launch_context import same_device_context
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

SEED = 7
# Qwen3-8B's hidden and intermediate widths at a 512-token row; the GLU width takes Liger's tiled SwiGLU on Blackwell.
ROWS, HIDDEN, INTERMEDIATE = 512, 4096, 12288
CE_ROWS, VOCAB = 64, 32000


def _rms_norm(device):
    x = torch.randn(1, ROWS, HIDDEN, dtype=torch.bfloat16).to(device).requires_grad_(True)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16).to(device).requires_grad_(True)
    dy = torch.randn(1, ROWS, HIDDEN, dtype=torch.bfloat16).to(device)
    y = LigerRMSNormFunction.apply(x, w, 1e-6, 0.0, "llama", False, None)
    y.backward(dy)
    return y, x.grad, w.grad


def _glu(function):
    def run(device):
        a = torch.randn(1, ROWS, INTERMEDIATE, dtype=torch.bfloat16).to(device).requires_grad_(True)
        b = torch.randn(1, ROWS, INTERMEDIATE, dtype=torch.bfloat16).to(device).requires_grad_(True)
        dc = torch.randn(1, ROWS, INTERMEDIATE, dtype=torch.bfloat16).to(device)
        c = function.apply(a, b)
        c.backward(dc)
        return c, a.grad, b.grad

    return run


def _fused_ce_tvd(device):
    student = torch.randn(CE_ROWS, VOCAB).to(device).requires_grad_(True)
    teacher = torch.randn(CE_ROWS, VOCAB).to(device)
    target = torch.randint(0, VOCAB, (CE_ROWS,)).to(device)
    ce, tvd = LigerFusedCETVDFunction.apply(student, teacher, target, -100)
    (ce.sum() + tvd.sum()).backward()
    return ce, tvd, student.grad


# Each op's module, as the pin rebinds it, and a forward+backward returning its outputs and gradients.
OPS = {
    "liger_kernel.ops.rms_norm": _rms_norm,
    "liger_kernel.ops.swiglu": _glu(LigerSiLUMulFunction),
    "liger_kernel.ops.geglu": _glu(LigerGELUMulFunction),
    "liger_kernel.ops.fused_ce_tvd": _fused_ce_tvd,
}


@contextlib.contextmanager
def _spied():
    """Record the ``torch.cuda.device`` switches entered and the device Triton reads at each launch."""
    switches, launches = [], []
    stock_device, active = torch.cuda.device, driver.active
    stock_current = active.get_current_device

    class CountingDevice(stock_device):
        def __enter__(self):
            switches.append(self.idx)
            return super().__enter__()

    def current_device():
        index = stock_current()
        launches.append(index)
        return index

    torch.cuda.device, active.get_current_device = CountingDevice, current_device
    try:
        yield switches, launches
    finally:
        torch.cuda.device, active.get_current_device = stock_device, stock_current


def _run(name, device):
    torch.manual_seed(SEED)
    with _spied() as (switches, launches):
        tensors = OPS[name](device)
    # Outside the spies: synchronize enters torch.cuda.device itself.
    torch.cuda.synchronize(device)
    return tensors, switches, launches


@contextlib.contextmanager
def _bound(name, context):
    module = sys.modules[name]
    pinned = module.device_context
    module.device_context = context
    try:
        yield
    finally:
        module.device_context = pinned


def _assert_bitwise(name, got, expected) -> None:
    for i, (a, b) in enumerate(zip(got, expected, strict=True)):
        assert torch.equal(a, b.to(a.device)), (
            f"{name}: tensor {i} differs (max {(a.float() - b.float()).abs().max()})"
        )


def _check_pinned() -> None:
    sanitize_liger_config({}, needs_ep_layers=False, model_config=CONFIG_MAPPING["qwen3"]())
    unpinned = [name for name in OPS if sys.modules[name].device_context is not same_device_context]
    assert not unpinned, f"sanitize_liger_config left Liger's device_context on {unpinned}"


def _check_same_device(own) -> None:
    for name in OPS:
        tensors, switches, launches = _run(name, own)
        log(f"  {name}: {len(switches)} switches, launches on {sorted(set(launches))}")
        assert not switches, f"{name}: the fast path entered torch.cuda.device {len(switches)} times on its own device"
        assert launches and set(launches) == {own.index}, f"{name}: launched on {sorted(set(launches))}, not {own}"
        with _bound(name, liger_device_context):
            expected, liger_switches, _ = _run(name, own)
        assert liger_switches, f"{name}: Liger's own context took no switch, so the control does not exercise it"
        _assert_bitwise(name, tensors, expected)


def _check_other_device(own, peer) -> None:
    for name in OPS:
        tensors, switches, launches = _run(name, peer)
        log(f"  {name}: {len(switches)} switches to {sorted(set(switches))}, launches on {sorted(set(launches))}")
        assert torch.cuda.current_device() == own.index, f"{name}: left the current device on {peer}"
        assert peer.index in switches, f"{name}: never switched to {peer}"
        assert launches and set(launches) == {peer.index}, f"{name}: launched on {sorted(set(launches))}, not {peer}"
        with torch.cuda.device(peer):
            expected, _, _ = _run(name, peer)
        _assert_bitwise(name, tensors, expected)


@gpu_test_main(exact_world_size=2, prefix="test_liger_launch_context", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"rank {ctx.rank} sees {torch.cuda.device_count()} GPU(s); this test needs both visible")
    own = torch.device("cuda", ctx.local_rank)
    peer = torch.device("cuda", 1 - ctx.local_rank)
    torch.cuda.set_device(own)

    log("\n--- pinned by sanitize_liger_config ---")
    record_check(checks, "pinned_by_sanitize", _check_pinned)
    log(f"\n--- inputs on the current device ({own}) ---")
    record_check(checks, "same_device_skips_switch_and_matches", lambda: _check_same_device(own))
    log(f"\n--- inputs on the other GPU ({peer}, current {own}) ---")
    record_check(checks, "other_device_launches_on_tensor_device", lambda: _check_other_device(own, peer))
    return {"checks": checks}


if __name__ == "__main__":
    run()
