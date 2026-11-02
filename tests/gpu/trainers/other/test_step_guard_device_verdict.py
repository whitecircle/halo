#!/usr/bin/env python
"""The non-finite step guard's late verdict on CUDA, where its copy is asynchronous and its read must not stall.

``tests/cpu/trainers/test_nonfinite_step_guard.py`` pins the guard's logic; on CPU its copy of the norm to host
memory is a plain copy. Here, on one GPU, under AdamWBF16's fused step:

1. **The late read waits for its copy.** The stream is held busy (``torch.cuda._sleep``) ahead of a NaN step, so the
   copy of its norm to pinned memory is still queued when the host reads the verdict straight after the step. The
   read must see the NaN, not the previous step's finite norm the buffer still holds: a read that skips the wait on
   the copy's event, or an event recorded before the copy or on another stream, reads the stale norm.
   Negative control: the same read with the event wait removed does read the stale norm.
2. **A device-path step adds no host sync.** Under ``torch.cuda.set_sync_debug_mode("error")`` two finite steps go
   through the guard: the late read of the step before, the copy of the norm, the hand-over to the optimizer,
   AdamWBF16's step and a ``_maybe_log_save_evaluate`` pass that logs nothing. Any synchronizing CUDA call raises
   there; an event wait does not. The verdict buffer must be pinned too, since a non-blocking copy into pageable
   memory stalls the host without tripping the mode.
   Negative control: the host path (torch's AdamW) reads the norm with ``.item()`` and raises under the same mode.

Run: torchrun --nproc_per_node=1 tests/gpu/trainers/other/test_step_guard_device_verdict.py
"""

from types import SimpleNamespace

import torch

from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log
from tests.common.step_guard import Guarded, guarded, guarded_step

# GPU cycles torch.cuda._sleep holds the stream for: about a second on current GPUs, far longer than the host takes
# from queuing the NaN step to reading its verdict.
BUSY_CYCLES = 2**31


def _device_guard() -> Guarded:
    """The guard over bf16 parameters on the GPU, which AdamWBF16 steps with its fused kernel; a pass after each step
    logs, evaluates and saves nothing."""
    me = guarded(optimizer_cls=AdamWBF16, device="cuda", dtype=torch.bfloat16)
    me.control = SimpleNamespace(should_log=False, should_evaluate=False, should_save=False)
    me._globalstep_last_logged = 0
    return me


def _step_and_pass(me: Guarded, grad: float) -> None:
    guarded_step(me, grad)
    me._maybe_log_save_evaluate(me._tr_loss)


def _warm_up(me: Guarded) -> None:
    """Two settled finite steps: they create the optimizer state, the pinned verdict buffer and the pinned launch
    tables, and compile the kernel, all of which the steps after them reuse."""
    for _ in range(2):
        _step_and_pass(me, 0.5)
    torch.cuda.synchronize()
    me._read_device_verdict()


def _weights(me: Guarded) -> list[torch.Tensor]:
    return [param.detach().clone() for param in me.model.parameters()]


def _read_behind_a_busy_stream(me: Guarded) -> list[torch.Tensor]:
    """Queue a NaN step behind a busy stream and read its verdict at once; the weights before the step."""
    weights = _weights(me)
    torch.cuda._sleep(BUSY_CYCLES)
    _step_and_pass(me, float("nan"))
    assert not torch.cuda.current_stream().query(), "premise: the NaN step's copy is still queued at the read"
    me._read_device_verdict()
    return weights


def check_late_read_waits_for_its_copy():
    me = _device_guard()
    _warm_up(me)

    weights = _read_behind_a_busy_stream(me)

    assert me._nonfinite_steps_skipped == 1, "the late read returned before its copy landed and read a stale norm"
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(weights, _weights(me), strict=True)), "the NaN step was applied"


def check_late_read_without_the_wait_reads_a_stale_norm():
    """Negative control for the check above: without the event wait the busy stream leaves the read stale."""
    me = _device_guard()
    _warm_up(me)
    me._verdict_copied = None

    _read_behind_a_busy_stream(me)
    torch.cuda.synchronize()

    assert me._nonfinite_steps_skipped == 0, "the read saw the NaN without waiting: the busy stream did not delay it"


def check_device_path_adds_no_host_sync():
    me = _device_guard()
    _warm_up(me)

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(2):
            _step_and_pass(me, 0.5)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert me._verdict_norm.is_pinned(), "the verdict buffer is pageable: its non-blocking copy stalls the host"
    assert me._unread_verdict is not None, "premise: the last step's verdict is left to a late read"
    me._read_device_verdict()
    assert me._nonfinite_steps_skipped == 0 and me.state.global_step == 4


def check_host_path_read_trips_the_sync_check():
    """Negative control for the check above: the host path's read of the norm is a synchronizing copy."""
    me = guarded(optimizer_cls=torch.optim.AdamW, device="cuda")
    for param in me.model.parameters():
        param.grad = torch.full_like(param, 0.5)
    norm = torch.nn.utils.clip_grad_norm_(me.model.parameters(), float("inf"))
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        me._get_grad_norm(me.model, grad_norm=norm)
    except RuntimeError as error:
        assert "synchronizing" in str(error), error
    else:
        raise AssertionError("the host path's read passed the sync check: the check is blind")
    finally:
        torch.cuda.set_sync_debug_mode("default")


@gpu_test_main(exact_world_size=1, prefix="test_step_guard_device_verdict", partial_state=False)
def run(ctx) -> dict:
    torch.cuda.set_device(ctx.device)
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    checks: dict[str, bool] = {}
    record_check(checks, "late_read_waits_for_its_copy", check_late_read_waits_for_its_copy)
    record_check(
        checks, "late_read_without_the_wait_reads_a_stale_norm", check_late_read_without_the_wait_reads_a_stale_norm
    )
    record_check(checks, "device_path_adds_no_host_sync", check_device_path_adds_no_host_sync)
    record_check(checks, "host_path_read_trips_the_sync_check", check_host_path_read_trips_the_sync_check)
    return {"checks": checks}


if __name__ == "__main__":
    run()
