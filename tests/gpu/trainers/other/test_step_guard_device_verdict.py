#!/usr/bin/env python
"""The non-finite step guard's late verdict on CUDA, where its copy is asynchronous and its read must not stall.

A read skipping the wait on the copy's event, or on an event from before the copy or on another stream, reads the
previous step's stale norm. The verdict buffer must be pinned too, since a non-blocking copy into pageable memory
stalls the host without tripping ``set_sync_debug_mode``.
"""

from types import SimpleNamespace

import torch

from halo.optimizers.adamw_bf16 import AdamWBF16
from tests.common.harness import cuda_syncs_are_errors, gpu_test_main, record_check
from tests.common.rank_log import log
from tests.common.step_guard import Guarded, guarded, guarded_step

# About a second on current GPUs, far longer than the host takes from queuing the NaN step to reading its verdict.
BUSY_CYCLES = 2**31


def _device_guard(dtype: torch.dtype = torch.bfloat16) -> Guarded:
    """A pass after each step logs, evaluates and saves nothing."""
    me = guarded(optimizer_cls=AdamWBF16, device="cuda", dtype=dtype)
    me.control = SimpleNamespace(should_log=False, should_evaluate=False, should_save=False)
    me._globalstep_last_logged = 0
    return me


def _step_and_pass(me: Guarded, grad: float) -> None:
    guarded_step(me, grad)
    me._maybe_log_save_evaluate(me._tr_loss)


def _warm_up(me: Guarded) -> None:
    """Two settled steps create the optimizer state, pinned buffers and compiled kernel the later steps reuse."""
    for _ in range(2):
        _step_and_pass(me, 0.5)
    torch.cuda.synchronize()
    me._read_device_verdict()


def _weights(me: Guarded) -> list[torch.Tensor]:
    return [param.detach().clone() for param in me.model.parameters()]


def _read_behind_a_busy_stream(me: Guarded) -> list[torch.Tensor]:
    """Queue a NaN step behind a busy stream and read its verdict at once; returns the weights before the step."""
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
    """Negative control: without the event wait the busy stream leaves the read stale."""
    me = _device_guard()
    _warm_up(me)
    me._verdict_copied = None

    _read_behind_a_busy_stream(me)
    torch.cuda.synchronize()

    assert me._nonfinite_steps_skipped == 0, "the read saw the NaN without waiting: the busy stream did not delay it"


def check_device_path_adds_no_host_sync():
    for dtype in (torch.bfloat16, torch.float32):
        me = _device_guard(dtype)
        _warm_up(me)

        with cuda_syncs_are_errors():
            for _ in range(2):
                _step_and_pass(me, 0.5)

        assert me._verdict_norm.is_pinned(), "the verdict buffer is pageable: its non-blocking copy stalls the host"
        assert me._unread_verdict is not None, f"premise ({dtype}): the last step's verdict is left to a late read"
        me._read_device_verdict()
        assert me._nonfinite_steps_skipped == 0 and me.state.global_step == 4, dtype


def check_host_path_read_trips_the_sync_check():
    """Negative control: the host path's read of the norm is a synchronizing copy."""
    me = guarded(optimizer_cls=torch.optim.AdamW, device="cuda")
    for param in me.model.parameters():
        param.grad = torch.full_like(param, 0.5)
    norm = torch.nn.utils.clip_grad_norm_(me.model.parameters(), float("inf"))
    torch.cuda.synchronize()

    with cuda_syncs_are_errors():
        try:
            me._get_grad_norm(me.model, grad_norm=norm)
        except RuntimeError as error:
            assert "synchronizing" in str(error), error
        else:
            raise AssertionError("the host path's read passed the sync check: the check is blind")


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
