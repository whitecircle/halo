"""``gpu_test_main``: the shared lifecycle for torchrun GPU tests, so a body is just load, train, assert. The body
returns ``{"checks": {name: bool}, "metrics": {...}}``. Exits 0 on pass, 1 on a failure or a raise, and 2 on a bad
launch."""

import contextlib
import functools
import os
import random
import sys
import traceback
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
from accelerate import PartialState

from halo.callbacks.efficiency import EfficiencyCallback
from halo.hardware import is_blackwell_gpu
from halo.log import configure_training_logging
from tests.common.distributed import (
    cleanup_dirs,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.memory import cleanup_memory
from tests.common.rank_log import log, log_all
from tests.common.reporting import emit_result, format_table, snapshot_efficiency

_EXIT_PASS = 0
_EXIT_FAIL = 1
_EXIT_BAD_LAUNCH = 2
# causal_conv1d's default backward sums conv weight gradients with atomics, so two identical steps can round apart and
# an exact replay diverges. Set unless the caller exported its own value.
_DETERMINISTIC_KERNEL_ENV = {"CAUSAL_CONV1D_DETERMINISTIC": "1"}


def _efficiency_callback(trainer) -> EfficiencyCallback | None:
    handler = getattr(trainer, "callback_handler", None)
    if handler is None:
        return None
    return next((cb for cb in handler.callbacks if isinstance(cb, EfficiencyCallback)), None)


class Ctx:
    def __init__(self, rank: int, world_size: int, local_rank: int, output_dir: str, cache_dir: str):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self._finalizers: list[Callable[[], None]] = []

    def on_teardown(self, fn: Callable[[], None]) -> None:
        """Register a finalizer (e.g. ``trainer.cleanup_ep``) run before teardown."""
        self._finalizers.append(fn)

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def broadcast_seed(self, seed: int = 42) -> int:
        """Seed torch/random identically on all ranks (rank 0's value wins); returns the shared seed."""
        t = torch.tensor([seed], device=self.device)
        if dist.is_initialized():
            dist.broadcast(t, src=0)
        shared = int(t.item())
        torch.manual_seed(shared)
        torch.cuda.manual_seed_all(shared)
        random.seed(shared)
        return shared

    def broadcast_checks(self, checks: dict[str, bool]) -> dict[str, bool]:
        """AND rank 0's checks into each rank's own, without masking another rank's failure. Rank-0-only checks would
        otherwise leave rank 0 exiting 1 while peers exit 0, which the launcher reports as a teardown race."""
        if not dist.is_initialized() or self.world_size == 1:
            return checks
        payload = [checks if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        merged = dict(checks)
        for name, ok in (payload[0] or {}).items():
            merged[name] = ok and merged.get(name, True)
        return merged

    def metrics(self, trainer_or_cb) -> dict:
        """Snapshot headline metrics from a trainer or EfficiencyCallback; ``{}`` if none is attached."""
        cb = trainer_or_cb if isinstance(trainer_or_cb, EfficiencyCallback) else _efficiency_callback(trainer_or_cb)
        return snapshot_efficiency(cb) if cb is not None else {}

    def _run_finalizers(self) -> None:
        for fn in reversed(self._finalizers):
            try:
                fn()
            except Exception:
                log(
                    f"finalizer {getattr(fn, '__name__', fn)} raised during teardown",
                )
                traceback.print_exc()


def record_check(checks: dict[str, bool], name: str, fn: Callable[[], None]) -> None:
    """Run ``fn`` and record its verdict under ``name`` instead of aborting the body, so later independent properties
    still report. Every raise inside ``fn`` must be rank-symmetric or come after its last collective, or a rank is left
    misaligned."""
    try:
        fn()
    except Exception as e:
        checks[name] = False
        log_all(f"  FAIL: {name}: {e}")
        traceback.print_exc()
    else:
        checks[name] = True
        log(f"  PASS: {name}")


@contextlib.contextmanager
def cuda_syncs_are_errors() -> Iterator[None]:
    """CUDA's sync debug mode at ``error`` over the body, so an op that synchronizes host and device raises instead of
    stalling."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def skip_unless_local_checkpoint(path: str, env_var: str) -> None:
    """Decline to run when a suite's local checkpoint is absent; the launcher reads exit 0 with a ``SKIP:`` line and no
    result as a skip."""
    if not os.path.isdir(path):
        log(f"SKIP: local model path missing: {path} (set {env_var} to a present checkpoint)")
        sys.exit(0)


def skip_unless_blackwell(what: str) -> None:
    """Decline to run on a GPU older than Blackwell, for a suite using FA4; the launcher reads exit 0 with a ``SKIP:``
    line as a skip."""
    if not is_blackwell_gpu():
        log(f"SKIP: {what} runs FlashAttention 4, which needs a Blackwell GPU")
        sys.exit(0)


def gpu_test_main(
    *,
    min_world_size: int = 1,
    exact_world_size: int | None = None,
    prefix: str = "halo_test",
    partial_state: bool = True,
    process_group: bool = True,
):
    """Wrap a ``run(ctx) -> dict`` body with the full GPU-test lifecycle. ``process_group`` off is for a
    ``launcher="python"`` driver, one process that starts its own ranks."""

    def decorator(run: Callable[["Ctx"], dict]) -> Callable[[], int]:
        @functools.wraps(run)
        def wrapper() -> int:
            configure_training_logging()
            for name, value in _DETERMINISTIC_KERNEL_ENV.items():
                os.environ.setdefault(name, value)
            rank, world_size, local_rank = init_distributed() if process_group else (0, 1, 0)

            if partial_state:
                PartialState()

            bad = None
            if exact_world_size is not None and world_size != exact_world_size:
                bad = f"requires exactly {exact_world_size} GPUs, launched with {world_size}"
            elif world_size < min_world_size:
                bad = f"requires >= {min_world_size} GPUs, launched with {world_size}"
            if bad is not None:
                if rank == 0:
                    log(f"BAD LAUNCH: {bad}")
                    emit_result("error", error=bad)
                teardown_distributed()
                sys.exit(_EXIT_BAD_LAUNCH)

            output_dir, cache_dir = setup_cache_dirs(prefix, rank)
            ctx = Ctx(rank, world_size, local_rank, output_dir, cache_dir)

            status, checks, metrics, err = "error", {}, {}, None
            try:
                result = run(ctx) or {}
                checks = dict(result.get("checks", {}))
                metrics = dict(result.get("metrics", {}))
                if not checks:
                    status, err = "error", "body returned no checks"
                else:
                    status = "pass" if all(checks.values()) else "fail"
            except Exception as e:
                status = "error"
                err = f"{type(e).__name__}: {e}"
                # Every rank reports its own failure; a non-zero rank's exception is otherwise invisible to the
                # launcher.
                log_all(f"\nFATAL: {err}")
                traceback.print_exc()
            finally:
                ctx._run_finalizers()
                try:
                    cleanup_memory()
                except Exception:
                    # A body that faulted the device faults this too, and raising here would drop the failed verdict.
                    # Only a pass must not survive it.
                    if status == "pass":
                        raise
                    log_all("cleanup_memory raised after the body failed")
                    traceback.print_exc()
                cleanup_dirs(output_dir, cache_dir)
                # Clean path only. A rank whose body raised abandoned a collective its peers are still inside, so the
                # barrier and group teardown would block until the watchdog timeout. Exiting straight away makes the
                # launcher reap the group.
                if status != "error":
                    ctx.barrier()
                    teardown_distributed()

            failed = [k for k, v in checks.items() if not v]
            # Every rank announces its own failures, else a disagreement looks like a teardown race. Printed directly:
            # the process group is already gone, so log_all would label every line "[Rank 0]".
            if rank != 0 and failed:
                print(f"[Rank {rank}] FAILED CHECKS: {failed} (metrics: {metrics})", flush=True)

            if rank == 0:
                if metrics:
                    log("\n--- metrics ---\n" + format_table(metrics))
                log("\n--- checks ---")
                for name, ok in checks.items():
                    log(f"  {name}: {'PASS' if ok else 'FAIL'}")
                log(f"\nRESULT: {status.upper()}" + (f" (failed: {failed})" if failed else ""))
                emit_result(status, checks=checks, metrics=metrics, error=err)

            # The script's ``run()`` relies on this exiting with the launcher-visible code.
            sys.exit(_EXIT_PASS if status == "pass" else _EXIT_FAIL)

        return wrapper

    return decorator
