"""``gpu_test_main``: the shared lifecycle for torchrun-native GPU tests.

Covers the lifecycle every torchrun test needs: ``init_distributed`` →
``PartialState`` → validate world size → ``setup_cache_dirs`` → ``try`` body →
``finally`` (``cleanup_ep`` → ``cleanup_memory`` → ``cleanup_dirs`` → ``barrier``
→ ``teardown_distributed``) → ``sys.exit``. Hand-rolled copies drift (a skipped
barrier, a dir leaked on the error path, a missing ``cleanup_ep``), so with the
decorator handling it a test body is load, train, assert.

    from tests.common.harness import gpu_test_main

    @gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_dense")
    def run(ctx):
        pc = ParallelismConfig(world_size=ctx.world_size, ...)
        model, tok = load_distributed_model(..., parallelism_config=pc)
        trainer = DistributedSFTTrainer(..., parallelism_config=pc)
        ctx.on_teardown(trainer.cleanup_ep)        # finalizer the decorator can't reach
        trainer.train()
        return {
            "checks": {"loss_finite": ..., "loss_decreased": ...},
            "metrics": ctx.metrics(trainer),       # headline tok/s/GPU + peak mem
        }

    if __name__ == "__main__":
        run()                                       # decorator calls sys.exit itself

The body returns ``{"checks": {name: bool}, "metrics": {...}}``. The decorator
computes ``all(checks.values())``, emits the machine-readable result line
(:mod:`tests.common.reporting`), guarantees teardown on every exit path, and
exits ``0`` (pass) / ``1`` (fail or body raised) / ``2`` (bad launch: wrong world
size, reported as an infra error rather than a test failure).

A decorator rather than a bare context manager because world size is validated
per test, ``--cp/--ep/--tp`` change the required ``nproc``, and ``cleanup_ep`` is
trainer-scoped; none of that is reachable from a plain ``with`` block, and all of
it is handled here via ``ctx``.
"""

import functools
import random
import sys
import traceback
from collections.abc import Callable

import torch
import torch.distributed as dist

from tests.common.distributed import (
    cleanup_dirs,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.reporting import (
    emit_result,
    extract_efficiency_callback,
    format_table,
    snapshot_efficiency,
)
from tests.common.utils import cleanup_memory, log, log_all

# Exit codes the launcher keys on.
_EXIT_PASS = 0
_EXIT_FAIL = 1
_EXIT_BAD_LAUNCH = 2


class Ctx:
    """Per-run context handed to a ``gpu_test_main`` body."""

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
        """Seed torch/random identically on all ranks (rank-0 value wins).

        Use when every rank must generate the same data (a parallel-mode run
        compared against a single-GPU reference). Returns the shared seed.
        """
        t = torch.tensor([seed], device=self.device)
        if dist.is_initialized():
            dist.broadcast(t, src=0)
        shared = int(t.item())
        torch.manual_seed(shared)
        torch.cuda.manual_seed_all(shared)
        random.seed(shared)
        return shared

    def broadcast_checks(self, checks: dict[str, bool]) -> dict[str, bool]:
        """Share rank 0's verdict with every rank, without masking another rank's own failure.

        Checks only rank 0 can make (a served model's response, an HTTP probe) are missing on the
        other ranks, and the harness exits per rank, so a server-side failure would leave rank 0
        exiting 1 while its peers exit 0, which the launcher reports as a teardown race. Rank 0's
        entries are AND-ed into the local dict rather than replacing it, so a check that failed only
        on rank 1 survives the merge.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return checks
        payload = [checks if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        merged = dict(checks)
        for name, ok in (payload[0] or {}).items():
            merged[name] = ok and merged.get(name, True)
        return merged

    def metrics(self, trainer_or_cb) -> dict:
        """Snapshot headline metrics from a trainer (or an EfficiencyCallback).

        Returns ``{}`` if no ``EfficiencyCallback`` is attached; metrics are
        optional and correctness tests can omit them.
        """
        from src.callbacks.efficiency import EfficiencyCallback

        cb = (
            trainer_or_cb
            if isinstance(trainer_or_cb, EfficiencyCallback)
            else extract_efficiency_callback(trainer_or_cb)
        )
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
    """Run ``fn`` and record its verdict under ``name`` instead of aborting the body.

    For a suite that asserts many independent properties in one launch: a raise would end the body at
    the first failure and the harness would report a single error, hiding every later property.
    Recording keeps each verdict in the dict ``gpu_test_main`` reports and exits on.

    Every ``fn`` therefore has to be rank-symmetric and collective-free: continuing past a failure is
    only safe while no later check enters a collective the failed rank's peers would be misaligned on,
    which is why ``gpu_test_main`` skips its own barrier for a body that raised.
    """
    try:
        fn()
    except Exception as e:
        checks[name] = False
        log_all(f"  FAIL: {name}: {e}")
        traceback.print_exc()
    else:
        checks[name] = True
        log(f"  PASS: {name}")


def gpu_test_main(
    *,
    min_world_size: int = 1,
    exact_world_size: int | None = None,
    prefix: str = "halo_test",
    partial_state: bool = True,
):
    """Wrap a ``run(ctx) -> dict`` body with the full GPU-test lifecycle.

    Args:
        min_world_size: minimum GPUs the test needs; a smaller launch exits 2,
            reported as an infra error rather than a test failure.
        exact_world_size: if set, the launch must provide exactly this many GPUs.
        prefix: temp-dir prefix for this test's isolated output/cache dirs.
        partial_state: construct ``accelerate.PartialState()`` (needed by the
            Trainer-based tests; off for pure-kernel tests).
    """

    def decorator(run: Callable[["Ctx"], dict]) -> Callable[[], int]:
        @functools.wraps(run)
        def wrapper() -> int:
            rank, world_size, local_rank = init_distributed()

            if partial_state:
                from accelerate import PartialState

                PartialState()

            # ── Validate the launch before allocating anything ──────────────
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
                # Every rank reports its own failure. Under rank-0-only reporting a non-zero rank's
                # exception is invisible: the launcher shows exit code 1 with no message and no
                # traceback, indistinguishable from a teardown race.
                log_all(f"\nFATAL: {err}")
                traceback.print_exc()
            finally:
                ctx._run_finalizers()
                cleanup_memory()
                cleanup_dirs(output_dir, cache_dir)
                # Clean path only. A rank whose body raised has abandoned a collective its peers are
                # still inside, so both the barrier and the NCCL group teardown block until the
                # watchdog timeout, and the job hangs and reports an infra error instead of this
                # rank's traceback. Exiting immediately lets the launcher reap the group, so a
                # single-rank failure stays a failure.
                if status != "error":
                    ctx.barrier()
                    teardown_distributed()

            failed = [k for k, v in checks.items() if not v]
            # A non-zero rank whose checks disagree with rank 0's exits non-zero while rank 0 prints
            # RESULT: PASS, and reporting only from rank 0 would make that look like a teardown race
            # with no diagnosable cause, so every rank announces its own failures.
            # Printed directly rather than via log_all: the process group is torn down by here, so
            # log_all cannot read the rank and would label every line "[Rank 0]".
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

            # The script's `if __name__ == "__main__": run()` relies on this
            # exiting the process with the launcher-visible code.
            sys.exit(_EXIT_PASS if status == "pass" else _EXIT_FAIL)

        return wrapper

    return decorator
