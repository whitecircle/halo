"""Attach py-spy to a running training job: all-rank stack dumps and CPU flame graphs.

Out-of-process CLI over the :mod:`src.diagnostics.debugging` helpers. It needs no profiling flags at
launch time, so it works on a job that is already running or already hung. Discovers this node's
torchrun training ranks (or takes explicit ``--pid``) and either:

  - ``dump``   — instantaneous py-spy stack dump of every rank, for hang triage: one hung collective
    shows as N-1 ranks inside NCCL and one rank elsewhere,
  - ``record`` — per-rank CPU flame-graph SVGs sampled over ``--duration`` seconds, for dataloader
    stalls, tokenization and Python overhead between kernels.

GPU-side traces come from ``TorchProfilerCallback`` (``enable_torch_profiler: true``); this covers
the host side. Attach from the same pid namespace as the job: for a containerized run, ``docker
exec`` into the training container first.

Usage:
  python scripts/profiling/py_spy_diag.py dump
  python scripts/profiling/py_spy_diag.py record --duration 30
  python scripts/profiling/py_spy_diag.py record --pid 1234 --pid 1235 --native
"""

from __future__ import annotations

import argparse
import logging
import os

from src.diagnostics.debugging import (
    FLAMEGRAPH_DIR,
    STACK_DUMP_DIR,
    dump_distributed_stacks,
    record_distributed_flamegraph,
    torchrun_python_pids,
)
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shared options live on a parent parser so they are accepted after the subcommand (the
    # docstring's `record --pid N` form); options on `parser` itself would not be.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--pid",
        action="append",
        type=int,
        default=None,
        help="Explicit pid to target (repeatable). Default: this node's torchrun training ranks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser(
        "dump", parents=[common], help="Instantaneous stack dump of every rank (hang triage)."
    )
    dump.add_argument("--output-dir", default=STACK_DUMP_DIR, help="Default: %(default)s")

    record = subparsers.add_parser(
        "record", parents=[common], help="Per-rank CPU flame-graph SVGs over a sampling window."
    )
    record.add_argument("--output-dir", default=FLAMEGRAPH_DIR, help="Default: %(default)s")
    record.add_argument("--duration", type=int, default=30, help="Sampling window in seconds (default: %(default)s).")
    record.add_argument("--rate", type=int, default=100, help="Samples per second (default: %(default)s).")
    record.add_argument("--native", action="store_true", help="Also sample native (C/CUDA-runtime) frames.")

    args = parser.parse_args()
    pids = set(args.pid) if args.pid else torchrun_python_pids()
    pids.discard(os.getpid())  # never sample this CLI itself
    if not pids:
        parser.error("no torchrun training processes found on this node — pass --pid explicitly")

    if args.command == "dump":
        target = dump_distributed_stacks(args.output_dir, pids=pids)
    else:
        logger.info(f"Sampling {len(pids)} process(es) for {args.duration}s…")
        target = record_distributed_flamegraph(
            args.output_dir, duration=args.duration, rate=args.rate, native=args.native, pids=pids
        )

    if target is None:
        parser.error("py-spy is not installed (profiling dependency group) or not on PATH")
    logger.info(f"{args.command} artifacts for pids {sorted(pids)} → {target}")


if __name__ == "__main__":
    main()
