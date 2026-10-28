#!/usr/bin/env python
"""
Offline re-grading of recorded code-contest trajectories — contention-free and reproducible.

Online/large-scale evaluation generates trajectories with many concurrent rollout workers per process
and several processes per host. Grading runs untrusted solutions in subprocesses, so at extreme
aggregate concurrency the *wall-clock* a solution is measured against is inflated by host load — a
correct, fast solution can be spuriously TIME-LIMIT-EXCEEDED. The robust pattern is to **decouple
generation from grading**: generate in parallel (cheap, API-bound), then re-grade the recorded
submissions here in a single process with bounded concurrency, where each run gets an effectively
dedicated core and the wall-clock limit is real.

This reads the JSONL files written by ``write_trajectories_jsonl`` (a ``{"type": "meta", ...}`` line
then one episode per line), rebuilds each problem's hidden tests from its dataset by index — the same
order and contest selection ``build_examples`` used — and re-runs every recorded ``submit_solution``
through the *same* ``grade_solution`` the environment uses, reproducing the env's comparison /
checker / time-limit exactly. It reports, over the episodes that carry a verdict, the fraction whose
first admitted submission passes every test (``s@1``) and whose any submission within the episode's
budget does (``s@2``); an episode recorded with a ``generation_error`` has no verdict to re-grade and is
counted apart.

Usage::

    python scripts/environments/inference/regrade_trajectories.py /path/to/trajectories/*.jsonl \\
        --max_workers 64 --output_path /path/to/regraded.jsonl
"""

import argparse
import os

from halo.entrypoints.code_contests_regrade import regrade_trajectory_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trajectories", nargs="+", help="Trajectory JSONL file(s) to re-grade.")
    p.add_argument(
        "--max_workers",
        type=int,
        default=min(64, (os.cpu_count() or 8)),
        help="Concurrent gradings in this single process. Keep at/below the core count so each run "
        "gets a dedicated core and the wall-clock limit stays real.",
    )
    p.add_argument("--output_path", help="Optional JSONL file to append the corrected per-file metrics to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    regrade_trajectory_files(args.trajectories, args.max_workers, args.output_path)


if __name__ == "__main__":
    main()
