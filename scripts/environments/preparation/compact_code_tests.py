#!/usr/bin/env python
"""Compact a bulky test-suite corpus into one row per problem, capped, for ``prepare_code_dataset.py``.

Two sources: HardTests' encoded suites (``sigcp/hardtests_tests``, one row per problem, tests
base64-zlib-pickled) and open-r1/codeforces' generated tests (``generated_tests/*.parquet``, one row
per test, one shard per contest). Both are hundreds of gigabytes; the output is a directory of
parquet parts with ``key``, ``tests`` (a JSON list of ``{input, output}``) and ``checker``, read back
by ``prepare_code_dataset.py --tests_table``.

Caps keep a problem's payload trainable: at most ``--max_tests`` tests within ``--max_test_bytes``
each, of which up to ``--max_large_tests`` may reach ``--max_large_bytes`` so the suite keeps a
maximum-size input for time-limit discrimination. Order is preserved: HardTests' groups are taken as
LLM-written samples, then adversarial, then random, then special inputs; open-r1's generated tests
already come hardest-first. A part is named after its source shard and stamped with the caps that
built it, so an interrupted run resumes and a run with other caps rebuilds.

    python scripts/environments/preparation/compact_code_tests.py --source hardtests \\
        --input_dir "$HF_HOME/hub/datasets--sigcp--hardtests_tests/snapshots/<rev>/data" \\
        --output_dir "$HALO_DATA_ROOT/hardtests-tests-compact"
"""

import argparse

from halo.environments.tasks.coding.suite_compaction import CAP_DEFAULTS, COMPACTORS, compact_tests_table
from halo.log import configure_cli_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact a test-suite corpus into capped per-problem rows.")
    p.add_argument(
        "--source",
        required=True,
        choices=sorted(COMPACTORS),
        help="Corpus layout: hardtests (base64-zlib-pickled suites) or codeforces_generated (one row per test).",
    )
    p.add_argument("--input_dir", required=True, help="Directory of the source parquet shards.")
    p.add_argument("--output_dir", required=True, help="Directory for the compacted parquet parts.")
    p.add_argument(
        "--max_tests",
        type=int,
        default=CAP_DEFAULTS["max_tests"],
        help="Tests kept per problem (default: %(default)s).",
    )
    p.add_argument(
        "--max_test_bytes",
        type=int,
        default=CAP_DEFAULTS["max_test_bytes"],
        help="Largest ordinary test kept, in bytes (default: %(default)s).",
    )
    p.add_argument(
        "--max_large_tests",
        type=int,
        default=CAP_DEFAULTS["max_large_tests"],
        help="Tests per problem allowed past --max_test_bytes (default: %(default)s).",
    )
    p.add_argument(
        "--max_large_bytes",
        type=int,
        default=CAP_DEFAULTS["max_large_bytes"],
        help="Largest of those large tests, in bytes (default: %(default)s).",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Worker processes, one source shard each (default: the toolkit's dataset-processing default).",
    )
    return p.parse_args()


def main() -> None:
    configure_cli_logging()
    args = parse_args()
    caps = {name: getattr(args, name) for name in CAP_DEFAULTS}
    compact_tests_table(args.source, args.input_dir, args.output_dir, caps, args.num_proc)


if __name__ == "__main__":
    main()
