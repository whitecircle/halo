#!/usr/bin/env python
"""
Prepare a competitive-programming dataset for the coding RL environment
(``environment_type: code_contests`` / ``codeforces``).

Composes each problem's statement into a single prompt and packs its grading payload (tests +
optional special-judge checker + optional time limit) into one ``answer`` column, so the
Environmental-GRPO training script consumes it with ``prompt_field: prompt`` / ``answer_field:
answer``. Rows the stdin/stdout environment cannot grade are filtered out.

Prompt building, payload packing, and filtering come from the dataset adapters in
``halo/environments/tasks/coding/contest_datasets.py`` (``CODE_DATASET_ADAPTERS``), so preparation,
evaluation, and training share one definition per dataset. Add an adapter there to support a new one.

Usage (inside the training image):
    # Codeforces (open-r1/codeforces), rating-filtered
    python scripts/environments/preparation/prepare_code_dataset.py \\
        --adapter codeforces --dataset open-r1/codeforces --config verifiable \\
        --output_dir "$HALO_DATA_ROOT/datasets/codeforces-verifiable-rl" --min_rating 800 --max_rating 2200

    # DeepCoder (agentica-org/DeepCoder-Preview-Dataset), the taco config
    python scripts/environments/preparation/prepare_code_dataset.py \\
        --adapter deepcoder --dataset agentica-org/DeepCoder-Preview-Dataset --config taco \\
        --output_dir "$HALO_DATA_ROOT/datasets/deepcoder-taco-rl"

    # HardTests, suites joined from a compacted tests table, 100 problems per rating band held out,
    # pushed as one Hub config per training stage (full + one per band) over that shared test split
    python scripts/environments/preparation/prepare_code_dataset.py \\
        --adapter hardtests --dataset sigcp/hardtests_problems \\
        --tests_table "$HALO_DATA_ROOT/hardtests-tests-compact" --holdout_per_band 100 \\
        --min_rating 800 --push_to_hub org/hardtests-rl --push_bands

A bulky test corpus (open-r1's generated tests, HardTests' suites) is first reduced by
``compact_code_tests.py`` to one capped row per problem; ``--tests_table`` joins it onto the source
rows by problem id, and the adapter packs the joined suite. A table may cover some splits only; one
matching no row of any split exits.

A problem whose only tests are its statement's examples (a Codeforces row whose official tests are
the statement's samples and that no joined suite covers) is dropped and counted per split unless
``--include_examples_only``: a program printing a constant passes many of them.
"""

import argparse

from halo.entrypoints.code_pool_preparation import HOLDOUT_SEED, RATING_BANDS, prepare_code_pool
from halo.environments.tasks.coding.contest_datasets import CODE_DATASET_ADAPTERS
from halo.log import configure_cli_logging

# Adapters whose datasets load with a plain ``load_dataset`` (the RL training pools). Benchmark
# adapters with a custom ``load`` (LiveCodeBench, ICPC-Eval) are eval-only: they are scored directly
# by ``run_code_contests.py`` rather than materialized into a training set here.
_TRAINING_ADAPTERS = sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.load is None)


def _add_source_args(p: argparse.ArgumentParser) -> None:
    """The adapter, the source dataset, the joined tests table, the excluded ids and the map processes."""
    p.add_argument(
        "--adapter", required=True, choices=_TRAINING_ADAPTERS, help="Dataset adapter to apply (RL training pools)."
    )
    p.add_argument("--dataset", required=True, help="Source HF dataset id.")
    p.add_argument(
        "--config", default=None, help="Dataset config (e.g. 'verifiable' for codeforces, 'taco' for deepcoder)."
    )
    p.add_argument(
        "--tests_table",
        default=None,
        help="Directory of compacted per-problem test suites (compact_code_tests.py) joined onto the rows by id.",
    )
    p.add_argument(
        "--exclude_keys",
        default=None,
        help="File of problem ids to drop (one per line; a trailing * excludes a prefix): held-out and benchmark ids.",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Processes for the map/filter passes (default: the toolkit's dataset-processing default).",
    )


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    """The rating bounds and the suite checks a kept problem must pass."""
    p.add_argument(
        "--min_rating",
        type=int,
        default=None,
        help="Drop problems below this rating, and unrated ones with them. No-op on a dataset with no "
        "'rating' column (deepcoder).",
    )
    p.add_argument("--max_rating", type=int, default=None, help="Drop problems above this rating (see --min_rating).")
    p.add_argument(
        "--verify_checkers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop problems whose special judge rejects a test's own reference output, or accepts garbage on the first.",
    )
    p.add_argument(
        "--reject_constant_answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop problems whose suite accepts one constant answer (0, 1, -1, YES, NO, ...) on every test.",
    )
    p.add_argument(
        "--include_examples_only",
        action="store_true",
        help="Keep problems whose only tests are the statement's examples (a codeforces row whose official tests "
        "are the statement's samples and that no --tests_table suite covers), which a program printing a "
        "constant passes for many. Default: dropped and counted per split.",
    )


def _add_split_args(p: argparse.ArgumentParser) -> None:
    """The hold-out carved as the test split and the train-only drops after it."""
    p.add_argument(
        "--holdout_per_band",
        type=int,
        default=0,
        help="Hold out this many rows per rating band (and from the rows below the first band) as the test split "
        f"of a source that ships only a train split; deterministic (seed {HOLDOUT_SEED}).",
    )
    p.add_argument(
        "--drop_train_keys",
        default=None,
        help="File of problem ids (same format as --exclude_keys) dropped from the train split only, after the "
        "hold-out is carved, so the test split stays the same: known defects such as "
        "scripts/environments/preparation/hardtests_train_exclusions.txt.",
    )


def _add_publish_args(p: argparse.ArgumentParser) -> None:
    """Where the prepared pool goes: a local save, a Hub push, and its configs."""
    p.add_argument("--output_dir", default=None, help="save_to_disk path on a large mounted volume.")
    p.add_argument("--push_to_hub", default=None, help="Hub repo id to push the prepared dataset to.")
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push --push_to_hub as a private dataset (default: True).",
    )
    p.add_argument(
        "--config_name",
        default=None,
        help="Hub config to push as (one repo can hold one config per rating band); default config otherwise.",
    )
    p.add_argument(
        "--push_bands",
        action="store_true",
        help="With --push_to_hub: push the 'full' config plus one config per rating band "
        f"({', '.join(RATING_BANDS)}), each over the pool's shared test split.",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a competitive-programming dataset for the coding RL env.")
    _add_source_args(p)
    _add_filter_args(p)
    _add_split_args(p)
    _add_publish_args(p)
    return p.parse_args()


def main() -> None:
    configure_cli_logging()
    prepare_code_pool(parse_args())


if __name__ == "__main__":
    main()
