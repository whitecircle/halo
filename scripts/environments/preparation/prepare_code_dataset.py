#!/usr/bin/env python
"""
Prepare a competitive-programming dataset for the coding RL environment
(``environment_type: code_contests`` / ``codeforces``).

Composes each problem's statement into a single prompt and packs its grading payload (tests +
optional special-judge checker + optional time limit) into one ``answer`` column, so the
Environmental-GRPO training script consumes it with ``prompt_field: prompt`` / ``answer_field:
answer``. Rows the stdin/stdout environment cannot grade are filtered out.

Prompt building, payload packing, and filtering come from the dataset adapters in
``src/environments/envs/tasks/coding/datasets.py`` (``CODE_DATASET_ADAPTERS``), so preparation,
evaluation, and training share one definition per dataset. Add an adapter there to support a new one.

Usage (inside the training image):
    # Codeforces (open-r1/codeforces), rating-filtered
    python scripts/environments/preparation/prepare_code_dataset.py \
        --adapter codeforces --dataset open-r1/codeforces --config verifiable \
        --output_dir "$HALO_DATA_ROOT/datasets/codeforces-verifiable-rl" --min_rating 800 --max_rating 2200

    # DeepCoder (agentica-org/DeepCoder-Preview-Dataset), the taco config
    python scripts/environments/preparation/prepare_code_dataset.py \
        --adapter deepcoder --dataset agentica-org/DeepCoder-Preview-Dataset --config taco \
        --output_dir "$HALO_DATA_ROOT/datasets/deepcoder-taco-rl"

    # HardTests, suites joined from a compacted tests table, 100 problems per rating band held out,
    # pushed as one Hub config per training stage (full + one per band) over that shared test split
    python scripts/environments/preparation/prepare_code_dataset.py \
        --adapter hardtests --dataset sigcp/hardtests_problems \
        --tests_table "$HALO_DATA_ROOT/s3_datasets/hardtests-tests-compact" --holdout_per_band 100 \
        --min_rating 800 --push_to_hub org/hardtests-rl --push_bands

A bulky test corpus (open-r1's generated tests, HardTests' suites) is first reduced by
``compact_code_tests.py`` to one capped row per problem; ``--tests_table`` joins it onto the source
rows by problem id, and the adapter packs the joined suite.
"""

import argparse
import functools
import json
import logging
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, disable_caching, load_dataset

from scripts.environments.preparation._common import parquet_parts
from src.data.pipeline.processing import report_rejected_rows, resolve_map_num_proc
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS
from src.environments.envs.tasks.coding.grading import select_verdict
from src.environments.sandbox.resolve import resolve_sandbox
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)


# What a sound judge must reject: not a wrong answer to any problem, but no answer at all.
_GARBAGE_OUTPUT = "\x00 garbage \x00\n"

# Training stages on the Codeforces rating scale: ``--push_bands`` publishes one Hub config per band
# over the pool's shared test split, and ``--holdout_per_band`` stratifies the hold-out by them (plus
# the rows below the first band).
RATING_BANDS: dict[str, tuple[int, int]] = {"medium": (1500, 1999), "hard": (2000, 2599), "extra-hard": (2600, 3500)}
_FULL_CONFIG = "full"
_HOLDOUT_SEED = 42

# Adapters whose datasets load with a plain ``load_dataset`` (the RL training pools). Benchmark
# adapters with a custom ``load`` (LiveCodeBench, ICPC-Eval) are eval-only: they are scored directly
# by ``run_code_contests.py`` rather than materialized into a training set here.
_TRAINING_ADAPTERS = sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.load is None)


def rating_in_bounds(row: dict, min_rating: int | None, max_rating: int | None) -> bool:
    """Whether ``row`` satisfies the requested rating bounds.

    A dataset with no ``rating`` column (deepcoder) ignores the bounds rather than filtering every row
    out. Where the column exists, an unrated row cannot be shown to satisfy a bound and is dropped,
    so a rating-filtered pool does not carry problems of unknown difficulty.
    """
    if "rating" not in row:
        return True
    rating = row["rating"]
    if min_rating is not None and (rating is None or rating < min_rating):
        return False
    return max_rating is None or (rating is not None and rating <= max_rating)


def keep_row(row: dict, adapter, min_rating: int | None, max_rating: int | None) -> bool:
    """Whether the adapter accepts ``row`` and it satisfies the rating bounds."""
    return adapter.keep(row) and rating_in_bounds(row, min_rating, max_rating)


def load_exclusions(path: str | None) -> tuple[set[str], tuple[str, ...]]:
    """Problem ids to drop, one per line; a line ending in ``*`` excludes every id with that prefix.
    Blank lines and ``#`` comments are ignored."""
    if not path:
        return set(), ()
    exact: set[str] = set()
    prefixes: list[str] = []
    for line in Path(path).read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.endswith("*"):
            prefixes.append(entry[:-1])
        else:
            exact.add(entry)
    return exact, tuple(prefixes)


def excluded_by(problem_id: str, exact: set[str], prefixes: tuple[str, ...]) -> bool:
    """Whether ``problem_id`` is listed exactly or under one of the prefixes."""
    return problem_id in exact or problem_id.startswith(prefixes)


def load_tests_table(path: str) -> tuple[Dataset, dict[str, int]]:
    """A compacted tests table (parquet parts written by ``compact_code_tests.py``) and its key index.
    The table stays memory-mapped; only the keys are materialized. A key present twice means a
    problem's suite was split across parts, and the join would keep one half silently."""
    table = load_dataset("parquet", data_files=parquet_parts(path, non_empty=True), split="train")
    index: dict[str, int] = {}
    for position, key in enumerate(table["key"]):
        if key in index:
            raise SystemExit(f"tests table {path} carries {key!r} twice; rebuild it into an empty directory")
        index[key] = position
    return table, index


def join_tests(row: dict, table: Dataset, index: dict[str, int]) -> dict:
    """The compacted suite for ``row`` (matched on its prepared ``id``), or nulls when it has none."""
    position = index.get(row.get("id") or "")
    if position is None:
        return {"joined_tests": None, "joined_checker": None}
    entry = table[position]
    return {"joined_tests": entry["tests"], "joined_checker": entry["checker"]}


@functools.cache
def _sandbox():
    return resolve_sandbox()


def checker_is_sound(row: dict) -> bool:
    """Whether a prepared row's special judge accepts the first test's own reference output and rejects
    garbage. A judge failing either grades every answer the same way, so its problem is dropped. A sandbox
    backend failure propagates as :class:`CheckerInfraError` instead of dropping rows."""
    payload = json.loads(row["answer"])
    checker, tests = payload.get("checker"), payload.get("tests") or []
    if not checker or not tests:
        return True
    verdict = select_verdict(checker, "tokens", _sandbox())
    test_input, expected = tests[0].get("input", ""), tests[0].get("output", "")
    return verdict(test_input, expected, expected) and not verdict(test_input, expected, _GARBAGE_OUTPUT)


def band_of(rating: int | None) -> str | None:
    """The :data:`RATING_BANDS` name a rating falls in, ``None`` when it lies outside every band."""
    for name, (low, high) in RATING_BANDS.items():
        if rating is not None and low <= rating <= high:
            return name
    return None


def carve_holdout(train: Dataset, per_band: int, seed: int = _HOLDOUT_SEED) -> tuple[Dataset, Dataset]:
    """Split ``train`` into ``(train, test)`` with ``per_band`` rows from every rating band, and from the
    rows below the first band, drawn deterministically. A band short of ``per_band`` rows raises."""
    buckets: dict[str, list[int]] = {}
    for position, rating in enumerate(train["rating"]):
        buckets.setdefault(band_of(rating) or "below", []).append(position)
    held: list[int] = []
    for name in ("below", *RATING_BANDS):
        positions = buckets.get(name, [])
        if len(positions) < per_band:
            raise SystemExit(f"band {name!r} has {len(positions)} rows, fewer than the {per_band} to hold out")
        held.extend(random.Random(seed).sample(positions, per_band))
    held_set = set(held)
    return train.select([i for i in range(len(train)) if i not in held_set]), train.select(sorted(held))


def band_configs(prepared: DatasetDict) -> dict[str, DatasetDict]:
    """The stage configs of a pool: ``full`` plus one per rating band, every one over the same test split."""
    configs = {_FULL_CONFIG: prepared}
    for name, (low, high) in RATING_BANDS.items():
        band_train = prepared["train"].filter(lambda row, low=low, high=high: low <= (row["rating"] or 0) <= high)
        configs[name] = DatasetDict({"train": band_train, "test": prepared["test"]})
    return configs


def to_rl_row(row: dict, adapter) -> dict:
    """One source row as the environmental-GRPO contract: prompt + packed grading payload."""
    return {
        # Coerce to a stable string: a dataset without an "id" (deepcoder) would otherwise emit an
        # all-None column that Arrow types as `null`, breaking save_to_disk/push_to_hub.
        "id": str(row.get("id") or row.get("problem_id") or row.get("name") or ""),
        "prompt": adapter.format_prompt(row),
        "answer": json.dumps(adapter.pack_verification(row)),
        "rating": row.get("rating") or 0,
        "tags": row.get("tags") or [],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a competitive-programming dataset for the coding RL env.")
    p.add_argument(
        "--adapter", required=True, choices=_TRAINING_ADAPTERS, help="Dataset adapter to apply (RL training pools)."
    )
    p.add_argument("--dataset", required=True, help="Source HF dataset id.")
    p.add_argument(
        "--config", default=None, help="Dataset config (e.g. 'verifiable' for codeforces, 'taco' for deepcoder)."
    )
    p.add_argument("--output_dir", default=None, help="save_to_disk path on a large mounted volume.")
    p.add_argument("--push_to_hub", default=None, help="Hub repo id to push the prepared dataset to.")
    p.add_argument(
        "--min_rating",
        type=int,
        default=None,
        help="Drop problems below this rating, and unrated ones with them. No-op on a dataset with no "
        "'rating' column (deepcoder).",
    )
    p.add_argument("--max_rating", type=int, default=None, help="Drop problems above this rating (see --min_rating).")
    p.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Processes for the map/filter passes (default: the toolkit's dataset-processing default).",
    )
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
        "--holdout_per_band",
        type=int,
        default=0,
        help="Hold out this many rows per rating band (and from the rows below the first band) as the test split "
        f"of a source that ships only a train split; deterministic (seed {_HOLDOUT_SEED}).",
    )
    p.add_argument(
        "--push_bands",
        action="store_true",
        help="With --push_to_hub: push the 'full' config plus one config per rating band "
        f"({', '.join(RATING_BANDS)}), each over the pool's shared test split.",
    )
    p.add_argument(
        "--verify_checkers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop problems whose special judge rejects its own reference output or accepts garbage (first test).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_dir and not args.push_to_hub:
        raise SystemExit("Provide --output_dir and/or --push_to_hub.")

    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    ds = load_dataset(args.dataset, args.config)
    logger.info("Loaded %s (%s): %s", args.dataset, args.config, {k: len(v) for k, v in ds.items()})

    # An adapter's code is not part of the map fingerprint (imported functions hash by name), so a
    # cached step would outlive the adapter change that should have rewritten it.
    disable_caching()
    num_proc = resolve_map_num_proc(args.num_proc)
    keep_kwargs = {"adapter": adapter, "min_rating": args.min_rating, "max_rating": args.max_rating}
    exact, prefixes = load_exclusions(args.exclude_keys)
    tests_table = load_tests_table(args.tests_table) if args.tests_table else None

    prepared = {}
    for split, split_ds in ds.items():
        original = len(split_ds)
        if adapter.normalize is not None:
            split_ds = split_ds.map(adapter.normalize, num_proc=num_proc, desc=f"normalize[{split}]")
        if exact or prefixes:
            excluded = split_ds.filter(
                lambda row: not excluded_by(str(row.get("id") or ""), exact, prefixes),
                num_proc=num_proc,
                desc=f"exclude[{split}]",
            )
            report_rejected_rows(len(split_ds), len(excluded), f"exclude[{split}]")
            split_ds = excluded
        if tests_table is not None:
            # Single process: the join indexes a memory-mapped table, and forked workers would copy it.
            split_ds = split_ds.map(join_tests, fn_kwargs={"table": tests_table[0], "index": tests_table[1]})
            matched = len(split_ds) - split_ds.data.column("joined_tests").null_count
            if matched == 0:
                raise SystemExit(f"no {split} row matched a key of the tests table; the id spellings differ")
            logger.info("  %s: %d/%d rows matched a suite in the tests table", split, matched, len(split_ds))
        kept = split_ds.filter(keep_row, fn_kwargs=keep_kwargs, num_proc=num_proc, desc=f"filter[{split}]")
        report_rejected_rows(len(split_ds), len(kept), f"filter[{split}]")
        mapped = kept.map(
            to_rl_row,
            fn_kwargs={"adapter": adapter},
            num_proc=num_proc,
            remove_columns=kept.column_names,
            desc=f"format[{split}]",
        )
        if args.verify_checkers:
            sound = mapped.filter(checker_is_sound, num_proc=num_proc, desc=f"verify[{split}]")
            report_rejected_rows(len(mapped), len(sound), f"verify[{split}]")
            mapped = sound
        prepared[split] = mapped
        logger.info("  %s: %d -> %d kept", split, original, len(mapped))

    if args.holdout_per_band:
        if "test" in prepared:
            raise SystemExit("--holdout_per_band given, but the source already ships a test split")
        if "train" not in prepared:
            raise SystemExit(f"--holdout_per_band needs a train split; the source ships {sorted(prepared)}")
        prepared["train"], prepared["test"] = carve_holdout(prepared["train"], args.holdout_per_band)
        logger.info("  held out %d rows as the test split", len(prepared["test"]))

    prepared = DatasetDict(prepared)

    if args.output_dir:
        prepared.save_to_disk(args.output_dir)
        logger.info("Saved to %s", args.output_dir)
    if args.push_to_hub:
        if args.push_bands:
            if "test" not in prepared:
                raise SystemExit("--push_bands needs a test split to share across the band configs")
            configs = band_configs(prepared)
        else:
            configs = {args.config_name: prepared}
        for name, config in configs.items():
            push_kwargs = {"private": args.private}
            if name:
                push_kwargs["config_name"] = name
            config.push_to_hub(args.push_to_hub, **push_kwargs)
            logger.info("Pushed to hub: %s config=%s (private=%s)", args.push_to_hub, name, args.private)


if __name__ == "__main__":
    main()
