"""The code-contest pool preparation behind ``scripts/environments/preparation/prepare_code_dataset.py``: rows
normalized, filtered and joined with compacted suites, checked for a sound judge and constant-proof suite."""

import argparse
import functools
import json
import logging
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, disable_caching, load_dataset

from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD
from halo.data.pipeline.coordinated_map import resolve_map_num_proc
from halo.data.pipeline.rejected_rows import report_rejected_rows
from halo.environments.sandbox.factory import build_sandbox
from halo.environments.tasks.coding.contest_datasets import (
    CODE_DATASET_ADAPTERS,
    JOINED_CHECKER_FIELD,
    JOINED_TESTS_FIELD,
)
from halo.environments.tasks.coding.suite_compaction import parquet_parts
from halo.environments.tasks.coding.verdicts import build_verdict

logger = logging.getLogger(__name__)


# A sound judge must reject this: no answer at all.
_GARBAGE_OUTPUT = "\x00 garbage \x00\n"
# Answers a sound suite fails on at least one test: one every test accepts pays the problem's reward for no work.
CONSTANT_ANSWERS = ("0", "1", "-1", "YES", "NO", "Yes", "No", "IMPOSSIBLE", "Impossible", "")

# Codeforces rating stages: ``--push_bands`` publishes one Hub config per band, ``--holdout_per_band`` stratifies.
RATING_BANDS: dict[str, tuple[int, int]] = {"medium": (1500, 1999), "hard": (2000, 2599), "extra-hard": (2600, 3500)}
_FULL_CONFIG = "full"
HOLDOUT_SEED = 42


def rating_in_bounds(row: dict, min_rating: int | None, max_rating: int | None) -> bool:
    """A row with no ``rating`` key has nothing to bound. Where it exists an unrated row is dropped, so a
    rating-filtered pool carries no problems of unknown difficulty."""
    if "rating" not in row:
        return True
    rating = row["rating"]
    if min_rating is not None and (rating is None or rating < min_rating):
        return False
    return max_rating is None or (rating is not None and rating <= max_rating)


def keep_row(row: dict, adapter, min_rating: int | None, max_rating: int | None) -> bool:
    return adapter.keep(row) and rating_in_bounds(row, min_rating, max_rating)


def load_exclusions(path: str | None) -> tuple[set[str], tuple[str, ...]]:
    """Ids to drop, one per line; a line ending in ``*`` excludes that prefix. Blank and ``#`` lines are skipped."""
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
    return problem_id in exact or problem_id.startswith(prefixes)


def drop_listed_rows(split: Dataset, path: str, num_proc: int | None = None) -> Dataset:
    exact, prefixes = load_exclusions(path)
    kept = split.filter(lambda row: not excluded_by(str(row.get("id") or ""), exact, prefixes), num_proc=num_proc)
    report_rejected_rows(len(split), len(kept), f"drop {path}")
    return kept


def load_tests_table(path: str) -> tuple[Dataset, dict[str, int]]:
    """The table stays memory-mapped; only the keys are materialized. A duplicate key means a suite was split across
    parts, and the join would keep one half silently."""
    table = load_dataset("parquet", data_files=parquet_parts(path, non_empty=True), split="train")
    index: dict[str, int] = {}
    for position, key in enumerate(table["key"]):
        if key in index:
            raise SystemExit(f"tests table {path} carries {key!r} twice; rebuild it into an empty directory")
        index[key] = position
    return table, index


def join_suites(splits: dict[str, Dataset], table: Dataset, index: dict[str, int]) -> dict[str, Dataset]:
    """Join and log matches. A table may cover some splits only; one matching no row of any split is the wrong table,
    or keys spelled otherwise than the ids, and exits before any filtering."""
    joined: dict[str, Dataset] = {}
    for split, split_ds in splits.items():
        # Single process: the join indexes a memory-mapped table, and forked workers would copy it.
        joined[split] = split_ds.map(join_tests, fn_kwargs={"table": table, "index": index})
    matched = {split: len(ds) - ds.data.column(JOINED_TESTS_FIELD).null_count for split, ds in joined.items()}
    if not any(matched.values()):
        raise SystemExit("no row of any split matched a key of the tests table; the id spellings differ")
    for split, count in matched.items():
        logger.info("  %s: %d/%d rows matched a suite in the tests table", split, count, len(joined[split]))
    return joined


def join_tests(row: dict, table: Dataset, index: dict[str, int]) -> dict:
    """The compacted suite matched on the prepared ``id``, or nulls."""
    position = index.get(row.get("id") or "")
    if position is None:
        return {JOINED_TESTS_FIELD: None, JOINED_CHECKER_FIELD: None}
    entry = table[position]
    return {JOINED_TESTS_FIELD: entry["tests"], JOINED_CHECKER_FIELD: entry["checker"]}


@functools.cache
def _sandbox():
    return build_sandbox()


def checker_is_sound(row: dict) -> bool:
    """The special judge must accept every test's reference output and reject garbage on the first, else its problem is
    dropped. A sandbox failure propagates as :class:`CheckerInfraError` instead of dropping rows."""
    payload = json.loads(row[DEFAULT_ANSWER_FIELD])
    checker, tests = payload.get("checker"), payload.get("tests") or []
    if not checker or not tests:
        return True
    verdict = build_verdict(checker, "tokens", _sandbox())
    if verdict(tests[0].get("input", ""), tests[0].get("output", ""), _GARBAGE_OUTPUT):
        return False
    return all(verdict(test.get("input", ""), test.get("output", ""), test.get("output", "")) for test in tests)


def suite_rejects_constants(row: dict) -> bool:
    """Every constant answer must fail at least one test: a suite accepting one everywhere (a lax checker, one expected
    answer throughout) pays its reward to a program that never reads the input."""
    payload = json.loads(row[DEFAULT_ANSWER_FIELD])
    checker, tests = payload.get("checker"), payload.get("tests") or []
    if not tests:
        return True
    verdict = build_verdict(checker, "tokens", _sandbox() if checker else None)
    return not any(
        all(verdict(test.get("input", ""), test.get("output", ""), answer + "\n") for test in tests)
        for answer in CONSTANT_ANSWERS
    )


def band_of(rating: int | None) -> str | None:
    for name, (low, high) in RATING_BANDS.items():
        if rating is not None and low <= rating <= high:
            return name
    return None


def carve_holdout(train: Dataset, per_band: int, seed: int = HOLDOUT_SEED) -> tuple[Dataset, Dataset]:
    """``(train, test)`` with ``per_band`` rows from every band and from the rows below the first, drawn
    deterministically; a short band raises."""
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
    """``full`` plus one per rating band, every one over the same test split."""
    configs = {_FULL_CONFIG: prepared}
    for name, (low, high) in RATING_BANDS.items():
        band_train = prepared["train"].filter(lambda row, low=low, high=high: low <= (row["rating"] or 0) <= high)
        configs[name] = DatasetDict({"train": band_train, "test": prepared["test"]})
    return configs


def to_rl_row(row: dict, adapter) -> dict:
    return {
        # A dataset without an "id" (deepcoder) would emit an all-None column Arrow types as `null`, breaking
        # save_to_disk/push_to_hub.
        "id": str(row.get("id") or row.get("problem_id") or row.get("name") or ""),
        "prompt": adapter.format_prompt(row),
        DEFAULT_ANSWER_FIELD: json.dumps(adapter.pack_verification(row)),
        "rating": row.get("rating") or 0,
        "tags": row.get("tags") or [],
    }


def _stage_splits(ds: DatasetDict, adapter, args: argparse.Namespace, num_proc: int | None) -> dict[str, Dataset]:
    exact, prefixes = load_exclusions(args.exclude_keys)
    tests_table = load_tests_table(args.tests_table) if args.tests_table else None

    staged = {}
    for split, split_ds in ds.items():
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
        staged[split] = split_ds
    if tests_table is not None:
        staged = join_suites(staged, *tests_table)
    return staged


def _prepare_split(split: str, split_ds: Dataset, adapter, args: argparse.Namespace, num_proc: int | None) -> Dataset:
    """One staged split filtered, formatted as RL rows and passed through the suite checks, each drop reported."""
    keep_kwargs = {"adapter": adapter, "min_rating": args.min_rating, "max_rating": args.max_rating}
    kept = split_ds.filter(keep_row, fn_kwargs=keep_kwargs, num_proc=num_proc, desc=f"filter[{split}]")
    report_rejected_rows(len(split_ds), len(kept), f"filter[{split}]")
    if adapter.examples_only is not None and not args.include_examples_only:
        graded = kept.filter(
            lambda row: not adapter.is_examples_only(row), num_proc=num_proc, desc=f"examples_only[{split}]"
        )
        logger.info(
            "  %s: left out %d problems graded only on their statement's examples (--include_examples_only "
            "keeps them)",
            split,
            len(kept) - len(graded),
        )
        kept = graded
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
    if args.reject_constant_answers:
        strict = mapped.filter(suite_rejects_constants, num_proc=num_proc, desc=f"constants[{split}]")
        report_rejected_rows(len(mapped), len(strict), f"constants[{split}]")
        mapped = strict
    return mapped


def _settle_train_and_test(
    prepared: dict[str, Dataset], args: argparse.Namespace, num_proc: int | None
) -> DatasetDict:
    """The hold-out carved off a train-only source as its test split, then the train-only drops."""
    if args.holdout_per_band:
        if "test" in prepared:
            raise SystemExit("--holdout_per_band given, but the source already ships a test split")
        if "train" not in prepared:
            raise SystemExit(f"--holdout_per_band needs a train split; the source ships {sorted(prepared)}")
        prepared["train"], prepared["test"] = carve_holdout(prepared["train"], args.holdout_per_band)
        logger.info("  held out %d rows as the test split", len(prepared["test"]))
    if args.drop_train_keys:
        if "train" not in prepared:
            raise SystemExit(f"--drop_train_keys needs a train split; the source ships {sorted(prepared)}")
        # After the hold-out is carved, so dropping training rows never redraws the test split.
        prepared["train"] = drop_listed_rows(prepared["train"], args.drop_train_keys, num_proc)
    return DatasetDict(prepared)


def _publish(prepared: DatasetDict, args: argparse.Namespace) -> None:
    """Save and/or push, as one config or as the full config plus one per rating band."""
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


def _reject_unread_flags(args: argparse.Namespace, adapter) -> None:
    """Before the source downloads: each of these would run to the end and publish without it."""
    if not args.output_dir and not args.push_to_hub:
        raise SystemExit("Provide --output_dir and/or --push_to_hub.")
    for flag, value in (("--config_name", args.config_name), ("--push_bands", args.push_bands)):
        if value and not args.push_to_hub:
            raise SystemExit(f"{flag} names how --push_to_hub publishes, and this run pushes nothing")
    if args.push_bands and args.config_name:
        raise SystemExit("--push_bands names its own configs, one per rating band, so --config_name would be dropped")
    if args.include_examples_only and adapter.examples_only is None:
        raise SystemExit(f"--include_examples_only on --adapter {args.adapter}: it marks no problem examples-only")


def _reject_unusable_rating_bounds(args: argparse.Namespace, staged: dict[str, Dataset]) -> None:
    """On the staged rows, since an adapter's ``normalize`` may be what derives the rating."""
    if args.min_rating is None and args.max_rating is None:
        return
    if not any("rating" in split.column_names for split in staged.values()):
        raise SystemExit(f"--min_rating and --max_rating filter a 'rating' column, which {args.dataset} does not have")


def prepare_code_pool(args: argparse.Namespace) -> None:
    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    _reject_unread_flags(args, adapter)
    ds = load_dataset(args.dataset, args.config)
    logger.info("Loaded %s (%s): %s", args.dataset, args.config, {k: len(v) for k, v in ds.items()})

    # An adapter's code is not part of the map fingerprint (imported functions hash by name), so a cached step would
    # outlive an adapter change.
    disable_caching()
    num_proc = resolve_map_num_proc(args.num_proc)
    staged = _stage_splits(ds, adapter, args, num_proc)
    _reject_unusable_rating_bounds(args, staged)
    prepared = {}
    for split, split_ds in staged.items():
        prepared[split] = _prepare_split(split, split_ds, adapter, args, num_proc)
        logger.info("  %s: %d -> %d kept", split, len(ds[split]), len(prepared[split]))
    _publish(_settle_train_and_test(prepared, args, num_proc), args)
