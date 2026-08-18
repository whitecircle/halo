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
"""

import argparse
import json
import logging

from datasets import DatasetDict, load_dataset

from src.data.pipeline.processing import resolve_map_num_proc
from src.environments.envs.tasks.coding.datasets import CODE_DATASET_ADAPTERS
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)


# Adapters whose datasets load with a plain ``load_dataset`` (the RL training pools). Benchmark
# adapters with a custom ``load`` (LiveCodeBench, ICPC-Eval) are eval-only — they are scored directly
# by ``run_code_contests.py``, not materialized into a training set here.
_TRAINING_ADAPTERS = sorted(name for name, a in CODE_DATASET_ADAPTERS.items() if a.load is None)


def rating_in_bounds(row: dict, min_rating: int | None, max_rating: int | None) -> bool:
    """Whether ``row`` satisfies the requested rating bounds.

    A dataset with no ``rating`` column at all (deepcoder) ignores the bounds rather than filtering
    every row out. Where the column exists, an unrated row cannot be shown to satisfy a bound, so it
    is dropped — otherwise a rating-filtered pool silently carries problems of unknown difficulty.
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


def to_rl_row(row: dict, adapter) -> dict:
    """One source row as the environmental-GRPO contract: prompt + packed grading payload."""
    return {
        # Coerce to a stable string: a dataset without an "id" (e.g. deepcoder) would otherwise
        # emit an all-None column that Arrow types as `null`, breaking save_to_disk/push_to_hub.
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_dir and not args.push_to_hub:
        raise SystemExit("Provide --output_dir and/or --push_to_hub.")

    adapter = CODE_DATASET_ADAPTERS[args.adapter]
    ds = load_dataset(args.dataset, args.config)
    logger.info("Loaded %s (%s): %s", args.dataset, args.config, {k: len(v) for k, v in ds.items()})

    num_proc = resolve_map_num_proc(args.num_proc)
    keep_kwargs = {"adapter": adapter, "min_rating": args.min_rating, "max_rating": args.max_rating}

    prepared = {}
    for split, split_ds in ds.items():
        kept = split_ds.filter(keep_row, fn_kwargs=keep_kwargs, num_proc=num_proc, desc=f"filter[{split}]")
        mapped = kept.map(
            to_rl_row,
            fn_kwargs={"adapter": adapter},
            num_proc=num_proc,
            remove_columns=kept.column_names,
            desc=f"format[{split}]",
        )
        prepared[split] = mapped
        logger.info("  %s: %d -> %d kept", split, len(split_ds), len(mapped))

    prepared = DatasetDict(prepared)

    if args.output_dir:
        prepared.save_to_disk(args.output_dir)
        logger.info("Saved to %s", args.output_dir)
    if args.push_to_hub:
        prepared.push_to_hub(args.push_to_hub, private=args.private)
        logger.info("Pushed to hub: %s (private=%s)", args.push_to_hub, args.private)


if __name__ == "__main__":
    main()
