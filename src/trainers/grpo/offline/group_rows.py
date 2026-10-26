"""Offline GRPO's dataset rows: each stored ``(prompt, completions, rewards)`` group becomes one tokenized training
row per completion, carrying its group-relative advantage."""

import logging
from typing import Any

from src.data.pipeline.coordinated_map import coordinated_map, resolve_map_num_proc
from src.data.pipeline.prompt_completion import tokenize_prompt_completions
from src.data.turn_markers import resolve_eos_token_ids
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.runtime.ranks import is_global_main_process
from src.trainers.grpo.objective.advantages import compute_group_advantages

logger = logging.getLogger(__name__)


def tokenize_offline_grpo_rows(
    batch: dict[str, list],
    indices: list[int],
    *,
    processing_class: Any,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    advantage_method: str,
    best_completion_emphasis: float | str,
    eos_token_ids: frozenset[int] = frozenset(),
    drop_degenerate_groups: bool = False,
) -> dict[str, list]:
    """Tokenize a batch, expanding each ``(prompt, completions, rewards)`` group into one training row per
    completion with its computed advantage and group metadata."""
    tokenizer = resolve_tokenizer(processing_class)
    columns = {
        name: [] for name in ("prompt_input_ids", "completion_input_ids", "group_id", "group_size", "advantage")
    }
    for prompt, completions_list, rewards_list, idx in zip(
        batch["prompt"], batch["completions"], batch["rewards"], indices, strict=True
    ):
        if len(completions_list) != len(rewards_list):
            raise ValueError(
                f"Offline GRPO row {idx}: {len(completions_list)} completions but "
                f"{len(rewards_list)} rewards — each completion needs exactly one reward "
                f"(silent zip truncation would train on mispaired advantages and record a "
                f"wrong group_size)."
            )
        # Dropped at tokenization rather than in the loss, which also skips those rows' forwards and holds
        # across every mode, PP's metadata denominators included.
        if drop_degenerate_groups and (len(rewards_list) < 2 or max(rewards_list) == min(rewards_list)):
            continue
        advantages_list = compute_group_advantages(rewards_list, advantage_method, best_completion_emphasis)

        prompt_ids, completion_ids_list = tokenize_prompt_completions(
            prompt,
            completions_list,
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            truncation_mode="keep_end",
            joint_boundary=False,
            eos_token_ids=eos_token_ids,
        )
        # Raised here, inside the coordinated map, so every rank aborts: the collator is the one
        # rank that drew the row, and a raise there strands its peers in the next collective.
        if completions_list and not prompt_ids:
            raise ValueError(
                f"Offline GRPO row {idx}: the prompt tokenizes to no tokens, so its completions "
                f"would train against no context. Fix or drop the row."
            )
        for completion_ids, advantage in zip(completion_ids_list, advantages_list, strict=True):
            columns["prompt_input_ids"].append(prompt_ids)
            columns["completion_input_ids"].append(completion_ids)
            columns["group_id"].append(idx)
            columns["group_size"].append(len(completions_list))
            columns["advantage"].append(advantage)
    return columns


def tokenize_offline_splits(train_dataset, eval_dataset, args, processing_class, model_config):
    """``(train, eval)`` tokenized and expanded per completion; ``eval_dataset`` may be ``None``."""
    num_proc = resolve_map_num_proc(args.dataset_num_proc)
    # In fn_kwargs, not closed over: the map fn must stay picklable, and the cache key fingerprints fn_kwargs.
    tokenize_kwargs = {
        "processing_class": processing_class,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "advantage_method": args.advantage_method,
        "best_completion_emphasis": args.best_completion_emphasis,
        "eos_token_ids": resolve_eos_token_ids(resolve_tokenizer(processing_class), model_config),
        "drop_degenerate_groups": args.drop_degenerate_groups,
    }

    def tokenize_split(dataset, split: str):
        num_groups_in = len(dataset) if args.drop_degenerate_groups else 0
        tokenized = coordinated_map(
            dataset,
            tokenize_offline_grpo_rows,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing and expanding {split} dataset",
            batched=True,
            batch_size=32,
            with_indices=True,
            fn_kwargs=tokenize_kwargs,
        )
        verify_degenerate_drop(args, tokenized, num_groups_in, split)
        return tokenized

    train_dataset = tokenize_split(train_dataset, "training")
    if eval_dataset is not None:
        eval_dataset = tokenize_split(eval_dataset, "evaluation")
    return train_dataset, eval_dataset


def verify_degenerate_drop(args, dataset, num_groups_in: int, what: str) -> None:
    """Report ``drop_degenerate_groups``' effect and refuse an all-degenerate dataset, which would otherwise
    surface steps later as an opaque dataloader error and reads the same as a mis-scored dataset."""
    if not args.drop_degenerate_groups:
        return
    num_groups_kept = len(set(dataset["group_id"])) if len(dataset) else 0
    if is_global_main_process():
        logger.info(
            f"drop_degenerate_groups: kept {num_groups_kept}/{num_groups_in} {what} groups "
            f"(dropped groups scored all-alike or held < 2 completions)"
        )
    if num_groups_kept == 0:
        raise ValueError(
            f"drop_degenerate_groups removed every {what} group: all groups scored all-alike "
            f"rewards or held fewer than two completions. Such data carries no group-relative "
            f"signal — fix the rewards (or disable drop_degenerate_groups to train on "
            f"zero-advantage rows anyway)."
        )
