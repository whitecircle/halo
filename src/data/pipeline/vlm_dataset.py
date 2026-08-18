"""Raw-VLM dataset preparation: the training row map, its pinned Arrow schema and the over-length
filter the SFT and distillation scripts share.
"""

import datasets
from accelerate.logging import get_logger
from datasets.features import Json

from src.data.pipeline.processing import coordinated_filter, coordinated_map, dataset_total_size, require_render_column
from src.data.pipeline.row_processors import create_vlm_processor
from src.data.probe_consensus import agree_probe_across_ranks
from src.data.spans import require_response_marker
from src.data.vlm import render_vlm_text, vlm_row_tools
from src.distributed.runtime import is_global_main_process

logger = get_logger(__name__, log_level="INFO")


def vlm_map_features() -> datasets.Features:
    """Explicit Arrow schema for the raw VLM map output (``create_vlm_processor`` rows).

    Shard-wise type inference is unstable on mixed datasets: an all-text shard infers message
    ``content`` as ``List(struct)`` while an image shard infers ``Json`` (image parts have no
    ``text`` key), and an all-empty ``images`` shard infers ``List(null)``, so multiprocess maps fail
    to align shard schemas. Pinning the schema makes the map deterministic for every mix.
    """
    return datasets.Features(
        {
            "history": [{"role": datasets.Value("string"), "content": Json()}],
            "images": datasets.Sequence(datasets.Image()),
            "tools_json": datasets.Value("string"),
        }
    )


# The columns a mapped VLM dataset carries into collation: the map's own output plus the tokenized
# columns a preprocessed artifact arrives with. Derived from the pinned schema, so a column added to
# ``create_vlm_processor`` is not stripped here for want of a second edit.
_VLM_SIGNATURE_COLUMNS = ("input_ids", "labels", "attention_mask", *vlm_map_features())


def _vlm_extra_columns(dataset, *keep: str) -> list[str]:
    """Columns to ``remove_columns`` from a raw VLM dataset map: everything outside the collator
    signature, except any extra ``keep`` columns (e.g. self-distillation privileged fields)."""
    # sorted: this list feeds the coordinated-map cache key, and set order is hash-randomized per process.
    return sorted(set(dataset.column_names) - set(_VLM_SIGNATURE_COLUMNS) - set(keep))


def _filter_vlm_over_length(ds, processor, tokenizer, max_length: int, num_proc: int):
    """Coordinated drop of mapped VLM rows whose rendered text alone exceeds ``max_length``.

    The VLM collators raise on over-length batches (image tokens cannot be truncated), so
    pathological text rows are removed first. Vision tokens are not counted here: this drops the text
    tail cheaply and the collator remains the exact backstop. The emptied-everything refusal is
    world-agreed, since the surviving count is a per-rank fact on a presharded corpus.
    """

    def fits(row) -> bool:
        # The collator's own render, tools included, so the filter measures the sequence that will
        # actually be collated rather than a shorter approximation of it.
        text = render_vlm_text(processor, row["history"], tools=vlm_row_tools(row))
        return len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_length

    before = dataset_total_size(ds)
    ds = coordinated_filter(
        ds,
        fits,
        desc="Dropping over-length VLM rows",
        num_proc=num_proc,
    )
    after = dataset_total_size(ds)
    if agree_probe_across_ranks(after == 0, "the VLM dataset", "every row over max_length"):
        raise ValueError(
            f"All {before} rows exceed max_length={max_length} by text alone on at least one "
            f"data-parallel rank — raise the budget."
        )
    if after < before and is_global_main_process():
        logger.info(f"Dropped {before - after}/{before} VLM rows over max_length={max_length} (text tokens alone).")
    return ds


def prepare_vlm_dataset(
    ds,
    args,
    processor,
    tokenizer,
    max_length: int,
    num_proc: int,
    *,
    keep: tuple[str, ...] = (),
    features: datasets.Features | None = None,
    desc: str = "Processing VLM dataset",
):
    """Shared raw-VLM dataset preparation: ``create_vlm_processor`` map, then over-length filter.

    Maps every split through the training row processor (dropping columns outside the collator
    signature except ``keep``), then drops rows whose rendered text alone exceeds ``max_length``.
    ``features`` optionally pins the map's Arrow schema — only usable when ``keep`` is empty, since
    the pinned schema covers exactly the mapped columns.
    """
    # Here rather than at collation: the collator is built after this map, so a run missing its
    # marker would otherwise pay the whole VLM tokenization before failing.
    require_response_marker(
        getattr(args, "assistant_message_template", None),
        getattr(args, "train_on_completions_only", False),
        "VLM training",
    )
    if getattr(args, "interleaved_thinking", False):
        raise ValueError(
            "interleaved_thinking is text-only: no supported VLM family's chat template renders a "
            "clear_thinking kwarg, so on the VLM path the flag would silently do nothing."
        )
    if features is not None and keep:
        raise ValueError(
            f"prepare_vlm_dataset got both a pinned Arrow schema and keep={list(keep)}: the schema "
            f"declares exactly the mapped columns, so the kept ones have no slot in it and the map "
            f"would die on an Arrow cast inside a worker. Pass one or the other."
        )
    # Here rather than per script: without it, every prepare_vlm_dataset consumer (SFT and the
    # distillation scripts) would train text-only on a mistyped column name.
    images_field = getattr(args, "images_field", None)
    if images_field:
        require_render_column(ds, str(getattr(args, "dataset", "the loaded dataset")), "images_field", images_field)
    cache_key_extras = {
        "conversation_field": args.conversation_field,
        "system_prompt": args.system_prompt,
        "model_supports_system_role": args.model_supports_system_role,
        "tools_field": args.tools_field,
        "images_field": args.images_field,
        "add_generation_prompt": False,
    }
    ds = coordinated_map(
        ds,
        create_vlm_processor(**cache_key_extras),
        num_proc=num_proc,
        remove_columns=_vlm_extra_columns(ds["train"], *keep),
        desc=desc,
        cache_key_extras=cache_key_extras,
        **({"features": features} if features is not None else {}),
    )
    return _filter_vlm_over_length(ds, processor, tokenizer, max_length, num_proc)
