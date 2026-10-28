"""Script-side dataset loading, the run's VLM verdict, the prepare steps scripts share, and the switches for TRL's
own dataset prep."""

import logging
import os
from collections.abc import Callable
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from halo.callbacks.generate_examples import GenerateExamplesCallback
from halo.config_validation import reject_non_default_args, reject_unsupported_args
from halo.data.collators.vlm import VLMDataCollator
from halo.data.pipeline.coordinated_map import coordinated_map_and_filter, resolve_map_num_proc
from halo.data.pipeline.preferences import prepare_generative_dataset, prepare_preference_datasets
from halo.data.pipeline.text_rows import build_chat_row_processor, text_render_kwargs
from halo.data.pipeline.vlm_rows import prepare_vlm_dataset
from halo.data.sources.loading import is_presharded_dataset_load, load_datasets
from halo.data.vlm_detection import dataset_declares_images, is_vlm_run, reject_image_columns
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import RUN_LOG_DIR_NAME
from halo.model_loading.parallel.model_source import probe_checkpoint
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def reject_trl_dataset_prep_args(
    context: str, sft_config, *unread_fields: str, render_remedy: str | None = None
) -> None:
    """Refuse the ``SFTConfig`` knobs only TRL's own dataset prep and default collator read, which a script that
    renders and masks its rows replaces (:func:`disable_trl_dataset_prep`), and ``chat_template_path``, which
    TRL's trainer applies to the processing class after those rows were rendered with the run's template.
    ``unread_fields`` are further defaulted fields the caller's path leaves unread; ``render_remedy`` is what the
    script offers instead."""
    reject_unsupported_args(
        f"{context} (it masks completions through train_on_completions_only + assistant_message_template)",
        # Tri-state: an explicit False ("train on the full sequence") is ignored the same as True.
        completion_only_loss=sft_config.completion_only_loss is not None,
        assistant_only_loss=sft_config.assistant_only_loss,
    )
    reject_non_default_args(
        f"{context} ({render_remedy})" if render_remedy else context,
        sft_config,
        "dataset_text_field",
        "dataset_kwargs",
        *unread_fields,
    )
    reject_non_default_args(
        f"{context} (the rows are rendered before the trainer applies chat_template_path, so the checkpoint would "
        f"export a template they never saw; set chat_template instead, with force_chat_template: true over a "
        f"template the checkpoint ships and added_special_tokens for any tokens it needs)",
        sft_config,
        "chat_template_path",
    )


def load_script_datasets(
    args, parallelism_config: ParallelismConfig, *, loader: Callable[..., Any] = load_datasets, **loader_kwargs
) -> tuple[Any, bool]:
    """The script's datasets loaded for this data-parallel rank: ``(loader_result, dataset_presharded)``. ``loader``
    is any ``load_datasets``-shaped loader; ``loader_kwargs`` forward extras such as ``conversation_field`` (with
    ``conversation_knob`` when another config field names that column) or ``seed``."""
    data_parallel_rank = parallelism_config.get_data_parallel_rank()
    data_parallel_size = parallelism_config.data_parallel_size
    dataset_presharded = is_presharded_dataset_load(args.dataset, data_parallel_size)
    # Threaded for every script so a typo'd tools column fails the load instead of rendering the run without tools;
    # the scripts that cannot render it (KTO, environmental GRPO) refuse the knob before their load.
    loader_kwargs.setdefault("tools_field", args.tools_field)
    ds = loader(
        args.dataset,
        args.test_size,
        args.dataset_ratio,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        **loader_kwargs,
    )
    return ds, dataset_presharded


def _reject_images_under_text_only_model(args, datasets, *, text_only_model: bool) -> None:
    """Refuse image data on a run that loads a multimodal checkpoint through its text-only class: ``is_vlm_run``
    reads the still-multimodal checkpoint config, so each image declaration it counts would route the run to a VLM
    data path with no vision model behind it."""
    if not text_only_model:
        return
    images_field = getattr(args, "images_field", None)
    if images_field:
        raise ValueError(
            f"images_field={images_field!r} is set but text_only_model=True loads the text-only "
            f"CausalLM class, which has no vision path. Drop text_only_model to train the "
            f"multimodal wrapper, or drop images_field for a text-only run."
        )
    reject_image_columns(datasets, "text_only_model=True (text-only CausalLM load)")
    conversation_field = getattr(args, "conversation_field", None)
    if conversation_field and dataset_declares_images(datasets, conversation_field):
        raise ValueError(
            f"The {conversation_field!r} column embeds image content parts, but text_only_model=True "
            f"loads the text-only CausalLM class, which has no vision path. Drop text_only_model to "
            f"train the multimodal wrapper, or drop the image parts for a text-only run."
        )


def resolve_vlm_run(
    args, model_config, datasets, *, text_only_model: bool, vlm_checkpoint: bool | None = None
) -> bool:
    """Whether the run takes the VLM data path (:func:`is_vlm_run`), decided before the model load, which needs the
    checkpoint's processor for an image run. Without the ``vlm_checkpoint`` verdict the script already probed,
    :func:`probe_checkpoint` reads the checkpoint as the load does."""
    _reject_images_under_text_only_model(args, datasets, text_only_model=text_only_model)
    if vlm_checkpoint is None:
        _, vlm_checkpoint = probe_checkpoint(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
    return is_vlm_run(args, model_config.model_name_or_path, datasets, vlm_checkpoint=vlm_checkpoint)


def _write_decoded_samples(
    dataset: Dataset, *, name: str, num_examples: int, tokenizer: PreTrainedTokenizer, output_dir: str
) -> None:
    """Decode ``dataset``'s first rows into ``<output_dir>/log/<name>_sample.txt`` (the test split's as ``eval``)."""
    if "input_ids" not in dataset.column_names:
        return

    log_dir = os.path.join(output_dir, RUN_LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "eval_sample.txt" if name == "test" else f"{name}_sample.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num_examples):
            input_ids = dataset[i]["input_ids"]
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            f.write(f"=== Sample {i + 1} ({len(input_ids)} tokens) ===\n")
            f.write(decoded)
            f.write("\n\n")
    logger.info(f"Wrote {num_examples} decoded sample(s) to {out_path}")


def log_dataset_examples(
    datasets: dict[str, Dataset],
    num_examples: int = 1,
    *,
    tokenizer: PreTrainedTokenizer | None = None,
    output_dir: str | None = None,
    write_decoded_samples: bool = False,
) -> None:
    """Log each dataset's first rows on the global main process, and with ``write_decoded_samples`` decode the
    tokenized ones beside ``run.log`` on the filesystem save rank, which writes once per node on a non-shared
    filesystem."""
    log_examples = is_global_main_process()
    write_samples = write_decoded_samples and is_fs_save_rank()
    if not (log_examples or write_samples):
        return

    for name, dataset in datasets.items():
        if dataset is None:
            continue

        if isinstance(dataset, DatasetDict):
            if len(dataset) == 0:
                continue
            split_name = next(iter(dataset))
            split_dataset = dataset[split_name]
            header = f"Example(s) from {name} dataset (split: {split_name}):"
        else:
            split_dataset = dataset
            header = f"Example(s) from {name} dataset:"

        if len(split_dataset) == 0:
            continue

        n = min(num_examples, len(split_dataset))
        if log_examples:
            logger.info(header)
            for i in range(n):
                logger.info(split_dataset[i])

        if write_samples and tokenizer is not None and output_dir is not None:
            _write_decoded_samples(
                split_dataset, name=name, num_examples=n, tokenizer=tokenizer, output_dir=output_dir
            )


def log_script_dataset_examples(datasets: dict, tokenizer: PreTrainedTokenizer, args, training_config) -> None:
    """``log_dataset_examples`` into the run's ``output_dir`` under its decoded-sample switch; a
    ``None`` split is skipped."""
    log_dataset_examples(
        datasets,
        tokenizer=tokenizer,
        output_dir=training_config.output_dir,
        write_decoded_samples=args.log_decoded_samples,
    )


def prepare_chat_text_split(ds, tokenizer, args, training_config, *, desc: str, use_padding: bool):
    """COLLECTIVE: ``(train, eval)`` chat-templated and tokenized once, a row over ``max_length`` dropped rather than
    truncated mid-turn (measured on the ids it trains on); ``desc`` keys the map's cache."""
    render_kwargs = text_render_kwargs(args)
    ds = coordinated_map_and_filter(
        ds,
        build_chat_row_processor(
            tokenizer=tokenizer, max_length=training_config.max_length, use_padding=use_padding, **render_kwargs
        ),
        # sorted: this list feeds the coordinated-map cache key — set order is hash-randomized per process.
        remove_columns=sorted(set(ds["train"].column_names)),
        desc=desc,
        cache_key_extras={**render_kwargs, "add_generation_prompt": False},
        num_proc=resolve_map_num_proc(training_config.dataset_num_proc),
    )
    return ds["train"], ds["test"]


def prepare_raw_vlm_split(ds, args, training_config, processor, tokenizer, model_config, *, desc: str):
    """COLLECTIVE: ``(train, eval, collator)``, raw conversations mapped to ``history``/``images`` for
    :class:`VLMDataCollator`; ``desc`` keys the map's cache."""
    num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
    ds = prepare_vlm_dataset(ds, args, processor, tokenizer, training_config.max_length, num_proc, desc=desc)
    # The mapped columns (history/images/…) are not model-forward kwargs, so HF's default
    # remove_unused_columns=True would strip them before they reach the collator.
    training_config.remove_unused_columns = False
    collator = VLMDataCollator(
        processor,
        tokenizer,
        training_config.max_length,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        model_config=model_config,
    )
    return ds["train"], ds["test"], collator


def prepare_script_preference_data(
    args,
    training_config,
    ds,
    tokenizer: PreTrainedTokenizer,
    *,
    is_vlm_data: bool,
    generation_max_prompt_length: int | None,
):
    """Tokenize a preference dataset and its generation-eval split, and log the examples; returns ``(train_dataset,
    eval_dataset, generate_callback)``, the callback ``None`` when nothing generates. VLM rows pass through: both
    trainers chat-template and tokenize them themselves."""
    generate_dataset = None
    if is_vlm_data:
        train_dataset, eval_dataset = ds["train"], ds.get("test")
    else:
        num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
        train_dataset, eval_dataset = prepare_preference_datasets(
            ds["train"], ds["test"], tokenizer, num_proc=num_proc, tools_field=args.tools_field
        )
        # Only when the callback will consume it — the prep tokenizes the whole test split.
        if args.generate_eval_examples:
            generate_dataset = prepare_generative_dataset(
                ds["test"], tokenizer, generation_max_prompt_length, num_proc=num_proc, tools_field=args.tools_field
            )

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset}, tokenizer, args, training_config
    )
    generate_callback = GenerateExamplesCallback.from_config(args, training_config, generate_dataset, tokenizer)
    return train_dataset, eval_dataset, generate_callback


def disable_trl_dataset_prep(sft_config) -> None:
    """Turn TRL's own dataset preparation off on a config whose data the script already prepared: TRL reads
    ``packing`` and ``padding_free`` again at collator selection, and either one left on replaces the script's
    collator with TRL's own, over rows TRL never tokenized."""
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_config.padding_free = False
    sft_config.packing = False
