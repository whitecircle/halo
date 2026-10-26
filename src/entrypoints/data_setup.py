"""Script-side dataset loading, the run's VLM verdict, and the switches for TRL's own dataset prep."""

from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizer

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.config_validation import reject_non_default_args, reject_unsupported_args
from src.data.pipeline.preferences import prepare_generative_dataset, prepare_preference_datasets
from src.data.pipeline.processing import log_dataset_examples, resolve_map_num_proc
from src.data.sources.loading import is_presharded_dataset_load, load_datasets, reject_image_columns
from src.data.vlm import dataset_declares_images, is_vlm_run
from src.distributed.parallelism_config import ParallelismConfig
from src.model_loading.parallel.model_source import probe_checkpoint


def reject_trl_dataset_prep_args(
    context: str, sft_config, *unread_fields: str, render_remedy: str | None = None
) -> None:
    """Refuse the ``SFTConfig`` knobs only TRL's own dataset prep and default collator read.

    A script that renders and masks its rows itself replaces both (:func:`disable_trl_dataset_prep`),
    so each knob would parse and do nothing. ``unread_fields`` are further defaulted fields the
    caller's path leaves unread; ``render_remedy`` is what the script offers instead.
    """
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


def load_script_datasets(
    args,
    parallelism_config: ParallelismConfig,
    *,
    loader: Callable[..., Any] = load_datasets,
    **loader_kwargs,
) -> tuple[Any, bool]:
    """The script's datasets loaded for this data-parallel rank: ``(loader_result, dataset_presharded)``.

    ``loader`` is any ``load_datasets``-shaped loader (``load_datasets_auto``'s ``(DatasetDict,
    is_preprocessed)`` passes through); ``loader_kwargs`` forward extras such as ``conversation_field``
    (with ``conversation_knob`` when another config field names that column) or ``seed``.
    """
    data_parallel_rank = parallelism_config.get_data_parallel_rank()
    data_parallel_size = parallelism_config.data_parallel_size
    dataset_presharded = is_presharded_dataset_load(args.dataset, data_parallel_size)
    # Threaded for every script so a declared tools column is validated at load; a typo'd knob
    # otherwise renders the whole run without tools. The scripts that cannot render it (KTO,
    # environmental GRPO) reject the knob through reject_unsupported_args before their load.
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
    """Refuse image data on a run that loads a multimodal checkpoint through its text-only class.

    ``is_vlm_run`` reads the checkpoint's config, which still says multimodal, so each image
    declaration it counts (``images_field``, an image column, parts embedded in
    ``conversation_field``) would route the run to a VLM data path with no vision model behind it.
    """
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
    """Whether the run takes the VLM data path (:func:`is_vlm_run`), decided once the dataset is in hand
    and before the model load, which needs the checkpoint's processor for an image run.

    Without a ``vlm_checkpoint`` verdict the script already probed, :func:`probe_checkpoint` reads the
    checkpoint at ``model_revision`` under ``trust_remote_code``, as the load does.
    """
    _reject_images_under_text_only_model(args, datasets, text_only_model=text_only_model)
    if vlm_checkpoint is None:
        _, vlm_checkpoint = probe_checkpoint(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
    return is_vlm_run(args, model_config.model_name_or_path, datasets, vlm_checkpoint=vlm_checkpoint)


def log_script_dataset_examples(datasets: dict, tokenizer: PreTrainedTokenizer, args, training_config) -> None:
    """``log_dataset_examples`` into the run's ``output_dir`` under its decoded-sample switch; a
    ``None`` split is skipped."""
    log_dataset_examples(
        datasets,
        tokenizer=tokenizer,
        output_dir=training_config.output_dir,
        write_decoded_samples=args.log_decoded_samples,
    )


def prepare_script_preference_data(
    args,
    training_config,
    ds,
    tokenizer: PreTrainedTokenizer,
    *,
    is_vlm_data: bool,
    generation_max_prompt_length: int | None,
):
    """Tokenize a preference dataset and its generation-eval split, and log the examples; returns
    ``(train_dataset, eval_dataset, generate_callback)``, the callback ``None`` when nothing generates.

    The VLM arm passes the rows through: both trainers chat-template and tokenize vision preference
    rows themselves, so a prep here would render them twice.
    """
    generate_dataset = None
    if is_vlm_data:
        train_dataset, eval_dataset = ds["train"], ds.get("test")
    else:
        num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
        train_dataset, eval_dataset = prepare_preference_datasets(
            ds["train"],
            ds["test"],
            tokenizer,
            num_proc=num_proc,
            tools_field=args.tools_field,
        )
        # Only when the callback will consume it — the prep tokenizes the whole test split.
        if args.generate_eval_examples:
            generate_dataset = prepare_generative_dataset(
                ds["test"],
                tokenizer,
                generation_max_prompt_length,
                num_proc=num_proc,
                tools_field=args.tools_field,
            )

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset},
        tokenizer,
        args,
        training_config,
    )
    generate_callback = GenerateExamplesCallback.from_config(args, training_config, generate_dataset, tokenizer)
    return train_dataset, eval_dataset, generate_callback


def disable_trl_dataset_prep(sft_config) -> None:
    """Turn TRL's own dataset preparation off on a config whose data the script already prepared.

    ``skip_prepare_dataset`` alone is not enough: TRL reads ``packing`` and ``padding_free`` again at
    collator-selection time, and either one left on replaces the script's collator with TRL's own,
    over rows TRL never tokenized.
    """
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_config.padding_free = False
    sft_config.packing = False
