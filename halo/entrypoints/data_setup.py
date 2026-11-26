"""Script-side dataset loading (datatrove token sources included), the run's VLM verdict and vision options (the tower
freeze, the FSDP2 refusal of a batch holding no image), shared prepare steps and switches for TRL's dataset prep."""

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer

from halo.callbacks.generate_examples import GenerateExamplesCallback
from halo.callbacks.token_mix_checkpoint import TokenMixCheckpointCallback, find_saved_token_mix
from halo.checkpoint.format import TRAINER_STATE_FILE
from halo.config_validation import differs_from_default, reject_non_default_args, reject_unsupported_args
from halo.data.collators.vlm import VLMDataCollator
from halo.data.conversation import template_column_kwargs
from halo.data.pipeline.coordinated_map import coordinated_map_and_filter, resolve_map_num_proc, run_load_rank_first
from halo.data.pipeline.preferences import prepare_generative_dataset, prepare_preference_datasets
from halo.data.pipeline.text_rows import build_chat_row_processor, text_render_kwargs
from halo.data.pipeline.vlm_rows import prepare_vlm_dataset
from halo.data.sources.loading import is_presharded_dataset_load_across_ranks, load_datasets, share_s3_control_files
from halo.data.sources.token_mix import (
    TokenMix,
    TokenSource,
    build_held_out_shards_across_ranks,
    build_token_mix_across_ranks,
    verify_resumed_token_mix,
)
from halo.data.sources.token_shards import TokenShards, fetch_s3_token_folder
from halo.data.turn_markers import resolve_eos_token_ids
from halo.data.vlm_detection import VLM_IMAGE_COLUMNS, dataset_declares_images_across_ranks, is_vlm_run_across_ranks
from halo.data.vlm_render import apply_image_pixel_bounds
from halo.distributed.fsdp import fsdp2_param_groups
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import is_accelerate_fsdp_launch
from halo.log import RUN_LOG_DIR_NAME
from halo.model_loading.parallel.freeze_patterns import freeze_parameters_by_patterns
from halo.model_loading.parallel.model_source import probe_checkpoint
from halo.model_loading.parallel.peft_setup import wraps_in_peft_model
from halo.models.attention.prefix_lm import has_prefix_lm_attention
from halo.models.introspection.module_tree import (
    backbone_with_layers,
    decoder_layers,
    embedding_module,
    media_module_names,
)
from halo.models.introspection.unwrapping import find_peft_model, unwrap_framework_wrappers
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from halo.runtime.consensus import refusal_text, reject_across_ranks, run_on_rank0_raising_everywhere
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import is_global_main_process
from halo.trainers.construction.training_args_overrides import uses_evaluation
from halo.trainers.loop.dataloaders import resolve_data_seed

logger = logging.getLogger(__name__)


def reject_trl_dataset_prep_args(
    context: str, sft_config, *unread_fields: str, render_remedy: str | None = None
) -> None:
    """Refuse the ``SFTConfig`` knobs only TRL's dataset prep and default collator read, and ``chat_template_path``,
    which TRL applies after the rows were rendered with the run's template. ``unread_fields`` are further fields the
    caller leaves unread; ``render_remedy`` is what the script offers instead."""
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
    """``(loader_result, dataset_presharded)`` for this data-parallel rank; ``loader_kwargs`` forward extras such as
    ``conversation_field`` (with ``conversation_knob`` when another config field names that column) or ``seed``."""
    data_parallel_rank = parallelism_config.get_data_parallel_rank()
    data_parallel_size = parallelism_config.data_parallel_size
    share_s3_control_files(args.dataset)
    dataset_presharded = is_presharded_dataset_load_across_ranks(args.dataset, data_parallel_size)
    ds = loader(
        args.dataset,
        args.test_size,
        args.dataset_ratio,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        # For every script, so a typo'd template column fails the load; a script that cannot render one refuses the
        # knob.
        template_columns=template_column_kwargs(args),
        **loader_kwargs,
    )
    return ds, dataset_presharded


def _reject_images_under_text_only_model(args, datasets, *, text_only_model: bool) -> None:
    """COLLECTIVE under ``text_only_model``. Refuse image data on a multimodal checkpoint loaded through its text-only
    class, since ``is_vlm_run_across_ranks`` reads the still-multimodal config and would route the run to a VLM data
    path with no vision model. A presharded load answers per rank, so the verdict is agreed."""
    if not text_only_model:
        return
    images_field = getattr(args, "images_field", None)
    if images_field:
        raise ValueError(
            f"images_field={images_field!r} is set but text_only_model=True loads the text-only "
            f"CausalLM class, which has no vision path. Drop text_only_model to train the "
            f"multimodal wrapper, or drop images_field for a text-only run."
        )
    conversation_field = getattr(args, "conversation_field", None)
    if dataset_declares_images_across_ranks(datasets, conversation_field):
        declared = f"an image column ({', '.join(VLM_IMAGE_COLUMNS)})"
        if conversation_field:
            declared += f" or image content parts in its {conversation_field!r} column"
        raise ValueError(
            f"The dataset carries {declared}, but text_only_model=True loads the text-only CausalLM class, which has "
            f"no vision path. Drop text_only_model to train the multimodal wrapper, or drop the image data for a "
            f"text-only run."
        )


def reject_image_data_on_text_checkpoint(args, model_name_or_path: str, datasets) -> None:
    """COLLECTIVE: the VLM dispatch reads a text checkpoint as a text run whatever the data declares, and the text
    pipeline then drops the images. ``images_field`` is rank-uniform, so every rank takes the same branch."""
    images_field = getattr(args, "images_field", None)
    if images_field or dataset_declares_images_across_ranks(datasets, getattr(args, "conversation_field", None)):
        declared = f"images_field={images_field!r}" if images_field else "an image column or image content parts"
        raise ValueError(
            f"{model_name_or_path!r} is a text checkpoint, but this run declares image data ({declared}), which the "
            f"text pipeline would drop. Train a multimodal checkpoint, or remove the image data."
        )


def resolve_vlm_run(
    args, model_config, datasets, *, text_only_model: bool, vlm_checkpoint: bool | None = None
) -> bool:
    """Whether the run takes the VLM data path, decided before the model load, which needs the checkpoint's processor
    for an image run. Without ``vlm_checkpoint``, :func:`probe_checkpoint` reads the checkpoint as the load does."""
    _reject_images_under_text_only_model(args, datasets, text_only_model=text_only_model)
    if vlm_checkpoint is None:
        _, vlm_checkpoint = probe_checkpoint(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
        )
    if not vlm_checkpoint:
        reject_image_data_on_text_checkpoint(args, model_config.model_name_or_path, datasets)
    return is_vlm_run_across_ranks(args, model_config.model_name_or_path, datasets, vlm_checkpoint=vlm_checkpoint)


def reject_prefix_lm_image_preference(checkpoint_config, model_name_or_path: str, method: str) -> None:
    """Refuse an image preference run on a prefix-LM checkpoint (PaliGemma): its decoder needs ``token_type_ids``
    marking each completion as the causal suffix, and no vision preference collator writes them. TRL's calls the
    processor on completions without images, which raises; SMPO's marks the whole row as prefix, so a completion
    attends to its own next tokens."""
    if has_prefix_lm_attention(checkpoint_config):
        raise ValueError(
            f"{method} on image data cannot train {model_name_or_path!r} "
            f"(model_type={checkpoint_config.model_type!r}): its decoder is a prefix LM, and the {method} vision "
            f"collator marks no completion as the causal suffix in token_type_ids, so each completion would score "
            f"with its own next tokens in view. Train it with SFT or distillation on images, or with {method} on "
            f"text-only pairs."
        )


def _reject_trl_image_rows(checkpoint_config, model_name_or_path: str, method: str) -> None:
    family = find_moe_family_spec_for_config(checkpoint_config)
    if family is not None and family.refuses_trl_image_rows is not None:
        raise ValueError(
            f"{method} on image data cannot train {model_name_or_path!r} "
            f"(model_type={checkpoint_config.model_type!r}): {family.refuses_trl_image_rows}. Train it on images with "
            f"SFT or SMPO, or with {method} on text-only pairs."
        )


def resolve_preference_vlm_run(
    args, model_config, datasets, *, text_only_model: bool, method: str, trl_renders_rows: bool = False
) -> bool:
    """As :func:`resolve_vlm_run`, off one probe that also refuses an image run on a prefix-LM checkpoint, and with
    ``trl_renders_rows`` one on a family TRL's vision path does not carry."""
    checkpoint_config, vlm_checkpoint = probe_checkpoint(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    is_vlm = resolve_vlm_run(
        args, model_config, datasets, text_only_model=text_only_model, vlm_checkpoint=vlm_checkpoint
    )
    if is_vlm:
        reject_prefix_lm_image_preference(checkpoint_config, model_config.model_name_or_path, method)
        if trl_renders_rows:
            _reject_trl_image_rows(checkpoint_config, model_config.model_name_or_path, method)
    return is_vlm


def _first_trained_tower_param(model) -> str | None:
    for tower in media_module_names(model):
        for name, param in model.get_submodule(tower).named_parameters():
            if param.requires_grad:
                return f"{tower}.{name}"
    return None


def _why_a_batch_holding_no_image_cannot_pair(model) -> str | None:
    """Where the FSDP2 wrap's agreement on which parameters hold a gradient (``_agree_unsharded_grad_presence``) does not
    pair a text rank's reduces with an image rank's, with the first remedy. Checkpointing's input-grad hook gives a frozen
    embedding's output a gradient on every row, so a first layer then reduces in order on each rank; with no layer
    trained, only the groups reducing at the end of each rank's backward reduce at all. ZeRO-3 pairs neither way: a group
    re-gathers at its backward only where its output carries a gradient, which the tower's group does on an image rank
    alone."""
    if is_accelerate_fsdp_launch():
        return (
            "Accelerate-managed FSDP2 takes no agreement on which parameters hold a gradient before reducing. "
            "Launch with torchrun"
        )
    if model.is_gradient_checkpointing or any(p.requires_grad for p in embedding_module(model, "input").parameters()):
        return None
    layers = decoder_layers(backbone_with_layers(model))
    trains_a_layer = layers is not None and any(param.requires_grad for param in layers.parameters())
    if not trains_a_layer and not any(group.post_forward_mesh_info is not None for group in fsdp2_param_groups(model)):
        return None
    return (
        "a frozen input embedding leaves a text row's hidden states with no gradient, so the groups they pass through "
        "reduce out of order. Train the embedding, enable gradient checkpointing"
    )


def _refuse_a_batch_holding_no_image(model, args, kwargs) -> None:
    """On a training forward, where the step would otherwise hang to the collective timeout. Positional arguments are a
    redirected method's, which reads the inputs below this forward."""
    if args or not model.training or kwargs.get("pixel_values") is not None:
        return
    reason = _why_a_batch_holding_no_image_cannot_pair(model)
    if reason is None or not any(group.mesh_info.mesh.size() > 1 for group in fsdp2_param_groups(model)):
        return
    trained = _first_trained_tower_param(model)
    if trained is None:
        return
    remedy = (
        f"keep lora_target_modules off the tower (an adapter reached {trained})"
        if find_peft_model(model) is not None
        else "set freeze_vision_tower: true, train adapters"
    )
    raise ValueError(
        "This rank's batch holds no image while the vision tower trains under data parallelism, so its gradient "
        f"reduces would never pair with a peer's that holds one: {reason}, {remedy}, or drop the text-only rows from "
        "the image dataset."
    )


def carry_image_refusal_above_adapters(model) -> None:
    """Call on the trainer's model: the adapter wrap reaches what it wraps through ``.forward()``, which runs no
    pre-hook, and only past that wrap does a tower hold its adapters."""
    peft_model = find_peft_model(unwrap_framework_wrappers(model))
    if (
        peft_model is not None
        and _refuse_a_batch_holding_no_image in peft_model.get_base_model()._forward_pre_hooks.values()
    ):
        peft_model.register_forward_pre_hook(_refuse_a_batch_holding_no_image, with_kwargs=True)


def apply_vision_options(args, model, model_config, processing_class, *, is_vlm: bool) -> None:
    """COLLECTIVE under ``freeze_vision_tower``, after :func:`configure_adapter_training`, so the freeze outlasts its
    unfreeze patterns. A tower no text row reaches also stops being a parameter FSDP2 reduces on only some ranks; on an
    image run that still trains one, a batch holding no image is refused where the reduces could not pair. Under an
    adapter wrap the refusal waits for :func:`carry_image_refusal_above_adapters`, as the tower's adapters do not exist
    yet."""
    if args.min_pixels is not None or args.max_pixels is not None:
        if not is_vlm:
            raise ValueError("min_pixels / max_pixels bound images, and this run takes the text data path.")
        apply_image_pixel_bounds(processing_class, args.min_pixels, args.max_pixels)
    if args.freeze_vision_tower:
        if model_config.use_peft:
            raise ValueError(
                "freeze_vision_tower has nothing to do with adapters: they freeze the whole base already."
            )
        towers = media_module_names(model)
        reason = (
            None if towers else f"freeze_vision_tower: {type(model).__name__} holds no vision or audio tower here."
        )
        reject_across_ranks(reason, "freeze_vision_tower", ValueError)
        freeze_parameters_by_patterns(model, [f"{tower}.*" for tower in towers])
    elif is_vlm and (wraps_in_peft_model(model_config) or _first_trained_tower_param(model) is not None):
        model.register_forward_pre_hook(_refuse_a_batch_holding_no_image, with_kwargs=True)


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
    datasets: dict[str, Any],
    num_examples: int = 1,
    *,
    tokenizer: PreTrainedTokenizer | None = None,
    output_dir: str | None = None,
    write_decoded_samples: bool = False,
) -> None:
    """Log each dataset's first rows on the main process; ``write_decoded_samples`` decodes tokenized ones beside
    ``run.log`` on the filesystem save rank (once per node on a non-shared filesystem)."""
    log_examples = is_global_main_process()
    write_samples = write_decoded_samples and is_fs_save_rank()
    if not (log_examples or write_samples):
        return

    for name, dataset in datasets.items():
        if dataset is None:
            continue

        if isinstance(dataset, Mapping):
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
    """A ``None`` split is skipped."""
    log_dataset_examples(
        datasets,
        tokenizer=tokenizer,
        output_dir=training_config.output_dir,
        write_decoded_samples=args.log_decoded_samples,
    )


def reject_unsupported_token_source_options(args, sft_config, dist_args) -> None:
    """Refuse the training options a token_sources run cannot honor, naming each one it misses, before the model
    load."""
    evaluates = uses_evaluation(sft_config) or args.gates_mtp_acceptance
    held_out = args.eval_token_source_specs or []
    loader = sft_config.accelerator_config
    best = sft_config.metric_for_best_model
    requirements = {
        "max_steps set (the mix is laid out for the run's length)": sft_config.max_steps > 0,
        "train_sampling_strategy: sequential (the mix is shuffled already, and a random sampler would hold a list of "
        "every sample on each rank)": sft_config.train_sampling_strategy == "sequential",
        "packing: true (each window is a packed row of documents)": sft_config.packing,
        "packing_strategy left at its default (windows are cut before training, as the wrapped strategy cuts)": (
            not differs_from_default(sft_config, "packing_strategy")
        ),
        "eval_packing left unset (held-out windows are packed rows too)": sft_config.eval_packing is None,
        "padding_free off (the rows are packed already)": not sft_config.padding_free,
        "ignore_data_skip off (a resume skips the trained steps' rows by index without reading them, and without the "
        "skip it trains them again)": not sft_config.ignore_data_skip,
        "auto_find_batch_size off (a halved batch reads fewer than rows_per_step rows a step, and the resume skip "
        "miscounts)": not sft_config.auto_find_batch_size,
        "accelerator_config's dispatch_batches and split_batches off and even_batches on (each rank draws its own "
        "batches of the mix, which rows_per_step and the resume skip count, and with even_batches off a held-out "
        "folder that does not split evenly over the ranks hangs the evaluation)": (
            not loader.dispatch_batches and not loader.split_batches and loader.even_batches
        ),
        "context_parallel_size: 1 (a window is a packed row, which context parallelism refuses)": (
            dist_args.context_parallel_size == 1
        ),
        "eval_token_sources when the run evaluates, by eval_strategy, eval_on_start or the MTP acceptance gate (a "
        "token run evaluates on held-out folders only)": not evaluates or bool(held_out),
        "one eval_token_sources folder under the MTP acceptance gate (every folder logs its acceptance under the same "
        "key, so the gate would set one folder's reading against another's)": (
            not args.gates_mtp_acceptance or len(held_out) <= 1
        ),
        "metric_for_best_model unset or naming a held-out folder's eval_<name>_loss (a token run logs no eval_loss)": (
            best is None
            or (best if best.startswith("eval_") else f"eval_{best}")
            in {f"eval_{source.name}_loss" for source in held_out}
        ),
    }
    unmet = [requirement for requirement, met in requirements.items() if not met]
    if unmet:
        raise ValueError("token_sources needs: " + "; ".join(unmet) + ".")


def _fetched_token_sources(sources: list[TokenSource]) -> list[TokenSource]:
    """COLLECTIVE when a path is ``s3://``: each such source pointed at its download-cache copy, which each input
    filesystem's load rank fetches before its peers read the cache."""
    if not any(source.path.startswith("s3://") for source in sources):
        return sources

    def fetch() -> list[TokenSource]:
        fetched = []
        for source in sources:
            if source.path.startswith("s3://"):
                try:
                    source = replace(source, path=str(fetch_s3_token_folder(source.path)))
                except Exception as error:  # S3's own errors name neither the source nor its prefix
                    raise RuntimeError(
                        f"token source {source.name!r} ({source.path}): {refusal_text(error)}"
                    ) from error
            fetched.append(source)
        return fetched

    return run_load_rank_first(fetch, "fetching the s3:// token sources")


def _saved_token_mix(checkpoint: str) -> dict:
    saved = find_saved_token_mix(checkpoint)
    if saved is None:
        raise ValueError(
            f"Checkpoint {checkpoint} carries no token mix: its {TRAINER_STATE_FILE} holds no "
            f"{TokenMixCheckpointCallback.__name__} entry under stateful_callbacks, so the mix it trained on is "
            f"unknown, as for a checkpoint of a run without token_sources. Resume from a checkpoint a token_sources "
            f"run saved with this entry, or start a new run."
        )
    return saved


def _rows_per_step(training_config, parallelism_config: ParallelismConfig) -> int:
    """The rows one optimizer step reads across the data-parallel ranks. Under pipeline parallelism the accumulation
    becomes the microbatch count, which splits the per-device batch."""
    accumulation = 1 if parallelism_config.pp_size > 1 else training_config.gradient_accumulation_steps
    return training_config.per_device_train_batch_size * accumulation * parallelism_config.data_parallel_size


def reject_resume_from_a_token_run(resume_checkpoint: str | None) -> None:
    """COLLECTIVE: refuse a run without token_sources resuming a checkpoint a token_sources run saved: the resumed
    trainer state would carry the mix's fingerprint into every checkpoint this run saves, so a later token run could
    resume from one onto rows these steps never read."""
    if resume_checkpoint is None:
        return
    failure = None
    try:
        if find_saved_token_mix(resume_checkpoint) is not None:
            raise ValueError(
                f"Checkpoint {resume_checkpoint} was saved by a token_sources run: its {TRAINER_STATE_FILE} records "
                f"the token mix its steps trained on, so a run without token_sources cannot resume it. Resume with "
                f"the checkpoint's token_sources, or start a new run from its weights (model_name_or_path: "
                f"{resume_checkpoint}, merged with scripts/after_training/merge_ep_shards.py first if it is "
                f"EP-sharded)."
            )
    except Exception as error:  # a rank raising alone would strand its peers in the join below
        failure = refusal_text(error)
    reject_across_ranks(failure, "Checking the resumed checkpoint for a token mix", ValueError)


def _labelled_folders(mix: TokenMix, held_out: dict[str, TokenShards] | None) -> dict[str, TokenShards]:
    return {
        **{f"token_sources {source.name!r}": shards for source, shards in zip(mix.sources, mix.shards, strict=True)},
        **{f"eval_token_sources {name!r}": shards for name, shards in (held_out or {}).items()},
    }


def load_token_sources(
    args, training_config, parallelism_config: ParallelismConfig, *, resume_checkpoint: str | None
) -> tuple[TokenMix, dict[str, TokenShards] | None]:
    """COLLECTIVE: the run's token mix, ``max_steps`` global batches long and checked against the mix
    ``resume_checkpoint`` trained on, and the held-out folders by name, all of one token width. A failure on one rank
    raises on every rank."""
    mix = build_token_mix_across_ranks(
        _fetched_token_sources(args.token_source_specs),
        max_length=training_config.max_length,
        num_steps=training_config.max_steps,
        rows_per_step=_rows_per_step(training_config, parallelism_config),
        seed=resolve_data_seed(training_config),
    )
    if resume_checkpoint is not None:
        failure = None
        try:
            verify_resumed_token_mix(_saved_token_mix(resume_checkpoint), mix.fingerprint())
        except Exception as error:  # a rank raising alone would strand its peers in the join below
            failure = refusal_text(error)
        reject_across_ranks(failure, "Checking the token mix against the resumed checkpoint", ValueError)
    held_out = (
        None
        if args.eval_token_source_specs is None
        else build_held_out_shards_across_ranks(_fetched_token_sources(args.eval_token_source_specs), mix=mix)
    )
    # Every rank agreed on each folder's width, so each raises alike.
    widths = {label: shards.token_bytes for label, shards in _labelled_folders(mix, held_out).items()}
    if len(set(widths.values())) > 1:
        listed = ", ".join(f"{label} {width} bytes" for label, width in widths.items())
        raise ValueError(
            f"The token folders hold ids of different widths: {listed}. datatrove writes 2-byte ids up to a vocabulary "
            f"of 65,536 and 4-byte ids past it, so these come from two tokenizers, and one's ids would train or score "
            f"as the other's. Re-tokenize every folder with the model's tokenizer."
        )
    return mix, held_out


def verify_token_sources_fit_model(
    mix: TokenMix, evaluated: dict[str, TokenShards] | None, vocab_size: int, tokenizer_name: str
) -> None:
    """COLLECTIVE: raise on every rank when a folder holds an id past the model's vocabulary; warn when the folders'
    metadata names more than one tokenizer or one other than the run's, which a local path or another spelling of the
    same one also trips. Rank 0 reads the windows alone: the ranks agreed on their digest."""
    folders = _labelled_folders(mix, evaluated)

    def verify_ids() -> None:
        for shards in folders.values():
            shards.verify_token_ids_below(vocab_size)

    run_on_rank0_raising_everywhere(verify_ids, refusal_text, ValueError)
    tokenizers = {label: sorted({file.tokenizer for file in shards.files}) for label, shards in folders.items()}
    named = sorted({name for names in tokenizers.values() for name in names})
    if named != [tokenizer_name] and is_global_main_process():
        listed = "; ".join(f"{label} names {names}" for label, names in tokenizers.items())
        logger.warning(
            f"The token folders' metadata names {named}, not only this run's tokenizer {tokenizer_name!r}: {listed}. "
            f"A folder of another tokenizer trains the model on ids that mean other text; re-tokenize it with this "
            f"run's. A local path or another spelling of the same tokenizer also reads as another name here."
        )


def prepare_chat_text_split(ds, tokenizer, args, training_config, model_config, generation_config, *, desc: str):
    """COLLECTIVE: ``(train, eval)`` chat-templated and tokenized once; a row over ``max_length`` is dropped, never
    truncated mid-turn, and a last assistant turn the template leaves open is closed against the terminators the model
    and generation configs name. ``desc`` keys the map's cache."""
    render_kwargs = text_render_kwargs(args)
    eos_token_ids = resolve_eos_token_ids(tokenizer, model_config, generation_config)
    ds = coordinated_map_and_filter(
        ds,
        build_chat_row_processor(
            tokenizer=tokenizer, max_length=training_config.max_length, eos_token_ids=eos_token_ids, **render_kwargs
        ),
        # sorted: this list feeds the coordinated-map cache key, and set order is hash-randomized per process.
        remove_columns=sorted(set(ds["train"].column_names)),
        desc=desc,
        cache_key_extras={**render_kwargs, "add_generation_prompt": False, "eos_token_ids": sorted(eos_token_ids)},
        num_proc=resolve_map_num_proc(training_config.dataset_num_proc),
    )
    return ds["train"], ds["test"]


def prepare_raw_vlm_split(
    ds, args, training_config, processor, tokenizer, model_config, generation_config, *, desc: str
):
    """COLLECTIVE: ``(train, eval, collator)``, raw conversations mapped for :class:`VLMDataCollator`."""
    num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
    ds = prepare_vlm_dataset(
        ds,
        args,
        processor,
        tokenizer,
        training_config.max_length,
        num_proc,
        eos_token_ids=resolve_eos_token_ids(tokenizer, model_config, generation_config),
        desc=desc,
    )
    # The mapped columns are not model-forward kwargs, so HF's default remove_unused_columns=True would strip them.
    training_config.remove_unused_columns = False
    collator = VLMDataCollator(
        processor,
        tokenizer,
        training_config.max_length,
        response_prompt_template=args.assistant_message_template,
        train_on_completions_only=args.train_on_completions_only,
        model_config=model_config,
        generation_config=generation_config,
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
    """``(train_dataset, eval_dataset, generate_callback)``, the callback ``None`` when nothing generates. VLM rows
    pass through: both trainers chat-template and tokenize them themselves."""
    generate_dataset = None
    if is_vlm_data:
        train_dataset, eval_dataset = ds["train"], ds.get("test")
    else:
        num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
        template_columns = template_column_kwargs(args)
        train_dataset, eval_dataset = prepare_preference_datasets(
            ds["train"], ds["test"], tokenizer, num_proc=num_proc, **template_columns
        )
        # Only when the callback will consume it: the prep tokenizes the whole test split.
        if args.generate_eval_examples:
            generate_dataset = prepare_generative_dataset(
                ds["test"], tokenizer, generation_max_prompt_length, num_proc=num_proc, **template_columns
            )

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset}, tokenizer, args, training_config
    )
    generate_callback = GenerateExamplesCallback.from_config(args, training_config, generate_dataset, tokenizer)
    return train_dataset, eval_dataset, generate_callback


def disable_trl_dataset_prep(sft_config) -> None:
    """For data the script already prepared: TRL reads ``packing`` and ``padding_free`` again at collator selection,
    and either left on replaces the script's collator with TRL's own, over rows TRL never tokenized."""
    sft_config.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_config.padding_free = False
    sft_config.packing = False
