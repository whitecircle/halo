#!/usr/bin/env python
"""Distributed embedding training: sentence-transformers losses over pairs, triplets, scored pairs and labeled texts.
Text-only (the parallel loader's preloaded transformer only tokenizes text); CP is not supported, since pooling reads
the complete sequence."""

import json
import logging
from collections.abc import Iterable
from typing import NamedTuple

from peft import inject_adapter_in_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Normalize, Pooling
from sentence_transformers.util import load_file_path
from torch import nn
from transformers import AutoModel
from trl import get_quantization_config

from halo.checkpoint.format import ST_CARRIED_MODEL_SETTINGS, ST_MODEL_SETTINGS_FILE, ST_MODULES_FILE
from halo.checkpoint.model_card import HUB_TAGS
from halo.config_validation import reject_non_default_args, reject_unsupported_args
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.vlm_detection import reject_image_columns
from halo.entrypoints.data_setup import load_script_datasets
from halo.entrypoints.model_setup import load_model_and_tokenizer, plan_row_attention
from halo.entrypoints.script import (
    ScriptContext,
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.model_loading.checkpoint_coverage import full_weights_on_every_rank
from halo.model_loading.dtype import cast_parameters_to_run_dtype, resolve_training_dtype
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import apply_family_patches, apply_run_model_policies, resolve_build_config
from halo.model_loading.parallel.model_source import fetch_checkpoint_config, resolve_model_source
from halo.model_loading.parallel.peft_setup import build_peft_config
from halo.model_loading.parallel.whole_model import warm_attention_kernels
from halo.models.attention.gpt_oss_sinks import SinksPolicy, reject_trainable_sinks_without_sinks
from halo.models.external_tables import EXTERNAL_TABLES_STAMP
from halo.models.introspection.config_access import drop_toolkit_stamps
from halo.models.introspection.context_window import resolve_length_to_context
from halo.models.moe.family_specs import wraps_experts
from halo.models.peft_adapters import tuner_adapter_param_ids
from halo.runtime.collectives import barrier
from halo.runtime.filesystem import hub_metadata_main_first
from halo.runtime.load_coordination import joined_node_load
from halo.runtime.ranks import is_global_main_process
from halo.trainers.embedding.embedding_rules import (
    reject_injected_lora_parallelism,
    reject_uneven_sentence_transformers_batches,
)
from halo.trainers.embedding.preloaded_transformer import PreloadedTransformer
from halo.trainers.embedding.trainer import EmbeddingTrainer
from halo.trainers.loop.dataloaders import uses_dp_sharded_loader

logger = logging.getLogger(__name__)

# What the parallel loader's pipeline rebuilds past its backbone.
REBUILT_MODULES = (Pooling, Normalize)


class SourcePipeline(NamedTuple):
    """A checkpoint's sentence-transformers pipeline as the parallel loader rebuilds it around its own backbone:
    rebuilt modules, ``modules.json`` types of any other, backbone settings for :class:`PreloadedTransformer` and
    model-level settings. All empty for a bare backbone."""

    modules: list[nn.Module]
    unrebuilt: list[str]
    backbone_settings: dict
    model_settings: dict


def resolve_embedding_max_length(embedding_config: EmbeddingConfig, transformer_module) -> int:
    """Both loading paths land here before the length reaches a tokenizer, so ``null`` means the backbone's context
    window on either. ``transformer_module`` holds the backbone as ``auto_model``."""
    backbone = getattr(transformer_module, "auto_model", None)
    if backbone is None:
        raise ValueError(
            f"The loaded SentenceTransformer pipeline opens with {type(transformer_module).__name__}, "
            "which carries no transformer backbone — there is nothing to train or to read a context "
            "window from. Point model_name_or_path at a transformer/ST checkpoint."
        )
    embedding_config.max_length = resolve_length_to_context(
        embedding_config.max_length, backbone, transformer_module.tokenizer
    )
    return embedding_config.max_length


def _align_pipeline_modules(modules: Iterable[nn.Module], embedding_config: EmbeddingConfig) -> bool:
    """Apply ``pooling_mode`` and hold Normalize to ``normalize_embeddings``; True when a Normalize must be appended.
    Warns on every change, since the config overrides a checkpoint's own pooling (Qwen3-Embedding is ``lasttoken``)."""
    pooling_modules = [module for module in modules if isinstance(module, Pooling)]
    if not pooling_modules:
        raise ValueError(
            "The loaded SentenceTransformer pipeline has no Pooling module, so `pooling_mode` cannot be "
            "applied. Point model_name_or_path at a transformer/ST checkpoint (SentenceTransformer adds "
            "mean pooling to a bare backbone) instead of a pre-pooled encoder."
        )
    for module in pooling_modules:
        if module.pooling_mode != embedding_config.pooling_mode:
            if is_global_main_process():
                logger.warning(
                    f"Overriding the checkpoint's pooling_mode {module.pooling_mode!r} with the configured "
                    f"{embedding_config.pooling_mode!r}."
                )
            module.pooling_mode = embedding_config.pooling_mode

    has_normalize = any(isinstance(module, Normalize) for module in modules)
    if not embedding_config.normalize_embeddings and has_normalize:
        raise ValueError(
            "normalize_embeddings=False, but the loaded checkpoint's pipeline ends in a Normalize module. "
            "Dropping it would change what the checkpoint's downstream similarity thresholds mean — set "
            "normalize_embeddings: true, or use a checkpoint without the Normalize module."
        )
    appends_normalize = embedding_config.normalize_embeddings and not has_normalize
    if appends_normalize and is_global_main_process():
        logger.warning("Appending a Normalize module (normalize_embeddings=True); the checkpoint had none.")
    return appends_normalize


def align_st_pipeline_to_config(st_model: SentenceTransformer, embedding_config: EmbeddingConfig) -> None:
    """``max_length`` must already be resolved; it becomes SentenceTransformers' ``max_seq_length``, the length its
    tokenizer truncates to."""
    st_model.max_seq_length = embedding_config.max_length
    if _align_pipeline_modules(st_model, embedding_config):
        st_model.append(Normalize())


def _rebuilt_module_class(module_type: str) -> type[nn.Module] | None:
    """The class of :data:`REBUILT_MODULES` a ``modules.json`` type names, its legacy spelling included."""
    package, _, name = module_type.rpartition(".")
    if package.partition(".")[0] != "sentence_transformers":
        return None
    return next((cls for cls in REBUILT_MODULES if cls.__name__ == name), None)


def read_source_pipeline(source: str, revision: str | None, *, local_files_only: bool) -> SourcePipeline:
    """Read from the cache alone under ``local_files_only``."""
    location = {"revision": revision, "local_files_only": local_files_only}

    def read_json(filename: str, subfolder: str = ""):
        path = load_file_path(source, filename, subfolder=subfolder, **location)
        if path is None:
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    entries = read_json(ST_MODULES_FILE)
    if not entries:
        return SourcePipeline([], [], {}, {})
    backbone, *rest = entries
    modules, unrebuilt = [], []
    for entry in rest:
        module_class = _rebuilt_module_class(entry["type"])
        if module_class is None:
            unrebuilt.append(entry["type"])
        else:
            modules.append(module_class.load(source, subfolder=entry["path"], **location))
    backbone_config = read_json(PreloadedTransformer.config_file_name, backbone["path"]) or {}
    model_config = read_json(ST_MODEL_SETTINGS_FILE) or {}
    return SourcePipeline(
        modules,
        unrebuilt,
        {key: backbone_config[key] for key in PreloadedTransformer.config_keys if key in backbone_config},
        {key: model_config[key] for key in ST_CARRIED_MODEL_SETTINGS if key in model_config},
    )


def _assemble_preloaded_pipeline(
    context: ScriptContext,
    embedding_config: EmbeddingConfig,
    model_config: ModelArguments,
    args: EmbeddingScriptArguments,
) -> SentenceTransformer:
    """The parallel loader's backbone, then the checkpoint's own pipeline aligned to the config, or over a bare
    backbone the pooling and normalization the config describes."""
    source = hub_metadata_main_first(
        "embedding_source_pipeline",
        lambda local_files_only: read_source_pipeline(
            context.model_source, model_config.model_revision, local_files_only=local_files_only
        ),
    )
    if source.unrebuilt:
        raise ValueError(
            f"{context.model_source} is a sentence-transformers pipeline holding {source.unrebuilt} past its "
            "backbone, which the parallel loader's pipeline does not rebuild (it builds the backbone, Pooling and "
            "Normalize), so the run would train and export the pipeline without them. Train it where "
            "sentence-transformers loads the whole pipeline: no EP or TP, and use_grouped_gemm: false on a MoE "
            "(one GPU or accelerate DDP if a module outside the backbone trains), or start from the backbone alone."
        )
    modules = source.modules
    # Before the load, so refusing to drop the checkpoint's Normalize costs no weight read.
    if modules and _align_pipeline_modules(modules, embedding_config):
        modules.append(Normalize())

    # sentence-transformers pads each text column's batch: more than one row per batch reaches the varlen call.
    plan = plan_row_attention(
        model_config,
        embedding_config,
        sinks_policy=SinksPolicy.from_flags(reset_sinks=args.reset_sinks, train_sinks=args.train_sinks),
    )
    backbone, tokenizer = load_model_and_tokenizer(
        context, embedding_config, model_config, args, attn_default=plan.attn_default, model_class=AutoModel
    )

    transformer_module = PreloadedTransformer(auto_model=backbone, tokenizer=tokenizer, **source.backbone_settings)
    transformer_module.max_seq_length = resolve_embedding_max_length(embedding_config, transformer_module)

    if not modules:
        # get_text_config() resolves composite configs whose hidden_size lives in text_config (Qwen3.5, Gemma4).
        hidden_dim = backbone.config.get_text_config().hidden_size
        modules = [Pooling(word_embedding_dimension=hidden_dim, pooling_mode=embedding_config.pooling_mode)]
        if embedding_config.normalize_embeddings:
            modules.append(Normalize())

    st_model = SentenceTransformer(modules=[transformer_module, *modules], **source.model_settings)
    # The TP loader attaches the mesh to the backbone but the trainer reads it off the ST model, and refuses a TP run
    # without one.
    mesh = getattr(backbone, "_device_mesh", None)
    if mesh is not None:
        st_model._device_mesh = mesh
    return st_model


def _load_sentence_transformer(
    context: ScriptContext,
    embedding_config: EmbeddingConfig,
    model_config: ModelArguments,
    args: EmbeddingScriptArguments,
) -> SentenceTransformer:
    """The pipeline sentence-transformers loads itself, with the parallel loader's post-load steps and settings."""
    parallelism_config = context.parallelism_config
    model_kwargs: dict = {"dtype": resolve_training_dtype(embedding_config)}

    # The parallel loader's two seams, as sentence-transformers loads the backbone itself: the source fetched once per
    # scope and agreed across ranks, then a node-batched load joined over the store. The whole repo: pooling and dense
    # modules sit in subfolders.
    revision = resolve_model_source(
        context.model_source, model_config.model_revision, tag="embedding_model", whole_repo=True
    ).revision
    # The parallel loader's family patches, keyed on the backbone's config; unpinned, transformers picks the backend at
    # the build and only backend-independent patches apply (Zaya's checkpointing refusal).
    backbone_config = resolve_build_config(
        context.model_source, revision=revision, trust_remote_code=model_config.trust_remote_code
    )
    sinks_policy = SinksPolicy.from_flags(reset_sinks=args.reset_sinks, train_sinks=args.train_sinks)
    reject_trainable_sinks_without_sinks(backbone_config, sinks_policy)
    if attn_implementation := apply_family_patches(backbone_config, model_config.attn_implementation):
        model_kwargs["attn_implementation"] = attn_implementation
    # sentence-transformers reads the backbone's config.json itself, so the stamp would never reach its build.
    if getattr(backbone_config, EXTERNAL_TABLES_STAMP, False):
        raise ValueError(
            f"{backbone_config.model_type} builds its external tables (Qwen3.8-Flash-Next's n-gram table) in host "
            "memory only from the config its family patches stamp, and sentence-transformers reads the backbone's "
            "config itself, so the table would build as a trainable embedding on every rank. Embedding training "
            "takes no backbone with external tables; train a backbone without them."
        )
    with (
        joined_node_load(
            f"Embedding model load from {context.model_source}", parallelism_config.max_concurrent_loading
        ),
        full_weights_on_every_rank(),
    ):
        st_model = SentenceTransformer(
            context.model_source,
            revision=revision,
            trust_remote_code=model_config.trust_remote_code,
            model_kwargs=model_kwargs,
        )
    # Refuses a pipeline without a transformer backbone before anything reads it.
    resolve_embedding_max_length(embedding_config, st_model[0])
    # ST loads the backbone itself, so the parallel loader's post-load steps run here too (a GptOss backbone would
    # otherwise keep live sinks under a backend that drops them).
    backbone = st_model[0].auto_model
    drop_toolkit_stamps(backbone.config)
    cast_parameters_to_run_dtype(backbone, model_kwargs["dtype"], keep_fp32=parallelism_config.fp32_non_ep_params)
    finalize_loaded_model(backbone)
    apply_run_model_policies(
        backbone,
        backbone.config,
        sinks_policy=sinks_policy,
        attn_implementation=model_kwargs.get("attn_implementation"),
    )
    # Every rank builds the pipeline here, so the FA4 and flex-sliding compiles run fenced, not at a first forward.
    warm_attention_kernels(backbone, dtype=model_kwargs["dtype"])
    # The checkpoint's modules.json decides pooling/normalization/length; unaligned, those three knobs are inert.
    align_st_pipeline_to_config(st_model, embedding_config)
    return st_model


def _ep1_layers_requested(context: ScriptContext, model_config: ModelArguments) -> bool:
    """A MoE an EP layer class wraps, at ``use_grouped_gemm``, in a run without EP or TP. COLLECTIVE where the verdict
    needs the checkpoint's config."""
    parallelism_config = context.parallelism_config
    if parallelism_config.has_ep_group or parallelism_config.is_tp_mode or not parallelism_config.needs_ep_layers:
        return False
    return wraps_experts(fetch_checkpoint_config(model_config, tag="embedding_loader_route"), ep_layers_requested=True)


def _loads_through_parallel_loader(context: ScriptContext, model_config: ModelArguments) -> bool:
    """Under EP or TP, and where the grouped-GEMM EP layers take a MoE's experts, which only that loader puts in.
    COLLECTIVE as :func:`_ep1_layers_requested`."""
    parallelism_config = context.parallelism_config
    return (
        parallelism_config.has_ep_group
        or parallelism_config.is_tp_mode
        or _ep1_layers_requested(context, model_config)
    )


def reject_lora_before_the_load(context: ScriptContext, model_config: ModelArguments) -> None:
    """Refuse injected LoRA under EP or on the grouped-GEMM EP layers before the model loads; the trainer's own gate
    refuses a hand-built run after it. COLLECTIVE as :func:`_ep1_layers_requested`."""
    reject_injected_lora_parallelism(
        context.parallelism_config,
        has_injected_lora=model_config.use_peft,
        has_ep_layers=model_config.use_peft and _ep1_layers_requested(context, model_config),
    )


def reject_knobs_injected_lora_cannot_honor(model_config: ModelArguments, args: EmbeddingScriptArguments) -> None:
    """Injection freezes every non-adapter parameter, which these two would need trained."""
    if not model_config.use_peft:
        return
    # The freeze re-freezes trainable copies and the wrapper renames the base tensor, so the saved ST module lacks the
    # plain <mod>.weight.
    if model_config.lora_modules_to_save:
        raise ValueError(
            f"lora_modules_to_save={list(model_config.lora_modules_to_save)} is not supported for "
            "embedding training: the SentenceTransformer path injects adapters in place, which "
            "leaves the modules_to_save copies frozen and rewrites the base module's keys so the "
            "saved model no longer reloads. Train those modules with a full fine-tune instead."
        )
    if args.train_sinks:
        raise ValueError(
            "train_sinks: true needs full fine-tuning: embedding LoRA freezes every non-adapter "
            "parameter after injection, so the sinks would train nowhere. Keep the sinks live and "
            "frozen instead (reset_sinks: false without train_sinks), or drop the adapters."
        )


def reject_fp32_masters_without_ep_layers(
    context: ScriptContext, model_config: ModelArguments, dist_args: ParallelismArguments
) -> None:
    """A sentence-transformers load builds no EP layer, the one reader of these two; the parallel loader refuses them
    itself on a run building none. COLLECTIVE as :func:`_ep1_layers_requested`, once either is set."""
    if (dist_args.fp32_router or dist_args.fp32_experts) and not _loads_through_parallel_loader(context, model_config):
        reject_unsupported_args(
            "Embedding training on a sentence-transformers load, which builds no EP layers",
            fp32_router=dist_args.fp32_router,
            fp32_experts=dist_args.fp32_experts,
        )


def build_sentence_transformer(
    context: ScriptContext,
    embedding_config: EmbeddingConfig,
    model_config: ModelArguments,
    args: EmbeddingScriptArguments,
) -> SentenceTransformer:
    """The parallel loader's backbone in :class:`PreloadedTransformer` where that loader owns the load, otherwise
    ``SentenceTransformer(model_name)``."""
    # Neither path applies QLoRA/k-bit, so --load_in_4bit/--load_in_8bit would train at full precision.
    if get_quantization_config(model_config) is not None:
        raise ValueError(
            "QLoRA / k-bit quantization is not supported for embedding training. The SentenceTransformer "
            "loader does not apply a quantization_config, so --load_in_4bit/--load_in_8bit would be "
            "silently ignored. Use full-precision LoRA (--use_peft) or full fine-tuning."
        )
    if _loads_through_parallel_loader(context, model_config):
        st_model = _assemble_preloaded_pipeline(context, embedding_config, model_config, args)
    else:
        st_model = _load_sentence_transformer(context, embedding_config, model_config, args)
    # sentence-transformers writes its card from model_card_data, never from the backbone's model_tags.
    st_model.model_card_data.add_tags(list(HUB_TAGS))
    return st_model


def inject_lora(model: SentenceTransformer, model_config: ModelArguments, args: EmbeddingScriptArguments) -> None:
    """Inject LoRA into the transformer backbone in place and train the adapters alone; checkpoints fold and resume
    only its adapters. ``inject_adapter_in_model``, not ``add_adapter``: that goes through transformers' adapter API,
    which requires peft >= 0.20.0, above the image's pin (pyproject.toml). The trainer's gates reject EP, TP and the
    grouped-GEMM EP layers."""
    backbone = model[0].auto_model
    peft_config = build_peft_config(backbone, model_config)
    if peft_config is None:
        return
    inject_adapter_in_model(peft_config, backbone)
    adapters = tuner_adapter_param_ids(backbone)
    for param in model.parameters():
        param.requires_grad = id(param) in adapters
    if is_global_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Applied LoRA adapters (r={model_config.lora_r}); trainable params: {trainable / 1e6:.2f}M")


def _reject_unhonored_args(args: EmbeddingScriptArguments) -> None:
    """Refuse knobs this path would ignore: it has no configure_run_tokenizer / freeze stage, so a freeze request would
    become a full fine-tune."""
    reject_unsupported_args(
        "Embedding training",
        pad_token=args.pad_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        chat_template=args.chat_template,
        force_chat_template=args.force_chat_template,
        added_special_tokens=args.added_special_tokens,
        unfreeze_layers_patterns=args.unfreeze_layers_patterns,
        freeze_layers_patterns=args.freeze_layers_patterns,
        # No chat-template rendering and no log_dataset_examples stage on this path.
        tools_field=args.tools_field,
        log_decoded_samples=args.log_decoded_samples,
        # The parallel loader pins model_class=AutoModel and the SentenceTransformer branch never reads the flag.
        text_only_model=args.text_only_model,
    )
    # SentenceTransformer owns tokenization, so only the default backend is accepted.
    reject_non_default_args("Embedding training", args, "tokenizer_backend")


def main():
    args, embedding_config, model_config, dist_args = parse_script_args(
        EmbeddingScriptArguments, EmbeddingConfig, ModelArguments, ParallelismArguments
    )
    _reject_unhonored_args(args)

    # LoRA is injected below via inject_adapter_in_model, not the EP grouped-adapter split: EP and the
    # grouped-GEMM EP layers refuse injected LoRA (reject_lora_before_the_load), TP the mixin's LoRA gate.
    context = init_training_script(
        args,
        embedding_config,
        model_config,
        dist_args,
        script_prefix="embedding",
        trainer_cls=EmbeddingTrainer,
        split_expert_lora=False,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=None)
    reject_image_columns(ds, "Embedding training")
    train_dataset = ds["train"]
    eval_dataset = ds.get("test")
    # Here, ahead of the model load: the trainer would refuse the same run only after it.
    reject_uneven_sentence_transformers_batches(
        embedding_config, train_dataset, toolkit_loader=uses_dp_sharded_loader(parallelism_config, dataset_presharded)
    )
    reject_knobs_injected_lora_cannot_honor(model_config, args)
    reject_lora_before_the_load(context, model_config)
    reject_fp32_masters_without_ep_layers(context, model_config, dist_args)

    if is_global_main_process():
        logger.info(f"Train dataset: {len(train_dataset)} examples")
        logger.info(f"Train dataset columns: {train_dataset.column_names}")
        if eval_dataset is not None:
            logger.info(f"Eval dataset: {len(eval_dataset)} examples")

    model = build_sentence_transformer(context, embedding_config, model_config, args)

    if is_global_main_process():
        logger.info(f"Model: {model}")
        logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    inject_lora(model, model_config, args)

    barrier()

    callbacks = configure_balancing_and_build_callbacks(args, embedding_config, model, context)

    trainer = EmbeddingTrainer(
        model=model,
        args=embedding_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        context,
        method_name="Embedding",
        extra_start_log=[f"Loss: {embedding_config.loss_type}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
