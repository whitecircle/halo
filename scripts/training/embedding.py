#!/usr/bin/env python
"""Distributed embedding training: sentence-transformers losses over pairs, triplets, scored pairs and labeled texts.

Text-only (the EP/TP path's preloaded transformer only tokenizes text); CP is not supported, since pooling
reads the complete sequence. Usage:
    torchrun --nproc_per_node=8 scripts/training/embedding.py examples/embedding/gptoss/embedding-gptoss-20b-gooaq-ep.yaml
"""

import logging

from peft import inject_adapter_in_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling
from transformers import AutoConfig, AutoModel
from trl import get_quantization_config

from halo.checkpoint.model_card import HUB_TAGS
from halo.config_validation import reject_non_default_args, reject_unsupported_args
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.collators.collator_rules import reject_fa4_varlen_training, rows_reach_varlen_kernel
from halo.data.vlm_detection import reject_image_columns
from halo.entrypoints.data_setup import load_script_datasets
from halo.entrypoints.model_setup import default_training_rows_attn_implementation, load_model_and_tokenizer
from halo.entrypoints.script import (
    ScriptContext,
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.checkpoint_coverage import full_weights_on_every_rank
from halo.model_loading.dtype import cast_parameters_to_run_dtype, resolve_training_dtype
from halo.model_loading.model_preparation import apply_family_patches, apply_run_model_policies
from halo.model_loading.parallel.model_source import resolve_model_source
from halo.model_loading.parallel.peft_setup import build_peft_config
from halo.model_loading.parallel.whole_model import warm_attention_kernels
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.context_window import resolve_length_to_context
from halo.models.peft_adapters import tuner_adapter_param_ids
from halo.runtime.collectives import barrier
from halo.runtime.load_coordination import joined_node_load
from halo.runtime.ranks import is_global_main_process
from halo.trainers.dataloaders import uses_dp_sharded_loader
from halo.trainers.embedding.embedding_rules import reject_uneven_sentence_transformers_batches
from halo.trainers.embedding.preloaded_transformer import PreloadedTransformer
from halo.trainers.embedding.trainer import EmbeddingTrainer

logger = logging.getLogger(__name__)


def resolve_embedding_max_length(embedding_config: EmbeddingConfig, transformer_module) -> int:
    """Fix ``embedding_config.max_length`` against the pipeline's backbone and return it.

    Both loading paths land here before the length reaches a tokenizer, so ``null`` means the same
    thing on either: the backbone's context window. ``transformer_module`` is the pipeline's first
    module — SentenceTransformers' own ``Transformer`` or our :class:`PreloadedTransformer`, both of
    which hold the backbone as ``auto_model``.
    """
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


def align_st_pipeline_to_config(st_model: SentenceTransformer, embedding_config: EmbeddingConfig) -> None:
    """Apply ``pooling_mode`` / ``normalize_embeddings`` / ``max_length`` to a loaded ST pipeline.

    ``max_length`` must already be resolved (:func:`resolve_embedding_max_length`); it is installed as
    SentenceTransformers' ``max_seq_length``, the length its tokenizer truncates to.

    Warns on every value it changes, since the config overrides a checkpoint that ships its own
    pooling (Qwen3-Embedding is ``lasttoken``).
    """
    st_model.max_seq_length = embedding_config.max_length

    pooling_modules = [module for module in st_model if isinstance(module, Pooling)]
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

    has_normalize = any(isinstance(module, Normalize) for module in st_model)
    if embedding_config.normalize_embeddings and not has_normalize:
        if is_global_main_process():
            logger.warning("Appending a Normalize module (normalize_embeddings=True); the checkpoint had none.")
        st_model.append(Normalize())
    elif not embedding_config.normalize_embeddings and has_normalize:
        raise ValueError(
            "normalize_embeddings=False, but the loaded checkpoint's pipeline ends in a Normalize module. "
            "Dropping it would change what the checkpoint's downstream similarity thresholds mean — set "
            "normalize_embeddings: true, or use a checkpoint without the Normalize module."
        )


def _assemble_preloaded_pipeline(
    context: ScriptContext,
    embedding_config: EmbeddingConfig,
    model_config: ModelArguments,
    args: EmbeddingScriptArguments,
) -> SentenceTransformer:
    """The EP/TP pipeline: the parallel loader's backbone, its pooling and normalization from the config."""
    # sentence-transformers pads each text column's batch: more than one row per batch reaches the varlen call.
    rows = {
        "packing": False,
        "padding_free": False,
        "per_device_train_batch_size": embedding_config.per_device_train_batch_size,
    }
    varlen_rows = rows_reach_varlen_kernel(**rows)
    backbone, tokenizer = load_model_and_tokenizer(
        context,
        embedding_config,
        model_config,
        args,
        attn_default=default_training_rows_attn_implementation(
            model_config, **rows, dtype=resolve_training_dtype(embedding_config)
        ),
        model_class=AutoModel,
    )
    reject_fa4_varlen_training(backbone.config, varlen_rows=varlen_rows)

    transformer_module = PreloadedTransformer(auto_model=backbone, tokenizer=tokenizer)
    transformer_module.max_seq_length = resolve_embedding_max_length(embedding_config, transformer_module)

    # get_text_config() resolves composite (multimodal) configs whose hidden_size lives in
    # text_config (Qwen3.5, Gemma4); plain configs return themselves.
    hidden_dim = backbone.config.get_text_config().hidden_size
    pooling_module = Pooling(
        word_embedding_dimension=hidden_dim,
        pooling_mode=embedding_config.pooling_mode,
    )

    modules = [transformer_module, pooling_module]
    if embedding_config.normalize_embeddings:
        modules.append(Normalize())

    st_model = SentenceTransformer(modules=modules)
    # The TP loader attaches the (dp, tp) mesh to the backbone but the trainer reads it off the
    # top-level ST model. Without propagating it, TP+DP builds a conflicting mesh and pure TP skips
    # the TP grad-sync.
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
    """The pipeline sentence-transformers loads itself, given the parallel loader's post-load steps and the
    config's pooling, normalization and length."""
    parallelism_config = context.parallelism_config
    model_kwargs: dict = {"dtype": resolve_training_dtype(embedding_config)}

    # The parallel loader's two seams, since sentence-transformers loads the backbone itself: the
    # source fetched once per scope and agreed across ranks (revision threaded as on the EP/TP
    # branch, so a pinned checkpoint does not depend on flags), then a node-batched load joined
    # over the store. The whole repo: the pipeline's pooling and dense modules sit in subfolders.
    apply_remote_code_compat_shims()
    revision = resolve_model_source(
        context.model_source, model_config.model_revision, tag="embedding_model", whole_repo=True
    ).revision
    # Pinned, the backend takes the family patches the parallel loader applies, keyed on the backbone's
    # own config; unset, transformers picks it at the build.
    if model_config.attn_implementation:
        backbone_config = AutoConfig.from_pretrained(
            context.model_source, revision=revision, trust_remote_code=model_config.trust_remote_code
        )
        model_kwargs["attn_implementation"] = apply_family_patches(backbone_config, model_config.attn_implementation)
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
    # ST loads the backbone itself, so the post-load steps of the parallel loader run here too: the
    # run-dtype cast, the buffer repair, and the shared finalization (a GptOss backbone would
    # otherwise keep live sinks under an attention backend that drops them).
    backbone = st_model[0].auto_model
    cast_parameters_to_run_dtype(backbone, model_kwargs["dtype"], keep_fp32=parallelism_config.fp32_non_ep_params)
    finalize_loaded_model(backbone)
    apply_run_model_policies(
        backbone,
        backbone.config,
        sinks_policy=SinksPolicy.from_flags(reset_sinks=args.reset_sinks, train_sinks=args.train_sinks),
        attn_implementation=model_kwargs.get("attn_implementation"),
    )
    # Every rank builds the pipeline here, so the FA4 and flex-sliding compiles run fenced now rather than
    # at one rank's first forward while its peers wait in a collective.
    warm_attention_kernels(backbone, dtype=model_kwargs["dtype"])
    # The checkpoint's modules.json decides pooling/normalization/length here while the EP/TP branch
    # builds them from the config; align them or those three knobs are inert on the default path.
    align_st_pipeline_to_config(st_model, embedding_config)
    return st_model


def build_sentence_transformer(
    context: ScriptContext,
    embedding_config: EmbeddingConfig,
    model_config: ModelArguments,
    args: EmbeddingScriptArguments,
) -> SentenceTransformer:
    """Build the run's SentenceTransformer: under EP/TP the parallel loader's backbone wrapped in
    :class:`PreloadedTransformer` and assembled into a pipeline, otherwise ``SentenceTransformer(model_name)``."""
    # QLoRA/k-bit is unsupported here: neither the EP/TP loader nor the SentenceTransformer path
    # applies it, so --load_in_4bit/--load_in_8bit would train at full precision.
    if get_quantization_config(model_config) is not None:
        raise ValueError(
            "QLoRA / k-bit quantization is not supported for embedding training. The SentenceTransformer "
            "loader does not apply a quantization_config, so --load_in_4bit/--load_in_8bit would be "
            "silently ignored. Use full-precision LoRA (--use_peft) or full fine-tuning."
        )
    parallelism_config = context.parallelism_config
    if parallelism_config.has_ep_group or parallelism_config.is_tp_mode:
        st_model = _assemble_preloaded_pipeline(context, embedding_config, model_config, args)
    else:
        st_model = _load_sentence_transformer(context, embedding_config, model_config, args)
    # sentence-transformers writes its card from model_card_data, never from the backbone's model_tags.
    st_model.model_card_data.add_tags(list(HUB_TAGS))
    return st_model


def inject_lora(model: SentenceTransformer, model_config: ModelArguments, args: EmbeddingScriptArguments) -> None:
    """Inject the run's LoRA into ``model``'s transformer backbone in place and train the adapters
    alone; no-op without PEFT.

    The backbone only: checkpoints fold and resume its adapters, and the trainer refuses any outside
    it. ``inject_adapter_in_model``, not ``SentenceTransformer.add_adapter``: that goes through
    transformers' adapter API, which requires peft >= 0.20.0 while the image pins 0.18.1. EP/TP are
    rejected by the trainer's own gates, which see the injected adapters structurally.
    """
    backbone = model[0].auto_model
    peft_config = build_peft_config(backbone, model_config)
    if peft_config is None:
        return
    # modules_to_save is unsupported here: the trainable copies are created, the freeze below
    # re-freezes them, and the wrapper renames the base tensor, so the saved ST module lacks the
    # plain <mod>.weight.
    if model_config.lora_modules_to_save:
        raise ValueError(
            f"lora_modules_to_save={list(model_config.lora_modules_to_save)} is not supported for "
            "embedding training: the SentenceTransformer path injects adapters in place, which "
            "leaves the modules_to_save copies frozen and rewrites the base module's keys so the "
            "saved model no longer reloads. Train those modules with a full fine-tune instead."
        )
    # The freeze below re-freezes everything that is not an adapter — trainable sinks included.
    if args.train_sinks:
        raise ValueError(
            "train_sinks: true needs full fine-tuning: embedding LoRA freezes every non-adapter "
            "parameter after injection, so the sinks would train nowhere. Keep the sinks live and "
            "frozen instead (reset_sinks: false without train_sinks), or drop the adapters."
        )
    inject_adapter_in_model(peft_config, backbone)
    adapters = tuner_adapter_param_ids(backbone)
    for param in model.parameters():
        param.requires_grad = id(param) in adapters
    if is_global_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Applied LoRA adapters (r={model_config.lora_r}); trainable params: {trainable / 1e6:.2f}M")


def _reject_unhonored_args(args: EmbeddingScriptArguments) -> None:
    """Refuse the shared script knobs the sentence-transformers path would parse and ignore: it has no
    configure_run_tokenizer / freeze stage, so a freeze request would become a full fine-tune."""
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
        # Neither branch can honor it: the EP/TP loader pins model_class=AutoModel (the loader then
        # only warns), and the SentenceTransformer branch never reads the flag.
        text_only_model=args.text_only_model,
    )
    # SentenceTransformer owns tokenization, so only the default backend is accepted.
    reject_non_default_args("Embedding training", args, "tokenizer_backend")


def main():
    args, embedding_config, model_config, dist_args = parse_script_args(
        EmbeddingScriptArguments, EmbeddingConfig, ModelArguments, ParallelismArguments
    )
    _reject_unhonored_args(args)

    # LoRA on the SentenceTransformer path is injected below via inject_adapter_in_model (not the EP
    # grouped-adapter split, which is rejected together with EP/TP further down).
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
