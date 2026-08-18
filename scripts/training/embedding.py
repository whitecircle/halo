#!/usr/bin/env python
"""Distributed embedding training with Expert and Tensor Parallelism support.

Fine-tunes embedding models with sentence-transformers losses over the common dataset shapes (pairs,
triplets, scored pairs, labeled texts).

Modality: text-only. Multimodal embedding would need a multimodal preloaded transformer (the EP/TP
path here only tokenizes text); standard sentence-transformers multimodal mode is out of scope.

CP is not supported (pooling reads the complete sequence); use EP and/or TP.

Usage:
    torchrun --nproc_per_node=8 scripts/training/embedding.py \\
        examples/embedding/qwen3/embedding-qwen3-4b-nq.yaml --expert_parallel_size=8
"""

from accelerate.logging import get_logger
from peft import inject_adapter_in_model
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize, Pooling
from transformers import AutoModel
from trl import ModelConfig, get_peft_config, get_quantization_config

from src.args.distributed_args import DistributedArguments
from src.args.embedding_args import EmbeddingScriptArguments
from src.configs.embedding_config import EmbeddingConfig
from src.data.sources.loading import reject_image_columns
from src.distributed.filesystem import fs_aware_main_first
from src.distributed.runtime import barrier, is_global_main_process
from src.models.loading.dtype import resolve_training_dtype
from src.models.loading.tokenizer_setup import resolve_length_to_context
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy
from src.trainers.embedding.sentence_transformers_compat import PreloadedTransformer
from src.trainers.embedding.trainer import EmbeddingTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    ScriptRuntime,
    apply_distributed_trainer_config,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    load_script_datasets,
    load_script_model,
    reject_unsupported_args,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")


def build_sentence_transformer(
    runtime: ScriptRuntime,
    embedding_config: EmbeddingConfig,
    model_config: ModelConfig,
    dist_args: DistributedArguments,
) -> SentenceTransformer:
    """Build a SentenceTransformer with optional EP/TP parallelism.

    Standard mode uses ``SentenceTransformer(model_name)`` directly; EP/TP mode loads the backbone
    with ``load_script_model(model_class=AutoModel)``, wraps it in :class:`PreloadedTransformer` and
    builds the ``SentenceTransformer`` pipeline from modules.
    """
    # QLoRA/k-bit is unsupported here: neither the EP/TP loader nor the SentenceTransformer path
    # applies it, so --load_in_4bit/--load_in_8bit would train at full precision.
    if get_quantization_config(model_config) is not None:
        raise ValueError(
            "QLoRA / k-bit quantization is not supported for embedding training. The SentenceTransformer "
            "loader does not apply a quantization_config, so --load_in_4bit/--load_in_8bit would be "
            "silently ignored. Use full-precision LoRA (--use_peft) or full fine-tuning."
        )
    parallelism_config = runtime.parallelism_config
    if parallelism_config.is_ep_mode or parallelism_config.is_tp_mode:
        backbone, tokenizer = load_script_model(
            runtime,
            embedding_config,
            model_config,
            dist_args,
            model_class=AutoModel,
        )

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

    else:
        model_kwargs: dict = {"dtype": resolve_training_dtype(embedding_config)}
        if model_config.attn_implementation:
            model_kwargs["attn_implementation"] = model_config.attn_implementation

        # Main-first like every other load path, otherwise every rank downloads and materializes the
        # model at once. revision is threaded as on the EP/TP branch so a pin holds on both.
        with fs_aware_main_first("embedding_model"):
            st_model = SentenceTransformer(
                runtime.model_source,
                revision=model_config.model_revision,
                trust_remote_code=model_config.trust_remote_code,
                model_kwargs=model_kwargs,
            )
        # ST loads the backbone itself, so the parallel loader's sinks policy is applied here too: a
        # GptOss backbone would otherwise keep live sinks under an attention backend that drops them.
        backbone = getattr(st_model[0], "auto_model", None)
        if backbone is not None:
            apply_sinks_policy(
                backbone,
                backbone.config,
                policy=SinksPolicy.from_flags(reset_sinks=dist_args.reset_sinks, train_sinks=dist_args.train_sinks),
                attn_implementation=model_kwargs.get("attn_implementation"),
            )
        # The checkpoint's modules.json decides pooling/normalization/length here while the EP/TP branch
        # builds them from the config; align them or those three knobs are inert on the default path.
        resolve_embedding_max_length(embedding_config, st_model[0])
        align_st_pipeline_to_config(st_model, embedding_config)
        return st_model


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
            logger.warning(
                f"Overriding the checkpoint's pooling_mode {module.pooling_mode!r} with the configured "
                f"{embedding_config.pooling_mode!r}."
            )
            module.pooling_mode = embedding_config.pooling_mode

    has_normalize = any(isinstance(module, Normalize) for module in st_model)
    if embedding_config.normalize_embeddings and not has_normalize:
        logger.warning("Appending a Normalize module (normalize_embeddings=True); the checkpoint had none.")
        st_model.append(Normalize())
    elif not embedding_config.normalize_embeddings and has_normalize:
        raise ValueError(
            "normalize_embeddings=False, but the loaded checkpoint's pipeline ends in a Normalize module. "
            "Dropping it would change what the checkpoint's downstream similarity thresholds mean — set "
            "normalize_embeddings: true, or use a checkpoint without the Normalize module."
        )


def main():
    parser = H4ArgumentParser((EmbeddingScriptArguments, EmbeddingConfig, ModelConfig, DistributedArguments))
    args, embedding_config, model_config, dist_args = parser.parse()

    # The ST path has no setup_model_and_tokenizer / freeze stage, so these shared knobs would parse
    # and do nothing (a freeze request would become a full fine-tune). Reject them instead.
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
        # SentenceTransformer owns tokenization; "hf" (the default) is not a request.
        tokenizer_backend=args.tokenizer_backend if args.tokenizer_backend != "hf" else None,
        # No chat-template rendering and no log_dataset_examples stage on this path.
        tools_field=args.tools_field,
        log_decoded_samples=args.log_decoded_samples,
        # Neither branch can honor it: the EP/TP loader pins model_class=AutoModel (the loader then
        # only warns), and the SentenceTransformer branch never reads the flag.
        text_only_model=dist_args.text_only_model,
    )

    # LoRA on the SentenceTransformer path is injected below via inject_adapter_in_model (not the EP
    # grouped-adapter split, which is rejected together with EP/TP further down).
    runtime = init_training_script(
        args,
        embedding_config,
        model_config,
        dist_args,
        script_prefix="embedding",
        supports_cp=False,
        supports_pp=False,
        split_expert_lora=False,
    )
    parallelism_config = runtime.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=None)
    reject_image_columns(ds, "Embedding training")
    train_dataset = ds["train"]
    eval_dataset = ds.get("test") or ds.get("validation")

    if is_global_main_process():
        logger.info(f"Train dataset: {len(train_dataset)} examples")
        logger.info(f"Train dataset columns: {train_dataset.column_names}")
        if eval_dataset is not None:
            logger.info(f"Eval dataset: {len(eval_dataset)} examples")

    model = build_sentence_transformer(runtime, embedding_config, model_config, dist_args)

    if is_global_main_process():
        logger.info(f"Model: {model}")
        logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Inject LoRA with peft's inject_adapter_in_model, not SentenceTransformer.add_adapter (ST 5.5
    # gates that on peft >= 0.18.2 while the image pins 0.18.1). It does not freeze, so freeze every
    # non-adapter param below.
    peft_config = get_peft_config(model_config)
    if peft_config is not None:
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
        if dist_args.train_sinks:
            raise ValueError(
                "train_sinks: true needs full fine-tuning: embedding LoRA freezes every non-adapter "
                "parameter after injection, so the sinks would train nowhere. Keep the sinks live and "
                "frozen instead (reset_sinks: false without train_sinks), or drop the adapters."
            )
        # EP/TP are rejected by the trainer's own gates, which see the injected adapters structurally.
        inject_adapter_in_model(peft_config, model)
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
        if is_global_main_process():
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Applied LoRA adapters (r={model_config.lora_r}); trainable params: {trainable / 1e6:.2f}M")

    apply_distributed_trainer_config(embedding_config, parallelism_config)

    barrier()

    callbacks = build_training_callbacks(args, embedding_config, model, parallelism_config)

    trainer = EmbeddingTrainer(
        model=model,
        args=embedding_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="Embedding",
        extra_start_log=[f"Loss: {embedding_config.loss_type}"],
    )


if __name__ == "__main__":
    run_training(main)()
