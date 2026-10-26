"""Grow a model's vocabulary by the patterns that tokenize to more than one token, mean-initializing the new
embedding rows, and save the patched checkpoint as a new base (optionally with a chat template and neutralized sinks)."""

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.offline_export import (
    preflight_model_load_resources,
    reject_in_place_conversion,
    save_full_checkpoint,
)
from halo.checkpoint.reading import reject_partial_checkpoint
from halo.model_loading.dtype import DTYPE_BY_NAME
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.tokenizer_setup import (
    adopt_tokenizer_chat_template,
    load_chat_template,
    load_processing_class,
    resolve_tokenizer,
    set_run_chat_template,
)
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, stamped_sinks_policy

logger = logging.getLogger(__name__)


def filter_patterns_needing_multiple_tokens(
    tokenizer: PreTrainedTokenizer, patterns: list[str]
) -> list[tuple[str, int]]:
    """Filter patterns that currently use more than 1 token."""
    patterns_to_add = []
    for pattern in patterns:
        tokens = tokenizer.encode(pattern, add_special_tokens=False)
        if len(tokens) > 1:
            patterns_to_add.append((pattern, len(tokens)))
    return patterns_to_add


def add_tokens_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    patterns: list[str],
) -> int:
    """
    Add new tokens to tokenizer and resize model embeddings.

    Returns the number of tokens added.
    """
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size}")

    patterns_to_add = filter_patterns_needing_multiple_tokens(tokenizer, patterns)

    if not patterns_to_add:
        logger.info("No patterns need to be added (all already single tokens)")
        return 0

    logger.info(f"Found {len(patterns_to_add)} patterns that use >1 token:")
    for i, (text, num_tokens) in enumerate(patterns_to_add, 1):
        logger.info(f"{i:2d}. '{text}' - {num_tokens} tokens -> 1 token")

    new_tokens = [text for text, _ in patterns_to_add]
    num_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Added {num_added} new tokens to tokenizer")

    new_vocab_size = len(tokenizer)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    logger.info(f"Difference: +{new_vocab_size - original_vocab_size}")

    logger.info("Calculating average embeddings for initialization...")
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        avg_input_embedding = input_embeddings[:original_vocab_size].mean(dim=0)

        output_embeddings = model.get_output_embeddings().weight
        avg_output_embedding = output_embeddings[:original_vocab_size].mean(dim=0)

    logger.info(f"Model embeddings shape before: {model.get_input_embeddings().weight.shape}")

    # Grow only. Models like gpt-oss ship an embedding padded larger than len(tokenizer): the rows
    # above len(tokenizer) are unassigned and every special (harmony EOS, <|call|>) sits below it, so a
    # shrink recovers only padding and any target under len(tokenizer) cuts the specials.
    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size > current_embedding_size:
        model.resize_token_embeddings(new_vocab_size)
    else:
        logger.info(
            f"Keeping existing embedding size {current_embedding_size} (>= new vocab {new_vocab_size}); "
            "added tokens reuse existing padding rows — not shrinking (would drop high special tokens)."
        )

    logger.info(f"Model embeddings shape after resize: {model.get_input_embeddings().weight.shape}")

    with torch.no_grad():
        for i in range(original_vocab_size, new_vocab_size):
            model.get_input_embeddings().weight[i] = avg_input_embedding
            model.get_output_embeddings().weight[i] = avg_output_embedding

    logger.info(f"Successfully initialized {new_vocab_size - original_vocab_size} new token embeddings!")

    logger.info("Verification: Testing new tokens in tokenizer")
    for i, (text, _old_num_tokens) in enumerate(patterns_to_add[:5], 1):
        new_tokens_enc = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(new_tokens_enc)
        match = "✓" if decoded == text else "✗"
        logger.info(f"{i}. '{text}' -> {len(new_tokens_enc)} token(s), ID: {new_tokens_enc}, Match: {match}")

    return num_added


def patch_vocabulary(
    model_id: str,
    output_dir: str,
    *,
    patterns: list[str],
    chat_template: str | None = None,
    reset_sinks: bool = False,
    dtype: str = "bfloat16",
    device_map: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool = False,
) -> None:
    """Add ``patterns`` to ``model_id``'s vocabulary (mean-initialized rows), optionally set its chat template and
    neutralize its sinks, and save the patched checkpoint to ``output_dir`` as a new base."""
    # Refused up front: an in-place patch has no undo, and a per-rank EP/TP-sharded source would load
    # with randomly initialized experts (missing keys only warn) while the patched save looked
    # complete.
    reject_in_place_conversion(model_id, output_dir)
    reject_partial_checkpoint(model_id)
    # Without --device_map the whole model lands in host RAM, and the patched output is about the
    # source's size on disk (not reported for a Hub id, where there is nothing local to measure).
    preflight_model_load_resources(model_id, output_dir, tool="patch_vocab", device_map=device_map)

    logger.info("PATCH VOCABULARY SCRIPT")
    logger.info(f"Model: {model_id}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Dtype: {dtype}")
    logger.info(f"Reset sinks: {reset_sinks}")

    if patterns:
        logger.info(f"Patterns to process: {len(patterns)}")

    # Full processor for VLMs, whose processor config and chat template have to survive the save, else
    # a plain tokenizer. New tokens go into this object's tokenizer, so the patched VLM reloads as
    # multimodal.
    logger.info("Loading tokenizer/processor...")
    processing_class = load_processing_class(model_id, trust_remote_code=trust_remote_code)
    if processing_class is None:
        raise ValueError(f"{model_id} ships no tokenizer, so there is no vocabulary to add tokens to.")
    tokenizer = resolve_tokenizer(processing_class)

    if chat_template:
        set_run_chat_template(tokenizer, load_chat_template(chat_template))
        adopt_tokenizer_chat_template(processing_class)

    # auto_load_model resolves the widest Auto* class (AutoModelForCausalLM would drop a vision
    # tower) and routes through the checkpoint-coverage gate: a truncated or key-mismatched source
    # would otherwise load with randomly initialized tensors and be saved as a complete-looking
    # patch.
    logger.info("Loading model...")
    model = auto_load_model(
        model_id,
        trust_remote_code=trust_remote_code,
        dtype=DTYPE_BY_NAME[dtype],
        device_map=device_map,
    )
    logger.info(f"Loaded model class: {type(model).__name__} (model_type={model.config.model_type})")

    if patterns:
        logger.info("ADDING NEW TOKENS")
        add_tokens_to_model(model, tokenizer, patterns)

    if reset_sinks:
        logger.info("RESETTING ATTENTION SINKS")
        # The trainers' own seam rather than a second walk: it neutralizes what every trainer
        # neutralizes and stamps the sink policy the provenance record and the merge tools read, so a
        # patched checkpoint and a trained one agree on what their sinks mean.
        apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
        # It returns without error on a family that carries no sinks, and the stamp is the only
        # signal it did anything: unchecked, the flag would save an untouched checkpoint that
        # downstream tools (provenance, the merge tools, the RL sink gate) read as sink-free.
        if stamped_sinks_policy(model) is None:
            raise ValueError(
                f"--reset_sinks was requested, but {model_id} (model_type="
                f"{model.config.model_type!r}) carries no attention sinks — nothing was reset and the "
                f"saved checkpoint would be identical. Drop --reset_sinks; it applies to sink-carrying "
                f"families (gpt-oss)."
            )

    logger.info("SAVING MODEL AND TOKENIZER")
    logger.info(f"Saving to: {output_dir}")

    # A new base: a resume sidecar of a source training checkpoint would restore that run over it.
    save_full_checkpoint(
        model,
        output_dir,
        processing_class=processing_class,
        source_dir=model_id,
        include_resume_sidecars=False,
        max_shard_size=max_shard_size,
    )

    logger.info("DONE!")
