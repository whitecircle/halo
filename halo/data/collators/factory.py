"""Selects the SFT data collator a run's context-parallel, padding-free, packing and completion-only
options call for."""

import logging

from halo.data.collators.collator_rules import reject_unsupported_collator_options
from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlattening,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding, DataCollatorForCompletionOnlyLM
from halo.data.turn_markers import resolve_eos_token_ids
from halo.models.segment_markers import segment_markers_for
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def select_data_collator(
    tokenizer,
    padding_free: bool = False,
    packing: bool = False,
    train_on_completions_only: bool = False,
    assistant_message_template: str | None = None,
    pad_to_multiple_of: int | None = None,
    use_context_parallel: bool = False,
    train_on_last_assistant_only: bool = False,
    model_config=None,
    per_device_train_batch_size: int = 1,
    keeps_packed_rows: bool = False,
):
    """The data collator matching the training configuration, ``None`` for TRL's own; raises on an option
    combination no collator can serve. ``model_config`` supplies the eos set that ends assistant turns,
    which templates delimiting turns with role markers (GLM-4) need."""
    reject_unsupported_collator_options(
        tokenizer,
        padding_free=padding_free,
        packing=packing,
        train_on_completions_only=train_on_completions_only,
        assistant_message_template=assistant_message_template,
        use_context_parallel=use_context_parallel,
        train_on_last_assistant_only=train_on_last_assistant_only,
        model_config=model_config,
        per_device_train_batch_size=per_device_train_batch_size,
        keeps_packed_rows=keeps_packed_rows,
    )
    markers = segment_markers_for(model_config)
    completion_kwargs = {
        "response_prompt_template": assistant_message_template,
        "tokenizer": tokenizer,
        "train_on_last_assistant_only": train_on_last_assistant_only,
        "eos_token_ids": resolve_eos_token_ids(tokenizer, model_config),
    }
    last_only = ", last-only" if train_on_last_assistant_only else ""
    flat_markers = {"return_seq_idx": markers.seq_idx, "return_flash_attn_kwargs": True}
    packed_markers = {"return_seq_idx": markers.seq_idx, "return_flash_attn_kwargs": markers.cu_seq_lens}
    if use_context_parallel and train_on_completions_only:
        # pad_to_multiple_of keeps seqs divisible by cp_size.
        prefix, name = "📊", f"DataCollatorForCompletionOnlyLM (CP mode, pad_to={pad_to_multiple_of}{last_only})"
        collator = DataCollatorForCompletionOnlyLM(**completion_kwargs, pad_to_multiple_of=pad_to_multiple_of)
    elif use_context_parallel:
        prefix, name = "📊", f"DataCollatorForCausalLMWithPadding (CP mode, pad_to={pad_to_multiple_of})"
        collator = DataCollatorForCausalLMWithPadding(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
        )
    elif padding_free and train_on_completions_only:
        prefix, name = "⚡", f"DataCollatorWithFlatteningAndCompletionMask (padding-free, varlen{last_only})"
        collator = DataCollatorWithFlatteningAndCompletionMask(**completion_kwargs, **flat_markers)
    elif padding_free:
        prefix, name = "⚡", "DataCollatorWithFlattening (padding-free, varlen)"
        collator = DataCollatorWithFlattening(tokenizer=tokenizer, **flat_markers)
    elif packing and train_on_completions_only:
        prefix, name = "📦", f"DataCollatorForCompletionOnlyLMWithPacking{last_only}"
        collator = DataCollatorForCompletionOnlyLMWithPacking(**completion_kwargs, **packed_markers)
    elif packing:
        prefix, name = "📦", "DataCollatorWithPacking"
        collator = DataCollatorWithPacking(tokenizer=tokenizer, **packed_markers)
    elif train_on_completions_only:
        prefix, name = "📝", f"DataCollatorForCompletionOnlyLM{last_only}"
        collator = DataCollatorForCompletionOnlyLM(**completion_kwargs)
    else:
        prefix, name, collator = "ℹ️", "None (default TRL behavior)", None

    if is_global_main_process():
        logger.info(f"{prefix} Using {name}")
    return collator
