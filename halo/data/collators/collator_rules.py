"""The SFT collator option combinations no collator can serve, refused before one is selected."""

import logging

from halo.data.turn_markers import reject_unusable_response_marker
from halo.models.attention.selection import (
    VARLEN_ATTN_IMPLEMENTATIONS,
    effective_attn_implementation,
    is_flash_attn_implementation,
)
from halo.models.segment_markers import (
    has_dense_packing_leak,
    reject_compressed_kv_rows,
    require_segment_aware_kernels,
    segment_markers_for,
)
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def _reject_dense_attention_rows(model_config, *, padding_free: bool, packing: bool) -> None:
    """Refuse (or warn about) a multi-document row on a model whose decoder (read through a VLM wrapper's
    text sub-config) resolved to a dense kernel."""
    attn_impl = effective_attn_implementation(model_config)
    if model_config is None or is_flash_attn_implementation(attn_impl):
        return
    if padding_free:
        raise ValueError(
            f"padding_free requires a varlen attention kernel, but this model resolved to "
            f"attn_implementation={attn_impl!r}. Only {list(VARLEN_ATTN_IMPLEMENTATIONS)} consume "
            f"cu_seqlens, so the cu_seq_lens this collator exists to emit go unread — it buys "
            f"nothing here, while still paying a dense mask over the flattened batch's TOTAL "
            f"token count. Use packing: true instead (same isolation, and it raises tokens per "
            f"row on any kernel), or a model that resolves to flash attention."
        )
    if packing and has_dense_packing_leak(model_config):
        raise ValueError(
            f"packing=True on attn_implementation={attn_impl!r} is refused for "
            f"model_type={getattr(model_config, 'model_type', None)!r}: this family's forward does not pass "
            f"position_ids into its mask construction, so the packed row runs as ONE dense "
            f"causal sequence and documents attend across each other — silently. Use a varlen "
            f"attention implementation ({list(VARLEN_ATTN_IMPLEMENTATIONS)}, the family's "
            f"production path), or disable packing."
        )
    if packing and is_global_main_process():
        logger.warning(
            "packing=True on attn_implementation=%r: this backend materializes a dense mask "
            "from the packed position_ids rather than consuming cu_seqlens — and the packed "
            "rows are flattened into one sequence, so that mask's side is the whole batch's "
            "token count, up to per_device_train_batch_size * max_length. Keep "
            "per_device_train_batch_size at 1 and scale the batch with "
            "gradient_accumulation_steps (pipeline parallelism excepted: it keeps the rows and "
            "splits them into microbatches). Expected on models that cannot run a varlen "
            "kernel. Isolation is family-dependent: the families whose mixers carry state across "
            "document boundaries on every backend are listed in agent-docs/data/collators.md.",
            attn_impl,
        )


def reject_padding_free_conflicts(*, padding_free: bool, packing: bool, use_context_parallel: bool) -> None:
    """Refuse padding_free beside packing or context parallelism, reading only the flags, so a script can run it
    before the model load."""
    if packing and padding_free:
        raise ValueError(
            "Cannot use both 'packing' and 'padding_free' simultaneously. "
            "Choose one:\n"
            "  - packing=True: Traditional sequence packing (pad to max_length)\n"
            "  - padding_free=True: flattening for a varlen flash-attention kernel (no padding)"
        )
    if use_context_parallel and padding_free:
        raise ValueError(
            "Context Parallelism (CP) is NOT compatible with padding_free mode. "
            "CP requires fixed-length sequences for collective synchronization. "
            "Please disable padding_free when using CP."
        )


def reject_unsupported_collator_options(
    tokenizer,
    *,
    padding_free: bool,
    packing: bool,
    train_on_completions_only: bool,
    assistant_message_template: str | None,
    use_context_parallel: bool,
    train_on_last_assistant_only: bool,
    model_config,
    per_device_train_batch_size: int,
    keeps_packed_rows: bool,
) -> None:
    """Refuse (or warn about) the option combinations no collator can serve, reading only the caller's
    arguments, the model's attention backend and its segment markers."""
    reject_padding_free_conflicts(
        padding_free=padding_free, packing=packing, use_context_parallel=use_context_parallel
    )
    # Loss-less on every backend, so a warning; PP keeps its rows.
    if packing and per_device_train_batch_size > 1 and not keeps_packed_rows and is_global_main_process():
        logger.warning(
            "packing=True with per_device_train_batch_size=%d does not run %d independent sequences: "
            "the packed rows are merged into a single row of up to %d x max_length real tokens "
            "(transformers engages packed-sequence handling only at batch size 1). Document isolation "
            "is preserved and throughput is unaffected, but if the intent is a longer pack, raising "
            "max_length states it directly.",
            per_device_train_batch_size,
            per_device_train_batch_size,
            per_device_train_batch_size,
        )
    _reject_dense_attention_rows(model_config, padding_free=padding_free, packing=packing)
    reject_unusable_response_marker(
        tokenizer, assistant_message_template, train_on_completions_only, "select_data_collator"
    )
    if train_on_last_assistant_only and not train_on_completions_only:
        raise ValueError("train_on_last_assistant_only=True requires train_on_completions_only=True")
    if packing or padding_free:
        mode = "packing" if packing else "padding_free"
        reject_compressed_kv_rows(model_config, mode)
        require_segment_aware_kernels(model_config, mode)
    if packing and segment_markers_for(model_config).cu_seq_lens and keeps_packed_rows:
        raise ValueError(
            "packing under pipeline parallelism is not supported for the GatedDeltaNet families "
            "(Qwen3.5/3.6, Qwen3-Next): PP keeps the packed rows (no flattening), and the varlen "
            "cu_seq_lens the delta rule needs have no per-row convention — the conv would isolate "
            "while the delta rule silently crossed document boundaries. Disable packing for this "
            "run, or drop pipeline parallelism."
        )
    if use_context_parallel and packing:
        raise ValueError(
            "packing is not supported with Context Parallelism: the CP attention path has no "
            "per-document boundaries (dense causal kernel), so packed documents leak attention "
            "into each other. Disable packing for CP runs."
        )
