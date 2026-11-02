"""The SFT collator option combinations no collator, or no attention kernel on its rows, can serve, refused before
one is selected."""

import logging

from halo.data.turn_markers import reject_unusable_response_marker
from halo.models.attention.layout import has_mrope_rotary
from halo.models.attention.selection import (
    VARLEN_ATTN_IMPLEMENTATIONS,
    effective_attn_implementation,
    fa4_varlen_refusal,
    is_flash_attn_implementation,
)
from halo.models.segment_markers import (
    has_dense_packing_leak,
    reject_compressed_kv_rows,
    reject_pipeline_packed_rows,
    reject_unmarked_linear_attention_rows,
    require_segment_aware_kernels,
)
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def rows_reach_varlen_kernel(
    *, packing: bool, padding_free: bool, per_device_train_batch_size: int, per_device_eval_batch_size: int = 0
) -> bool:
    """Whether the run's batches reach FlashAttention's varlen call: packed and padding-free rows hold several
    documents, and transformers unpads any batch holding a padding token, which more than one row per batch does,
    a training batch or an evaluation one (``per_device_eval_batch_size``, :func:`evaluation_batch_size`)."""
    return packing or padding_free or max(per_device_train_batch_size, per_device_eval_batch_size) > 1


def evaluation_batch_size(training_args) -> int:
    """The rows per device of the run's evaluation batches, 0 for a run that never evaluates."""
    return training_args.per_device_eval_batch_size if training_args.eval_strategy != "no" else 0


def reject_fa4_varlen_training(model_config, *, varlen_rows: bool, attn_impl: str | None = None) -> None:
    """Refuse FA4 on batches that reach its varlen call (``varlen_rows``, training or evaluation) when its varlen
    kernels refuse the model's attention (:func:`fa4_varlen_refusal`): on B300 the first such forward would not
    finish, elsewhere the first backward would raise. ``attn_impl`` is the backend the load resolves, read off
    ``model_config`` when omitted."""
    if attn_impl is None:
        attn_impl = effective_attn_implementation(model_config)
    if attn_impl != "flash_attention_4" or not varlen_rows:
        return
    refused = fa4_varlen_refusal(model_config)
    if refused is None:
        return
    raise ValueError(
        f"attn_implementation='flash_attention_4' cannot train this model on packed, padding-free or padded rows: "
        f"they reach FA4's varlen kernels, which take no {refused} (the backward asserts, and on B300 the forward "
        f"does not finish). "
        f"Leave attn_implementation unset (these rows then take flash_attention_2) or set flash_attention_2, or "
        f"train unpacked at per_device_train_batch_size: 1 (and per_device_eval_batch_size: 1 where the run "
        f"evaluates), whose rows run FA4's dense kernel."
    )


def reject_dense_attention_rows(model_config, attn_impl: str | None, *, padding_free: bool, packing: bool) -> None:
    """Refuse the multi-document rows a model whose decoder runs the dense kernel ``attn_impl`` cannot serve: a
    padding-free row, whose ``cu_seq_lens`` only a varlen kernel reads, and a packed row on a family whose dense
    forward attends across its documents. ``attn_impl`` is the loaded model's backend, or the one a script
    resolves for its checkpoint before the load."""
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
            f"attention implementation ({list(VARLEN_ATTN_IMPLEMENTATIONS)}) where the family has "
            f"one, or disable packing."
        )


def _reject_dense_attention_rows(model_config, *, padding_free: bool, packing: bool) -> None:
    """Refuse (or warn about) a multi-document row on a model whose decoder (read through a VLM wrapper's
    text sub-config) resolved to a dense kernel."""
    attn_impl = effective_attn_implementation(model_config)
    if model_config is None or is_flash_attn_implementation(attn_impl):
        return
    reject_dense_attention_rows(model_config, attn_impl, padding_free=padding_free, packing=packing)
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


def reject_mrope_padding_free(model_config, *, padding_free: bool) -> None:
    """Refuse ``padding_free`` on a decoder that rotates by M-RoPE (:func:`has_mrope_rotary`), on every backend:
    its flattened row faults the flash varlen kernels, and no dense kernel serves one."""
    if not padding_free or model_config is None or not has_mrope_rotary(model_config):
        return
    raise ValueError(
        f"padding_free is refused for model_type={getattr(model_config, 'model_type', None)!r}: its decoder "
        f"rotates by M-RoPE, and a flattened M-RoPE row faults the flash varlen kernels (cudaErrorIllegalAddress "
        f"under flash_attention_2), while the dense backends take no cu_seqlens at all. Set padding_free: false; "
        f"under SFT, packing: true keeps the documents apart on the same kernels."
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
    allow_cross_document_linear_attention: bool = False,
    per_device_eval_batch_size: int = 0,
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
    reject_mrope_padding_free(model_config, padding_free=padding_free)
    reject_fa4_varlen_training(
        model_config,
        varlen_rows=rows_reach_varlen_kernel(
            packing=packing,
            padding_free=padding_free,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
        ),
    )
    reject_unusable_response_marker(
        tokenizer, assistant_message_template, train_on_completions_only, "select_data_collator"
    )
    if train_on_last_assistant_only and not train_on_completions_only:
        raise ValueError("train_on_last_assistant_only=True requires train_on_completions_only=True")
    if packing or padding_free:
        mode = "packing" if packing else "padding_free"
        reject_compressed_kv_rows(model_config, mode)
        require_segment_aware_kernels(model_config, mode)
        reject_unmarked_linear_attention_rows(
            model_config, mode, allow_cross_document_linear_attention=allow_cross_document_linear_attention
        )
    if packing and keeps_packed_rows:
        reject_pipeline_packed_rows(model_config)
    if use_context_parallel and packing:
        raise ValueError(
            "packing is not supported with Context Parallelism: the CP attention path has no "
            "per-document boundaries (dense causal kernel), so packed documents leak attention "
            "into each other. Disable packing for CP runs."
        )
