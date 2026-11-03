"""The SFT collator option combinations no collator or attention kernel can serve, and which rows reach
FlashAttention's varlen kernel."""

import logging

from halo.data.turn_markers import reject_unusable_response_marker
from halo.models.attention.layout import has_mrope_rotary
from halo.models.attention.selection import (
    VARLEN_ATTN_IMPLEMENTATIONS,
    declares_flash_attention,
    effective_attn_implementation,
    fa4_varlen_refusal,
    is_flash_attn_implementation,
)
from halo.models.segment_markers import (
    has_dense_packing_leak,
    packing_isolation_is_measured,
    reject_compressed_kv_rows,
    reject_pipeline_packed_rows,
    reject_unmarked_sequence_mixer_rows,
    require_segment_aware_kernels,
    segment_markers_for,
)
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def rows_reach_varlen_kernel(
    *, packing: bool, padding_free: bool, per_device_train_batch_size: int, per_device_eval_batch_size: int = 0
) -> bool:
    """Whether batches reach the varlen call whatever the model: packed and padding-free rows hold several documents,
    and transformers unpads any batch holding a padding token, which more than one row per batch (training or
    evaluation) does. A one-document batch reaches it only where :func:`supports_one_document_varlen_set`."""
    return packing or padding_free or max(per_device_train_batch_size, per_device_eval_batch_size) > 1


def supports_one_document_varlen_set(model_config) -> bool:
    """Whether one-document rows take their varlen set, running the varlen kernel where transformers would test them
    for packing in every layer before its dense kernel: a flash backend, no other mixer reading the set
    (GatedDeltaNet's delta rule would take its varlen path), and on FA4 kernels taking the attention."""
    attn_impl = effective_attn_implementation(model_config)
    if not is_flash_attn_implementation(attn_impl) or segment_markers_for(model_config).cu_seq_lens:
        return False
    return attn_impl != "flash_attention_4" or fa4_varlen_refusal(model_config) is None


def evaluation_batch_size(training_args) -> int:
    """0 for a run that never evaluates."""
    return training_args.per_device_eval_batch_size if training_args.eval_strategy != "no" else 0


def reject_fa4_varlen_training(model_config, *, varlen_rows: bool, attn_impl: str | None = None) -> None:
    """Refuse FA4 on batches reaching its varlen call when the model's attention needs what those kernels are not
    validated with. ``attn_impl`` is the backend the load resolves, read off ``model_config`` when omitted."""
    if attn_impl is None:
        attn_impl = effective_attn_implementation(model_config)
    if attn_impl != "flash_attention_4" or not varlen_rows:
        return
    refused = fa4_varlen_refusal(model_config)
    if refused is None:
        return
    raise ValueError(
        f"attn_implementation='flash_attention_4' cannot train this model on packed, padding-free or padded rows: "
        f"they reach FA4's varlen kernels, which are not validated with {refused}. "
        f"Leave attn_implementation unset (these rows then take flash_attention_2) or set flash_attention_2, or "
        f"train unpacked at per_device_train_batch_size: 1 (and per_device_eval_batch_size: 1 where the run "
        f"evaluates), whose rows run FA4's dense kernel."
    )


def has_dense_kernel_row_refusal(model_config, *, padding_free: bool, packing: bool) -> bool:
    """A padding-free row, whose ``cu_seq_lens`` only a varlen kernel reads, or a packed row on a family whose dense
    forward attends across its documents."""
    return model_config is not None and (padding_free or (packing and has_dense_packing_leak(model_config)))


def reject_dense_attention_rows(model_config, attn_impl: str | None, *, padding_free: bool, packing: bool) -> None:
    """Refuse the multi-document rows a dense ``attn_impl`` cannot serve; it is the loaded model's backend, or the one
    a script resolves before the load."""
    if is_flash_attn_implementation(attn_impl) or not has_dense_kernel_row_refusal(
        model_config, padding_free=padding_free, packing=packing
    ):
        return
    if padding_free:
        raise ValueError(
            f"padding_free requires a varlen attention kernel, but this model resolved to "
            f"attn_implementation={attn_impl!r}. Only {list(VARLEN_ATTN_IMPLEMENTATIONS)} consume "
            f"cu_seqlens, so the cu_seq_lens a flattened row exists to carry go unread — it buys "
            f"nothing here, while still paying a dense mask over the flattened batch's TOTAL "
            f"token count. Use a model that resolves to flash attention, or set padding_free: false; "
            f"under SFT, packing: true keeps the same isolation and raises tokens per row on any kernel."
        )
    raise ValueError(
        f"packing=True on attn_implementation={attn_impl!r} is refused for "
        f"model_type={getattr(model_config, 'model_type', None)!r}: this family's forward does not pass "
        f"position_ids into its mask construction, so the packed row runs as ONE dense "
        f"causal sequence and documents attend across each other — silently. Use a varlen "
        f"attention implementation ({list(VARLEN_ATTN_IMPLEMENTATIONS)}) where the family has "
        f"one, or disable packing."
    )


def _warn_dense_packed_rows(model_config, attn_impl: str | None, *, packing: bool) -> None:
    """Packed rows on a dense kernel run as one sequence under a dense mask."""
    if model_config is None or not packing or is_flash_attn_implementation(attn_impl) or not is_global_main_process():
        return
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


def _warn_unmeasured_multi_document_family(model_config, *, multi_document: bool) -> None:
    """On every backend: a family outside the table can leak on flash too (GPT-Neo, Moshi), with nothing in the loss."""
    if model_config is None or not multi_document or packing_isolation_is_measured(model_config):
        return
    if is_global_main_process():
        logger.warning(
            "Multi-document rows on model_type=%r: this family is outside the table in agent-docs/data/collators.md "
            "(Document isolation under packing), so a document may reach the next one, and nothing in the loss shows "
            "it. BERT- and BART-family decoders, OPT, BioGPT, GPT-Neo and Moshi do. Train without packing and "
            "padding_free unless you have measured it.",
            getattr(model_config, "model_type", None),
        )


def reject_mrope_padding_free(model_config, *, padding_free: bool) -> None:
    """On every backend: M-RoPE's flattened row faults the flash varlen kernels, and no dense kernel serves one."""
    if not padding_free or model_config is None or not has_mrope_rotary(model_config):
        return
    raise ValueError(
        f"padding_free is refused for model_type={getattr(model_config, 'model_type', None)!r}: its decoder "
        f"rotates by M-RoPE, and a flattened M-RoPE row faults the flash varlen kernels (cudaErrorIllegalAddress "
        f"under flash_attention_2), while the dense backends take no cu_seqlens at all. Set padding_free: false; "
        f"under SFT, packing: true keeps the documents apart on the same kernels."
    )


def reject_padding_free_conflicts(*, padding_free: bool, packing: bool, use_context_parallel: bool) -> None:
    """Reads only the flags, so a script can run it before the model load."""
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


def reject_context_parallel_packing(*, packing: bool, use_context_parallel: bool) -> None:
    """Reads only the flags, so a script can run it before the model load, once a baked corpus says what it packed."""
    if use_context_parallel and packing:
        raise ValueError(
            "packing is not supported with Context Parallelism: the CP attention path has no "
            "per-document boundaries (dense causal kernel), so packed documents leak attention "
            "into each other. Disable packing for CP runs."
        )


def resolve_multi_document_mode(*, packing: bool, padding_free: bool) -> str | None:
    """``"packing"``, ``"padding_free"``, or ``None`` for one document per row."""
    if packing:
        return "packing"
    return "padding_free" if padding_free else None


def reject_multi_document_rows(
    model_config,
    mode: str | None,
    *,
    attn_impl: str | None,
    allow_cross_document_linear_attention: bool,
    keeps_packed_rows: bool,
    varlen_rows: bool,
) -> None:
    """Refuse the rows ``model_config`` cannot keep apart or no kernel can serve: a multi-document row on an M-RoPE
    decoder (padding-free), compressed-KV layers, a dense ``attn_impl``, GatedDeltaNet layers without segment-aware
    kernels, or linear attention no segment marker reaches unless ``allow_cross_document_linear_attention``; a packed
    row a pipeline stage keeps; and FA4 on rows reaching its varlen call."""
    padding_free, packing = mode == "padding_free", mode == "packing"
    reject_mrope_padding_free(model_config, padding_free=padding_free)
    if mode is not None:
        reject_compressed_kv_rows(model_config, mode)
    reject_dense_attention_rows(model_config, attn_impl, padding_free=padding_free, packing=packing)
    reject_fa4_varlen_training(model_config, varlen_rows=varlen_rows, attn_impl=attn_impl)
    if mode is None:
        return
    require_segment_aware_kernels(model_config, mode)
    reject_unmarked_sequence_mixer_rows(
        model_config, mode, allow_cross_document_linear_attention=allow_cross_document_linear_attention
    )
    if packing and keeps_packed_rows:
        reject_pipeline_packed_rows(model_config)


def reject_unpackable_family(model_config, *, allow_cross_document_linear_attention: bool, flag: str) -> None:
    """The rules of :func:`reject_multi_document_rows` a bake can answer, for corpora training would then refuse. A bake
    fixes no backend, so the family takes its best: flash where it has one. The kernel rule reads the training host."""
    reject_compressed_kv_rows(model_config, flag)
    attn_impl = "flash_attention_2" if declares_flash_attention(model_config) else "sdpa"
    reject_dense_attention_rows(model_config, attn_impl, padding_free=False, packing=True)
    reject_unmarked_sequence_mixer_rows(
        model_config, flag, allow_cross_document_linear_attention=allow_cross_document_linear_attention
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
    """Refuse (or warn about) what no collator can serve, reading only arguments, backend and segment markers."""
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
    attn_impl = effective_attn_implementation(model_config)
    reject_multi_document_rows(
        model_config,
        resolve_multi_document_mode(packing=packing, padding_free=padding_free),
        attn_impl=attn_impl,
        allow_cross_document_linear_attention=allow_cross_document_linear_attention,
        keeps_packed_rows=keeps_packed_rows,
        varlen_rows=rows_reach_varlen_kernel(
            packing=packing,
            padding_free=padding_free,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
        ),
    )
    _warn_dense_packed_rows(model_config, attn_impl, packing=packing)
    _warn_unmeasured_multi_document_family(model_config, multi_document=packing or padding_free)
    reject_unusable_response_marker(
        tokenizer, assistant_message_template, train_on_completions_only, "select_data_collator"
    )
    if train_on_last_assistant_only and not train_on_completions_only:
        raise ValueError("train_on_last_assistant_only=True requires train_on_completions_only=True")
    reject_context_parallel_packing(packing=packing, use_context_parallel=use_context_parallel)
