"""Factory for selecting the data collator based on SFT training configuration."""

from accelerate import PartialState
from accelerate.logging import get_logger
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from src.data.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.data.collators.packing import (
    DataCollatorForCausalLMWithPadding,
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlattening,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from src.data.spans import resolve_eos_token_ids, verify_marker_renders_in_chat_template
from src.models.patches.attention import (
    GDN_MODEL_TYPE_PREFIXES,
    VARLEN_ATTN_IMPLEMENTATIONS,
    effective_attn_implementation,
    model_type_matches,
)

logger = get_logger(__name__)

# Families whose forward never passes ``position_ids`` into mask construction, so on a dense backend
# (eager/SDPA/flex) a packed row runs as one causal sequence and documents attend across each other.
# Packing is refused for them; a varlen kernel is their production path. Isolation matrix:
# ``agent-docs/data/collators.md``.
DENSE_PACKING_LEAK_MODEL_TYPES = frozenset({"gpt_oss"})


# Backends whose availability selects transformers' segment-aware GatedDeltaNet kernels; the torch
# fallbacks ignore ``seq_idx``/``cu_seqlens`` and run a packed row as one document while attention
# stays isolated. These are transformers' own fast-path predicates; re-spelling them as bare
# ``find_spec`` checks would drop the ``fla>=0.2.2`` floor and the CUDA-capable-torch conjunct.
GDN_SEGMENT_AWARE_BACKENDS = (
    ("causal_conv1d", is_causal_conv1d_available),
    ("fla>=0.2.2", is_flash_linear_attention_available),
)


def _validate_collator_options(
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
    """Refuse (or warn about) option combinations no collator can serve.

    Guards over the caller's arguments only: it reads nothing the dispatch in
    :func:`select_data_collator` produces and produces nothing that dispatch reads.
    """
    if packing and padding_free:
        raise ValueError(
            "Cannot use both 'packing' and 'padding_free' simultaneously. "
            "Choose one:\n"
            "  - packing=True: Traditional sequence packing (pad to max_length)\n"
            "  - padding_free=True: Flash Attention 2 flattening (no padding, ~2-3x faster)"
        )

    # Backend-independent, unlike the dense-mask warning below: flatten_packed_batch merges the
    # mini-batch into one row because transformers derives cu_seqlens from position_ids only at
    # batch size 1, so batch>1 is one row of up to B*max_length real tokens rather than B independent
    # sequences. Document isolation is preserved, hence a warning. PP keeps its rows and is exempt.
    if packing and per_device_train_batch_size > 1 and not keeps_packed_rows and PartialState().is_main_process:
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

    # The decoder's backend: a VLM wrapper records it on its text sub-config, so a top-level-only
    # read would report None and refuse padding_free on a model that does run flash attention.
    attn_impl = effective_attn_implementation(model_config)
    if model_config is not None and attn_impl not in VARLEN_ATTN_IMPLEMENTATIONS:
        if padding_free:
            raise ValueError(
                f"padding_free requires a varlen attention kernel, but this model resolved to "
                f"attn_implementation={attn_impl!r}. Only {list(VARLEN_ATTN_IMPLEMENTATIONS)} consume "
                f"cu_seqlens, so the cu_seq_lens this collator exists to emit go unread — it buys "
                f"nothing here, while still paying a dense mask over the flattened batch's TOTAL "
                f"token count. Use packing: true instead (same isolation, and it raises tokens per "
                f"row on any kernel), or a model that resolves to flash attention."
            )
        # model_type_matches reads the text sub-config too, so a VLM wrapper cannot slip the gate.
        if packing and model_type_matches(model_config, *DENSE_PACKING_LEAK_MODEL_TYPES):
            raise ValueError(
                f"packing=True on attn_implementation={attn_impl!r} is refused for "
                f"model_type={getattr(model_config, 'model_type', None)!r}: this family's forward does not pass "
                f"position_ids into its mask construction, so the packed row runs as ONE dense "
                f"causal sequence and documents attend across each other — silently. Use a varlen "
                f"attention implementation ({list(VARLEN_ATTN_IMPLEMENTATIONS)}, the family's "
                f"production path), or disable packing."
            )
        if packing and PartialState().is_main_process:
            logger.warning(
                "packing=True on attn_implementation=%r: this backend materializes a dense mask "
                "from the packed position_ids rather than consuming cu_seqlens — and the packed "
                "rows are flattened into one sequence, so that mask's side is the whole batch's "
                "token count, up to per_device_train_batch_size * max_length. Keep "
                "per_device_train_batch_size at 1 and scale the batch with "
                "gradient_accumulation_steps (pipeline parallelism excepted: it keeps the rows and "
                "splits them into microbatches). Expected on models that cannot run a varlen "
                "kernel (Gemma 4's head_dim=512, GLM-5 Next/Step-3.7/Inkling, Bailing/Ling's dispatchless "
                "remote code). "
                "Isolation is family-dependent — mixers without a reachable boundary parameter "
                "(Zaya CCA, DeepSeek-V4 compressors, Inkling convs, Bailing and GLM-5 KDA) "
                "carry state across document boundaries on every backend; see "
                "agent-docs/data/collators.md.",
                attn_impl,
            )

    if use_context_parallel and padding_free:
        raise ValueError(
            "Context Parallelism (CP) is NOT compatible with padding_free mode. "
            "CP requires fixed-length sequences for collective synchronization. "
            "Please disable padding_free when using CP."
        )

    if train_on_completions_only and assistant_message_template is None:
        raise ValueError("train_on_completions_only=True requires assistant_message_template to be set")

    if train_on_completions_only and getattr(tokenizer, "chat_template", None):
        verify_marker_renders_in_chat_template(tokenizer, assistant_message_template)

    if train_on_last_assistant_only and not train_on_completions_only:
        raise ValueError("train_on_last_assistant_only=True requires train_on_completions_only=True")


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
    """Select the data collator matching the training configuration.

    assistant_message_template is required when train_on_completions_only=True.
    model_config (the HF PretrainedConfig) supplies the eos_token_id set used to find assistant-turn
    ends; pass it so templates that delimit turns with role markers (e.g. GLM-4, no per-turn eos) are
    masked correctly. Returns None to fall back to TRL's default collator. Raises ValueError on
    incompatible options (packing+padding_free, CP+padding_free).
    """
    _validate_collator_options(
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

    eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)

    # Conv/linear-attention mixers cross document boundaries unless the modeling gets its segment
    # markers from forward kwargs (LFM2 ShortConv: ``seq_idx``; GatedDeltaNet: ``seq_idx`` plus
    # ``cu_seq_lens_q``). Family-gated: emitting them universally pushes unread kwargs everywhere.
    emit_seq_idx = (
        model_type_matches(model_config, "lfm2", *GDN_MODEL_TYPE_PREFIXES) if model_config is not None else False
    )
    emit_packed_flash_kwargs = (
        model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES) if model_config is not None else False
    )
    # Emitting the markers is not enough on its own: only the fast-path kernels read them.
    if (packing or padding_free) and emit_packed_flash_kwargs:
        missing = ", ".join(name for name, is_available in GDN_SEGMENT_AWARE_BACKENDS if not is_available())
        if missing:
            mode = "packing" if packing else "padding_free"
            raise ValueError(
                f"{mode}=True is refused for the GatedDeltaNet family "
                f"model_type={getattr(model_config, 'model_type', None)!r}: {missing} unavailable. "
                f"transformers selects its segment-aware linear-attention kernels on exactly these "
                f"checks (package installed, at the version floor, CUDA-capable torch), and the torch "
                f"fallbacks it takes instead IGNORE the document markers this collator emits — the "
                f"conv drops seq_idx, the chunked delta rule drops cu_seq_lens_q — so conv and "
                f"recurrent state cross packed document boundaries silently, while attention stays "
                f"isolated. Install {missing} (the production images pin both), or train this family "
                f"unpacked."
            )

    if packing and emit_packed_flash_kwargs and keeps_packed_rows:
        raise ValueError(
            "packing under pipeline parallelism is not supported for the GatedDeltaNet families "
            "(Qwen3.5/3.6, Qwen3-Next): PP keeps the packed rows (no flattening), and the varlen "
            "cu_seq_lens the delta rule needs have no per-row convention — the conv would isolate "
            "while the delta rule silently crossed document boundaries. Disable packing for this "
            "run, or drop pipeline parallelism."
        )

    last_only_suffix = ", last-only" if train_on_last_assistant_only else ""

    if use_context_parallel:
        if packing:
            raise ValueError(
                "packing is not supported with Context Parallelism: the CP attention path has no "
                "per-document boundaries (dense causal kernel), so packed documents leak attention "
                "into each other. Disable packing for CP runs."
            )
        # pad_to_multiple_of keeps seqs divisible by cp_size.
        prefix = "📊"
        if train_on_completions_only:
            collator_name = f"DataCollatorForCompletionOnlyLM (CP mode, pad_to={pad_to_multiple_of}{last_only_suffix})"
            collator = DataCollatorForCompletionOnlyLM(
                response_prompt_template=assistant_message_template,
                tokenizer=tokenizer,
                train_on_last_assistant_only=train_on_last_assistant_only,
                eos_token_ids=eos_token_ids,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        else:
            collator_name = f"DataCollatorForCausalLMWithPadding (CP mode, pad_to={pad_to_multiple_of})"
            collator = DataCollatorForCausalLMWithPadding(
                tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
            )

    elif padding_free and train_on_completions_only:
        prefix = "⚡"
        collator_name = f"DataCollatorWithFlatteningAndCompletionMask (padding-free + FA2{last_only_suffix})"
        collator = DataCollatorWithFlatteningAndCompletionMask(
            response_prompt_template=assistant_message_template,
            tokenizer=tokenizer,
            return_flash_attn_kwargs=True,
            return_seq_idx=emit_seq_idx,
            train_on_last_assistant_only=train_on_last_assistant_only,
            eos_token_ids=eos_token_ids,
        )

    elif padding_free:
        prefix = "⚡"
        collator_name = "DataCollatorWithFlattening (padding-free + FA2)"
        collator = DataCollatorWithFlattening(
            tokenizer=tokenizer,
            return_flash_attn_kwargs=True,
            return_seq_idx=emit_seq_idx,
        )

    elif packing and train_on_completions_only:
        prefix = "📦"
        collator_name = f"DataCollatorForCompletionOnlyLMWithPacking{last_only_suffix}"
        collator = DataCollatorForCompletionOnlyLMWithPacking(
            response_prompt_template=assistant_message_template,
            tokenizer=tokenizer,
            train_on_last_assistant_only=train_on_last_assistant_only,
            eos_token_ids=eos_token_ids,
            return_seq_idx=emit_seq_idx,
            return_flash_attn_kwargs=emit_packed_flash_kwargs,
        )

    elif packing:
        prefix = "📦"
        collator_name = "DataCollatorWithPacking"
        collator = DataCollatorWithPacking(
            tokenizer=tokenizer, return_seq_idx=emit_seq_idx, return_flash_attn_kwargs=emit_packed_flash_kwargs
        )

    elif train_on_completions_only:
        prefix = "📝"
        collator_name = f"DataCollatorForCompletionOnlyLM{last_only_suffix}"
        collator = DataCollatorForCompletionOnlyLM(
            response_prompt_template=assistant_message_template,
            tokenizer=tokenizer,
            train_on_last_assistant_only=train_on_last_assistant_only,
            eos_token_ids=eos_token_ids,
        )

    else:
        prefix = "ℹ️"
        collator_name = "None (default TRL behavior)"
        collator = None

    if PartialState().is_main_process:
        logger.info(f"{prefix} Using {collator_name}")

    return collator
