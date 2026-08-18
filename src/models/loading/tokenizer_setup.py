"""Tokenizer / processing-class setup: what a run pins on its tokenizer and reads back at save time.

The widest processing class for a checkpoint, the chat template, the context window and the length
budget resolved against it, the special-token ids recorded onto the model, and the
``model_max_length`` pin the exporter serves the tokenizer's own value back through.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from accelerate.logging import get_logger
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase

from src.data.pipeline.tokenizer_backend import resolve_tokenizer_backend
from src.models.loading.config_levels import set_config_field, text_config

if TYPE_CHECKING:
    # Annotation only: the entry scripts' argument layer sits above this loading leaf, and nothing
    # here reads the dataclass itself.
    from src.args.common_script_args import CommonScriptArguments

logger = get_logger(__name__)

# Where :func:`setup_model_and_tokenizer` parks the tokenizer's own ``model_max_length`` before
# pinning the run's, for :func:`pristine_model_max_length` to serve back at save time. A plain int on
# the instance, so the tokenizer stays picklable for the dataset-map fingerprint and the workers.
_PRISTINE_MODEL_MAX_LENGTH_ATTR = "_halo_pristine_model_max_length"

# HF's "this tokenizer declares no length" sentinel (``VERY_LARGE_INTEGER``): a positive int, so
# :func:`is_bounded_length` alone reads it as a real 1e9-token context window.
UNSET_MODEL_MAX_LENGTH = int(1e9)


def load_processing_class(path: str, *, trust_remote_code: bool = False):
    """Load the widest processing class: ``AutoProcessor`` for multimodal, else ``AutoTokenizer``, else ``None``.

    Re-save utilities MUST persist this — saving only the tokenizer for a VLM yields an unloadable
    checkpoint. Only "this directory holds no tokenizer" yields ``None``, because every caller reads
    that as "nothing to save": any other failure must propagate rather than ship a resized model
    beside the checkpoint's stale tokenizer.
    """
    try:
        return AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
    except (OSError, ValueError) as e:
        # The two ways "this repo has no processor" is reported (absent processor config,
        # unrecognized processing class) — the ordinary text-only path, hence info, not a warning.
        logger.info(f"No processor at {path} ({type(e).__name__}: {e}); falling back to the tokenizer.")
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        except (OSError, ValueError) as e:
            logger.warning(
                f"No processing class could be loaded from {path} ({type(e).__name__}: {e}). Any "
                f"checkpoint written from it carries no tokenizer of its own."
            )
            return None


def resolve_peft_processing_class(adapter_dir: str, base_model_path: str, *, trust_remote_code: bool = False):
    """Resolve the processing class for a merged / converted PEFT checkpoint.

    Prefer the base model's full processor when the adapter carries only a tokenizer (a VLM adapter dir
    drops the image preprocessor); keep the adapter's class when it is itself a full processor.
    """
    adapter_pc = load_processing_class(adapter_dir, trust_remote_code=trust_remote_code)
    base_pc = load_processing_class(base_model_path, trust_remote_code=trust_remote_code)
    base_is_full = base_pc is not None and not isinstance(base_pc, PreTrainedTokenizerBase)
    adapter_is_full = adapter_pc is not None and not isinstance(adapter_pc, PreTrainedTokenizerBase)
    if base_is_full and not adapter_is_full:
        return base_pc
    return adapter_pc or base_pc


def load_chat_template(chat_template: str) -> str:
    """Load a chat template from a ``.jinja``/``.jinja2``/``.j2`` file, or return the
    string directly if it isn't such a path."""
    if chat_template is None:
        return None

    if chat_template.endswith((".jinja", ".jinja2", ".j2")):
        if os.path.isfile(chat_template):
            logger.info(f"Loading chat template from file: {chat_template}")
            with open(chat_template, encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError(
                f"Chat template file not found: {chat_template}. "
                f"Make sure the file exists or provide the template string directly."
            )
    if os.path.isfile(chat_template):
        raise ValueError(
            f"chat_template points at an existing file ({chat_template}) whose extension is not "
            f".jinja/.jinja2/.j2 — rename it or pass the template string directly."
        )

    return chat_template


def context_window_from_config(config) -> int | None:
    """A model's context window as ``config.json`` alone states it, or ``None`` when it states none.

    Read off the text sub-config, where a composite (VLM) config keeps the position budget. Split out
    of :func:`get_model_context_window` for the config-time gates, which judge a run before any model
    or tokenizer exists and must not re-spell the lookup.
    """
    decoder_config = text_config(config)
    for attr in ("max_position_embeddings", "max_seq_length", "n_positions"):
        value = getattr(decoder_config, attr, None)
        if is_bounded_length(value):
            return int(value)
    return None


def get_model_context_window(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> int:
    """Resolve a model's max context window in tokens.

    Reads model config first (:func:`context_window_from_config`), then
    ``tokenizer.model_max_length``, rejecting HF's ~1e9 "unset" sentinel. Raises when nothing trustworthy.
    """
    window = context_window_from_config(model.config)
    if window is not None:
        return window

    tok_max = getattr(tokenizer, "model_max_length", None)
    if is_bounded_length(tok_max) and tok_max < UNSET_MODEL_MAX_LENGTH:
        return int(tok_max)

    raise ValueError(
        "Could not infer the model context window (no positive max_position_embeddings / "
        "max_seq_length / n_positions on model.config, and tokenizer.model_max_length is unset). "
        "Set the max-length field explicitly in the config."
    )


def is_bounded_length(value: int | None) -> bool:
    """Whether a length knob states a real bound.

    A positive int is a bound; ``None`` and any non-positive value mean "unset". Shared rather than
    re-spelled per call site because every consequence of getting it wrong is silent: HF resolves
    ``truncation=True, max_length=None`` against ``tokenizer.model_max_length``, and ``ids[-0:]`` is
    the whole list.
    """
    return value is not None and value > 0


def resolve_length_to_context(value: int | None, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> int:
    """Return ``value`` when it is a positive length, else the model's context window.

    ``None`` and any non-positive value both mean "use the model's own limit" — a config writing
    ``max_length: null`` opts into truncating at the real context window rather than a small
    dataclass default.
    """
    if is_bounded_length(value):
        return value
    context = get_model_context_window(model, tokenizer)
    logger.info("max_length unset → resolved to the model context window (%d)", context)
    return context


def sync_special_token_id(model: PreTrainedModel | None, field: str, token_id) -> None:
    """Record a tokenizer special-token id on the model, everywhere the model reads it back.

    Writes every config level that declares ``field``, plus the generation config when the head
    carries one (transformers attaches it only to models that ``can_generate()``). The per-level
    write is what makes composite families work: pooling keys on
    ``config.get_text_config().pad_token_id``, so a top-level-only write leaves the decoder's id
    unset (batch > 1 raises) or stale at the checkpoint's (pooling picks the wrong last token).
    """
    if model is None:
        return
    if not set_config_field(model.config, field, token_id, only_declared=True):
        logger.warning(
            f"{type(model.config).__name__} declares no '{field}' at either the top level or on its "
            f"text sub-config, so the tokenizer's id could not be recorded on the model config. Any "
            f"consumer reading it (sequence-classification pooling, packing terminators) sees None."
        )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        setattr(generation_config, field, token_id)


def setup_model_and_tokenizer(
    args: CommonScriptArguments,
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int | None = None,
    *,
    embeddings_sharded: Callable[[PreTrainedModel], bool] = lambda _model: False,
) -> PreTrainedTokenizer:
    """Apply args-driven tokenizer/model token setup; returns the tokenizer to use.

    The return value is the tokenizer resolved through ``args.tokenizer_backend`` — callers must use
    it in place of the one passed in. Resolution runs last, so the backend snapshots the final
    tokenizer state (chat template, added special tokens).

    ``embeddings_sharded`` says whether a vocabulary grow is impossible because the model's input
    embedding is a parallelism shard — a parameter rather than an import, since what sharding is
    lives under ``src.distributed`` and this layer stays sharding-agnostic. Asked only about a run
    that actually grows the vocabulary; the default suits a caller that shards nothing.
    """
    if max_seq_len is not None:
        # Recorded once, before the first pin overwrites it (a preference/distillation script runs
        # this seam twice against the same tokenizer), so every export can be written with the
        # tokenizer's OWN bound — see :func:`pristine_model_max_length`.
        if not hasattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR):
            setattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR, tokenizer.model_max_length)
        tokenizer.model_max_length = max_seq_len
    # Added BEFORE the special-token roles below, because those read an id back: a token added
    # afterwards has none at the time of the read, so ``--pad_token <new>`` with the same token in
    # ``--added_special_tokens`` would pad with an id the model config never recorded.
    if args.added_special_tokens is not None:
        # UNION, not the transformers default: ``replace_extra_special_tokens=True`` swaps the whole
        # extra-special list for the request, dropping every control token the checkpoint shipped
        # (Qwen's 13, GLM's role enders) from ``all_special_ids``, the trainers' special-token masks
        # and the exported tokenizer_config.json.
        tokenizer.add_special_tokens(
            {"additional_special_tokens": args.added_special_tokens}, replace_extra_special_tokens=False
        )
        # Grow only: shrinking drops special tokens above len(tokenizer) and the model can no longer stop.
        if model is not None and len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            if embeddings_sharded(model):
                raise ValueError(
                    f"--added_special_tokens grows the vocabulary to {len(tokenizer)} rows, but under HF-native "
                    f"tensor parallelism the embedding is a sharded DTensor that resize_token_embeddings cannot "
                    f"re-shard. Grow the vocabulary before the run with scripts/before_training/patch_vocab.py."
                )
            model.resize_token_embeddings(len(tokenizer))
    # The tokenizer write is conditional, the MODEL write is not: a preference/distillation script
    # runs this seam once per model against the SAME tokenizer, so a shared guard would skip the
    # second sync and leave that model's config on the checkpoint's eos.
    if args.eos_token is not None:
        if tokenizer.eos_token != args.eos_token:
            tokenizer.eos_token = args.eos_token
        sync_special_token_id(model, "eos_token_id", tokenizer.eos_token_id)
    if args.bos_token is not None:
        if tokenizer.bos_token != args.bos_token:
            tokenizer.bos_token = args.bos_token
        sync_special_token_id(model, "bos_token_id", tokenizer.bos_token_id)
    if args.pad_token is not None and tokenizer.pad_token != args.pad_token:
        tokenizer.pad_token = args.pad_token
    elif args.pad_token is None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Recorded whatever settled it, unlike eos/bos above: pooling reads the pad id back to locate
    # each row's last content token, so it must equal the id the collator pads with — none raises at
    # batch > 1, a stale one pools on padding in silence.
    if tokenizer.pad_token_id is not None:
        sync_special_token_id(model, "pad_token_id", tokenizer.pad_token_id)
        if tokenizer.pad_token_id == tokenizer.eos_token_id and model is not None:
            # A live pairing, not a hypothetical: DeepSeek-V4's tokenizer pads with eos and its config
            # ships no pad id, so this is the first place the cost can be named.
            logger.info(
                "Tokenizer pad token IS eos (id %s), so the recorded pad id binds "
                "nn.Embedding(padding_idx=%s) on the next load and that row's input-embedding gradient "
                "is masked from then on (the output side still trains, and under tied embeddings so "
                "does the row). Give the tokenizer a pad token distinct from eos — --pad_token for one "
                "already in the vocabulary, --added_special_tokens to add one — to avoid it.",
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
            )
    elif model is not None:
        logger.warning(
            f"The tokenizer settled on pad_token={tokenizer.pad_token!r} with no id, so the model "
            f"config keeps whatever the checkpoint shipped. An out-of-vocab pad_token is the usual "
            f"cause; both the padding collators and sequence-classification pooling read this id."
        )
    if tokenizer.chat_template is None or (args.chat_template is not None and args.force_chat_template):
        chat_template = load_chat_template(args.chat_template)
        tokenizer.chat_template = chat_template

    return resolve_tokenizer_backend(tokenizer, args.tokenizer_backend)


def _length_pinned_tokenizer(processing_class):
    """The object :func:`setup_model_and_tokenizer` pinned, or ``None`` if the run never pinned one.

    Reached through a processor wrapper as well as directly: a VLM run hands the trainer the
    processor while the pin lands on its nested tokenizer.
    """
    for candidate in (processing_class, getattr(processing_class, "tokenizer", None)):
        if candidate is not None and hasattr(candidate, _PRISTINE_MODEL_MAX_LENGTH_ATTR):
            return candidate
    return None


@contextlib.contextmanager
def pristine_model_max_length(processing_class):
    """Serve the tokenizer's OWN ``model_max_length`` for the duration of a save.

    The run's budget is pinned onto ``tokenizer.model_max_length`` because HF resolves every
    ``truncation=True, max_length=None`` call against it, but ``save_pretrained`` writes the LIVE
    attribute into ``tokenizer_config.json``: unrestored, an SFT run at ``max_length: 40000`` exports
    a 262k-context model whose served context is 40k. The pin is put back on exit, and a run that
    never pinned one has no value to keep off disk.
    """
    tokenizer = _length_pinned_tokenizer(processing_class)
    if tokenizer is None:
        yield
        return
    pinned = tokenizer.model_max_length
    tokenizer.model_max_length = getattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR)
    try:
        yield
    finally:
        tokenizer.model_max_length = pinned
