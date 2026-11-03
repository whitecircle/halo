"""Tokenizer and processing-class setup: what a run pins on its tokenizer and reads back at save time."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.utils import cached_file

from halo.model_loading.tokenizer_backend import resolve_tokenizer_backend
from halo.models.config_access import set_config_field
from halo.models.unwrapping import base_transformers_model
from halo.runtime.ranks import is_global_main_process

if TYPE_CHECKING:
    # Annotation only: the entry scripts' argument layer sits above this package.
    from halo.configs.script_arguments import CommonScriptArguments

logger = logging.getLogger(__name__)

# The tokenizer's own ``model_max_length``, served back at save time. A plain int, so the tokenizer stays picklable for
# the dataset-map fingerprint and the workers.
_PRISTINE_MODEL_MAX_LENGTH_ATTR = "_halo_pristine_model_max_length"

# Marks a chat template the run chose, which its processor then adopts; a checkpoint's template stays each one's.
_RUN_CHAT_TEMPLATE_ATTR = "_halo_run_chat_template"

# Marks a model grown past its checkpoint's rows; those rows exist in no base, so every adapter save carries them.
_GROWN_VOCABULARY_ATTR = "_halo_grown_vocabulary"

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model")


def _holds_no_tokenizer(processing_class, path: str) -> bool:
    """transformers 5 builds a ``model_type``'s tokenizer class with no file behind it, a vocabulary of its special
    tokens alone."""
    tokenizer = resolve_tokenizer(processing_class)
    if not isinstance(tokenizer, PreTrainedTokenizerBase) or len(tokenizer) > len(set(tokenizer.all_special_tokens)):
        return False
    lookup = {"_raise_exceptions_for_missing_entries": False, "_raise_exceptions_for_connection_errors": False}
    return not any(cached_file(path, name, **lookup) for name in _TOKENIZER_FILES)


def load_processing_class(path: str, *, trust_remote_code: bool = False):
    """``AutoProcessor`` for multimodal, else ``AutoTokenizer``, else ``None`` when the directory holds no tokenizer.
    Re-save utilities must persist this, as a tokenizer-only VLM save is unloadable; any other failure must propagate
    rather than ship a resized model beside the checkpoint's stale tokenizer."""
    try:
        try:
            processing_class = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
        except (OSError, ValueError) as e:
            # The two ways a repo reports no processor, the ordinary text-only path, hence info.
            if is_global_main_process():
                logger.info(f"No processor at {path} ({type(e).__name__}: {e}); falling back to the tokenizer.")
            processing_class = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    except (OSError, ValueError) as e:
        reason = f"{type(e).__name__}: {e}"
    else:
        if not _holds_no_tokenizer(processing_class, path):
            return processing_class
        reason = "it holds no tokenizer file"
    if is_global_main_process():
        logger.warning(
            f"No processing class could be loaded from {path} ({reason}). Any checkpoint written from it carries no "
            f"tokenizer of its own."
        )
    return None


def resolve_tokenizer(processing_class):
    """Unwrap a VLM processor to its tokenizer; a plain tokenizer passes through."""
    return getattr(processing_class, "tokenizer", processing_class)


def resolve_peft_processing_class(adapter_processor, base_processor, *, text_only: bool):
    """The processing class a merged or converted PEFT checkpoint ships. ``text_only`` ships the run's tokenizer, else
    the base's; otherwise the base's full processor wins over an adapter with only a tokenizer, as a tokenizer-only
    save of a multimodal checkpoint is unloadable, and an adapter's own full processor is kept."""
    if text_only:
        chosen = adapter_processor or base_processor
        return None if chosen is None else resolve_tokenizer(chosen)
    base_is_full = base_processor is not None and not isinstance(base_processor, PreTrainedTokenizerBase)
    adapter_is_full = adapter_processor is not None and not isinstance(adapter_processor, PreTrainedTokenizerBase)
    if base_is_full and not adapter_is_full:
        return base_processor
    return adapter_processor or base_processor


def load_chat_template(chat_template: str | None) -> str | None:
    """A ``.jinja``/``.jinja2``/``.j2`` file's contents, or the string itself if it is no such path."""
    if chat_template is None:
        return None
    if chat_template.endswith((".jinja", ".jinja2", ".j2")):
        if not os.path.isfile(chat_template):
            raise FileNotFoundError(
                f"Chat template file not found: {chat_template}. "
                f"Make sure the file exists or provide the template string directly."
            )
        if is_global_main_process():
            logger.info(f"Loading chat template from file: {chat_template}")
        with open(chat_template, encoding="utf-8") as f:
            return f.read()
    if os.path.isfile(chat_template):
        raise ValueError(
            f"chat_template points at an existing file ({chat_template}) whose extension is not "
            f".jinja/.jinja2/.j2 — rename it or pass the template string directly."
        )

    return chat_template


def sync_special_token_id(model: PreTrainedModel | None, field: str, token_id) -> None:
    """Record a special-token id on every config level declaring ``field`` and the generation config. Composite
    families need the per-level write: pooling keys on ``get_text_config().pad_token_id``, so a top-level-only write
    leaves the decoder's id unset (batch > 1 raises) or stale (pooling picks the wrong last token)."""
    if model is None:
        return
    recorded = set_config_field(model.config, field, token_id, only_declared=True)
    if not recorded and is_global_main_process():
        logger.warning(
            f"{type(model.config).__name__} declares no '{field}' at either the top level or on its "
            f"text sub-config, so the tokenizer's id could not be recorded on the model config. Any "
            f"consumer reading it (sequence-classification pooling, packing terminators) sees None."
        )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        setattr(generation_config, field, token_id)


def reject_stale_eos_copies(model: PreTrainedModel | None, eos_token_id) -> None:
    """Refuse ``eos_token_id`` on a model holding a module that copied its config's eos id at build; the copy keeps the
    checkpoint's id while exports record the new one (Qwen3.8-Flash-Next's n-gram history restarts after its copy)."""
    if model is None:
        return
    stale = sorted(
        {
            type(module).__name__
            for module in model.modules()
            if getattr(module, "eos_token_id", eos_token_id) != eos_token_id
        }
    )
    if stale:
        raise ValueError(
            f"eos_token sets eos_token_id={eos_token_id}, but {stale} copied the checkpoint's eos id from the config "
            f"when the model was built and computes with that copy, which the run cannot change, while every export "
            f"records the new id: the served model would differ from the trained one. Leave eos_token unset (or "
            f"set it to the checkpoint's own eos token) and end turns through the chat template."
        )


def configure_run_tokenizer(
    args: CommonScriptArguments,
    model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int | None = None,
    *,
    embeddings_sharded: Callable[[PreTrainedModel], bool],
) -> PreTrainedTokenizer:
    """Apply args-driven token setup and return the tokenizer to use, resolved last through ``args.tokenizer_backend``,
    which callers must use in place of the one passed. ``embeddings_sharded`` says whether a vocabulary grow is
    impossible; a parameter, since this package stays sharding-agnostic."""
    if max_seq_len is not None:
        # Recorded once, as a run with a frozen reference calls this twice on one tokenizer.
        if not hasattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR):
            setattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR, tokenizer.model_max_length)
        tokenizer.model_max_length = max_seq_len
    # Before the special-token roles, which read an id back: ``--pad_token <new>`` with the same token in
    # ``--added_special_tokens`` would otherwise pad with an id the model config never recorded.
    if args.added_special_tokens is not None:
        # A union: the transformers default replaces the extra-special list, dropping the checkpoint's control tokens.
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
            rows = model.get_input_embeddings().weight.shape[0]
            grow_vocabulary(model, len(tokenizer))
            if is_global_main_process():
                logger.info(
                    f"added_special_tokens grew {type(model).__name__}'s vocabulary from {rows} to {len(tokenizer)} "
                    "rows; a PEFT adapter save of this run carries the input embedding and output head whole, since "
                    "no base checkpoint holds the added rows."
                )
    # The tokenizer write is conditional, the MODEL write is not: a frozen reference runs this once per model on the
    # SAME tokenizer, and a shared guard would leave that model's config on the checkpoint's eos.
    if args.eos_token is not None:
        if tokenizer.eos_token != args.eos_token:
            tokenizer.eos_token = args.eos_token
        reject_stale_eos_copies(model, tokenizer.eos_token_id)
        sync_special_token_id(model, "eos_token_id", tokenizer.eos_token_id)
    if args.bos_token is not None:
        if tokenizer.bos_token != args.bos_token:
            tokenizer.bos_token = args.bos_token
        sync_special_token_id(model, "bos_token_id", tokenizer.bos_token_id)
    if args.pad_token is not None and tokenizer.pad_token != args.pad_token:
        tokenizer.pad_token = args.pad_token
    elif args.pad_token is None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Recorded whatever settled it: pooling reads the pad id to locate each row's last content token, so it must equal
    # the collator's; none raises at batch > 1, a stale one pools on padding in silence.
    if tokenizer.pad_token_id is not None:
        sync_special_token_id(model, "pad_token_id", tokenizer.pad_token_id)
        if tokenizer.pad_token_id == tokenizer.eos_token_id and model is not None and is_global_main_process():
            logger.info(
                "Tokenizer pad token IS eos (id %s), so the recorded pad id binds "
                "nn.Embedding(padding_idx=%s) on the next load and that row's input-embedding gradient "
                "is masked from then on (the output side still trains, and under tied embeddings so "
                "does the row). Give the tokenizer a pad token distinct from eos — --pad_token for one "
                "already in the vocabulary, --added_special_tokens to add one — to avoid it.",
                tokenizer.pad_token_id,
                tokenizer.pad_token_id,
            )
    elif model is not None and is_global_main_process():
        logger.warning(
            f"The tokenizer settled on pad_token={tokenizer.pad_token!r} with no id, so the model "
            f"config keeps whatever the checkpoint shipped. An out-of-vocab pad_token is the usual "
            f"cause; both the padding collators and sequence-classification pooling read this id."
        )
    if tokenizer.chat_template is None or (args.chat_template is not None and args.force_chat_template):
        set_run_chat_template(tokenizer, load_chat_template(args.chat_template))

    return resolve_tokenizer_backend(tokenizer, args.tokenizer_backend)


def grow_vocabulary(model: PreTrainedModel, num_tokens: int, *, mean_resizing: bool = True) -> None:
    """Resize the embeddings (and an untied head) up to ``num_tokens`` rows and mark the plain transformers model
    grown, which an adapter save reads back. ``mean_resizing=False`` is for a caller overwriting the rows."""
    model.resize_token_embeddings(num_tokens, mean_resizing=mean_resizing)
    setattr(base_transformers_model(model), _GROWN_VOCABULARY_ATTR, True)


def vocabulary_grown(model) -> bool:
    """Whether the run grew the vocabulary past its checkpoint's, read through every wrapper."""
    return getattr(base_transformers_model(model), _GROWN_VOCABULARY_ATTR, False)


def set_run_chat_template(tokenizer: PreTrainedTokenizerBase, template: str | None) -> None:
    """Set the chat template the run chose on ``tokenizer``, marked for its processor to adopt."""
    tokenizer.chat_template = template
    setattr(tokenizer, _RUN_CHAT_TEMPLATE_ATTR, True)


def adopt_tokenizer_chat_template(processing_class) -> None:
    """Give a processor the chat template the run chose. A processor's ``chat_template`` is its own attribute, rendered
    and saved over the tokenizer's ``chat_template.jinja``, so a tokenizer-only set would train and export the
    original."""
    if isinstance(processing_class, PreTrainedTokenizerBase):
        return
    tokenizer = resolve_tokenizer(processing_class)
    if getattr(tokenizer, _RUN_CHAT_TEMPLATE_ATTR, False) and tokenizer.chat_template is not None:
        processing_class.chat_template = tokenizer.chat_template


@contextlib.contextmanager
def pristine_model_max_length(processing_class):
    """Serve the tokenizer's OWN ``model_max_length`` during a save, as ``save_pretrained`` writes the live pin: an SFT
    run at ``max_length: 40000`` would export a 262k-context model served at 40k. A VLM run's pin sits on the nested
    tokenizer."""
    tokenizer = next(
        (
            candidate
            for candidate in (processing_class, getattr(processing_class, "tokenizer", None))
            if candidate is not None and hasattr(candidate, _PRISTINE_MODEL_MAX_LENGTH_ATTR)
        ),
        None,
    )
    if tokenizer is None:
        yield
        return
    pinned = tokenizer.model_max_length
    tokenizer.model_max_length = getattr(tokenizer, _PRISTINE_MODEL_MAX_LENGTH_ATTR)
    try:
        yield
    finally:
        tokenizer.model_max_length = pinned
