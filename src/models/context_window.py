"""A model's context window, and whether a length knob states a real bound: what a ``max_length``
left unset resolves to.
"""

from accelerate.logging import get_logger
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.config_access import text_config

logger = get_logger(__name__)

# Bound below which a ``model_max_length`` is a real context window. HF's "this tokenizer declares no
# length" sentinel (``VERY_LARGE_INTEGER``, 1e30) sits above it and is a positive int, so
# :func:`is_bounded_length` alone would read it as one.
UNSET_MODEL_MAX_LENGTH = int(1e9)


def is_bounded_length(value: int | None) -> bool:
    """Whether a length knob states a real bound.

    A positive int is a bound; ``None`` and any non-positive value mean "unset". Shared rather than
    re-spelled per call site because every consequence of getting it wrong is silent: HF resolves
    ``truncation=True, max_length=None`` against ``tokenizer.model_max_length``, and ``ids[-0:]`` is
    the whole list.
    """
    return value is not None and value > 0


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

    Reads model config first (:func:`context_window_from_config`), then ``tokenizer.model_max_length``
    below :data:`UNSET_MODEL_MAX_LENGTH`, which HF's "unset" sentinel exceeds. Raises when nothing
    trustworthy.
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
