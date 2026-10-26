"""A model's context window, and whether a length knob states a real bound: what a ``max_length``
left unset resolves to.
"""

from accelerate.logging import get_logger
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from src.models.config_access import text_config

logger = get_logger(__name__)

# Bound below which a ``model_max_length`` is a real context window. HF's "this tokenizer declares no
# length" sentinel (``VERY_LARGE_INTEGER``, 1e30) sits above it and is a positive int, so
# :func:`is_bounded_length` alone would read it as one.
UNSET_MODEL_MAX_LENGTH = int(1e9)
# The config fields a family spells its position budget with, in precedence order.
_CONTEXT_WINDOW_FIELDS = ("max_position_embeddings", "max_seq_length", "n_positions")


def is_bounded_length(value: int | None) -> bool:
    """Whether a length knob states a real bound: a positive int; ``None`` and any non-positive value
    mean "unset" (getting it wrong is silent: ``ids[-0:]`` is the whole list)."""
    return value is not None and value > 0


def context_window_from_config(config: PretrainedConfig) -> int | None:
    """A model's context window as ``config.json`` alone states it, or ``None`` when it states none.

    Read off the text sub-config, where a composite (VLM) config keeps the position budget; the
    config-time gates read it before any model or tokenizer exists.
    """
    decoder_config = text_config(config)
    for attr in _CONTEXT_WINDOW_FIELDS:
        value = getattr(decoder_config, attr, None)
        if is_bounded_length(value):
            return int(value)
    return None


def get_model_context_window(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> int:
    """A model's max context window in tokens: the config's (:func:`context_window_from_config`), else
    ``tokenizer.model_max_length`` below :data:`UNSET_MODEL_MAX_LENGTH`; raises when neither states one."""
    window = context_window_from_config(model.config)
    if window is not None:
        return window

    tok_max = getattr(tokenizer, "model_max_length", None)
    if is_bounded_length(tok_max) and tok_max < UNSET_MODEL_MAX_LENGTH:
        return int(tok_max)

    raise ValueError(
        f"Could not infer the model context window (no positive {' / '.join(_CONTEXT_WINDOW_FIELDS)} on "
        f"model.config, and tokenizer.model_max_length is unset). Set the max-length field explicitly in "
        f"the config."
    )


def resolve_length_to_context(value: int | None, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> int:
    """``value`` when it is a positive length, else the model's context window (``max_length: null``
    truncates at the real window rather than a small dataclass default)."""
    if is_bounded_length(value):
        return value
    context = get_model_context_window(model, tokenizer)
    logger.info("max_length unset → resolved to the model context window (%d)", context)
    return context


def bounded_row_length(*sources) -> int | None:
    """The token cap ``sources`` put on one row: the first bounded ``max_length`` across them, else the
    first bounded ``max_prompt_length`` plus the first bounded ``max_completion_length`` when both
    exist, else None (a one-sided budget leaves the other half unbounded).

    Several sources because on the RL surface the bound is split across objects: ``trl.GRPOConfig``
    declares neither ``max_length`` nor ``max_prompt_length``, and the scripts carry the latter.
    """

    def first_bounded(field: str) -> int | None:
        return next(
            (int(value) for source in sources if is_bounded_length(value := getattr(source, field, None))), None
        )

    max_length = first_bounded("max_length")
    if max_length is not None:
        return max_length
    prompt, completion = first_bounded("max_prompt_length"), first_bounded("max_completion_length")
    return prompt + completion if prompt is not None and completion is not None else None
