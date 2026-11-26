"""A model's context window, and whether a length knob states a real bound."""

import logging

from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from halo.models.introspection.config_access import text_config
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# HF's no-length sentinel (``VERY_LARGE_INTEGER``, 1e30) sits above this and is a positive int, so
# :func:`is_bounded_length` alone would read it as a real window.
UNSET_MODEL_MAX_LENGTH = int(1e9)
_CONTEXT_WINDOW_FIELDS = ("max_position_embeddings", "max_seq_length", "n_positions")


def is_bounded_length(value: int | None) -> bool:
    """``None`` and any non-positive value mean unset; a wrong read is silent, as ``ids[-0:]`` is the whole list."""
    return value is not None and value > 0


def context_window_from_config(config: PretrainedConfig) -> int | None:
    """Read off the text sub-config, where a composite (VLM) config keeps the position budget."""
    decoder_config = text_config(config)
    for attr in _CONTEXT_WINDOW_FIELDS:
        value = getattr(decoder_config, attr, None)
        if is_bounded_length(value):
            return int(value)
    return None


def resolve_model_context_window(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> int:
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
    """``max_length: null`` truncates at the real window rather than a small dataclass default."""
    if is_bounded_length(value):
        return value
    context = resolve_model_context_window(model, tokenizer)
    if is_global_main_process():
        logger.info("max_length unset → resolved to the model context window (%d)", context)
    return context


def bounded_row_length(*sources, one_sided: bool = False) -> int | None:
    """The first bounded ``max_length``, else the prompt plus completion budgets when both exist, else None;
    ``one_sided`` takes a lone bounded half. Several sources because ``trl.GRPOConfig`` declares neither ``max_length``
    nor ``max_prompt_length``, and the scripts carry the latter."""

    def first_bounded(field: str) -> int | None:
        return next(
            (int(value) for source in sources if is_bounded_length(value := getattr(source, field, None))), None
        )

    max_length = first_bounded("max_length")
    if max_length is not None:
        return max_length
    prompt, completion = first_bounded("max_prompt_length"), first_bounded("max_completion_length")
    if prompt is None or completion is None:
        return (prompt or completion) if one_sided else None
    return prompt + completion
