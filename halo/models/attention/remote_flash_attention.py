"""Varlen flash attention for remote modeling files built on transformers v4's Llama flash template, whose mask-only
``_flash_attention_forward`` never sees ``position_ids``, so a packed row runs as one causal sequence."""

from __future__ import annotations

import functools
import inspect
import logging

from transformers import PreTrainedModel
from transformers import modeling_flash_attention_utils as flash_utils

from halo.models.attention.packed_position_ids import (
    forward_names_position_ids,
    get_stashed_position_ids,
    install_position_ids_stash,
)
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS, is_flash_attn_implementation
from halo.models.attention.tensor_memo import TensorMemo
from halo.models.segment_markers import packed_row_varlen_kwargs
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The template's own flash call; a file whose method takes anything else (a sliding-window flag) is left alone.
_TEMPLATE_FLASH_PARAMS = (
    "self",
    "query_states",
    "key_states",
    "value_states",
    "attention_mask",
    "query_length",
    "dropout",
    "softmax_scale",
)
_PATCHED_ATTR = "_halo_varlen_flash"

# The template's model forward passes its layers no varlen set, so it is derived here, once per forward.
_packed_row_varlen = TensorMemo(packed_row_varlen_kwargs)


def _template_flash_class(module) -> type | None:
    table = getattr(module, "ATTENTION_CLASSES", None)
    flash_cls = table.get("flash_attention_2") if isinstance(table, dict) else None
    method = None if flash_cls is None else flash_cls.__dict__.get("_flash_attention_forward")
    if method is None or tuple(inspect.signature(method).parameters) != _TEMPLATE_FLASH_PARAMS:
        return None
    return flash_cls if forward_names_position_ids(flash_cls) else None


def _varlen_flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
):
    """Through transformers' dispatch, where ``position_ids`` turn a packed row's documents into varlen sequences."""
    position_ids = get_stashed_position_ids(self)
    varlen = {}
    if attention_mask is None and position_ids is not None and query_states.shape[0] == position_ids.shape[0] == 1:
        varlen = _packed_row_varlen(position_ids)
    return flash_utils._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        is_causal=self.is_causal,
        dropout=dropout,
        position_ids=position_ids,
        softmax_scale=softmax_scale,
        attn_implementation=self.config._attn_implementation,
        **varlen,
    )


def _route_mask_switch_by_flash_family(model_cls) -> None:
    """Make the ``_use_flash_attention_2`` 2D-mask switch hold for every flash label, so FA3/FA4 get no 4D mask."""
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if hasattr(self, "_use_flash_attention_2"):
            self._use_flash_attention_2 = is_flash_attn_implementation(self.config._attn_implementation)

    model_cls.__init__ = __init__


def patch_remote_llama_flash_template(module) -> bool:
    """Packed-document isolation on every flash label; ``False`` when ``module`` is not a v4 template, or is already
    patched. The flash class keeps its name and weights, as the CP wrapper claims it by name."""
    flash_cls = _template_flash_class(module)
    if flash_cls is None or getattr(flash_cls, _PATCHED_ATTR, False):
        return False
    install_position_ids_stash(flash_cls, lambda attention: (attention,))
    flash_cls._flash_attention_forward = _varlen_flash_attention_forward
    setattr(flash_cls, _PATCHED_ATTR, True)
    for label in VARLEN_ATTN_IMPLEMENTATIONS:
        module.ATTENTION_CLASSES.setdefault(label, flash_cls)
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__ or not issubclass(cls, PreTrainedModel):
            continue
        if cls.__dict__.get("_supports_flash_attn_2"):
            cls._supports_flash_attn = True
        if "__init__" in cls.__dict__:
            _route_mask_switch_by_flash_family(cls)
    if is_global_main_process():
        logger.info(
            f"Patched {flash_cls.__name__} ({module.__name__}) for varlen flash attention on "
            f"{list(VARLEN_ATTN_IMPLEMENTATIONS)} (packed-document isolation)"
        )
    return True
