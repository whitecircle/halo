"""Compatibility shims for remote-code MoE models with transformers>=5.

Some models (e.g. inclusionAI/Ling-mini-2.0) reference APIs removed in v5; these restore them. Apply
before from_pretrained(trust_remote_code=True). Idempotent.
"""

import functools
import inspect

import torch.nn.functional as F
import transformers.utils.import_utils
from transformers.cache_utils import DynamicLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from src.models.patches.buffer_fixes import default_rope_parameters
from src.models.patches.remote_code_hooks import register_remote_class_hook

# Names a published modeling file calls but never imports, so the bare global raises NameError.
# Bound into the offending module rather than worked around at the call site: the file's own logic is
# correct, only its import list is incomplete.
_UNIMPORTED_REMOTE_GLOBALS = {"DynamicLayer": DynamicLayer}

_applied = False


def apply_remote_code_compat_shims() -> None:
    """Apply all compatibility shims for remote-code models."""
    global _applied
    if _applied:
        return
    _patch_is_torch_fx_available()
    _patch_rope_init_default()
    register_remote_class_hook(_repair_remote_module)
    _applied = True


def _halo_is_torch_fx_available() -> bool:
    """Stand-in for the v4 predicate: torch.fx ships with every torch this toolkit runs on.

    Named rather than a lambda because it is bound into a third-party module, where ``<lambda>``
    names nothing in a traceback."""
    return True


def _patch_is_torch_fx_available() -> None:
    """Restore is_torch_fx_available removed in transformers v5."""
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        transformers.utils.import_utils.is_torch_fx_available = _halo_is_torch_fx_available


def _patch_rope_init_default() -> None:
    """Restore the ``'default'`` RoPE init entry transformers v5 dropped.

    v5 native rotaries call ``compute_default_rope_parameters`` instead, but remote-code models
    written against v4 still look the name up in the registry. Registered as the buffer-recompute
    pass's own :func:`~src.models.patches.buffer_fixes.default_rope_parameters` so the two cannot
    disagree, and so the entry reads ``rope_theta`` out of ``rope_parameters``; a hardcoded fallback
    would retune every position.
    """
    ROPE_INIT_FUNCTIONS.setdefault("default", default_rope_parameters)


def _repair_remote_module(module) -> None:
    """Per-module repairs only the loading module itself can receive, run on every remote class load.

    ``inclusionAI/Ring-mini-linear-2.0`` appends ``DynamicLayer()`` to ``past_key_value.layers`` while
    importing only ``Cache`` and ``DynamicCache``, so every cached forward raises ``NameError``.
    Training runs with the cache off, so the fault only appears at generation time. The name is never
    imported, so unlike the other shims here only the calling module can be given it. Absent names
    only.
    """
    for name, symbol in _UNIMPORTED_REMOTE_GLOBALS.items():
        if not hasattr(module, name):
            setattr(module, name, symbol)
    _shim_dispatchless_eager_attention(module)
    _fix_legacy_tied_weights_keys(module)


def _fix_legacy_tied_weights_keys(module) -> None:
    """Convert v4-era list-form ``_tied_weights_keys`` on remote classes to the v5 dict form.

    transformers 5.12+ reads ``_tied_weights_keys`` as a ``{duplicate: source}`` dict, so a remote
    modeling file shipping the old list form (every Bailing/Ling class) crashes ``save_pretrained``.
    ``{}`` is safe only because these checkpoints store ``lm_head`` untied; for a genuinely tied
    config no tie would form at all, so the conversion is marked on the class and
    ``from_pretrained_verified`` refuses tied configs that carried the legacy list.
    """
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__ and isinstance(cls.__dict__.get("_tied_weights_keys"), list):
            cls._tied_weights_keys = {}
            cls._halo_tied_keys_were_list = True


def _shim_dispatchless_eager_attention(module) -> None:
    """Route ``attn_implementation: sdpa`` to ``F.scaled_dot_product_attention`` in remote modeling
    files that define the HF ``eager_attention_forward`` helper but no attention dispatch.

    ``modeling_bailing_moe_v3`` (Ling 3.0) copies the helper without the ``ALL_ATTENTION_FUNCTIONS``
    lookup, so the requested implementation changes only the mask format while every layer still runs
    the eager score matmul: an ``[B, H, S, S]`` bf16 plane plus its fp32 softmax copy, around 190 GiB
    on a packed 80k row. The wrapper reads the attention module's own config per call, so everything
    but ``sdpa`` keeps the file's own eager path. Replacing the module global is what makes it stick,
    since the forwards resolve ``eager_attention_forward`` by name at every call.
    """
    original = getattr(module, "eager_attention_forward", None)
    if original is None or hasattr(module, "ALL_ATTENTION_FUNCTIONS") or getattr(original, "_halo_sdpa_shim", False):
        return

    @functools.wraps(original)
    def eager_or_sdpa(mod, query, key, value, attention_mask, scaling=None, dropout: float = 0.0, **kwargs):
        # SDPA cannot return attention weights, so an explicit output_attentions keeps the eager path.
        if getattr(mod.config, "_attn_implementation", "eager") != "sdpa" or kwargs.get("output_attentions"):
            return original(mod, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout,
            # The v4 SDPA convention this file targets: causal only for an unmasked multi-token
            # forward (a cached single-token step needs neither mask nor causality).
            is_causal=mask is None and query.shape[2] > 1,
            scale=scaling,
            # repeat_kv semantics (grouped, not interleaved), matching the eager helper's expansion.
            enable_gqa=key.shape[1] != query.shape[1],
        )
        return attn_output.transpose(1, 2).contiguous(), None

    eager_or_sdpa._halo_sdpa_shim = True
    module.eager_attention_forward = eager_or_sdpa
