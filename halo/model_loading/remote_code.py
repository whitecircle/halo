"""Remote-code (``trust_remote_code``) compatibility: the one wrap of transformers' remote-class funnel, the v4-era
shims and attention repairs, and whole-file module-cache writes, so a rank never imports a module a peer is still
writing."""

import functools
import inspect
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch.nn.functional as F
import transformers.dynamic_module_utils
import transformers.utils.import_utils
from transformers.cache_utils import DynamicLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from halo.checkpoint.atomic import create_staged_file, publish_staged_file
from halo.model_loading.buffer_repair import default_rope_parameters
from halo.models.attention.remote_flash_attention import patch_remote_llama_flash_template
from halo.models.families.nemotron_h import set_composite_head_accessors
from halo.models.introspection.config_access import ties_word_embeddings

# Names a published modeling file calls but never imports (Ring-mini-linear-2.0's cached forward builds a
# ``DynamicLayer``); only the module whose global lookup misses can be given them.
_UNIMPORTED_REMOTE_GLOBALS = {"DynamicLayer": DynamicLayer}
# Marks a remote class whose list-form ``_tied_weights_keys`` was converted to ``{}``.
TIED_KEYS_WERE_LIST_ATTR = "_halo_tied_keys_were_list"
_FUNNEL_WRAPPED_MARKER = "_halo_remote_class_hook"
_WHOLE_COPY_MARKER = "_halo_whole_copy"

_hooks: list[Callable[[ModuleType], None]] = []


def _halo_is_torch_fx_available() -> bool:
    """The v4 ``is_torch_fx_available``; a def, so a traceback through the calling third-party module names it."""
    return True


def _shim_dispatchless_eager_attention(module) -> None:
    """Route ``attn_implementation: sdpa`` to SDPA in a remote file copying HF's ``eager_attention_forward`` without
    the ``ALL_ATTENTION_FUNCTIONS`` dispatch, where the setting changes only the mask format and every layer still runs
    the quadratic eager matmul (Ling 3.0). The forwards resolve the helper by name per call, so replacing the global
    sticks."""
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
            # The v4 SDPA convention: causal only for an unmasked multi-token forward.
            is_causal=mask is None and query.shape[2] > 1,
            scale=scaling,
            # repeat_kv semantics (grouped, not interleaved), matching the eager helper's expansion.
            enable_gqa=key.shape[1] != query.shape[1],
        )
        return attn_output.transpose(1, 2).contiguous(), None

    eager_or_sdpa._halo_sdpa_shim = True
    module.eager_attention_forward = eager_or_sdpa


def _fix_legacy_tied_weights_keys(module) -> None:
    """Convert v4 list-form ``_tied_weights_keys`` to the v5 dict, as transformers 5.12+ crashes ``save_pretrained`` on
    a list (every Bailing/Ling class). ``{}`` ties nothing, safe only for the untied checkpoints these families ship,
    so the class is marked for :func:`reject_untieable_list_ties`."""
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module.__name__ and isinstance(cls.__dict__.get("_tied_weights_keys"), list):
            cls._tied_weights_keys = {}
            setattr(cls, TIED_KEYS_WERE_LIST_ATTR, True)


def _repair_remote_module(module) -> None:
    """The per-module repairs only the loading module itself can receive, run on every remote class load."""
    for name, symbol in _UNIMPORTED_REMOTE_GLOBALS.items():
        if not hasattr(module, name):
            setattr(module, name, symbol)
    _shim_dispatchless_eager_attention(module)
    patch_remote_llama_flash_template(module)
    _fix_legacy_tied_weights_keys(module)
    set_composite_head_accessors(module)


def register_remote_class_hook(hook: Callable[[ModuleType], None]) -> None:
    """Run ``hook(module)`` for every remote modeling module transformers loads a class from, through the one funnel
    ``get_class_in_module``, wrapped once per process since re-application would stack wrappers."""
    if hook not in _hooks:
        _hooks.append(hook)
    original = transformers.dynamic_module_utils.get_class_in_module
    if getattr(original, _FUNNEL_WRAPPED_MARKER, False):
        return

    @functools.wraps(original)
    def get_class_in_module_hooked(class_name, module_path, **kwargs):
        cls = original(class_name, module_path, **kwargs)
        module = sys.modules.get(cls.__module__)
        if module is not None:
            for hook in _hooks:
                hook(module)
        return cls

    setattr(get_class_in_module_hooked, _FUNNEL_WRAPPED_MARKER, True)
    transformers.dynamic_module_utils.get_class_in_module = get_class_in_module_hooked


def apply_remote_code_compat_shims() -> None:
    """Restore the v4 surface remote files still reach and repair each such module as it loads; idempotent. The
    ``"default"`` RoPE entry v5 dropped is the buffer repair's own function, so the two cannot disagree."""
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        transformers.utils.import_utils.is_torch_fx_available = _halo_is_torch_fx_available
    ROPE_INIT_FUNCTIONS.setdefault("default", default_rope_parameters)
    register_remote_class_hook(_repair_remote_module)


def reject_untieable_list_ties(model) -> None:
    """Refuse a tied config whose remote class shipped the legacy list-form ``_tied_weights_keys``."""
    if getattr(type(model), TIED_KEYS_WERE_LIST_ATTR, False) and ties_word_embeddings(model.config):
        raise RuntimeError(
            f"{type(model).__name__} declares tie_word_embeddings=True but shipped the legacy "
            "list-form _tied_weights_keys, which transformers cannot tie (the compat shim "
            "converts it to {}) — the head would silently train untied. Fix the remote modeling "
            "file's _tied_weights_keys to the dict form."
        )


def _copyfile_whole(src: str | os.PathLike, dst: str | os.PathLike, *, follow_symlinks: bool = True):
    """``shutil.copyfile`` publishing ``dst`` once all written; a reader sees the old file or the complete new one."""
    destination = Path(dst)
    staged = create_staged_file(destination.parent, destination.name)
    try:
        shutil.copyfile(src, staged, follow_symlinks=follow_symlinks)
        publish_staged_file(staged, destination)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return dst


def _write_module_cache_whole() -> None:
    """Route ``dynamic_module_utils``' ``shutil`` through :func:`_copyfile_whole`, once per process. Only ``copyfile``
    is exposed, so a release calling another ``shutil`` function fails by name."""
    bound = getattr(transformers.dynamic_module_utils, "shutil", None)
    if getattr(bound, _WHOLE_COPY_MARKER, False):
        return
    if bound is not shutil:
        raise RuntimeError(
            "transformers.dynamic_module_utils no longer reaches shutil as a module attribute, so its "
            "module-cache copies cannot be made whole; update _write_module_cache_whole for this "
            "transformers release."
        )
    transformers.dynamic_module_utils.shutil = SimpleNamespace(copyfile=_copyfile_whole, **{_WHOLE_COPY_MARKER: True})


_write_module_cache_whole()
