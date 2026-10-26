"""The meta-device model shell the lazy loaders plan against: zero RAM, real buffers.

:func:`instantiate_on_meta` builds it from ``from_pretrained(device_map="meta")`` where the
architecture allows and from the config alone where it does not, grafting back the non-persistent
buffers the meta dispatch strands.
"""

from __future__ import annotations

import logging
import os
import traceback

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import GenerationConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from halo.model_loading.dtype import resolve_load_dtype
from halo.models.param_roles import fp32_pin_matches

logger = logging.getLogger(__name__)

# The transformers model-build step that refuses an attention implementation the class does not
# support, on every load path; a ValueError raised under it is a config choice, not a lazy-load limit.
_ATTN_IMPLEMENTATION_CHECK = "_check_and_adjust_attn_implementation"


class MetaShellBuildError(RuntimeError):
    """A model class that cannot be built from its config on the meta device; each caller states what that
    blocks and any setting past it."""


def _resolve_remote_code_class(model_class, config, trust_remote_code: bool, revision: str | None):
    """The concrete remote-code model class, resolved from the config's hub id at the run's ``revision``.

    Auto dispatch against the local snapshot dir hits a transformers HF-cache-symlink bug (relative imports
    fail); unpinned, the modeling file would come from the repo's ``main`` while the weights come from the
    run's revision.
    """
    auto_map = getattr(config, "auto_map", None)
    class_name = getattr(model_class, "__name__", "")
    if not (trust_remote_code and auto_map and class_name in auto_map):
        return model_class
    ref = getattr(config, "_name_or_path", None)
    if not ref or os.path.isdir(ref):
        return model_class
    try:
        return get_class_from_dynamic_module(auto_map[class_name], ref, revision=revision)
    except Exception as exc:  # best-effort; fall back to the Auto class on any failure
        logger.warning(f"Could not pre-resolve remote-code class {auto_map[class_name]!r} from {ref!r}: {exc}")
        return model_class


def _restore_checkpoint_generation_config(model: nn.Module, model_name_or_path: str, revision=None) -> None:
    """Re-read ``generation_config.json`` onto a config-only shell, which ``from_config`` derives from the model
    config alone and the next save would write without the checkpoint's eos set and sampling defaults.

    An absent file keeps the derived defaults, as ``from_pretrained`` does; an unreadable one raises.
    """
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, revision=revision)
    except OSError:
        if os.path.isfile(os.path.join(model_name_or_path, "generation_config.json")):
            raise


def _apply_fp32_dtype_plan(model: nn.Module, dtype: torch.dtype) -> None:
    """Give a config-built shell the fp32 pins (``_keep_in_fp32_modules[_strict]``) ``from_pretrained``'s dtype
    plan applies and ``from_config`` does not: the lazy loaders read which parameters load fp32 off the shell.
    Matched with transformers' own rule on meta parameters, so nothing is allocated.
    """
    if not isinstance(model, PreTrainedModel):
        return
    for name in fp32_pin_matches((name for name, _ in model.named_parameters()), model._get_dtype_plan(dtype)):
        param = model.get_parameter(name)
        param.data = param.data.to(torch.float32)


def _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code: bool, **model_kwargs) -> nn.Module:
    """The model shell built from the config alone: parameters on meta, buffers real.

    Buffers must be real: checkpoints do not carry the non-persistent ones, and not every family's can be
    recomputed afterwards (a config-less rotary derives ``inv_freq`` from ctor args it never stores). Only
    ``dtype`` and ``attn_implementation`` are forwarded, since ``from_config`` hands anything else to
    ``__init__``; the fp32 pins follow the build, so both shells carry the same parameter dtypes.
    """
    kwargs: dict = {"dtype": dtype}
    if "attn_implementation" in model_kwargs:
        kwargs["attn_implementation"] = model_kwargs["attn_implementation"]
    dropped = sorted(set(model_kwargs) - set(kwargs))
    if dropped:
        logger.info(f"Config-only meta shell for {model_class.__name__} ignores load-only kwargs: {dropped}")
    # Auto classes take trust_remote_code and resolve the concrete class; a concrete one (already
    # resolved by _resolve_remote_code_class) would forward it to __init__ and raise.
    factory = getattr(model_class, "from_config", None)
    if factory is not None:
        kwargs["trust_remote_code"] = trust_remote_code
    else:
        factory = model_class._from_config
    try:
        with init_empty_weights(include_buffers=False):
            model = factory(config, **kwargs)
    except Exception as e:
        if isinstance(e, ValueError) and any(
            frame.name == _ATTN_IMPLEMENTATION_CHECK for frame in traceback.extract_tb(e.__traceback__)
        ):
            raise ValueError(
                f"{model_class.__name__} refuses attn_implementation={kwargs.get('attn_implementation')!r} "
                f"at model build ({type(e).__name__}: {e}), on every load path, lazy or not. Set "
                f"attn_implementation to one the architecture supports: sdpa, or eager where sdpa is the "
                f"one refused."
            ) from e
        # On the fallback path the caller's warning has already named the from_pretrained failure;
        # on the config_only path this build is the only attempt, so the message claims no more.
        raise MetaShellBuildError(
            f"Could not build a config-only meta-device shell for {model_class.__name__} ({type(e).__name__}: {e})."
        ) from e
    _apply_fp32_dtype_plan(model, dtype)
    return model


def _materialize_nonpersistent_buffers_from_config_twin(
    model: nn.Module, model_class, config, dtype, trust_remote_code: bool, **model_kwargs
) -> None:
    """Graft the non-persistent buffers ``from_pretrained(device_map="meta")`` strands on meta from a
    config-only twin, whose buffers are real by construction. Persistent ones stay meta (the checkpoint
    carries them); one the twin cannot supply either is named by the trainer's meta-buffer gate.
    """
    stranded = [
        (module_name, attr)
        for module_name, module in model.named_modules()
        for attr, buf in module._buffers.items()
        if buf is not None and buf.is_meta and attr in module._non_persistent_buffers_set
    ]
    if not stranded:
        return
    twin = _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code, **model_kwargs)
    grafted = 0
    for module_name, attr in stranded:
        source = twin.get_submodule(module_name)._buffers.get(attr)
        if source is None or source.is_meta:
            continue
        model.get_submodule(module_name)._buffers[attr] = source.detach().clone()
        grafted += 1
    logger.info(
        f"Materialized {grafted}/{len(stranded)} non-persistent buffer(s) the meta dispatch stranded, "
        f"from a config-only twin of {type(model).__name__}"
    )


def instantiate_on_meta(
    model_name_or_path: str,
    model_class,
    config,
    dtype: torch.dtype | None = None,
    *,
    trust_remote_code: bool,
    config_only: bool = False,
    **model_kwargs,
) -> nn.Module:
    """Instantiate a HuggingFace model on the meta device (zero RAM), at the run's ``dtype``.

    ``from_pretrained(device_map="meta")`` first (it handles weight tying and custom init), with the buffers
    its dispatch strands repaired; a config-only build when that fails (GLM4's non-Parameter attrs) or under
    ``config_only``, for a checkpoint whose index names shards this filesystem lacks (a per-node pipeline
    save), which ``from_pretrained`` would open. The run's dtype, not the checkpoint's: what the checkpoint
    omits (a task head) keeps the shell's dtype.
    """
    model_class = _resolve_remote_code_class(model_class, config, trust_remote_code, model_kwargs.get("revision"))
    dtype = resolve_load_dtype(dtype, config)
    if not config_only:
        try:
            model = model_class.from_pretrained(
                model_name_or_path,
                device_map="meta",
                config=config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        except (AttributeError, ValueError) as e:
            logger.warning(
                f"from_pretrained(device_map='meta') failed for {model_class.__name__} "
                f"({type(e).__name__}: {e}); building the shell from the config alone instead."
            )
        else:
            _materialize_nonpersistent_buffers_from_config_twin(
                model, model_class, config, dtype, trust_remote_code, **model_kwargs
            )
            return model
    model = _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code, **model_kwargs)
    _restore_checkpoint_generation_config(model, model_name_or_path, model_kwargs.get("revision"))
    return model
