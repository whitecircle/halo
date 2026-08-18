"""The meta-device model shell the lazy loaders plan against — zero RAM, real buffers.

:func:`instantiate_on_meta` builds it from ``from_pretrained(device_map="meta")`` where the
architecture allows and from the config alone where it does not, grafting back the non-persistent
buffers the meta dispatch strands.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import GenerationConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from src.models.loading.lazy_safetensors.weights import resolve_run_dtype
from src.models.patches.attention import validate_attn_implementation

logger = logging.getLogger(__name__)


def _resolve_remote_code_class(model_class, config, trust_remote_code: bool):
    """Resolve a concrete remote-code model class from the hub-id-bearing config.

    Auto dispatch against the local snapshot dir hits a transformers HF-cache-symlink bug
    (``FileNotFoundError`` on relative imports); resolving from the hub id avoids it, and a concrete
    class skips Auto resolution entirely.
    """
    auto_map = getattr(config, "auto_map", None)
    class_name = getattr(model_class, "__name__", "")
    if not (trust_remote_code and auto_map and class_name in auto_map):
        return model_class
    ref = getattr(config, "_name_or_path", None)
    if not ref or os.path.isdir(ref):
        return model_class
    try:
        return get_class_from_dynamic_module(auto_map[class_name], ref)
    except Exception as exc:  # best-effort; fall back to the Auto class on any failure
        logger.warning(f"Could not pre-resolve remote-code class {auto_map[class_name]!r} from {ref!r}: {exc}")
        return model_class


def _is_attn_dispatch_error(exc: Exception) -> bool:
    """True when a from_pretrained failure is an attention-impl dispatch rejection.

    Raised at model build when an architecture cannot dispatch the requested impl (e.g. linear-attention
    Bailing rejecting FlashAttention-4).
    """
    msg = str(exc)
    return any(
        marker in msg
        for marker in (
            "Flash Attention",
            "flash_attention",
            "scaled_dot_product",
            "does not support",
            "attn_implementation",
        )
    )


def _restore_checkpoint_generation_config(model: nn.Module, model_name_or_path: str, revision=None) -> None:
    """Re-read ``generation_config.json`` onto a config-only shell.

    ``from_config`` derives the generation config from the model config alone (every generation
    parameter is popped at ``PretrainedConfig.__init__``), so the checkpoint's own file — custom eos
    sets, sampling defaults — would silently vanish from the next save. An absent file keeps the
    derived defaults, as a full ``from_pretrained`` would; a file that EXISTS but cannot be read
    raises, since swallowing it drops the EOS set and the export never stops generating.
    """
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, revision=revision)
    except OSError:
        if os.path.isfile(os.path.join(model_name_or_path, "generation_config.json")):
            raise


def _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code: bool, **model_kwargs) -> nn.Module:
    """Build the model shell from the CONFIG alone — parameters on meta, buffers REAL.

    ``init_empty_weights(include_buffers=False)`` puts only parameters on meta, so every buffer
    computes normally in ``__init__``. They must be real: checkpoints do not carry the non-persistent
    ones and not every family's can be recomputed afterwards — a config-less rotary (Qwen VL vision)
    derives ``inv_freq`` from ctor args it never stores, and the trainer's meta-buffer gate then
    rejects the model.

    Deliberately not ``from_pretrained`` inside the context: it would stream every checkpoint tensor
    into HOST RAM. Only ``dtype`` / ``attn_implementation`` are forwarded — ``from_config`` hands
    anything else straight to ``__init__`` — and the rest is already resolved into ``config``.
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
            return factory(config, **kwargs)
    except Exception as e:
        # On the fallback path the caller's warning has already named the from_pretrained failure;
        # on the config_only path this build is the only attempt — so the message claims no more.
        raise RuntimeError(
            f"Lazy loading could not build a config-only meta-device shell for {model_class.__name__} "
            f"({type(e).__name__}: {e}). There is no safe fallback — instantiating this "
            f"architecture from the checkpoint would materialize every tensor in host RAM. Load it "
            f"with ep_lazy_loading=False (PP: without a lazy stage load)."
        ) from e


def _materialize_nonpersistent_buffers_from_config_twin(
    model: nn.Module, model_class, config, dtype, trust_remote_code: bool, **model_kwargs
) -> None:
    """Repair non-persistent buffers ``from_pretrained(device_map="meta")`` strands on meta.

    The meta dispatch moves BUFFERS to meta too and their values are then lost for good, while a
    config-only twin's are real by construction — so graft them across. Persistent meta buffers are
    left alone (the checkpoint carries them); a buffer the twin cannot supply either stays meta and
    is named by the trainer's meta-buffer gate.
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
    trust_remote_code: bool = True,
    config_only: bool = False,
    **model_kwargs,
) -> nn.Module:
    """Instantiate a HuggingFace model on the meta device (zero RAM).

    Tries ``from_pretrained(device_map="meta")`` first (it handles weight tying and custom init) and
    repairs the buffers its dispatch strands; falls back to a config-only build if that fails (GLM4's
    non-Parameter attrs). ``config_only=True`` skips ``from_pretrained`` entirely, which opens every
    shard the index names — impossible on a legitimately incomplete checkpoint (a per-node pipeline
    save holds only this node's stage shard plus the full index).

    ``dtype`` is the RUN's, not the checkpoint config's: every materialized tensor is cast to it, so
    a shell at the checkpoint's dtype leaves what the checkpoint omits (a task head) in the wrong one.
    """
    model_class = _resolve_remote_code_class(model_class, config, trust_remote_code)
    dtype = resolve_run_dtype(dtype, config)
    if config_only:
        model = _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code, **model_kwargs)
        _restore_checkpoint_generation_config(model, model_name_or_path, model_kwargs.get("revision"))
        return model
    common = dict(
        config=config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    effective_kwargs = model_kwargs
    try:
        model = model_class.from_pretrained(model_name_or_path, device_map="meta", **common)
    except (AttributeError, ValueError) as e:
        # Exotic architectures (e.g. Bailing) reject an auto-detected FA4 at build; retry on SDPA.
        attn = model_kwargs.get("attn_implementation")
        if isinstance(e, ValueError) and attn not in (None, "sdpa", "eager") and _is_attn_dispatch_error(e):
            # Re-validated, not assumed safe: sdpa silently drops unreset gpt-oss sinks.
            retry_impl = validate_attn_implementation(config, "sdpa")
            logger.warning(
                f"{model_class.__name__} cannot dispatch attn_implementation={attn!r} ({e}); "
                f"retrying with {retry_impl!r}."
            )
            model = model_class.from_pretrained(
                model_name_or_path, device_map="meta", **dict(common, attn_implementation=retry_impl)
            )
            # The twin must build with the implementation that WORKED, not the one that failed.
            effective_kwargs = dict(model_kwargs, attn_implementation=retry_impl)
        else:
            logger.warning(
                f"from_pretrained(device_map='meta') failed for {model_class.__name__} "
                f"({type(e).__name__}: {e}); building the shell from the config alone instead."
            )
            model = _instantiate_from_config_on_meta(model_class, config, dtype, trust_remote_code, **model_kwargs)
            _restore_checkpoint_generation_config(model, model_name_or_path, model_kwargs.get("revision"))
            return model
    _materialize_nonpersistent_buffers_from_config_twin(
        model, model_class, config, dtype, trust_remote_code, **effective_kwargs
    )
    return model
