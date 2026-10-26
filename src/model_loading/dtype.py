"""Run precision: ``dtype`` resolution, quantized-load detection, the loaded-parameter cast and the
process-global fp32 matmul pin. Imports no loader, so any CLI can resolve a dtype or apply the pin."""

import warnings

import torch
import torch.nn as nn
from trl import get_quantization_config

from src.env import env_str

# Both spellings of each dtype, so a `--dtype` value accepted by one checkpoint tool is accepted by
# every sibling; derive each CLI's `choices=` from the keys rather than re-listing them.
DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

_VALID_FP32_MATMUL_PRECISION = ("highest", "high", "medium")


def normalize_model_init_dtype(model_init_kwargs: dict) -> dict:
    """Normalize the dtype in model_init_kwargs to transformers' ``dtype`` key, resolved (in place).

    ``model_init_kwargs`` is user YAML handed straight to ``from_pretrained``, so the pre-v5
    ``torch_dtype`` spelling a config may still carry is folded into ``dtype`` here rather than
    passed on as transformers' deprecated alias (``dtype`` wins if a config sets both). Handles
    "auto", None, ``torch.dtype`` instances, and names like "bfloat16"; raises on an invalid name.
    """
    if "torch_dtype" in model_init_kwargs:
        legacy = model_init_kwargs.pop("torch_dtype")
        model_init_kwargs.setdefault("dtype", legacy)
    if "dtype" not in model_init_kwargs:
        return model_init_kwargs

    dtype = model_init_kwargs["dtype"]
    if isinstance(dtype, torch.dtype) or dtype in ("auto", None):
        return model_init_kwargs

    if isinstance(dtype, str):
        # DTYPE_BY_NAME first: torch has no ``bf16``/``fp16``/``fp32`` attribute, so a getattr-only
        # resolve would reject the spellings every ``--dtype`` CLI accepts.
        resolved = DTYPE_BY_NAME.get(dtype) or getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                f"Invalid dtype '{dtype}'. Expected 'auto', one of "
                f"{sorted(DTYPE_BY_NAME)}, or a torch.dtype attribute name (e.g. 'float64')."
            )
        model_init_kwargs["dtype"] = resolved
        return model_init_kwargs

    raise ValueError(f"Invalid dtype type {type(dtype)}. Expected str or torch.dtype.")


def resolve_training_dtype(config) -> torch.dtype:
    """Map a config's precision flags to a ``torch.dtype`` (``bf16`` → bfloat16, ``fp16`` → float16, else float32)."""
    if getattr(config, "bf16", False):
        return torch.bfloat16
    if getattr(config, "fp16", False):
        return torch.float16
    return torch.float32


def reject_fp8_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype | None) -> None:
    """Raise when ``tensor`` is a 1-byte float (fp8): a quantized checkpoint's weight, whose block scales
    live in a separate tensor that a plain cast to ``dtype`` would drop, leaving unscaled values."""
    if tensor.is_floating_point() and tensor.element_size() == 1:
        raise ValueError(
            f"{name!r} is stored in {tensor.dtype}: this is a quantized fp8 checkpoint, and casting it "
            f"to {dtype} would drop its block scales. Dequantize it to bf16 once and train from that "
            f"(scripts/before_training/convert_*_bf16.py, for the families that ship a converter)."
        )


def is_packed_4bit_parameter(param: nn.Parameter) -> bool:
    """Whether ``param`` is bnb's ``Params4bit``, whose storage holds packed codes even when its declared
    storage dtype is floating point. Read off its ``quant_state`` attribute, without importing bnb."""
    return hasattr(param, "quant_state")


def has_packed_quantized_params(model: torch.nn.Module) -> bool:
    """Whether ``model`` carries quantized (non-floating-point) params, i.e. bnb 4/8-bit packed storage (QLoRA)."""
    return any(not p.dtype.is_floating_point for p in model.parameters())


def is_kbit_quantized(model: torch.nn.Module) -> bool:
    """Whether ``model`` was loaded quantized, the condition ``build_peft_model`` runs peft's k-bit
    preparation on: bitsandbytes sets the 8/4-bit pair, torchao/quanto only ``is_quantized``."""
    return any(getattr(model, flag, False) for flag in ("is_loaded_in_8bit", "is_loaded_in_4bit", "is_quantized"))


def cast_parameters_to_run_dtype(
    model: nn.Module, dtype: torch.dtype | str | None, *, keep_fp32: bool = False
) -> None:
    """Cast every floating parameter of a freshly loaded ``model`` to the run's ``dtype`` (in place).

    ``from_pretrained`` keeps a family's ``_keep_in_fp32_modules[_strict]`` parameters in fp32, which
    FSDP2 rejects inside one shard group and which promote activations into run-dtype projections, so
    the loaders call this right after the load, before any parallel wrap. ``keep_fp32`` leaves
    parameters that already arrived in fp32 untouched. Parameters only (a float buffer may be fp32 by
    design); "auto" and None leave the model as loaded; bnb's packed 4-bit storage is skipped.

    Raises on any other ``dtype``, on an fp8 parameter (:func:`reject_fp8_tensor`), and on a
    ``Parameter`` or tensor subclass off the run dtype: rebinding a DTensor's ``.data`` leaves its local
    shard in the old dtype.
    """
    if dtype in ("auto", None):
        return
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"cast_parameters_to_run_dtype takes a torch.dtype, 'auto' or None, not {dtype!r}.")
    for name, param in model.named_parameters():
        if is_packed_4bit_parameter(param):
            continue
        if not param.is_floating_point() or param.dtype == dtype or (keep_fp32 and param.dtype == torch.float32):
            continue
        reject_fp8_tensor(name, param, dtype)
        if type(param) is not nn.Parameter or type(param.data) is not torch.Tensor:
            raise TypeError(
                f"Cannot cast {name!r}, a {type(param).__name__} ({param.dtype}), to the run dtype {dtype}: "
                f"cast_parameters_to_run_dtype casts plain parameters of the freshly loaded model, before "
                f"any parallel wrap."
            )
        param.data = param.data.to(dtype)


def resolve_quantization_config(model_config, training_config):
    """The run's BitsAndBytes config, with the 4-bit compute dtype pinned to the run's own dtype.

    TRL derives ``bnb_4bit_compute_dtype`` from ``ModelConfig.dtype``, which defaults to ``"float32"``,
    so a bf16 QLoRA run would otherwise compute every 4-bit matmul in fp32. Every loader accepting a
    quantized base goes through this rather than ``get_quantization_config``.
    """
    quantization_config = get_quantization_config(model_config)
    if getattr(quantization_config, "load_in_4bit", False):
        quantization_config.bnb_4bit_compute_dtype = resolve_training_dtype(training_config)
    return quantization_config


def configure_float32_matmul_precision() -> None:
    """Pin fp32 matmuls and cuDNN convolutions to true fp32, overriding the image's TF32 default.

    TF32's 10-bit mantissa collapses adjacent RoPE token positions past 2048, corrupting long context
    on every model; pinning it in-process keeps the guarantee independent of the image's env. bf16
    matmuls are unaffected. ``HALO_FP32_MATMUL_PRECISION=high`` opts back into TF32, for cuDNN too,
    whose ``allow_tf32`` is a separate switch defaulting to True (Zaya's fp32 Conv1d).
    """
    precision = env_str("HALO_FP32_MATMUL_PRECISION") or "highest"
    if precision not in _VALID_FP32_MATMUL_PRECISION:
        # warnings.warn, not the accelerate logger (which needs an initialized PartialState).
        warnings.warn(
            f"Ignoring invalid HALO_FP32_MATMUL_PRECISION={precision!r} "
            f"(expected one of {_VALID_FP32_MATMUL_PRECISION}); using 'highest'.",
            stacklevel=2,
        )
        precision = "highest"
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = precision != "highest"


def resolve_load_dtype(dtype: torch.dtype | None, config) -> torch.dtype:
    """The dtype a load builds and casts to: the explicit override, else the checkpoint config's, else bf16.

    Keyed on the value, not the attribute: a config declaring no dtype carries ``dtype = None``, which the
    meta shell would resolve to torch's fp32 and the materializer to no cast, mixing a family's fp32-pinned
    modules with bf16 ones in one FSDP2 shard group.
    """
    return dtype or getattr(config, "dtype", None) or torch.bfloat16
