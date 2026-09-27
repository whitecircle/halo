"""Run precision: ``dtype`` resolution, the loaded-parameter cast and the process-global fp32 matmul pin.

Depends on torch and TRL only: the name-to-``torch.dtype`` table every ``--dtype`` flag derives its
choices from, the ``model_init_kwargs`` normalizer, the run-dtype and quantization resolvers, the
parameter cast the training loaders apply, and the matmul pin. A CLI or entry point can therefore
resolve a dtype, or apply the pin, without importing the model-loading stack (which reaches
``src.args`` and back into ``src.training``).
"""

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


def resolve_model_dtype(model_init_kwargs: dict) -> dict:
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


def cast_parameters_to_run_dtype(
    model: nn.Module, dtype: torch.dtype | str | None, *, keep_fp32: bool = False
) -> None:
    """Cast every floating parameter of a freshly loaded ``model`` to the run's ``dtype`` (in place).

    ``from_pretrained`` keeps a family's ``_keep_in_fp32_modules[_strict]`` parameters in fp32
    (DeepSeek-V4 norms and hyper-connections, GLM-5 Next KDA state, Inkling short convolutions).
    FSDP2 rejects mixed dtypes in one shard group, and DeepSeek-V4's fp32 norms promote activations
    into its run-dtype projections. The training and scoring loaders call this right after the load,
    before any parallel wrapper, so a family trains in one precision whatever the parallelism (the EP
    lazy loader casts per tensor to the same effect); conversion tools keep the pins.

    ``keep_fp32`` leaves fp32 parameters as stored, for a run that upcasts to fp32 masters anyway
    (``fp32_non_ep_params``): a round trip through the run dtype would discard the checkpoint's
    precision before the upcast. The training loaders apply it outside the MoE blocks EP wraps
    (``cast_loaded_parameters``), and the EP lazy loader materializes the same keys in fp32. Parameters only: a float buffer may be fp32 by design (Zaya's
    balancing biases). A ``dtype`` that is not a ``torch.dtype`` ("auto", None) leaves the model as
    loaded.

    Raises on a 1-byte float (fp8) parameter, whose block scales live beside it and would be dropped by
    a plain cast, and on a tensor-subclass parameter: rebinding a DTensor's ``.data`` leaves its local
    shard in the old dtype, which is why the dense TP loader, loading straight into DTensors, does not
    call this (no dense family pins a parameter).
    """
    if not isinstance(dtype, torch.dtype):
        return
    for name, param in model.named_parameters():
        if not param.is_floating_point() or param.dtype == dtype or (keep_fp32 and param.dtype == torch.float32):
            continue
        if param.element_size() == 1:
            raise ValueError(
                f"{name!r} is stored in {param.dtype}: this is a quantized fp8 checkpoint, and casting it "
                f"to {dtype} would drop its block scales. Convert it to bf16 once "
                f"(scripts/before_training/convert_*_bf16.py where the family has one) and train from that."
            )
        if type(param.data) is not torch.Tensor:
            raise TypeError(
                f"Cannot cast {name!r}, a {type(param.data).__name__} ({param.dtype}), to the run dtype "
                f"{dtype}: cast_parameters_to_run_dtype runs on the freshly loaded model, before any "
                f"parallel wrap."
            )
        param.data = param.data.to(dtype)


def resolve_quantization_config(model_config, training_config):
    """The run's BitsAndBytes config, with the 4-bit compute dtype pinned to the run's own dtype.

    TRL derives ``bnb_4bit_compute_dtype`` from ``ModelConfig.dtype``, which defaults to ``"float32"``
    and is read by nothing else here; every other consumer takes the training config's
    ``bf16``/``fp16`` through :func:`resolve_training_dtype`. Left alone, a bf16 QLoRA run dequantizes
    and computes every 4-bit matmul in fp32. Every loader that accepts a quantized base must go
    through this rather than ``get_quantization_config`` directly.
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
