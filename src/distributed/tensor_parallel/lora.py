"""Read-only validation of PEFT's native LoRA path on a pure dense TP mesh."""

from __future__ import annotations

import copy
import math

import peft.import_utils as peft_import_utils
import torch.nn as nn
from peft import LoraConfig, PeftModel
from peft.tuners.lora import model as peft_lora_model
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTunerLayer, _maybe_include_all_linear_layers, check_target_module_exists
from torch.distributed.tensor import DTensor, Shard

try:
    from transformers.distributed import tensor_parallel as hf_tp
except ImportError:
    # Older supported training dependencies do not expose the native DTensor TP API.
    hf_tp = None


_CONFIG_FIELDS = frozenset(
    {
        "auto_mapping",
        "base_model_name_or_path",
        "exclude_modules",
        "init_lora_weights",
        "layers_pattern",
        "layers_to_transform",
        "lora_alpha",
        "lora_dropout",
        "peft_type",
        "peft_version",
        "r",
        "revision",
        "target_modules",
        "task_type",
        "use_rslora",
    }
)
_NATIVE_STYLES = frozenset({"colwise", "rowwise"})
_RUNTIME_ERROR = (
    "Native TP LoRA requires PEFT's DTensor integration and Transformers' native TP styles "
    "(PEFT >= 0.21.1 and Transformers >= 5.17). The installed runtime lacks that API; "
    "use LoRA without TP or use a compatible training image."
)


def validate_native_tp_lora_config(model: nn.Module, peft_config: LoraConfig, *, tp_size: int | None = None) -> None:
    """Reject unsupported adapter settings and live targets before PEFT injects them.

    This does not change the model, configuration, initialization or TP transforms. The caller
    owns the dense-model, trainer and parallel-axis eligibility gates.
    """
    _require_native_runtime()
    _validate_config(peft_config)
    mesh, plan = _native_mesh_and_plan(model, tp_size)
    if isinstance(model, PeftModel):
        raise ValueError("An existing PEFT model needs validate_native_tp_lora_model, not adapter reinjection.")

    scan_config = copy.deepcopy(peft_config)
    if scan_config.target_modules is None:
        model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
        scan_config = peft_lora_model.LoraModel._prepare_adapter_config(scan_config, model_config)
    scan_config = _maybe_include_all_linear_layers(scan_config, model)
    embedding_ids = _embedding_module_ids(model)
    matched = []
    # An embedding/head alias must not disappear behind another registration of the same module.
    for name, module in model.named_modules(remove_duplicate=False):
        if name and check_target_module_exists(scan_config, name):
            _reject_embedding_target(name, module, embedding_ids)
            style = _target_style(name, plan)
            _validate_base(name, module, style, mesh)
            matched.append(name)
    if not matched:
        raise ValueError("Native TP LoRA target_modules matched no supported colwise/rowwise nn.Linear layer.")


def validate_native_tp_lora_model(model: nn.Module, *, tp_size: int | None = None) -> None:
    """Verify the native adapter geometry, transforms and frozen-base training contract.

    Layer metadata must describe the global projection, not its local shard. This exposes an
    initializer using local fan-in without guessing from weights that may already be trained.
    """
    _require_native_runtime()
    if not isinstance(model, PeftModel):
        raise ValueError("Native TP LoRA requires a PEFT model with one active LoRA adapter.")
    configs = model.peft_config
    if len(configs) != 1:
        raise ValueError("Native TP LoRA supports exactly one adapter, not multiple or mixed adapters.")
    adapter, config = next(iter(configs.items()))
    if adapter != "default":
        raise ValueError("Native TP LoRA requires the adapter name 'default' for checkpoint save and resume.")
    _validate_config(config)
    if model.active_adapters != [adapter]:
        raise ValueError("Native TP LoRA requires its single adapter to be active.")

    base_model = model.get_base_model()
    mesh, plan = _native_mesh_and_plan(base_model, tp_size)
    embedding_ids = _embedding_module_ids(base_model)
    adapter_param_ids = set()
    for name, layer in base_model.named_modules():
        if not isinstance(layer, BaseTunerLayer):
            continue
        if type(layer) is not LoraLinear:
            raise ValueError(f"Native TP LoRA target {name!r} must be PEFT's ordinary Linear adapter.")
        _reject_embedding_target(name, layer, embedding_ids)
        if layer.disable_adapters or layer.merged or layer.active_adapters != [adapter]:
            raise ValueError(f"Native TP LoRA target {name!r} must have one active, enabled, unmerged adapter.")
        if getattr(layer, "lora_variant", None):
            raise ValueError(f"Native TP LoRA target {name!r} uses an unsupported LoRA variant.")
        style = _target_style(name, plan)
        base = layer.get_base_layer()
        _reject_embedding_target(name, base, embedding_ids)
        _validate_base(name, base, style, mesh)
        if getattr(base, "_hf_tp_plan", None) != style or getattr(base, "_hf_device_mesh", None) != mesh:
            raise ValueError(f"Native TP LoRA target {name!r} is missing its native PEFT TP plan/mesh metadata.")
        expected_geometry = (base.in_features, base.out_features)
        if (layer.in_features, layer.out_features) != expected_geometry:
            raise ValueError(
                f"Native TP LoRA target {name!r} has local-shard layer metadata instead of global dimensions "
                f"{expected_geometry}. Its initializer can use the wrong fan-in; a compatible native PEFT "
                "shape fix is required. No local reinitialization is applied."
            )
        _validate_adapter_metadata(name, layer, adapter, config)
        factors = (("lora_A", (config.r, base.in_features)), ("lora_B", (base.out_features, config.r)))
        for factor_name, shape in factors:
            mapping = getattr(layer, factor_name)
            if set(mapping) != {adapter}:
                raise ValueError(f"Native TP LoRA target {name!r} has unexpected {factor_name} adapters.")
            factor = mapping[adapter]
            if type(factor) is not nn.Linear or factor.bias is not None or tuple(factor.weight.shape) != shape:
                raise ValueError(f"Native TP LoRA {name}.{factor_name} must be a bias-free Linear of shape {shape}.")
            if (factor.in_features, factor.out_features) != (shape[1], shape[0]):
                raise ValueError(f"Native TP LoRA {name}.{factor_name} has incorrect global Linear metadata.")
            is_sharded = factor_name == ("lora_B" if style == "colwise" else "lora_A")
            if is_sharded:
                _validate_shard(f"{name}.{factor_name}", factor.weight, 0 if style == "colwise" else 1, mesh)
                _validate_native_forward(name, factor, style, mesh)
            elif isinstance(factor.weight, DTensor):
                raise ValueError(f"Native TP LoRA {name}.{factor_name} must be a plain replicated factor.")
            if not factor.weight.requires_grad:
                raise ValueError(f"Native TP LoRA {name}.{factor_name} is unexpectedly frozen.")
            adapter_param_ids.add(id(factor.weight))
    if not adapter_param_ids:
        raise ValueError("Native TP LoRA found no supported adapter factors on the model.")
    unfrozen = [
        name for name, param in model.named_parameters() if param.requires_grad and id(param) not in adapter_param_ids
    ]
    if unfrozen:
        raise ValueError(
            f"Native TP LoRA must freeze every base parameter; unexpected trainable parameters: {unfrozen}."
        )


def _require_native_runtime() -> None:
    if not getattr(peft_import_utils, "is_transformers_dtensor_tp", False):
        raise RuntimeError(_RUNTIME_ERROR)
    if not callable(getattr(peft_lora_model, "add_lora_tp_hooks_dtensor", None)):
        raise RuntimeError(_RUNTIME_ERROR)
    if hf_tp is None or not callable(getattr(hf_tp, "_get_parameter_tp_plan", None)):
        raise RuntimeError(_RUNTIME_ERROR)
    styles = getattr(hf_tp, "ALL_PARALLEL_STYLES", {})
    for name in _NATIVE_STYLES:
        style = styles.get(name)
        if style is None or any(
            not callable(getattr(style, api, None)) for api in ("validate_param", "shard_param", "install_forward")
        ):
            raise RuntimeError(_RUNTIME_ERROR)


def _validate_config(config: LoraConfig) -> None:
    if type(config) is not LoraConfig:
        raise ValueError("Native TP supports ordinary LoraConfig only, not other PEFT methods.")
    if config.init_lora_weights is not True:
        raise ValueError(
            "Native TP LoRA currently requires init_lora_weights=True. Other initializers have unvalidated "
            "shard initialization or replica synchronization; use default LoRA initialization or LoRA without TP."
        )
    if config.lora_dropout != 0.0:
        raise ValueError(
            "Native TP LoRA requires lora_dropout=0.0; rank-dependent dropout masks are not synchronized."
        )
    if config.task_type not in (None, "CAUSAL_LM"):
        raise ValueError("Native TP LoRA supports causal language models only.")
    if not isinstance(config.r, int) or isinstance(config.r, bool) or config.r <= 0:
        raise ValueError("Native TP LoRA requires a positive integer rank r.")
    if not math.isfinite(config.lora_alpha) or config.lora_alpha <= 0:
        raise ValueError("Native TP LoRA requires a finite positive lora_alpha.")
    defaults = vars(LoraConfig())
    unsupported = [
        name for name, value in vars(config).items() if name not in _CONFIG_FIELDS and value != defaults.get(name)
    ]
    if unsupported:
        raise ValueError(f"Native TP LoRA does not support non-default options {sorted(unsupported)}.")


def _native_mesh_and_plan(model: nn.Module, tp_size: int | None):
    mesh = getattr(model, "_device_mesh", None)
    if mesh is None or mesh.ndim != 1 or mesh.mesh_dim_names != ("tp",) or mesh.size() <= 1:
        raise ValueError("Native TP LoRA requires the model's named one-dimensional ('tp',) mesh with TP > 1.")
    if tp_size is not None and tp_size != mesh.size():
        raise ValueError(
            f"Native TP LoRA mesh size {mesh.size()} does not match the requested Halo tp_size={tp_size}."
        )
    distributed = getattr(getattr(model, "config", None), "distributed_config", None)
    if getattr(distributed, "tp_size", None) != mesh.size():
        raise ValueError("Native TP LoRA requires the native distributed_config.tp_size to match the model mesh.")
    if any(getattr(distributed, axis, 1) != 1 for axis in ("fsdp_size", "pp_size")):
        raise ValueError("Native TP LoRA does not support FSDP/DP or PP combined with TP.")
    plan = getattr(model, "tp_plan", None)
    if not isinstance(plan, dict) or not plan:
        raise ValueError("Native TP LoRA requires the model's applied Transformers TP plan.")
    if getattr(model, "_tp_sharded_non_dtensor", None):
        raise ValueError("Native TP LoRA does not support manually sliced TP parameters.")
    return mesh, plan


def _target_style(name: str, plan: dict[str, str]) -> str:
    style = hf_tp._get_parameter_tp_plan(name, plan, is_weight=False)
    if style not in _NATIVE_STYLES:
        raise ValueError(
            f"Native TP LoRA target {name!r} has unsupported TP style {style!r}; only exact colwise/rowwise "
            "Linear targets are supported, not embeddings, lm_head, packed, replicated or unplanned targets."
        )
    return style


def _embedding_module_ids(model: nn.Module) -> set[int]:
    modules = []
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, accessor, None)
        if callable(getter) and (module := getter()) is not None:
            modules.append(module)
    return {id(module) for module in modules}


def _reject_embedding_target(name: str, module: nn.Module, embedding_ids: set[int]) -> None:
    if id(module) in embedding_ids:
        raise ValueError(f"Native TP LoRA target {name!r} is an input/output embedding module, which is unsupported.")


def _validate_base(name: str, base: nn.Module, style: str, mesh) -> None:
    if type(base) is not nn.Linear:
        raise ValueError(f"Native TP LoRA target {name!r} must be an ordinary non-quantized nn.Linear.")
    if not base.weight.dtype.is_floating_point or base.weight.dtype.itemsize < 2:
        raise ValueError(f"Native TP LoRA target {name!r} must have ordinary floating-point base weights.")
    if getattr(base, "_hf_quantized_needs_local_tp", False):
        raise ValueError(f"Native TP LoRA target {name!r} uses an unsupported quantized TP path.")
    if tuple(base.weight.shape) != (base.out_features, base.in_features):
        raise ValueError(f"Native TP LoRA target {name!r} does not retain its global base Linear geometry.")
    _validate_shard(name, base.weight, 0 if style == "colwise" else 1, mesh)


def _validate_shard(name: str, tensor, dim: int, mesh) -> None:
    if not isinstance(tensor, DTensor) or tensor.device_mesh != mesh or len(tensor.placements) != 1:
        raise ValueError(f"Native TP LoRA {name!r} must be a DTensor on the model's same one-dimensional TP mesh.")
    placement = tensor.placements[0]
    if not isinstance(placement, Shard) or placement.dim % tensor.ndim != dim:
        raise ValueError(f"Native TP LoRA {name!r} has incorrect TP placement; expected Shard({dim}).")
    if tensor.shape[dim] % mesh.size() != 0:
        raise ValueError(f"Native TP LoRA {name!r} needs its sharded dimension divisible by TP size {mesh.size()}.")
    local_shape = list(tensor.shape)
    local_shape[dim] //= mesh.size()
    if tuple(tensor.to_local().shape) != tuple(local_shape):
        raise ValueError(f"Native TP LoRA {name!r} has incorrect local shard geometry.")


def _validate_native_forward(name: str, factor: nn.Linear, style: str, mesh) -> None:
    forward = factor.forward
    code, closure = getattr(forward, "__code__", None), getattr(forward, "__closure__", None)
    parts = (
        dict(zip(code.co_freevars, (cell.cell_contents for cell in closure), strict=True)) if code and closure else {}
    )
    original = parts.get("original_forward")
    if (
        parts.get("self") is not hf_tp.ALL_PARALLEL_STYLES[style]
        or parts.get("mesh") != mesh
        or getattr(original, "__self__", None) is not factor
        or getattr(original, "__func__", None) is not nn.Linear.forward
    ):
        raise ValueError(f"Native TP LoRA {name!r} is missing the native Transformers factor forward transform.")


def _validate_adapter_metadata(name: str, layer: LoraLinear, adapter: str, config: LoraConfig) -> None:
    expected_scale = config.lora_alpha / (math.sqrt(config.r) if config.use_rslora else config.r)
    if (
        layer.r.get(adapter) != config.r
        or layer.lora_alpha.get(adapter) != config.lora_alpha
        or not math.isclose(layer.scaling.get(adapter, math.nan), expected_scale)
        or layer.use_dora.get(adapter, False)
        or layer.fan_in_fan_out
        or layer.lora_embedding_A
        or layer.lora_embedding_B
        or (getattr(layer, "lora_bias", {}) or {}).get(adapter, False)
    ):
        raise ValueError(f"Native TP LoRA target {name!r} has unsupported or inconsistent adapter metadata.")
    if set(layer.lora_dropout) != {adapter}:
        raise ValueError(f"Native TP LoRA target {name!r} has unexpected dropout adapters.")
    dropout = layer.lora_dropout[adapter]
    if not isinstance(dropout, nn.Identity) and not (isinstance(dropout, nn.Dropout) and dropout.p == 0.0):
        raise ValueError(f"Native TP LoRA target {name!r} has nonzero or unsupported adapter dropout.")
