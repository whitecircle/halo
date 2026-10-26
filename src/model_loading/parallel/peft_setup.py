"""PEFT for a loaded model: the expert-LoRA target peel, the adapter config and the adapter build."""

from __future__ import annotations

import contextlib
import copy
import warnings

import torch
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.tuners_utils import BaseTunerLayer, _maybe_include_all_linear_layers, check_target_module_exists
from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND
from torch.distributed.tensor import DTensor
from transformers import PreTrainedModel
from trl import ModelConfig, get_peft_config

from src.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from src.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from src.model_loading.dtype import is_kbit_quantized
from src.model_loading.parallel.freeze_patterns import freeze_modules_by_patterns, unfreeze_modules_by_patterns
from src.model_loading.parallel.model_source import fetch_checkpoint_config
from src.models.attention.gpt_oss_sinks import SinksPolicy, stamped_sinks_policy
from src.models.moe.config_fields import has_routed_experts
from src.models.moe.expert_lora_spec import ExpertLoraSpec, expert_target_projections
from src.models.param_roles import EMBEDDING_HEAD_MARKERS, is_normalization_module
from src.runtime.consensus import reject_across_ranks, reject_divergent_settings

# The forward-probe verdict cache key: a class can be grouped at one width and plain at another.
_ProbeKey = tuple[type, int, int]


def split_expert_lora_targets(model_config: ModelConfig) -> ExpertLoraSpec | None:
    """Peel the expert-weight entries out of ``lora_target_modules`` (in place) into an
    :class:`ExpertLoraSpec` for native EP grouped-LoRA, since stock PEFT cannot wrap grouped expert tensors.

    COLLECTIVE-EQUIVALENT (a main-rank-first config read). Runs before ``load_distributed_model``; ``None``
    when PEFT is off, the model is dense, or no entry names an expert projection.
    """
    if not model_config.use_peft:
        return None
    targets = model_config.lora_target_modules
    # TRL's ModelConfig.__post_init__ collapses a one-element list to a bare string, so a lone expert
    # target ("experts") would otherwise skip the peel and reach PEFT as a never-matching regex. A
    # string naming no projection is a sentinel or regex ("all-linear", ".*gate_proj"): peel nothing.
    if isinstance(targets, str):
        targets = [targets] if expert_target_projections(targets) else []
    elif not isinstance(targets, (list, tuple, set)):
        targets = []
    if not targets:
        return None

    # Dense gate/up/down_proj are plain nn.Linear stock PEFT wraps; peeling them drops the MLP adapters.
    config = fetch_checkpoint_config(model_config, tag="expert_lora_probe")
    if not has_routed_experts(config):
        return None

    projections: set[str] = set()
    remaining: list[str] = []
    for target in targets:
        logical = expert_target_projections(target)
        if logical:
            projections |= logical
        else:
            remaining.append(target)

    if not projections:
        return None

    # The peeled projections become an ExpertLoraSpec, which carries only r/alpha/dropout/use_rslora:
    # any other LoraConfig knob would apply to the attention half alone.
    if getattr(model_config, "use_dora", False):
        raise ValueError(
            f"use_dora=True cannot be combined with expert projections {sorted(projections)} in "
            "lora_target_modules: DoRA's magnitude decomposition is not implemented for the native "
            "grouped expert adapters, so it would apply to the attention adapters only. Drop use_dora, "
            "or drop the expert projections and LoRA the attention modules alone."
        )
    # The peel is name-based, not module-based: on a hybrid MoE the same names also spell plain
    # nn.Linear MLPs (dense prefix layers, shared experts) that neither half then adapts.
    warnings.warn(
        f"Expert projections {sorted(projections)} peeled from lora_target_modules to native EP "
        f"grouped-LoRA; {remaining or 'no'} target(s) left for PEFT. Plain nn.Linear MLPs sharing "
        f"those names (dense prefix layers, shared experts) are adapted by NEITHER half — name them "
        f"under their own module paths if you need them.",
        stacklevel=2,
    )
    model_config.lora_target_modules = remaining
    return ExpertLoraSpec(
        r=model_config.lora_r,
        alpha=model_config.lora_alpha,
        dropout=model_config.lora_dropout or 0.0,
        projections=frozenset(projections),
        use_rslora=bool(getattr(model_config, "use_rslora", False)),
    )


def wraps_in_peft_model(model_config: ModelConfig) -> bool:
    """Whether stock PEFT has targets left after the expert peel, i.e. the run gets a ``PeftModel``.

    ``None`` targets mean the architecture defaults; ``[]`` means every target was an expert
    projection (expert-only LoRA), and conflating the two would full-finetune a ``use_peft`` run.
    """
    targets = model_config.lora_target_modules
    return model_config.use_peft and (targets is None or bool(targets))


def _inherits_its_layer_forward(cls: type) -> bool:
    """Whether ``cls`` runs the ``forward`` of a concrete ``torch.nn`` layer rather than its own."""
    forward = cls.forward
    return any(
        base.__module__.startswith("torch.nn.modules.") and vars(base).get("forward") is forward
        for base in cls.__mro__
    )


def _computes_its_layers_affine_map(linear: torch.nn.Linear, verdicts: dict[_ProbeKey, bool]) -> bool:
    """Whether the layer maps ``(1, in_features)`` to ``(1, out_features)``, as ``F.linear`` does.

    Measured, once per class and geometry: an ``nn.Linear.forward`` override may only change how the map
    is computed (a quantized kernel) or compute a different one (a block-diagonal grouped projection),
    and nothing in the class tells the two apart. A device fault is not an answer about the layer.
    """
    key = (type(linear), linear.in_features, linear.out_features)
    if (verdict := verdicts.get(key)) is None:
        probe = torch.zeros(1, linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype)
        try:
            with torch.no_grad():
                verdict = linear(probe).shape[-1] == linear.out_features
        except (torch.OutOfMemoryError, torch.AcceleratorError):
            raise
        except Exception:
            verdict = False
        verdicts[key] = verdict
    return verdict


def _is_probeable_linear(module: torch.nn.Module) -> bool:
    """Whether a forward probe on this module is both meaningful and free of side effects: a float
    ``nn.Linear`` weight of at least 2 bytes (fp8 has no CPU kernel), neither meta nor a ``DTensor``
    (the probe would become a collective)."""
    weight = getattr(module, "weight", None)
    return (
        isinstance(module, torch.nn.Linear)
        and isinstance(weight, torch.Tensor)
        and not weight.is_meta
        and weight.dtype.is_floating_point
        and weight.dtype.itemsize >= 2
        and not isinstance(weight.data, DTensor)
    )


def _unadaptable_reason(module: torch.nn.Module, verdicts: dict[_ProbeKey, bool]) -> str | None:
    """Why stock LoRA cannot adapt this module, or ``None`` when it can.

    PEFT decomposes the matched module's own weight and adds the delta to its output, which misses a
    wrapper holding its weight in a child module and a layer computing something other than its affine
    map. An adapted layer is read through its base layer, as PEFT reads it.
    """
    base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
    if not any(True for _ in base.parameters(recurse=False)):
        return "hold their weight in a child module" if any(True for _ in base.parameters()) else None
    if _inherits_its_layer_forward(type(base)) or not _is_probeable_linear(base):
        return None
    if _computes_its_layers_affine_map(base, verdicts):
        return None
    return "compute something other than their layer's affine map"


def _excluded_detail(model: torch.nn.Module, excluded: dict[str, list[str]]) -> str:
    """One counted clause per reason, with an example path and its class."""
    return "; ".join(
        f"{len(names)} {reason} (e.g. {names[0]}, a {type(model.get_submodule(names[0])).__name__})"
        for reason, names in excluded.items()
    )


def _exclude_unadaptable_lora_targets(model: torch.nn.Module, peft_config) -> None:
    """COLLECTIVE: exclude, by full module path, the target-name matches PEFT cannot adapt.

    Targets match by name suffix over the whole tree (a vision tower's ``q_proj`` too, and ``all-linear``
    every custom projection), through PEFT's own matcher. The rejection and the exclusion list are agreed
    across ranks: a rank adapting a different parameter set would only surface as a hang.
    """
    # `all-linear` is a sentinel PEFT resolves against the model at injection time; resolve a copy the
    # same way so the scan sees what injection will target while the config the caller holds — the one
    # the warning and raise below quote — still reads the sentinel; PEFT rewrites it in place at injection.
    scan_config = _maybe_include_all_linear_layers(copy.deepcopy(peft_config), model)
    preset = peft_config.exclude_modules
    if isinstance(preset, str):
        raise ValueError(
            f"exclude_modules={preset!r} is a regex; the scan adds module paths to the list form. Spell the "
            f"exclusion as a list of module names."
        )
    excluded: dict[str, list[str]] = {}
    verdicts: dict[_ProbeKey, bool] = {}
    adaptable = 0
    for name, module in model.named_modules():
        # The scan config carries the preset exclusions, so PEFT's matcher already skips those.
        if not name or not check_target_module_exists(scan_config, name):
            continue
        if reason := _unadaptable_reason(module, verdicts):
            excluded.setdefault(reason, []).append(name)
        else:
            adaptable += 1

    reason = None
    if excluded and not adaptable:
        total = sum(len(names) for names in excluded.values())
        # Native grouped expert adapters, when present, would train on without any attention adapter.
        consequence = (
            "The native expert adapters would train alone, with no attention adapter at all"
            if model_has_expert_lora(model)
            else "The run would train nothing"
        )
        reason = (
            f"lora_target_modules={peft_config.target_modules} matched {total} module(s) and PEFT can "
            f"adapt none of them — {_excluded_detail(model, excluded)}. {consequence}. Name the projections "
            f"of the backbone you mean to adapt, or use 'all-linear'."
        )
    reject_across_ranks(reason, "lora_target_modules", ValueError)
    found = sorted(name for names in excluded.values() for name in names)
    reject_divergent_settings(
        {"lora_exclude_modules": tuple(found)},
        "The LoRA exclusion list",
        "The module tree and lora_target_modules are the same on every rank, so a differing list means "
        "the forward probe faulted on one rank.",
    )
    if not excluded:
        return
    total = len(found)
    warnings.warn(
        f"Excluded {total} of {total + adaptable} lora_target_modules matches from PEFT injection — "
        f"{_excluded_detail(model, excluded)}. No stock LoRA adapter fits those; the other {adaptable} "
        f"module(s) are adapted.",
        stacklevel=2,
    )
    peft_config.exclude_modules = sorted(set(preset or ()) | set(found))


def build_peft_config(model: torch.nn.Module, model_config: ModelConfig) -> object | None:
    """COLLECTIVE: the run's ``LoraConfig``, with the name matches PEFT cannot adapt excluded."""
    targets = model_config.lora_target_modules
    # PEFT reads its sentinel only as a bare string, case-insensitively; TRL collapses a one-entry YAML
    # list to that at construction, but a CLI override lands after construction and arrives as a list.
    if isinstance(targets, (list, tuple, set)) and [str(t).lower() for t in targets] == [
        INCLUDE_LINEAR_LAYERS_SHORTHAND
    ]:
        model_config.lora_target_modules = INCLUDE_LINEAR_LAYERS_SHORTHAND
    peft_config = get_peft_config(model_config)
    if peft_config is not None and getattr(peft_config, "target_modules", None):
        _exclude_unadaptable_lora_targets(model, peft_config)
    return peft_config


def configure_adapter_training(
    args,
    model: PreTrainedModel,
    model_config: ModelConfig,
    expected_task_type: str = "CAUSAL_LM",
) -> object | None:
    """COLLECTIVE: set which parameters train and return the PEFT config to wrap the model with, if any.

    No adapters: the freeze/unfreeze patterns, else a full fine-tune. Any adapter: a frozen base with the
    native EP expert adapters trainable, and the ``LoraConfig`` unless the run is expert-LoRA only.
    """
    # lora_target_parameters makes PEFT wrap the module holding the parameter, which for an expert weight
    # is the EP layer itself: its adapters would sit outside the EP validators and gradient sync. Without
    # EP wrappers the experts are ordinary modules stock PEFT handles.
    if getattr(model_config, "lora_target_parameters", None) and named_ep_layers(model):
        raise ValueError(
            f"lora_target_parameters={list(model_config.lora_target_parameters)} is not supported under "
            "Expert Parallelism: PEFT would wrap the EP layer itself and replace it in its parent, putting "
            "the adapters outside the EP gradient sync and outside the EP validators. Name the expert "
            "projections in lora_target_modules instead (gate_proj/up_proj/down_proj/gate_up_proj/experts) "
            "— they route to native grouped LoRA, which is EP-sharded and synced with the experts."
        )
    expert_lora_active = model_has_expert_lora(model)
    attention_peft = wraps_in_peft_model(model_config)

    if not attention_peft and not expert_lora_active:
        if model_config.use_peft:
            raise ValueError(
                "use_peft: true but no adapter would be created — the base model stays fully trainable "
                "and the run would full-finetune at the LoRA learning rate. lora_target_modules is empty "
                "(after the expert peel, if any), so there is nothing for PEFT to wrap and no EP layer "
                "built native grouped adapters. Name the modules to adapt (attention projections and/or "
                "expert projections: gate_proj/up_proj/down_proj/gate_up_proj/experts), or set "
                "use_peft: false for a deliberate full fine-tune."
            )
        if args.unfreeze_layers_patterns:
            unfreeze_modules_by_patterns(model, args.unfreeze_layers_patterns)
        if args.freeze_layers_patterns:
            freeze_modules_by_patterns(model, args.freeze_layers_patterns)
        return None

    set_patterns = [
        name for name in ("unfreeze_layers_patterns", "freeze_layers_patterns") if getattr(args, name, None)
    ]
    if set_patterns:
        raise ValueError(
            f"{' and '.join(set_patterns)} cannot combine with adapters: an adapter run freezes every base "
            "parameter and trains only the adapters, so the patterns would select nothing. Drop the "
            "patterns, or drop the adapters (use_peft: false, no expert LoRA targets) for a partial "
            "fine-tune."
        )
    if stamped_sinks_policy(model) is SinksPolicy.TRAINABLE:
        raise ValueError(
            "train_sinks: true needs full fine-tuning: an adapter run freezes every base parameter, and "
            "the adapter artifact has no slot for the sinks, so they would train nowhere. Keep the sinks "
            "live and frozen instead (reset_sinks: false without train_sinks), or drop the adapters."
        )
    for p in model.parameters():
        p.requires_grad = False
    # The native grouped expert adapters come back on; PEFT re-enables its own after the wrap.
    for _path, layer in named_ep_layers(model):
        for _name, adapter in layer.expert_lora_named_params():
            adapter.requires_grad_(True)

    if not attention_peft:
        # Expert-LoRA only: no PeftModel exists, so every remaining LoraConfig field is inert.
        # modules_to_save is the one a user can reasonably expect to apply, and ignoring it would
        # leave the router/head frozen without an error.
        if model_config.lora_modules_to_save:
            raise ValueError(
                f"lora_modules_to_save={list(model_config.lora_modules_to_save)} cannot be honoured by an "
                "expert-only LoRA run: lora_target_modules names expert projections exclusively, so the "
                "model is never PEFT-wrapped and those modules would stay frozen. Add at least one "
                "attention target (they route to stock PEFT, which owns modules_to_save), or drop "
                "lora_modules_to_save. Either way merge_expert_lora_on_save still produces a merged "
                "servable checkpoint — it folds the attention half of a mixed run too."
            )
        return None  # expert-LoRA only

    if model_config.lora_task_type != expected_task_type:
        warnings.warn(
            f"You are using a `task_type` that is different than `{expected_task_type}` for PEFT. "
            f"This will lead to silent bugs. Make sure to pass --lora_task_type {expected_task_type}.",
            stacklevel=2,
        )

    return build_peft_config(model, model_config)


def _peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module.to(torch.bfloat16)
        elif is_normalization_module(module):
            # Class-derived rather than name-derived: norm modules at a path without "norm" in it are
            # still kept in fp32, and a non-norm module under a "norm"-named parent is not upcast.
            module.to(torch.float32)
        elif any(marker in name for marker in EMBEDDING_HEAD_MARKERS):  # noqa: SIM102  keep guard separate from the weight-dtype check
            if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)


def build_peft_model(model, peft_config, args, *, merge_existing: bool = True):
    """Wrap ``model`` with ``peft_config``: k-bit prep, adapters, optional bf16 cast. Returns
    ``(model, casted_to_bf16)``; ``merge_existing=False`` returns a ``PeftModel`` unchanged.

    The k-bit prep freezes every parameter, so it must precede ``get_peft_model``.
    """
    if isinstance(model, PeftModel):
        if not merge_existing:
            return model, False
        model = model.merge_and_unload()

    if is_kbit_quantized(model):
        prepare_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
        gc_kwargs = getattr(args, "gradient_checkpointing_kwargs", None)
        if gc_kwargs is not None:
            prepare_kwargs["gradient_checkpointing_kwargs"] = gc_kwargs
        model = prepare_model_for_kbit_training(model, **prepare_kwargs)

    model = get_peft_model(model, peft_config)

    casted = bool(getattr(args, "bf16", False) and getattr(model, "is_loaded_in_4bit", False))
    if casted:
        _peft_module_casting_to_bf16(model)
    return model, casted


def peft_bf16_autocast(casted_to_bf16: bool, device: torch.device) -> contextlib.AbstractContextManager:
    """bf16 autocast around a bf16-cast PEFT forward, inert otherwise: the cast leaves adapter modules
    bf16 while activations stay fp32. bf16 is explicit because torch's CUDA autocast defaults to fp16."""
    if not casted_to_bf16:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)
