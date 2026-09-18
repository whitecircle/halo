"""PEFT for a loaded model: expert-LoRA target peeling, the EP/PP guards, and the adapter build.

Folding a saved adapter back into its base is in :mod:`src.checkpoint.adapters`."""

from __future__ import annotations

import contextlib
import copy
import fnmatch
import warnings

import torch
from accelerate.logging import get_logger
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.tuners_utils import BaseTunerLayer, _maybe_include_all_linear_layers, check_target_module_exists
from peft.utils.constants import INCLUDE_LINEAR_LAYERS_SHORTHAND
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, PreTrainedModel
from trl import ModelConfig, get_peft_config

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import ExpertLoraSpec, expert_target_projections
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.filesystem import hub_metadata_main_first
from src.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from src.distributed.runtime import reject_across_ranks, reject_divergent_settings
from src.models.moe_balancing import config_has_experts
from src.models.patches.gpt_oss_sinks import SinksPolicy, stamped_sinks_policy
from src.models.structure import DECODER_LAYER_LIST_ATTRS, EMBEDDING_HEAD_MARKERS, is_normalization_module

logger = get_logger(__name__)

_TORCH_LAYER_PACKAGE = "torch.nn.modules."
# The forward-probe verdict cache key: a class can be grouped at one width and plain at another.
_ProbeKey = tuple[type, int, int]


def _reject_lora_target_parameters_under_ep(model: PreTrainedModel, model_config: ModelConfig) -> None:
    """Reject PEFT's fused-``nn.Parameter`` adapter mechanism once EP wrappers own the experts.

    ``lora_target_parameters`` makes PEFT wrap the module holding the parameter and swap it into its
    parent; for an expert weight that module is the ``EPMoELayerBase`` itself, so the adapters land
    above the EP layer, outside both the EP validators and the EP gradient sync. Gated on the live
    model rather than on the config: without EP wrappers (``ep_size<=1`` + ``use_grouped_gemm: false``)
    the experts are ordinary modules and stock PEFT handles them.
    """
    if not getattr(model_config, "lora_target_parameters", None):
        return
    if not any(isinstance(module, EPMoELayerBase) for module in model.modules()):
        return
    raise ValueError(
        f"lora_target_parameters={list(model_config.lora_target_parameters)} is not supported under "
        "Expert Parallelism: PEFT would wrap the EP layer itself and replace it in its parent, putting "
        "the adapters outside the EP gradient sync and outside the EP validators. Name the expert "
        "projections in lora_target_modules instead (gate_proj/up_proj/down_proj/gate_up_proj/experts) "
        "— they route to native grouped LoRA, which is EP-sharded and synced with the experts."
    )


def _reject_unsupported_expert_lora_knobs(model_config: ModelConfig, projections: set[str]) -> None:
    """Reject ``LoraConfig`` knobs the native grouped expert adapters cannot honour.

    Expert projections are peeled to :class:`ExpertLoraSpec`, which carries only
    ``r``/``alpha``/``dropout``/``use_rslora``. Any other knob would apply to the attention half of
    the run and not the expert half, so it is rejected rather than half-applied.
    """
    if getattr(model_config, "use_dora", False):
        raise ValueError(
            f"use_dora=True cannot be combined with expert projections {sorted(projections)} in "
            "lora_target_modules: DoRA's magnitude decomposition is not implemented for the native "
            "grouped expert adapters, so it would apply to the attention adapters only. Drop use_dora, "
            "or drop the expert projections and LoRA the attention modules alone."
        )


def split_expert_lora_targets(model_config: ModelConfig) -> ExpertLoraSpec | None:
    """Route expert-weight entries in ``lora_target_modules`` to native EP grouped-LoRA.

    EP experts are grouped tensors stock PEFT cannot wrap, so expert names are peeled out (mutating
    ``lora_target_modules`` in place) and returned as an :class:`ExpertLoraSpec`. Call before
    ``load_distributed_model`` and assign to ``parallelism_config.expert_lora``. Returns ``None`` when PEFT
    is off, the model is dense, or no entry names an expert projection.
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
    # Main-rank-first: this runs before the weight download's own coordination, so uncoordinated it
    # is one hub request per rank for a file every rank is about to read from the same cache.
    config = hub_metadata_main_first(
        "expert_lora_probe",
        lambda: AutoConfig.from_pretrained(
            model_config.model_name_or_path,
            revision=getattr(model_config, "model_revision", None),
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
        ),
    )
    if not config_has_experts(config):
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

    _reject_unsupported_expert_lora_knobs(model_config, projections)
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


def _reenable_expert_lora_grads(model: PreTrainedModel) -> None:
    """Re-mark native grouped-LoRA expert adapters trainable after the global PEFT freeze clobbers them.

    Base experts stay frozen."""
    for module in model.modules():
        if isinstance(module, EPMoELayerBase):
            for attr in module._expert_lora_attrs:
                getattr(module, f"{attr}_lora_A").requires_grad_(True)
                getattr(module, f"{attr}_lora_B").requires_grad_(True)


def _pins_a_decoder_layer_index(pattern: str) -> bool:
    """Whether an fnmatch pattern selects decoder layers by their position in the layer list.

    Structural rather than a digit scan of the whole pattern: only the segment following the
    decoder-layer list (:data:`DECODER_LAYER_LIST_ATTRS`) counts, and it pins an index whenever it
    carries a digit. That covers character-class ranges (``model.layers.5[6-9].*``,
    ``model.layers.[6-8][0-9].*``), which a per-segment ``str.isdigit`` would read as index-free, and
    leaves ``*.mlp.experts.0.*`` allowed since an expert index is identical on every stage.
    ``model.layers.*`` selects every layer the stage holds and stays allowed.
    """
    segments = pattern.split(".")
    return any(
        segment in DECODER_LAYER_LIST_ATTRS and any(char.isdigit() for char in next_segment)
        for segment, next_segment in zip(segments, segments[1:], strict=False)
    )


def _reject_layer_indexed_patterns_under_pp(model, patterns, flag: str) -> None:
    """Reject layer-indexed freeze/unfreeze patterns on a pipeline-sliced model.

    Under PP the stage-aware loader returns a model holding only this stage's layers, re-based to
    index 0. A global-space pattern like ``model.layers.30.*`` therefore matches nothing on most
    stages and a different layer wherever local index 30 exists. Index-free patterns
    (``*.self_attn.sinks``) mean the same thing on every stage and stay allowed.
    """
    if getattr(model, PP_STAGE_PARTITION_ATTR, None) is None:
        return
    indexed = [pattern for pattern in patterns if _pins_a_decoder_layer_index(pattern)]
    if indexed:
        raise ValueError(
            f"{flag} carries layer-indexed patterns {indexed} under pipeline parallelism. Each stage "
            f"holds only its own layers, re-based to index 0, so a global layer index selects nothing "
            f"on most stages and the wrong layer on the rest. Use index-free patterns (e.g. "
            f"'*.self_attn.sinks'), or train without pipeline parallelism to target layers by index."
        )


def _inherits_its_layer_forward(cls: type) -> bool:
    """Whether ``cls`` runs the ``forward`` of a concrete ``torch.nn`` layer rather than its own."""
    forward = cls.forward
    return any(
        base.__module__.startswith(_TORCH_LAYER_PACKAGE) and vars(base).get("forward") is forward
        for base in cls.__mro__
    )


def _computes_its_layers_affine_map(linear: torch.nn.Linear, verdicts: dict[_ProbeKey, bool]) -> bool:
    """Whether the layer maps ``(1, in_features)`` to ``(1, out_features)``, as ``F.linear`` does.

    A subclass overriding ``nn.Linear.forward`` may change only *how* the affine map is computed (the
    low-precision compute drop-in, a quantized kernel) or may compute a different map entirely (a
    block-diagonal grouped projection, whose output is one group wide). Nothing in the class tells the
    two apart, and only the second breaks the adapter delta — so it is measured, once per class and
    geometry (a class can be grouped at one width and plain at another). A layer that cannot take an
    ``nn.Linear`` input at all is the answer; a device fault is not an answer about the layer.
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
    """Whether a forward probe on this module is both meaningful and free of side effects.

    A packed quantized weight is not a float matrix — PEFT routes those to its own adapter layer, and
    their kernels are not always loadable where the config is built. An fp8 weight is a float matrix
    with no CPU kernel behind it, so probing it would report a raise as a different affine map. A meta
    or ``DTensor`` weight has nothing to multiply, or would turn the probe into a collective.
    """
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

    PEFT decomposes the *matched* module's own weight and adds the delta to that module's output
    (``dispatch_default`` in ``peft.tuners.lora.layer``), which leaves two shapes out of its reach: a
    wrapper holding its weight in a child module (nothing to decompose — injection raises), and a
    layer subclass computing something other than its base layer's affine map (the delta has the wrong
    geometry — the first forward raises). Already-adapted layers are read through their base layer, as
    PEFT reads them.
    """
    base = module.get_base_layer() if isinstance(module, BaseTunerLayer) else module
    if not any(True for _ in base.parameters(recurse=False)):
        return "hold their weight in a child module" if any(True for _ in base.parameters()) else None
    if _inherits_its_layer_forward(type(base)) or not _is_probeable_linear(base):
        return None
    if _computes_its_layers_affine_map(base, verdicts):
        return None
    return "compute something other than their layer's affine map"


def _exclude_unadaptable_lora_targets(model: torch.nn.Module, peft_config) -> None:
    """Keep target-name matches PEFT cannot adapt out of the injection, via ``exclude_modules``.

    ``lora_target_modules`` is matched by name suffix over the whole module tree, so a vision or audio
    tower whose projections carry the language model's ``q_proj``…``o_proj`` spelling is matched too,
    and ``all-linear`` reaches every custom projection a family defines. Where the match is an ordinary
    leaf layer it is adapted as before; the two shapes stock LoRA cannot reach (Gemma 4's
    ``Gemma4ClippableLinear`` wrapper, DeepSeek-V4's grouped ``o_a_proj``) are excluded by full module
    path (PEFT also honours an entry as a dotted suffix), so an identically named leaf elsewhere in the
    tree is untouched.

    The matcher is PEFT's own, so ``all-linear`` (resolved to ``nn.Linear`` paths at injection time),
    regex targets, ``layers_to_transform`` and ``modules_to_save`` all keep their semantics.

    Collective: the tree and the targets are rank-uniform, so the verdict is too unless a rank's probe
    faulted; both the rejection and the exclusion list are agreed across ranks, since a rank adapting a
    different parameter set than its peers would only surface as a hang in the first collective.
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
            if has_ep_lora(model)
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


def _excluded_detail(model: torch.nn.Module, excluded: dict[str, list[str]]) -> str:
    """One counted clause per reason, with an example path and its class."""
    return "; ".join(
        f"{len(names)} {reason} (e.g. {names[0]}, a {type(model.get_submodule(names[0])).__name__})"
        for reason, names in excluded.items()
    )


def build_peft_config(model: torch.nn.Module, model_config: ModelConfig) -> object | None:
    """The run's ``LoraConfig``, with name matches PEFT cannot adapt excluded.

    The one seam where a PEFT config meets the live model: every training script reaches it through
    :func:`setup_peft_model`, and the SentenceTransformer path calls it directly.
    """
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


def setup_peft_model(
    args,
    model: PreTrainedModel,
    model_config: ModelConfig,
    expected_task_type: str = "CAUSAL_LM",
) -> object | None:
    """Set up adapter training (attention PEFT and/or native EP expert LoRA).

    Three cases: (1) no adapters → unfreeze/freeze patterns + full/partial finetune; (2) any adapter run →
    freeze base then re-enable native EP expert adapters (PEFT re-enables attention after); (3) expert-LoRA
    only → frozen base + trainable expert adapters, no PEFT config.
    """
    _reject_lora_target_parameters_under_ep(model, model_config)
    expert_lora_active = has_ep_lora(model)
    # None targets means architecture defaults, [] means expert-only; conflating the two would
    # full-finetune a `use_peft` run.
    targets = model_config.lora_target_modules
    attention_peft = model_config.use_peft and (targets is None or bool(targets))

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
            _reject_layer_indexed_patterns_under_pp(model, args.unfreeze_layers_patterns, "unfreeze_layers_patterns")
            unfreeze_modules_by_patterns(model, args.unfreeze_layers_patterns)
        if args.freeze_layers_patterns:
            _reject_layer_indexed_patterns_under_pp(model, args.freeze_layers_patterns, "freeze_layers_patterns")
            freeze_modules_by_patterns(model, args.freeze_layers_patterns)
        return None

    if stamped_sinks_policy(model) is SinksPolicy.TRAINABLE:
        raise ValueError(
            "train_sinks: true needs full fine-tuning: an adapter run freezes every base parameter, and "
            "the adapter artifact has no slot for the sinks, so they would train nowhere. Keep the sinks "
            "live and frozen instead (reset_sinks: false without train_sinks), or drop the adapters."
        )
    for p in model.parameters():
        p.requires_grad = False
    if expert_lora_active:
        _reenable_expert_lora_grads(model)

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

    for patterns_arg in ("unfreeze_layers_patterns", "freeze_layers_patterns"):
        if getattr(args, patterns_arg, None):
            warnings.warn(
                f"You can't use non-empty {patterns_arg} and peft together, only peft config will be used",
                stacklevel=2,
            )

    return build_peft_config(model, model_config)


def freeze_modules_by_patterns(model, patterns):
    """Freeze parameters whose names match at least one fnmatch pattern (wildcards
    ``*``/``?``), e.g. ``["*.self_attn.sinks", "model.layers.0.mlp.*"]``.

    Raises when a pattern matches nothing, as :func:`unfreeze_modules_by_patterns` does: freezing is
    additive, so a typo'd entry leaves the parameters it was meant to hold still training while every
    other pattern works. Under pipeline parallelism this holds each stage to patterns every stage
    carries.
    """
    unmatched = set(patterns)
    for param_name, param in model.named_parameters():
        for pattern in patterns:
            if fnmatch.fnmatch(param_name, pattern):
                param.requires_grad = False
                unmatched.discard(pattern)

    reason = None
    if unmatched:
        param_names = [name for name, _ in model.named_parameters()]
        reason = (
            f"freeze_layers_patterns {sorted(unmatched)} matched no parameter of "
            f"{type(model).__name__}, so whatever they name would keep training — a typo here has "
            f"no other symptom. Patterns are fnmatch against full PARAMETER names (not module "
            f"names, unlike unfreeze_layers_patterns); e.g. {param_names[:5]}. Under pipeline "
            f"parallelism this model is one stage, so name only parameters every stage carries."
        )
    # World-uniform: under PP a stage-local raise would leave peer stages hanging in the next
    # collective, so the verdict is agreed across ranks.
    reject_across_ranks(reason, "freeze_layers_patterns", ValueError)


def unfreeze_modules_by_patterns(model, patterns):
    """Freeze all parameters, then unfreeze modules whose full names match at least
    one fnmatch pattern (wildcards ``*``/``?``), e.g.
    ``["*.mlp.up_proj", "score", "model.layers.0.self_attn.*"]``.

    Raises when the patterns leave nothing trainable: everything is frozen first, so a typo'd pattern
    would train a fully frozen model (the optimizer accepts empty parameter groups).
    """
    for param in model.parameters():
        param.requires_grad = False

    for module_name, module in model.named_modules():
        for pattern in patterns:
            if fnmatch.fnmatch(module_name, pattern):
                for param in module.parameters():
                    param.requires_grad = True
                break

    reason = None
    if not any(param.requires_grad for param in model.parameters()):
        module_names = [name for name, _ in model.named_modules() if name]
        reason = (
            f"unfreeze_layers_patterns {list(patterns)} matched no parameter-bearing module of "
            f"{type(model).__name__}, so every parameter stayed frozen and the run would train "
            f"nothing. Patterns are fnmatch against full MODULE names (not parameter names); e.g. "
            f"{module_names[:5]}."
        )
    # Same world-uniform contract as freeze_modules_by_patterns.
    reject_across_ranks(reason, "unfreeze_layers_patterns", ValueError)


def prepare_peft_model(model, peft_config, args, *, merge_existing: bool = True):
    """Canonical PEFT wrap: k-bit prep → adapters → optional bf16 cast. Returns ``(model, casted_to_bf16)``.

    ``prepare_model_for_kbit_training`` must run before ``get_peft_model``: it freezes every param, so
    running it after would re-freeze the adapters. ``merge_existing=False`` returns a PeftModel unchanged.
    """
    if isinstance(model, PeftModel):
        if not merge_existing:
            return model, False
        model = model.merge_and_unload()

    quantized = (
        getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_quantized", False)
    )
    if quantized:
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
    """bf16 autocast around a bf16-cast PEFT forward, inert otherwise.

    ``prepare_peft_model``'s cast leaves adapter modules bf16 while activations stay fp32, so every
    adapted linear needs an autocast region. The device type comes from the caller's accelerator
    (a hardcoded "cuda" would be inert on other backends); bf16 is explicit because torch's CUDA
    autocast defaults to fp16.
    """
    if not casted_to_bf16:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


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
