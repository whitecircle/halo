"""Fold a saved LoRA adapter (attention and dense, DoRA, native expert) and its run's trained router balancing biases
into its base one safetensors tensor at a time and write the merged checkpoint through :class:`StreamingShardWriter`,
holding the adapter, one tensor, the writer's parts and one expert layer's fp32 deltas at a time."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from huggingface_hub.errors import StrictDataclassError
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.utils.other import get_pattern_key

from halo.checkpoint.adapters import (
    EXPERT_LORA_CONFIG_KEY,
    EXPERT_LORA_PEFT_TYPE,
    MIXED_EXPERT_LORA_PEFT_TYPE,
    find_adapter_weight_file,
    is_expert_lora_key,
    read_adapter_config,
    read_adapter_file,
)
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHT_NAMES,
    DEFAULT_MAX_SHARD_SIZE,
    MERGED_ADAPTER_CONFIG_DIR,
    ROUTER_BALANCING_BIASES_FILE,
    TRAINING_PROVENANCE_FILE,
)
from halo.checkpoint.hub_layout import CheckpointNamespace
from halo.checkpoint.reading import (
    checkpoint_shard_files,
    detect_model_type,
    iter_checkpoint_shard_entries,
    iter_checkpoint_tensors,
    read_checkpoint_config,
    reject_partial_checkpoint,
)
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS, copy_training_sidecars
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_adapter_base_revision,
    resolve_checkpoint_source,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.model_loading.checkpoint_keys import build_key_plan
from halo.model_loading.lazy_safetensors.meta_shell import MetaShellBuildError, instantiate_on_meta
from halo.model_loading.model_preparation import resolve_auto_model_class, resolve_build_config
from halo.model_loading.tokenizer_setup import load_processing_class
from halo.models.attention.gpt_oss_sinks import SinksPolicy, is_sink_key, neutralized_sink_value
from halo.models.introspection.unwrapping import PEFT_BASE_MODEL_PREFIX, plain_module_name
from halo.models.moe.balancing import BALANCING_BIASES_ATTR
from halo.models.moe.expert_layout import (
    SEPARATE_GLU_PROJECTIONS,
    SEPARATE_MLP_PROJECTIONS,
    ExpertLayout,
    ExpertLayoutKind,
    to_hub_expert_tensors,
)
from halo.models.moe.expert_lora_spec import ExpertLoraSpec, expert_lora_delta
from halo.models.moe.family_specs import (
    MoEFamilySpec,
    find_moe_family_spec,
    inkling2_pre_524_adapter_reason,
    moe_model_types,
)
from halo.models.peft_adapters import lora_folded

logger = logging.getLogger(__name__)

_ADAPTER_NAME = "default"
_LORA_WEIGHT_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<half>[AB])\.(?P<param>weight|bias)$")
_MAGNITUDE_KEY = re.compile(r"^(?P<module>.+)\.lora_magnitude_vector$")
_EXPERT_KEY = re.compile(r"^(?P<layer>.+)\.experts\.(?P<attr>[^.]+)\.lora_(?P<half>[AB])$")
_DELTA_BYTES_PER_ELEMENT = 4
# Stored dtypes (safetensors names) whose values are the weight itself, so a delta adds to them.
_FOLDABLE_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_LAYER_COPIES_IN_FLIGHT = 3

# A key's fold: the merged tensor from the base's.
Fold = Callable[[torch.Tensor], torch.Tensor]


class _StoredTensor(NamedTuple):
    shape: tuple[int, ...]
    dtype: str


@dataclass
class _LinearAdapter:
    """One module's saved tensors, at fp32 as PEFT's autocast loads them."""

    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    b_bias: torch.Tensor | None = None
    magnitude: torch.Tensor | None = None


@dataclass
class _ExpertLayer:
    """One EP layer's deltas: computed when the first of its hub tensors streams past and released after the last."""

    name: str
    spec: MoEFamilySpec
    scaling: float
    halves: dict[str, dict[str, torch.Tensor]]
    hub_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pending: set[str] = field(default_factory=set)
    deltas: dict[str, torch.Tensor] | None = None

    def __post_init__(self):
        self.hub_shapes = {key: tuple(delta.shape) for key, delta in self.hub_deltas(meta=True).items()}
        self.pending = set(self.hub_shapes)

    def hub_deltas(self, *, meta: bool = False) -> dict[str, torch.Tensor]:
        """The fp32 delta of every hub tensor the layer's adapters touch, in the family's expert layout."""
        banks = {
            attr: (
                torch.empty(ab["A"].shape[0], ab["A"].shape[1], ab["B"].shape[2], device="meta")
                if meta
                else expert_lora_delta(ab["A"], ab["B"], self.scaling)
            )
            for attr, ab in self.halves.items()
        }
        layout = self.spec.expert_layout
        hub = to_hub_expert_tensors(layout, _with_zero_partners(layout, banks), device="meta" if meta else "cpu")
        return {f"{self.name}.{self.spec.to_hub_key(key)}": tensor for key, tensor in hub.items()}

    def fold(self, base: torch.Tensor, key: str) -> torch.Tensor:
        if self.deltas is None:
            self.deltas = self.hub_deltas()
        delta = self.deltas.pop(key)
        self.pending.discard(key)
        if not self.pending:
            self.deltas = None
        # The gather's own arithmetic, so a merge-on-save and this agree bit for bit.
        return (base.float() + delta).to(base.dtype)


_FUSED_GATE_UP_KINDS = frozenset(
    {ExpertLayoutKind.FUSED, ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, ExpertLayoutKind.INTERLEAVED_GLU}
)


def _required_attrs(kind: ExpertLayoutKind, present: set[str]) -> set[str]:
    """The runtime attributes ``kind``'s layout reads when ``present`` ones carry an adapter."""
    if kind is ExpertLayoutKind.SEPARATE_GLU:
        return set(SEPARATE_GLU_PROJECTIONS)
    if kind is ExpertLayoutKind.SEPARATE_MLP:
        return set(SEPARATE_MLP_PROJECTIONS)
    if kind not in _FUSED_GATE_UP_KINDS:
        raise ValueError(f"No streaming-merge attribute set for expert layout kind {kind.name}.")
    halves = [attr for attr in ("gate_proj", "up_proj") if attr in present or f"{attr}_gmm" in present]
    if halves:
        suffix = "_gmm" if any(attr.endswith("_gmm") for attr in present) else ""
        return {"gate_proj" + suffix, "up_proj" + suffix, "down_proj"}
    return {"gate_up_proj", "down_proj"}


def _with_zero_partners(layout: ExpertLayout, banks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``banks`` plus a zero view for every attribute the layout reads that no adapter touches, shaped from the others
    (gate and up ``[E, H, M]``, fused ``[E, H, 2M]``, down ``[E, M, H]``), since the layout converts a whole layer."""
    experts, hidden, inter = None, None, None
    for attr, bank in banks.items():
        experts = bank.shape[0]
        if attr == "down_proj":
            inter, hidden = bank.shape[1], bank.shape[2]
        elif attr == "gate_up_proj":
            hidden, inter = bank.shape[1], bank.shape[2] // 2
        else:
            hidden, inter = bank.shape[1], bank.shape[2]
    shapes = {
        "down_proj": (experts, inter, hidden),
        "gate_up_proj": (experts, hidden, 2 * inter),
    }
    completed = dict(banks)
    first = next(iter(banks.values()))
    for attr in _required_attrs(layout.kind, set(banks)) - set(banks):
        shape = shapes.get(attr, (experts, hidden, inter))
        completed[attr] = (
            torch.empty(shape, device="meta")
            if first.device.type == "meta"
            else torch.zeros((), dtype=first.dtype).expand(shape)
        )
    return completed


def _linear_scalars(config: dict, module: str) -> tuple[float, bool, bool]:
    """``(alpha, use_rslora, use_dora)`` PEFT would build ``module`` with, ``alpha_pattern`` first."""
    alpha_pattern = config.get("alpha_pattern") or {}
    alpha = alpha_pattern.get(get_pattern_key(alpha_pattern.keys(), module), config["lora_alpha"])
    return alpha, bool(config.get("use_rslora")), bool(config.get("use_dora"))


def _lora_layer(config: dict, module: str, adapter: _LinearAdapter, weight: torch.Tensor, bias) -> LoraLinear:
    """PEFT's own tuner layer over a shell holding ``weight``, loaded with the saved tensors, so the fold is the one
    ``merge_and_unload`` runs."""
    out_features, in_features = weight.shape
    base = torch.nn.Linear(in_features, out_features, bias=bias is not None, device="meta")
    base.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        base.bias = torch.nn.Parameter(bias, requires_grad=False)
    alpha, use_rslora, use_dora = _linear_scalars(config, module)
    layer = LoraLinear(
        base,
        _ADAPTER_NAME,
        r=adapter.a.shape[0],
        lora_alpha=alpha,
        use_rslora=use_rslora,
        use_dora=use_dora,
        lora_bias=adapter.b_bias is not None,
    )
    layer.lora_A[_ADAPTER_NAME].weight.data = adapter.a
    layer.lora_B[_ADAPTER_NAME].weight.data = adapter.b
    if adapter.b_bias is not None:
        layer.lora_B[_ADAPTER_NAME].bias.data = adapter.b_bias
    if use_dora:
        layer.lora_magnitude_vector[_ADAPTER_NAME].weight.data = adapter.magnitude
    return layer


def _linear_folds(config: dict, module: str, adapter: _LinearAdapter) -> dict[str, Fold]:
    """``{base key: fold}`` for one module's weight, and its bias where the adapter trained one."""

    def fold_weight(weight: torch.Tensor) -> torch.Tensor:
        layer = _lora_layer(config, module, adapter, weight, None)
        return lora_folded(layer.get_base_layer().weight, [layer])

    folds = {f"{module}.weight": fold_weight}
    if adapter.b_bias is not None:

        def fold_bias(bias: torch.Tensor) -> torch.Tensor:
            # Never read: only the bias folds, and an unwritten allocation costs no memory.
            shell = torch.empty(bias.shape[0], adapter.a.shape[1])
            layer = _lora_layer(config, module, adapter, shell, bias)
            return lora_folded(layer.get_base_layer().bias, [layer])

        folds[f"{module}.bias"] = fold_bias
    return folds


def _replacement(saved: torch.Tensor, dtype: torch.dtype | None = None) -> Fold:
    """``saved`` in place of the base's tensor, at ``dtype``, else at the base's."""
    return lambda base: saved.to(dtype or base.dtype)


@dataclass
class _MergePlan:
    """Every tensor the merge rewrites, resolved against the base's keys before anything is written."""

    folds: dict[str, Fold]
    expert_layer_of: dict[str, _ExpertLayer]
    delta_bytes: int
    # The run trained with neutralized GptOss sinks, and the base stores them live.
    neutralizes_sinks: bool = False

    def fold(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.neutralizes_sinks and is_sink_key(key):
            return torch.full_like(tensor, neutralized_sink_value(tensor.dtype))
        layer = self.expert_layer_of.get(key)
        if layer is not None:
            return layer.fold(tensor, key)
        fold = self.folds.get(key)
        return tensor if fold is None else fold(tensor)


def _unmergeable_remedy(holds_expert_lora: bool) -> str:
    """Where an adapter the streaming merge refuses folds instead; the in-memory merge drops native expert deltas."""
    if holds_expert_lora:
        return (
            "an adapter holding native expert LoRA has no in-memory merge, which would drop its expert deltas: train "
            "with merge_expert_lora_on_save: true, whose gathered save folds both halves"
        )
    return "a dense adapter merges in memory with merge_peft_adapters.py without --streaming"


def _classify_adapter_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, _LinearAdapter], dict[str, dict[str, dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    """``(per-module linear adapters, expert halves by layer then attribute, whole-tensor replacements)``; a LoRA tensor
    of any other kind (embedding LoRA, an unmatched half) raises."""
    linear: dict[str, _LinearAdapter] = {}
    experts: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    replacements: dict[str, torch.Tensor] = {}
    unsupported = []
    for key, tensor in tensors.items():
        # A mixed run's saver walks from the PeftModel, so its expert keys carry the prefix, and under CP the wrapper's
        # level too.
        if is_expert_lora_key(key) and (match := _EXPERT_KEY.match(plain_module_name(key))):
            experts.setdefault(match["layer"], {}).setdefault(match["attr"], {})[match["half"]] = tensor
            continue
        name = key.removeprefix(PEFT_BASE_MODEL_PREFIX)
        if match := _LORA_WEIGHT_KEY.match(name):
            adapter = linear.setdefault(match["module"], _LinearAdapter())
            if match["param"] == "bias" and match["half"] == "B":
                adapter.b_bias = tensor.float()
            elif match["param"] == "weight":
                setattr(adapter, match["half"].lower(), tensor.float())
            else:
                unsupported.append(key)
        elif match := _MAGNITUDE_KEY.match(name):
            linear.setdefault(match["module"], _LinearAdapter()).magnitude = tensor.float()
        elif "lora_" in name:
            unsupported.append(key)
        else:
            replacements[name] = tensor
    incomplete = [module for module, adapter in linear.items() if adapter.a is None or adapter.b is None]
    half_experts = [
        f"{layer}.experts.{attr}"
        for layer, attrs in experts.items()
        for attr, ab in attrs.items()
        if set(ab) != {"A", "B"}
    ]
    if unsupported or incomplete or half_experts:
        raise ValueError(
            f"The adapter holds LoRA tensors the streaming merge does not fold: {sorted(unsupported)[:4]} "
            f"(only linear LoRA, its DoRA magnitude and bias, and native expert LoRA are supported), modules missing "
            f"an A or B half {incomplete[:4]}, expert weights missing one {half_experts[:4]}; "
            f"{_unmergeable_remedy(any(map(is_expert_lora_key, tensors)))}."
        )
    return linear, experts, replacements


def resolve_expert_lora_spec(config: dict) -> ExpertLoraSpec:
    """The native expert adapters' recipe a saved ``adapter_config.json`` records; raises for one that records none."""
    peft_type = config.get("peft_type")
    if peft_type == MIXED_EXPERT_LORA_PEFT_TYPE:
        config = config[EXPERT_LORA_CONFIG_KEY]
    elif peft_type != EXPERT_LORA_PEFT_TYPE:
        raise ValueError(f"adapter_config.json peft_type {peft_type!r} carries no native expert LoRA recipe.")
    return ExpertLoraSpec(r=config["r"], alpha=config["lora_alpha"], use_rslora=bool(config.get("use_rslora")))


def _base_tensors(base_dir: str) -> dict[str, _StoredTensor]:
    stored = {}
    for _shard, reader, key in iter_checkpoint_shard_entries(base_dir):
        view = reader.get_slice(key)
        stored[key] = _StoredTensor(tuple(view.get_shape()), view.get_dtype())
    return stored


def _expert_layers(
    experts: dict[str, dict[str, dict[str, torch.Tensor]]], config: dict, spec: MoEFamilySpec | None, base_dir: str
) -> dict[str, _ExpertLayer]:
    """The expert layers by the hub keys they rewrite, under the base's family layout."""
    if not experts:
        return {}
    if spec is None:
        raise ValueError(
            f"The base at {base_dir} is model_type {detect_model_type(base_dir)!r}, which no MoE family spec claims, so "
            f"there is no expert layout to fold its expert adapters through. Supported: "
            f"{', '.join(sorted(moe_model_types()))}."
        )
    scaling = resolve_expert_lora_spec(config).scaling
    return {name: _ExpertLayer(name, spec, scaling, attrs) for name, attrs in experts.items()}


def _release_slot_dtypes(layers: dict[str, _ExpertLayer], base: dict[str, _StoredTensor]) -> dict[str, str]:
    """The stored dtype of every tensor a family's packed release holds in place of an expert tensor the adapter
    rewrites. The release spells the layer's prefix its own way, so the slots are found by their stem."""
    stems = {
        slot.hub_stem
        for layer in layers.values()
        for slot in layer.spec.frozen_expert_slots
        if any(key.rpartition(".")[2] == slot.weight for key in layer.hub_shapes)
    }
    return {key: stored.dtype for key, stored in base.items() if key.rpartition(".")[2] in stems}


def _stored_keys_by_live_name(
    base_dir: str, base_keys: Collection[str], required: Collection[str], trust_remote_code: bool, remedy: str
) -> dict[str, str]:
    """Each tensor name of the base's module tree → the key ``base_dir`` stores it under, planned as ``from_pretrained``
    renames the checkpoint onto the config's widest class, built without weights. A name the load fuses or splits out
    of other keys, or that no stored key supplies, has no entry. Raises for a remote-code config (an ``auto_map``)
    under ``trust_remote_code=False``. A class the meta device cannot build maps every stored key to itself when the
    base stores each ``required`` name verbatim, and raises naming ``remedy`` otherwise."""
    try:
        config = resolve_build_config(base_dir, revision=None, trust_remote_code=trust_remote_code)
    except (ValueError, StrictDataclassError) as e:
        # Untrusted, a registered model_type reads an auto_map config through transformers' own class, whose strict
        # dataclass may refuse a field the remote class types otherwise.
        if trust_remote_code or "auto_map" not in (read_checkpoint_config(base_dir, missing_ok=True) or {}):
            raise
        raise ValueError(
            f"{e} The streaming merge reads the base's config to build its class without weights and find the key "
            f"each adapter target is stored under: pass --trust_remote_code for a remote-code base you trust."
        ) from e
    try:
        shell = instantiate_on_meta(
            base_dir, resolve_auto_model_class(config), config, trust_remote_code=trust_remote_code, config_only=True
        )
    except MetaShellBuildError as e:
        if not set(required) <= set(base_keys):
            raise ValueError(
                f"{e} The streaming merge reads the base's module names off this shell to find the key each adapter "
                f"target is stored under; {remedy}."
            ) from e
        logger.warning(f"{e} The base stores every tensor the adapter rewrites under its module path; folding there.")
        return {key: key for key in base_keys}
    namespace = CheckpointNamespace.of(shell)
    plan = build_key_plan(base_keys, shell.state_dict(), namespace.conversions, namespace.base_model_prefix)
    return plan.live_renames()


def _read_balancing_sidecar(adapter_dir: str) -> dict[str, torch.Tensor]:
    path = os.path.join(adapter_dir, ROUTER_BALANCING_BIASES_FILE)
    if not os.path.isfile(path):
        return {}
    return torch.load(path, map_location="cpu", weights_only=True)


def _balancing_folds(
    sidecar: dict[str, torch.Tensor],
    spec: MoEFamilySpec | None,
    stored: dict[str, str],
    base_shapes: dict[str, tuple[int, ...]],
    remedy: str,
) -> dict[str, Fold]:
    """``{stored key: fold}`` writing each router's trained balancing bias into its native slot at fp32, the in-memory
    merge's ``apply_training_sidecars``. A router with no slot in the base trained a transient bias no checkpoint
    carries, skipped with a warning as that merge does. Raises for a module the base lacks, a reshaped bias, and a
    config-gated slot the base lacks, which only a merge that rewrites the config adds."""
    slot = spec.balancing_slot if spec is not None else None
    attrs = (BALANCING_BIASES_ATTR, *(slot.attrs if slot is not None else ()))
    folds: dict[str, Fold] = {}
    transient = []
    for name, bias in sidecar.items():
        module = plain_module_name(name)
        key = next((stored[f"{module}.{attr}"] for attr in attrs if f"{module}.{attr}" in stored), None)
        if key is None:
            if not any(live.startswith(f"{module}.") for live in stored):
                raise ValueError(
                    f"{ROUTER_BALANCING_BIASES_FILE} names module {name!r}, which the base holds no tensor under: the "
                    f"sidecar belongs to a different model tree."
                )
            if slot is not None and slot.config_flag:
                raise ValueError(
                    f"{ROUTER_BALANCING_BIASES_FILE} trained router {module!r}'s balancing bias in the {slot.attrs} "
                    f"slot, which the base builds only under config {slot.config_flag}: true and the streaming merge "
                    f"cannot add; {remedy}."
                )
            transient.append(module)
            continue
        if base_shapes[key] != tuple(bias.shape):
            raise ValueError(
                f"{ROUTER_BALANCING_BIASES_FILE} entry {name!r} has shape {tuple(bias.shape)} but the base stores its "
                f"balancing slot {key} at {base_shapes[key]}: the sidecar belongs to another expert count."
            )
        folds[key] = _replacement(bias, torch.float32)
    if transient:
        logger.warning(
            f"{len(transient)} router(s) trained a transient balancing bias the base has no checkpoint slot for, so the "
            f"merged model routes without it (near-tied top-k picks flip against training): {transient[:4]}"
        )
    return folds


def _has_neutralized_sinks(adapter_dir: str) -> bool:
    """Whether the adapter's training provenance records neutralized GptOss sinks."""
    path = os.path.join(adapter_dir, TRAINING_PROVENANCE_FILE)
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return json.load(fh).get(PROVENANCE_GPT_OSS_SINKS) == SinksPolicy.NEUTRALIZED


def _plan_merge(adapter_dir: str, config: dict, base_dir: str, trust_remote_code: bool) -> _MergePlan:
    path = find_adapter_weight_file(adapter_dir)
    if path is None:
        raise FileNotFoundError(
            f"{adapter_dir} holds none of the adapter weight files {', '.join(ADAPTER_WEIGHT_NAMES)}: point at a "
            f"saved adapter directory."
        )
    linear, experts, replacements = _classify_adapter_tensors(read_adapter_file(path))
    by_live_name: dict[str, Fold] = {}
    for module, adapter in linear.items():
        by_live_name.update(_linear_folds(config, module, adapter))
    by_live_name.update({name: _replacement(saved) for name, saved in replacements.items()})
    model_type = detect_model_type(base_dir)
    spec = find_moe_family_spec(model_type)
    layers = _expert_layers(experts, config, spec, base_dir)
    by_key = {key: layer for layer in layers.values() for key in layer.hub_shapes}
    sidecar = _read_balancing_sidecar(adapter_dir)

    base = _base_tensors(base_dir)
    base_shapes = {key: stored.shape for key, stored in base.items()}
    remedy = _unmergeable_remedy(bool(experts))
    stored = (
        _stored_keys_by_live_name(base_dir, base_shapes, by_live_name, trust_remote_code, remedy)
        if by_live_name or sidecar
        else {}
    )
    neutralizes_sinks = _has_neutralized_sinks(adapter_dir)
    sinks = {key for key in base if is_sink_key(key)} if neutralizes_sinks else set()
    rewritten = {stored[name] for name in by_live_name if name in stored} | (set(by_key) & set(base)) | sinks
    stored_dtypes = {key: base[key].dtype for key in rewritten}
    _reject_packed_tensors(base_dir, adapter_dir, stored_dtypes | _release_slot_dtypes(layers, base))
    absent = sorted((set(by_live_name) - set(stored)) | (set(by_key) - set(base_shapes)))
    pre_524 = absent and inkling2_pre_524_adapter_reason(model_type, (name.rpartition(".")[0] for name in absent))
    if pre_524:
        raise ValueError(f"The adapter at {adapter_dir} cannot be merged on this release: {pre_524}.")
    if absent:
        raise ValueError(
            f"The adapter at {adapter_dir} rewrites {len(absent)} tensor(s) the base at {base_dir} does not store "
            f"under a key of their own (e.g. {absent[:3]}): its load fuses or splits them out of other keys, its "
            f"class builds no such module (a text-only class's keys), they are tied to another tensor (a tied head), "
            f"its hub layout respells an expert layer's path, or the adapter belongs to another base. The streaming "
            f"merge folds a module's tensor under the key the base's load reads it from and an expert tensor under "
            f"its layer's module path; {remedy}."
        )
    folds = {stored[name]: fold for name, fold in by_live_name.items()}
    folds.update(_balancing_folds(sidecar, spec, stored, base_shapes, remedy))
    resized = [
        f"{stored[name]}: base {base_shapes[stored[name]]}, adapter {tuple(saved.shape)}"
        for name, saved in replacements.items()
        if base_shapes[stored[name]] != tuple(saved.shape)
    ]
    if resized:
        raise ValueError(
            f"The run resized {len(resized)} tensor(s) the adapter stores whole ({resized[:3]}), as a grown "
            f"vocabulary does, which only the in-memory merge reproduces."
        )
    mismatched = [
        f"{key}: base {base_shapes[key]}, adapter {shape}"
        for layer in layers.values()
        for key, shape in layer.hub_shapes.items()
        if base_shapes[key] != shape
    ]
    if mismatched:
        raise ValueError(
            f"The base's expert tensors are not in the family's export layout, so the adapter's deltas cannot be "
            f"folded by key: {mismatched[:3]}."
        )
    if neutralizes_sinks and not sinks:
        raise ValueError(
            f"{os.path.join(adapter_dir, TRAINING_PROVENANCE_FILE)} records neutralized GptOss attention sinks, and "
            f"the base at {base_dir} stores no sink tensor to neutralize: the adapter belongs to another base."
        )
    delta_bytes = max(
        (
            _DELTA_BYTES_PER_ELEMENT * sum(math.prod(shape) for shape in layer.hub_shapes.values())
            for layer in layers.values()
        ),
        default=0,
    )
    return _MergePlan(folds, by_key, delta_bytes, neutralizes_sinks)


def _merged_tensors(base_dir: str, plan: _MergePlan) -> Iterator[tuple[str, torch.Tensor]]:
    for key, tensor in iter_checkpoint_tensors(base_dir):
        yield key, plan.fold(key, tensor)


def _reject_packed_tensors(base_dir: str, adapter_dir: str, rewritten: dict[str, str]) -> None:
    """``rewritten`` maps each base tensor the merge would rewrite to its stored dtype. A packed tensor (NVFP4
    nibbles, MXFP8 or fp8 bytes with their scales apart) is no weight, and a fold adds its delta to the stored
    values at the stored dtype. A quantized base whose rewritten tensors are all stored as floats merges."""
    packed = sorted(f"{key} ({dtype})" for key, dtype in rewritten.items() if dtype not in _FOLDABLE_DTYPES)
    if not packed:
        return
    raise ValueError(
        f"The adapter at {adapter_dir} rewrites {len(packed)} tensor(s) that the base at {base_dir} stores packed "
        f"(e.g. {packed[:3]}), and a delta folded into the stored values gives no weight. The adapter has no merged "
        f"export. An adapter trained over a packed release (frozen_quantized_experts) resumes training from its "
        f"adapter directory; any other adapter merges over the base's bf16 conversion."
    )


def merge_adapter_streaming(
    adapter_dir: str,
    output_dir: str,
    *,
    base_revision: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> None:
    """Fold ``adapter_dir`` into its base and write ``output_dir``: tensors keep the base's stored dtype and key
    spelling, the base's config and tokenizer files are copied (the adapter's tokenizer replaces them), and the
    adapter config lands in ``original_adapter_config/``. A Hub base is read at the commit the adapter records, or at
    ``base_revision``. A rewrite of a tensor the base stores packed is refused, and every key the adapter rewrites is
    resolved against the base before the first write, so a refusal leaves no output. A run that trained with
    neutralized GptOss sinks gets them back: the base stores them live, and the merge writes each at its dtype
    minimum."""
    reject_partial_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    config = read_adapter_config(adapter_dir)
    base_path = config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(f"{os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)} names no base_model_name_or_path.")
    base_dir = resolve_checkpoint_source(
        base_path, resolve_adapter_base_revision(base_path, config.get("revision"), base_revision)
    )
    reject_partial_checkpoint(base_dir)
    reject_in_place_conversion(base_dir, output_dir)
    plan = _plan_merge(adapter_dir, config, base_dir, trust_remote_code)

    shards = checkpoint_shard_files(base_dir)
    with StreamingShardWriter(output_dir, max_shard_size, enabled=True) as writer:
        preflight_resource_warning(
            "merge_peft_adapters",
            output_dir,
            disk_bytes=sum(os.path.getsize(shard) for shard in shards),
            ram_bytes=writer.max_resident_bytes + _LAYER_COPIES_IN_FLIGHT * plan.delta_bytes,
        )
        os.makedirs(output_dir, exist_ok=True)
        log = logger.info if verbose else lambda message: None
        log(
            f"Streaming {adapter_dir} into {base_dir}: {len(plan.folds)} tensors and {len(set(map(id, plan.expert_layer_of.values())))} expert layers"
        )
        for key, tensor in _merged_tensors(base_dir, plan):
            writer.add(key, tensor)
        writer.close_as_hf_checkpoint()
    copy_checkpoint_aux_files(base_dir, output_dir, include_resume_sidecars=False)
    processing_class = load_processing_class(adapter_dir, trust_remote_code=trust_remote_code)
    if processing_class is not None:
        processing_class.save_pretrained(output_dir)
    copy_training_sidecars(adapter_dir, output_dir)
    original_config_dir = os.path.join(output_dir, MERGED_ADAPTER_CONFIG_DIR)
    os.makedirs(original_config_dir, exist_ok=True)
    shutil.copy2(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE), original_config_dir)
    log(f"Merged model saved to: {output_dir}")
