"""Fold a saved LoRA adapter (attention and dense, DoRA, native expert) into its base one safetensors tensor at a time and
write the merged checkpoint through :class:`StreamingShardWriter`, holding the adapter, one tensor and the expert deltas
of the layers a shard touches."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.utils.other import get_pattern_key
from safetensors import safe_open

from halo.checkpoint.adapters import (
    EXPERT_LORA_CONFIG_KEY,
    EXPERT_LORA_PEFT_TYPE,
    MIXED_EXPERT_LORA_PEFT_TYPE,
    adapter_weight_paths,
    is_expert_lora_key,
    read_adapter_config,
    read_adapter_file,
)
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    DEFAULT_MAX_SHARD_SIZE,
    MERGED_ADAPTER_CONFIG_DIR,
    TRAINING_PROVENANCE_FILE,
)
from halo.checkpoint.reading import checkpoint_shard_files, detect_model_type, reject_partial_checkpoint
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS, copy_training_sidecars
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.model_loading.tokenizer_setup import load_processing_class
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.expert_layout import (
    SEPARATE_GLU_PROJECTIONS,
    SEPARATE_MLP_PROJECTIONS,
    ExpertLayout,
    ExpertLayoutKind,
    to_hub_expert_tensors,
)
from halo.models.moe.expert_lora_spec import ExpertLoraSpec, expert_lora_delta
from halo.models.moe.family_specs import MoEFamilySpec, find_moe_family_spec, moe_model_types
from halo.models.peft_adapters import lora_folded
from halo.models.unwrapping import PEFT_BASE_MODEL_PREFIX

logger = logging.getLogger(__name__)

_ADAPTER_NAME = "default"
_LORA_WEIGHT_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<half>[AB])\.(?P<param>weight|bias)$")
_MAGNITUDE_KEY = re.compile(r"^(?P<module>.+)\.lora_magnitude_vector$")
_EXPERT_KEY = re.compile(r"^(?P<layer>.+)\.experts\.(?P<attr>[^.]+)\.lora_(?P<half>[AB])$")
_DELTA_BYTES_PER_ELEMENT = 4
_LAYER_COPIES_IN_FLIGHT = 3

# A key's fold: the merged tensor from the base's.
Fold = Callable[[torch.Tensor], torch.Tensor]


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


def _required_attrs(kind: ExpertLayoutKind, present: set[str]) -> set[str]:
    """The runtime attributes ``kind``'s layout reads when ``present`` ones carry an adapter."""
    if kind is ExpertLayoutKind.SEPARATE_GLU:
        return set(SEPARATE_GLU_PROJECTIONS)
    if kind is ExpertLayoutKind.SEPARATE_MLP:
        return set(SEPARATE_MLP_PROJECTIONS)
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


def _replacement(saved: torch.Tensor) -> Fold:
    return lambda base: saved.to(base.dtype)


@dataclass
class _MergePlan:
    """Every tensor the merge rewrites, resolved against the base's keys before anything is written."""

    folds: dict[str, Fold]
    expert_layer_of: dict[str, _ExpertLayer]
    delta_bytes: int

    def fold(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        layer = self.expert_layer_of.get(key)
        if layer is not None:
            return layer.fold(tensor, key)
        fold = self.folds.get(key)
        return tensor if fold is None else fold(tensor)


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
        if is_expert_lora_key(key) and (match := _EXPERT_KEY.match(key)):
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
            f"an A or B half {incomplete[:4]}, expert weights missing one {half_experts[:4]}. Merge it in memory with "
            f"merge_peft_adapters.py without --streaming."
        )
    return linear, experts, replacements


def _expert_spec(config: dict) -> ExpertLoraSpec:
    peft_type = config.get("peft_type")
    if peft_type == MIXED_EXPERT_LORA_PEFT_TYPE:
        config = config[EXPERT_LORA_CONFIG_KEY]
    elif peft_type != EXPERT_LORA_PEFT_TYPE:
        raise ValueError(f"adapter_config.json peft_type {peft_type!r} carries no native expert LoRA recipe.")
    return ExpertLoraSpec(r=config["r"], alpha=config["lora_alpha"], use_rslora=bool(config.get("use_rslora")))


def _base_tensor_shapes(base_dir: str) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in checkpoint_shard_files(base_dir):
        with safe_open(shard, framework="pt") as reader:
            for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
                shapes[key] = tuple(reader.get_slice(key).get_shape())
    return shapes


def _expert_layers(
    experts: dict[str, dict[str, dict[str, torch.Tensor]]], config: dict, base_dir: str
) -> dict[str, _ExpertLayer]:
    """The expert layers by the hub keys they rewrite, under the base's family layout."""
    if not experts:
        return {}
    model_type = detect_model_type(base_dir)
    spec = find_moe_family_spec(model_type)
    if spec is None:
        raise ValueError(
            f"The base at {base_dir} is model_type {model_type!r}, which no MoE family spec claims, so there is no "
            f"expert layout to fold its expert adapters through. Supported: {', '.join(sorted(moe_model_types()))}."
        )
    scaling = _expert_spec(config).scaling
    return {name: _ExpertLayer(name, spec, scaling, attrs) for name, attrs in experts.items()}


def _plan_merge(adapter_dir: str, config: dict, base_dir: str) -> _MergePlan:
    path = next(path for path in adapter_weight_paths(adapter_dir) if os.path.isfile(path))
    linear, experts, replacements = _classify_adapter_tensors(read_adapter_file(path))
    folds: dict[str, Fold] = {}
    for module, adapter in linear.items():
        folds.update(_linear_folds(config, module, adapter))
    folds.update({key: _replacement(saved) for key, saved in replacements.items()})
    layers = _expert_layers(experts, config, base_dir)
    by_key = {key: layer for layer in layers.values() for key in layer.hub_shapes}

    base_shapes = _base_tensor_shapes(base_dir)
    absent = sorted((set(folds) | set(by_key)) - set(base_shapes))
    if absent:
        raise ValueError(
            f"The adapter at {adapter_dir} rewrites {len(absent)} tensor(s) the base at {base_dir} does not store "
            f"under that name (e.g. {absent[:3]}): the base spells them another way (a text-only class's keys, a "
            f"fused or renamed layout, a tied head) or the adapter belongs to another base. The streaming merge folds "
            f"by stored key; a dense adapter merges in memory with merge_peft_adapters.py without --streaming."
        )
    resized = [
        f"{key}: base {base_shapes[key]}, adapter {tuple(saved.shape)}"
        for key, saved in replacements.items()
        if base_shapes[key] != tuple(saved.shape)
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
    delta_bytes = max(
        (
            _DELTA_BYTES_PER_ELEMENT * sum(math.prod(shape) for shape in layer.hub_shapes.values())
            for layer in layers.values()
        ),
        default=0,
    )
    return _MergePlan(folds, by_key, delta_bytes)


def _merged_tensors(base_dir: str, plan: _MergePlan) -> Iterator[tuple[str, torch.Tensor]]:
    for shard in checkpoint_shard_files(base_dir):
        with safe_open(shard, framework="pt") as reader:
            for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
                yield key, plan.fold(key, reader.get_tensor(key))


def _reject_unreplayable_sidecars(adapter_dir: str) -> None:
    """The one training sidecar that rewrites model state the stream has no module tree to apply."""
    path = os.path.join(adapter_dir, TRAINING_PROVENANCE_FILE)
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        provenance = json.load(fh)
    if provenance.get(PROVENANCE_GPT_OSS_SINKS) == SinksPolicy.NEUTRALIZED:
        raise ValueError(
            f"{path} records neutralized GptOss attention sinks, which the streaming merge cannot re-apply (it holds "
            f"no model). Merge in memory, or run reset_sinks.py on the merged checkpoint."
        )


def merge_adapter_streaming(
    adapter_dir: str,
    output_dir: str,
    *,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> None:
    """Fold ``adapter_dir`` into its base and write ``output_dir``: tensors keep the base's stored dtype and key
    spelling, the base's config and tokenizer files are copied (the adapter's tokenizer replaces them), and the
    adapter config lands in ``original_adapter_config/``. Every key the adapter rewrites is resolved against the base
    before the first write, so a refusal leaves no output."""
    reject_partial_checkpoint(adapter_dir)
    reject_in_place_conversion(adapter_dir, output_dir)
    _reject_unreplayable_sidecars(adapter_dir)
    config = read_adapter_config(adapter_dir)
    base_path = config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(f"{os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)} names no base_model_name_or_path.")
    base_dir = resolve_checkpoint_source(base_path)
    reject_partial_checkpoint(base_dir)
    reject_in_place_conversion(base_dir, output_dir)
    plan = _plan_merge(adapter_dir, config, base_dir)

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
