"""Lazy safetensors-based model loader for Expert Parallelism.

Reads the safetensors index and materializes only each rank's expert slice directly from disk, all
ranks in parallel. :func:`load_ep_model_lazy` orchestrates: meta-device shell → format detect →
:class:`EPWeightPlanner` → :class:`ExpertFuser` → ``SafetensorsWeightLoader``.

Handles fused-3D checkpoints (LFM2/GLM4 ``[E,2M,H]``, GptOss ``[E,H,2M]``) sliced directly, and
individual per-expert checkpoints (Qwen3/Qwen3.5/Bailing) loaded local-only and fused into 3D tensors.

This module owns the EXPERT-domain knowledge: the checkpoint key patterns derived from the layer
classes, the planner that classifies keys into shard/replicate/ignore, the per-expert fuser, and the
family resolution (hub renames and conversion steps off the EP layer registry). The generic
machinery — index resolution, key alignment, ``WeightPlan`` materialization, absent-module init, the
meta shell — lives in ``src/models/loading/lazy_safetensors/`` and is shared with the pipeline
stage loader, which also runs :func:`lazy_load_prologue` (index → meta shell → whole-checkpoint plan)
and then narrows the plan to its own stage.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from src.distributed.expert_parallel.config import (
    EPConfig,
    get_num_experts,
)
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    experts_container_attrs,
    hf_fused_expert_keys,
    hub_to_module_key_renames,
    per_expert_fusion_map,
)
from src.distributed.expert_parallel.hub_conversion import resolve_conversion_steps
from src.distributed.expert_parallel.patching import (
    create_ep_buffers,
    patch_moe_model_for_ep,
)
from src.distributed.filesystem import fs_aware_main_first
from src.distributed.runtime import DeferredRankFailure, get_local_rank
from src.log import KEY_PREVIEW_COUNT
from src.models.loading.config_levels import text_config
from src.models.loading.lazy_safetensors.conversion import Concat
from src.models.loading.lazy_safetensors.meta_shell import instantiate_on_meta
from src.models.loading.lazy_safetensors.weights import (
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
    assign_tensor_to_model,
    build_key_mapping,
    init_checkpoint_absent_modules,
    resolve_run_dtype,
    resolve_safetensors_index,
    verify_loaded_shape,
)
from src.models.patches.attention import revalidate_attn_kwarg
from src.models.patches.buffer_fixes import finalize_loaded_model

logger = logging.getLogger(__name__)


# The checkpoint-key vocabulary is DERIVED from the layer classes (container attrs and fused tensor
# names they declare), never restated here: a hand-written alternation that omitted a family's
# container silently mis-planned every one of its expert keys.
_CONTAINER_ALT = "|".join(re.escape(a) for a in experts_container_attrs())

# Fused 3D expert keys (dim-0 = expert index): GptOss ``[E,H,2M]``, Qwen3/GLM4/LFM2 ``[E,2M,H]``.
_FUSED_EXPERT_PATTERN = re.compile(
    rf"\.(?:{_CONTAINER_ALT})\.({'|'.join(re.escape(k) for k in hf_fused_expert_keys())})$"
)

# Individual expert keys (Qwen3/Qwen3.5/Bailing checkpoints): model.layers.X.mlp.experts.N.gate_proj.weight
# Groups: (1) the expert-container attribute, (2) the expert index.
_INDIVIDUAL_EXPERT_PATTERN = re.compile(rf"\.({_CONTAINER_ALT})\.(\d+)\.")

# Expert index nested under an extra module (``...experts.local_experts.N.linear_fc{1,2}``):
# unmatched above, so detect() would misclassify it as FUSED and silently drop the expert plan.
# The structural backstop for any such layout.
_UNSUPPORTED_LAZY_EXPERT_PATTERN = re.compile(rf"\.(?:{_CONTAINER_ALT})\.[A-Za-z_]\w*\.\d+\.")


class CheckpointFormat(Enum):
    """How expert weights are stored on disk."""

    FUSED = auto()  # 3D tensors with expert dim — slice directly
    INDIVIDUAL = auto()  # per-expert keys — load local + fuse into 3D

    @staticmethod
    def detect(weight_map: dict[str, str]) -> CheckpointFormat:
        """Detect whether checkpoint uses fused or individual expert keys."""
        for key in weight_map:
            if _FUSED_EXPERT_PATTERN.search(key):
                return CheckpointFormat.FUSED
            if _INDIVIDUAL_EXPERT_PATTERN.search(key):
                return CheckpointFormat.INDIVIDUAL
        return CheckpointFormat.FUSED  # non-MoE or unknown → treat as fused (no-op)


def _read_checkpoint_config(model_path: str) -> dict:
    """A checkpoint dir's ``config.json``, or ``{}`` when it is absent or unreadable.

    Both probes below run on the same directory, so the parse lives here once rather than being
    repeated per field — this runs on every rank, before any weight is read.
    """
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def lazy_loader_supports_checkpoint(model_path: str) -> bool:
    """Whether the lazy loader can materialize this checkpoint's expert layout.

    The family's own :attr:`~EPMoELayerBase._supports_lazy_loading` declaration decides (resolved from
    ``config.model_type``); callers route an unsupported family to the HF from_pretrained + EP-patch
    path. The structural probe stays as the backstop for a checkpoint whose ``model_type`` no
    registered family claims: an expert index nested under an extra module (Zaya's
    ``...experts.local_experts.N.linear_fc{1,2}``) matches neither key pattern, so ``detect()`` would
    misclassify it as FUSED and silently drop every expert plan. Best-effort: ``True`` when neither
    the config nor the index can be read.
    """
    config = _read_checkpoint_config(model_path)
    layer_cls = ep_layer_class_by_model_type().get(config.get("model_type", "") or "")
    if layer_cls is not None and not layer_cls._supports_lazy_loading:
        return False
    if config.get("quantization_config"):
        # A checkpoint-native quantization (fp8 hub checkpoints and friends), distinct from the RUN's
        # TRL quantization knobs: the lazy loaders read weight tensors raw and map nothing for the
        # scale tensors, whose live counterparts do not exist — the coverage gate cannot fire, so the
        # experts would silently load as garbage. from_pretrained owns quantized checkpoints.
        return False
    try:
        weight_map, _ = resolve_safetensors_index(model_path)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return True
    return not any(_UNSUPPORTED_LAZY_EXPERT_PATTERN.search(k) for k in weight_map)


def build_family_key_mapping(
    model: nn.Module,
    disk_keys: list[str],
) -> tuple[dict[str, str], dict[str, tuple]]:
    """:func:`~src.models.loading.lazy_safetensors.weights.build_key_mapping` with the family's
    conversion steps and hub renames resolved off the EP layer registry (see ``hub_conversion``).

    The family is read through ``text_config``: a composite (VLM-wrapper) config keeps model_type on
    its text sub-config, and a wrapper-only read returns "" rather than raising — which would
    silently return no renames, i.e. a whole module absent from the load.
    """
    model_type = getattr(text_config(model.config), "model_type", "")
    return build_key_mapping(
        model,
        disk_keys,
        steps=resolve_conversion_steps(model_type, model),
        hub_renames=hub_to_module_key_renames(model_type),
    )


def _check_fused_experts_mapped(ckpt_format: CheckpointFormat, weight_map: dict[str, str], n_shard: int) -> None:
    """Fail loud when a fused checkpoint's expert weights all missed the model's namespace.

    A silently empty expert plan would train freshly-initialized experts on real attention weights.
    """
    if ckpt_format != CheckpointFormat.FUSED or n_shard > 0:
        return
    sample = next((key for key in weight_map if _FUSED_EXPERT_PATTERN.search(key)), None)
    if sample is not None:
        raise RuntimeError(
            "The checkpoint contains fused expert weights, but none map to the instantiated model. "
            f"Example unmapped checkpoint key: {sample!r}. Check the checkpoint-to-model namespace mapping."
        )


class EPWeightPlanner:
    """Classify each checkpoint key into a loading action for EP.

    For fused 3D expert weights, produces EXPERT_SHARD plans (slice dim 0).
    For individual expert weights, produces REPLICATE (local) or IGNORE (remote).
    Everything else (attention, norms, router, embeddings) is REPLICATE.

    ``ep_config=None`` is the no-expert-sharding case (a dense model, or a pipeline stage loaded
    without EP): every key replicates, so the planner is the whole-checkpoint plan builder that the
    pipeline loader filters, rather than a second copy of it.
    """

    def __init__(self, ep_config: EPConfig | None):
        # An unfinalized config carries no expert range, and would classify EVERY key as REPLICATE:
        # each rank silently materializes all experts unsharded on a job that asked for EP. "Shard no
        # experts" is spelled ``ep_config=None``, so an unfinalized config is a caller-order bug.
        if ep_config is not None and ep_config.expert_start_idx is None:
            raise RuntimeError(
                "EPWeightPlanner received an EPConfig whose expert assignment is not finalized "
                "(expert_start_idx is None); every expert key would be planned as a full replicate. "
                "Call EPConfig.finalize_expert_assignment(num_experts) before building the plan, or "
                "pass ep_config=None for a load that deliberately shards no experts."
            )
        self.start = ep_config.expert_start_idx if ep_config is not None else None
        self.end = ep_config.expert_end_idx if ep_config is not None else None
        self.num_experts = ep_config.num_experts if ep_config is not None else None

    def build(
        self,
        weight_map: dict[str, str],
        disk_to_model: dict[str, str],
        model_keys: set[str],
        fanout: dict[str, tuple] | None = None,
    ) -> list[WeightPlan]:
        plans: list[WeightPlan] = []
        claimed_by: dict[str, str] = {}
        unexpected: list[str] = []
        fanout = fanout or {}
        # Siblings of a fan-in entry are read by the FIRST source's plan (its leading Concat) and
        # never planned or reported on their own.
        consumed = {
            sibling
            for targets in fanout.values()
            for _, ops in targets
            for op in ops
            if isinstance(op, Concat)
            for sibling in op.siblings
        }

        for disk_key, shard_file in weight_map.items():
            if disk_key in consumed:
                continue
            targets = fanout.get(disk_key) or ((disk_to_model.get(disk_key, disk_key), ()),)
            claimed = False
            for model_key, ops in targets:
                if model_key not in model_keys:
                    continue
                # Two disk keys aligning to one model tensor is always a layout error — the second
                # read would either overwrite the first or be dropped on index order, picking a
                # winner by chance.
                if model_key in claimed_by:
                    raise RuntimeError(
                        f"Checkpoint keys {claimed_by[model_key]!r} and {disk_key!r} both align to "
                        f"model tensor {model_key!r}; only one of them can be the weight it holds. "
                        f"Check the checkpoint-to-model key mapping (hub renames / conversion entries)."
                    )

                # Classified on the CANONICAL key: a converted family's disk spelling never matches
                # the fused-expert pattern (Inkling's ``w13_weight``, Step-3.7's ``moe.gate_proj``);
                # for every canonical family the two spellings are identical, so this is the same
                # verdict as classifying disk keys.
                action, shard_dim, start, end = self._classify(model_key)
                if action == WeightAction.EXPERT_SHARD and any(op.dim == shard_dim for op in ops):
                    raise ValueError(
                        f"Conversion op on the expert shard axis for {disk_key!r} — the ranged "
                        f"per-expert read cannot commute with it."
                    )
                plans.append(
                    WeightPlan(
                        action,
                        shard_file,
                        disk_key,
                        model_key,
                        shard_dim,
                        start,
                        end,
                        ops,
                        shard_total=self.num_experts or 0,
                    )
                )
                claimed_by[model_key] = disk_key
                claimed = True

            if not claimed and not self._is_expected_per_expert_key(disk_key):
                unexpected.append(disk_key)

        if unexpected:
            logger.warning(
                f"{len(unexpected)} checkpoint key(s) align to no model tensor and will not be loaded "
                f"(from_pretrained reports these as unexpected keys). First few: {sorted(unexpected)[:KEY_PREVIEW_COUNT]}"
            )

        return plans

    def _classify(self, disk_key: str) -> tuple[WeightAction, int, int, int]:
        if self.start is None or self.end is None:
            return WeightAction.REPLICATE, 0, 0, 0

        if _FUSED_EXPERT_PATTERN.search(disk_key):
            return WeightAction.EXPERT_SHARD, 0, self.start, self.end

        m = _INDIVIDUAL_EXPERT_PATTERN.search(disk_key)
        if m:
            expert_idx = int(m.group(2))
            if self.start <= expert_idx < self.end:
                return WeightAction.REPLICATE, 0, 0, 0
            return WeightAction.IGNORE, 0, 0, 0

        return WeightAction.REPLICATE, 0, 0, 0

    def _is_expected_per_expert_key(self, disk_key: str) -> bool:
        """A per-expert key the ExpertFuser (or a per-expert live model) will consume.

        Those map to no fused model key by construction, and a fusion that then misses an index or
        a GLU half raises there — so they are not "unexpected". Reported instead: an UNDECLARED
        projection suffix (``detect_tasks`` skips it and nothing would ever read it), and an expert
        index at or beyond the config's expert count (no rank owns it under any ep_size — the
        checkpoint declares more experts than the config).
        """
        m = _INDIVIDUAL_EXPERT_PATTERN.search(disk_key)
        if not m:
            return False
        if disk_key[m.end() :] not in per_expert_fusion_map():
            return False
        return self.num_experts is None or int(m.group(2)) < self.num_experts


def _enclosing_layer_prefix(container_prefix: str) -> str | None:
    """``model.layers.47.mlp.experts.`` → ``model.layers.47.``, or ``None`` if the path has no index.

    Walks back to the last numeric component rather than matching a ``layers`` segment: families
    spell the container differently (``model.layers.N``, ``transformer.h.N``).
    """
    parts = [p for p in container_prefix.split(".") if p]
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[: i + 1]) + "."
    return None


class ExpertFuser:
    """Fuse per-expert checkpoint keys into the 3D tensors the model expects.

    E.g. Qwen3 checkpoint has ``experts.0.gate_proj.weight [M, H]`` but the model state dict has
    ``experts.gate_up_proj [E, 2M, H]``. Loads only local experts.
    """

    def __init__(self, ep_start: int, ep_end: int):
        self.ep_start = ep_start
        self.ep_end = ep_end

    def detect_tasks(
        self,
        weight_map: dict[str, str],
        disk_to_model: dict[str, str],
        model_keys: set[str],
    ) -> list[tuple[str, str, dict[int, dict[str, tuple[str, str]]]]]:
        """Detect individual expert keys that need fusing.

        The fusion target is resolved through ``disk_to_model`` — the real disk→model mapping built by
        :func:`build_family_key_mapping` — never by re-deriving model names from the disk key: a family whose
        keys need the ``base_model_prefix`` or a declared hub rename would otherwise resolve to a name
        the model does not have and lose every expert to a silent skip.

        Returns list of (model_key, fusion_type, experts_data) where:
        - model_key: target fused key (e.g. "model.layers.0.mlp.experts.gate_up_proj")
        - fusion_type: "gate_up" or "down"
        - experts_data: {expert_idx: {suffix: (disk_key, shard_file)}} for local experts
        """
        groups: dict[str, dict[str, dict[int, dict[str, tuple[str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        for disk_key, shard_file in weight_map.items():
            m = _INDIVIDUAL_EXPERT_PATTERN.search(disk_key)
            if not m:
                continue

            expert_idx = int(m.group(2))
            if expert_idx < self.ep_start or expert_idx >= self.ep_end:
                continue

            suffix = disk_key[m.end() :]
            if suffix not in per_expert_fusion_map():
                continue

            fusion_type, _ = per_expert_fusion_map()[suffix]
            # Prefix in MODEL space: the mapped key carries whatever renaming the model applies.
            mapped = disk_to_model.get(disk_key, disk_key)
            mapped_m = _INDIVIDUAL_EXPERT_PATTERN.search(mapped)
            if mapped_m is None:
                raise RuntimeError(
                    f"EP lazy load: per-expert checkpoint key '{disk_key}' maps to model key "
                    f"'{mapped}', which carries no expert index — the disk→model mapping renamed the "
                    f"expert container. Its experts would stay on meta and fail later as 'Cannot copy "
                    f"out of meta tensor'. Load with ep_lazy_loading=False."
                )
            prefix = f"{mapped[: mapped_m.start()]}.{mapped_m.group(1)}."
            groups[prefix][fusion_type][expert_idx][suffix] = (disk_key, shard_file)

        tasks = []
        for layer_prefix, fusion_groups in groups.items():
            for fusion_type, expert_dict in fusion_groups.items():
                model_key = layer_prefix + ("gate_up_proj" if fusion_type == "gate_up" else "down_proj")
                if model_key in model_keys:
                    tasks.append((model_key, fusion_type, expert_dict))
                    continue

                # No fused target: legitimate only when the model itself stores one module per expert,
                # in which case EPWeightPlanner already covers these keys. Anything else would leave the
                # experts on meta with no error until the first _to_device.
                unresolved = sorted(
                    disk_key
                    for suffixes in expert_dict.values()
                    for disk_key, _ in suffixes.values()
                    if disk_to_model.get(disk_key, disk_key) not in model_keys
                )
                if not unresolved:
                    continue

                # A layer the model instantiates NOTHING of is a declared-but-absent MTP tail (GLM-4
                # num_nextn_predict_layers): dropping its experts is correct, and EPWeightPlanner
                # already drops that layer's non-expert keys the same way.
                enclosing_layer = _enclosing_layer_prefix(layer_prefix)
                if enclosing_layer is not None and not any(key.startswith(enclosing_layer) for key in model_keys):
                    logger.info(
                        "EP lazy load: skipping %d expert tensors under '%s' — the model instantiates "
                        "no parameter of layer '%s' (multi-token-prediction tail).",
                        len(unresolved),
                        layer_prefix,
                        enclosing_layer,
                    )
                    continue

                raise RuntimeError(
                    f"EP lazy load: per-expert checkpoint keys under '{layer_prefix}' match neither "
                    f"the fused model parameter '{model_key}' nor any per-expert model parameter, so "
                    f"{len(unresolved)} expert tensors would never load — and the model DOES "
                    f"instantiate the rest of that layer, so this is a layout mismatch, not an "
                    f"absent MTP tail. First few: {unresolved[:KEY_PREVIEW_COUNT]}. Load with ep_lazy_loading=False."
                )

        return tasks

    def execute(
        self,
        tasks: list[tuple[str, str, dict[int, dict[str, tuple[str, str]]]]],
        model: nn.Module,
        model_path: str,
        dtype: torch.dtype | None,
        device: str,
    ) -> set[str]:
        """Load and fuse individual expert weights into 3D model parameters.

        Returns the set of model keys that were assigned (so main loader can skip them).
        """
        if not tasks:
            return set()

        needed_shards: set[str] = set()
        for _, _, expert_dict in tasks:
            for suffixes in expert_dict.values():
                for _, shard_file in suffixes.values():
                    needed_shards.add(shard_file)

        handles = {sf: safe_open(os.path.join(model_path, sf), framework="pt", device=device) for sf in needed_shards}

        fused_keys: set[str] = set()
        try:
            for model_key, fusion_type, expert_dict in tasks:
                self._check_local_range(expert_dict, model_key)
                if fusion_type == "gate_up":
                    tensor = self._fuse_gate_up(expert_dict, handles)
                else:
                    tensor = self._fuse_down(expert_dict, handles)

                verify_loaded_shape(
                    model,
                    model_key,
                    f"{len(expert_dict)} fused per-expert checkpoint keys",
                    tensor,
                    shard_dim=0,
                    shard_len=self.ep_end - self.ep_start,
                )
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)

                assign_tensor_to_model(model, model_key, tensor)
                fused_keys.add(model_key)
        finally:
            handles.clear()

        return fused_keys

    def _check_local_range(self, expert_dict: dict[int, dict[str, tuple[str, str]]], model_key: str) -> None:
        """Refuse a fusion that did not collect this rank's FULL contiguous expert range.

        The fuses below stack ``sorted(expert_dict)`` into local slots ``0..E_local-1``, so a global
        index the checkpoint does not carry shifts every later expert down a slot — this rank then
        trains and exports the wrong experts under every name, with nothing to notice it. The
        individual-checkpoint counterpart of :func:`_check_fused_experts_mapped`.
        """
        missing = sorted(set(range(self.ep_start, self.ep_end)) - set(expert_dict))
        if missing:
            raise RuntimeError(
                f"EP lazy load: the checkpoint carries {len(expert_dict)} of this rank's "
                f"{self.ep_end - self.ep_start} experts for {model_key!r} — global expert index(es) "
                f"{missing[:KEY_PREVIEW_COUNT]} are absent from it. Fusing the rest would shift every later expert "
                f"into the wrong local slot."
            )

    def _fuse_gate_up(
        self,
        expert_dict: dict[int, dict[str, tuple[str, str]]],
        handles: dict,
    ) -> torch.Tensor:
        """Fuse the gate + up halves → gate_up_proj [E_local, 2M, H] (halves resolved by position)."""
        slices = []
        for expert_idx in sorted(expert_dict.keys()):
            by_pos = {per_expert_fusion_map()[s][1]: dk_sf for s, dk_sf in expert_dict[expert_idx].items()}
            if set(by_pos) != {0, 1}:
                raise RuntimeError(
                    f"EP lazy load: expert {expert_idx} carries only {sorted(expert_dict[expert_idx])} "
                    f"of its two GLU halves in the checkpoint — fusing a half-present expert would "
                    f"install garbage for the missing half."
                )
            gate_dk, gate_sf = by_pos[0]
            up_dk, up_sf = by_pos[1]
            gate = handles[gate_sf].get_slice(gate_dk)[...]
            up = handles[up_sf].get_slice(up_dk)[...]
            slices.append(torch.cat([gate, up], dim=0))  # [M,H]+[M,H] → [2M,H]
        return torch.stack(slices, dim=0)  # [E_local, 2M, H]

    def _fuse_down(
        self,
        expert_dict: dict[int, dict[str, tuple[str, str]]],
        handles: dict,
    ) -> torch.Tensor:
        """Fuse the single down projection → [E_local, H, M] (``down_proj`` or ``w2``)."""
        slices = []
        for expert_idx in sorted(expert_dict.keys()):
            dk, sf = next(iter(expert_dict[expert_idx].values()))
            slices.append(handles[sf].get_slice(dk)[...])
        return torch.stack(slices, dim=0)  # [E_local, H, M]


@dataclass(frozen=True)
class LazyLoadBase:
    """The meta shell and whole-checkpoint weight plan both lazy loaders start from."""

    model: nn.Module
    plans: list[WeightPlan]
    weight_map: dict[str, str]
    shard_files: list[str]
    missing_shard_files: list[str]
    ckpt_format: CheckpointFormat
    disk_to_model: dict[str, str]
    model_keys: set[str]
    dtype: torch.dtype | None


def lazy_load_prologue(
    model_path: str,
    config,
    ep_config: EPConfig | None,
    *,
    rank: int,
    dtype,
    meta_init_phase: str,
    build_meta_shell: Callable[[bool], nn.Module],
) -> LazyLoadBase:
    """Resolve what the EP and PP lazy loaders both need before the first rank-local read.

    The order is load-bearing: the expert assignment is finalized before any plan exists (the planner
    slices by this rank's expert range), and the meta shell exists before the checkpoint keys are
    aligned against it. Everything here is rank-uniform — ``meta_init_phase`` is a store rendezvous
    every rank must enter. ``build_meta_shell`` receives whether the index names shard files this
    filesystem does not hold, the case a caller answers with a config-only build.
    """
    dtype = resolve_run_dtype(dtype, config)
    if ep_config is not None and ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(get_num_experts(config))

    weight_map, shard_files = resolve_safetensors_index(model_path)
    ckpt_format = CheckpointFormat.detect(weight_map)
    missing_shard_files = sorted(f for f in shard_files if not os.path.isfile(os.path.join(model_path, f)))
    logger.info(f"[Rank {rank}] {len(weight_map)} keys, {len(shard_files)} shards, format={ckpt_format.name}")

    logger.info(f"[Rank {rank}] Instantiating model on meta device...")
    with fs_aware_main_first(meta_init_phase):
        model = build_meta_shell(bool(missing_shard_files))

    model_keys = set(model.state_dict().keys())
    disk_to_model, conversion_fanout = build_family_key_mapping(model, list(weight_map.keys()))
    plans = EPWeightPlanner(ep_config).build(weight_map, disk_to_model, model_keys, fanout=conversion_fanout)
    return LazyLoadBase(
        model=model,
        plans=plans,
        weight_map=weight_map,
        shard_files=shard_files,
        missing_shard_files=missing_shard_files,
        ckpt_format=ckpt_format,
        disk_to_model=disk_to_model,
        model_keys=model_keys,
        dtype=dtype,
    )


def load_ep_model_lazy(
    model_name_or_path: str,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    **model_kwargs,
) -> nn.Module:
    """Load a MoE model for EP using lazy safetensors slicing.

    All ranks run in parallel, each reading only its own expert slice from disk. Handles both fused 3D
    checkpoints (GptOss, LFM2, GLM4) and individual-expert checkpoints (Qwen3, Qwen3.5, Bailing).
    """
    if model_class is None:
        model_class = AutoModelForCausalLM

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = get_local_rank()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    logger.info(f"[Rank {rank}] EP lazy loader: {model_name_or_path}")

    revalidate_attn_kwarg(model_kwargs, config)

    base = lazy_load_prologue(
        model_name_or_path,
        config,
        ep_config,
        rank=rank,
        dtype=dtype,
        meta_init_phase="ep_lazy_meta_init",
        build_meta_shell=lambda _incomplete: instantiate_on_meta(
            model_name_or_path,
            model_class,
            config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        ),
    )
    model, plans, dtype = base.model, base.plans, base.dtype
    weight_map, ckpt_format = base.weight_map, base.ckpt_format

    # Every step from here to the reject below is rank-LOCAL: each rank reads only the shards holding
    # its own experts, and its expert range decides which keys it fuses and how long each slice is. A
    # torn shard, a checkpoint short of one rank's experts, an OOM in the fuse — each raises on that
    # rank alone while its peers sit in the collective below until the NCCL watchdog fires, blaming
    # the collective for a disk error. Same discipline as the save side (see saving.save_ep_model).
    guard = DeferredRankFailure(f"EP lazy load from {model_name_or_path}")

    fused_keys: set[str] = set()
    if ckpt_format == CheckpointFormat.INDIVIDUAL:
        fuser = ExpertFuser(ep_config.expert_start_idx, ep_config.expert_end_idx)
        fusion_tasks = guard.run(partial(fuser.detect_tasks, weight_map, base.disk_to_model, base.model_keys)) or []
        if fusion_tasks:
            fused_keys = (
                guard.run(
                    partial(
                        fuser.execute,
                        fusion_tasks,
                        model,
                        model_name_or_path,
                        dtype=dtype,
                        device=device,
                    )
                )
                or set()
            )
        if fused_keys:
            plans = [p for p in plans if p.model_key not in fused_keys]
            logger.info(f"[Rank {rank}] Fused {len(fused_keys)} expert parameters from individual keys")

    n_shard = sum(1 for p in plans if p.action == WeightAction.EXPERT_SHARD)
    n_repl = sum(1 for p in plans if p.action == WeightAction.REPLICATE)
    n_skip = sum(1 for p in plans if p.action == WeightAction.IGNORE)
    # Ahead of the random-init sweep below: when every expert key missed, ``planned`` is empty and
    # that sweep would re-initialize the WHOLE model before this raised, burying the real diagnostic
    # under an alarming init log.
    _check_fused_experts_mapped(ckpt_format, weight_map, n_shard)

    # Before the load, with the planned key set final: a task head the checkpoint does not carry
    # (``score`` on a CausalLM base for reward / classification) is nobody else's job here — nothing
    # downstream initializes it, and the trainer's meta sweep would hand it uninitialized memory that
    # also differs per rank.
    planned = {plan.model_key for plan in plans} | fused_keys
    guard.run(
        partial(
            init_checkpoint_absent_modules,
            model,
            planned,
            device,
            f"EP lazy load (rank {rank})",
            dtype=dtype,
        )
    )

    logger.info(
        f"[Rank {rank}] Loading: {n_shard} expert-sharded, {n_repl} replicated, "
        f"{n_skip} ignored, {len(fused_keys)} fused"
    )

    # Only the shards this rank actually reads: SafetensorsWeightLoader safe_opens every file it is
    # given, and at ep64 most of them hold other ranks' experts. The PP stage loader does the same.
    live_shards = sorted({plan.shard_file for plan in plans if plan.action != WeightAction.IGNORE})
    loader = SafetensorsWeightLoader(model_name_or_path, live_shards, device=device)
    guard.run(partial(loader.load_into_model, model, plans, dtype=dtype))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # COLLECTIVE — every rank enters it, which is what makes it the fence: a rank that failed above
    # surfaces its real reason on all of them, and the ranks that succeeded still synchronize here
    # before the EP patching below.
    guard.reject()

    logger.info(f"[Rank {rank}] Applying EP patching...")
    # Derived, never asserted: the layers slice their own experts out of a full tensor unless this
    # load actually pre-sliced them. A checkpoint whose expert keys match no known pattern is planned
    # as REPLICATE, and claiming "already sharded" there would give EVERY rank global experts
    # 0..E_local-1 — no error, 7/8 of the experts never trained.
    weights_already_sharded = bool(fused_keys) or n_shard > 0
    model = patch_moe_model_for_ep(model, ep_config, weights_already_sharded=weights_already_sharded)
    create_ep_buffers(model)
    # Non-persistent buffers stayed meta (the shell never materialized them) and the shadow tied
    # lm_head stays meta until tying.
    finalize_loaded_model(model)

    logger.info(f"[Rank {rank}] EP lazy loading complete")
    return model
