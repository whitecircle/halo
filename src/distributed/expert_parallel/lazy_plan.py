"""Planning a lazy EP load: the expert-key vocabulary, the lazy gate, the per-key weight plan and the prologue
the EP and PP lazy loaders share.

The vocabulary is derived from the family specs; a fused-3D checkpoint is planned as
ranged expert slices, a per-expert one as this rank's keys for :mod:`.expert_fuser` to stack.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn

from src.checkpoint.reading import read_checkpoint_config, resolve_checkpoint_weights
from src.distributed.expert_parallel.config import EPConfig
from src.log import KEY_PREVIEW_COUNT
from src.model_loading.dtype import resolve_load_dtype
from src.model_loading.lazy_safetensors.conversion import Concat, ConvertTargets, build_key_mapping
from src.model_loading.lazy_safetensors.conversion_selection import replay_loaded_conversions, resolve_conversion_steps
from src.model_loading.lazy_safetensors.materialization import WeightAction, WeightPlan
from src.models.config_access import text_config
from src.models.moe.config_fields import require_expert_count
from src.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS
from src.models.moe.family_specs import experts_container_attrs, find_moe_family_spec, per_expert_fusion_map
from src.runtime.filesystem import fs_aware_main_first
from src.runtime.ranks import get_global_rank

logger = logging.getLogger(__name__)

# Derived from the family specs' container attrs, never restated: a hand-written alternation that omitted
# a family's container silently mis-planned every one of its expert keys.
_CONTAINER_ALT = "|".join(re.escape(a) for a in experts_container_attrs())
# A fused 3D expert tensor, expert index on dim 0: GptOss ``[E,H,2M]``, Qwen3/GLM4/LFM2 ``[E,2M,H]``.
FUSED_EXPERT_KEY_PATTERN = re.compile(
    rf"\.(?:{_CONTAINER_ALT})\.({'|'.join(re.escape(k) for k in HF_FUSED_EXPERT_KEYS)})$"
)
# One expert's tensor (``…mlp.experts.N.gate_proj.weight``): (1) the container attribute, (2) the expert index.
PER_EXPERT_KEY_PATTERN = re.compile(rf"\.({_CONTAINER_ALT})\.(\d+)\.")
# An expert index nested under an extra module (``…experts.local_experts.N.linear_fc1``), which neither
# pattern above matches, so the layout would read as FUSED and every expert plan would drop.
_NESTED_EXPERT_KEY_PATTERN = re.compile(rf"\.(?:{_CONTAINER_ALT})\.[A-Za-z_]\w*\.\d+\.")


class ExpertKeyLayout(Enum):
    """How a checkpoint stores its expert weights."""

    FUSED = auto()  # 3D tensors with the expert dim: sliced directly
    PER_EXPERT = auto()  # one key per expert: this rank's read and fused

    @staticmethod
    def detect(weight_map: dict[str, str]) -> ExpertKeyLayout:
        """The layout of the first expert key, FUSED for a checkpoint with none (a no-op)."""
        for key in weight_map:
            if FUSED_EXPERT_KEY_PATTERN.search(key):
                return ExpertKeyLayout.FUSED
            if PER_EXPERT_KEY_PATTERN.search(key):
                return ExpertKeyLayout.PER_EXPERT
        return ExpertKeyLayout.FUSED


def is_lazy_loadable(model_path: str) -> bool:
    """Whether the lazy loaders can materialize this checkpoint; the caller routes a refusal to
    ``from_pretrained``.

    The family spec's ``supports_lazy_loading`` decides, read off ``config.model_type``; a checkpoint-native
    quantization is refused (its scale tensors map to nothing live, so the experts would load as garbage
    unseen); a nested expert index is the structural backstop for a ``model_type`` no family claims.
    Best-effort: ``True`` when neither the config nor the index can be read.
    """
    try:
        config = read_checkpoint_config(model_path, missing_ok=True) or {}
    except (OSError, ValueError):
        config = {}
    spec = find_moe_family_spec(config.get("model_type"))
    if spec is not None and not spec.supports_lazy_loading:
        return False
    if config.get("quantization_config"):
        return False
    try:
        weight_map = resolve_checkpoint_weights(model_path).require_safetensors("checkpoint").weight_map
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return True
    return not any(_NESTED_EXPERT_KEY_PATTERN.search(k) for k in weight_map)


def family_key_mapping(model: nn.Module, disk_keys: list[str]) -> tuple[dict[str, str], dict[str, ConvertTargets]]:
    """:func:`~src.model_loading.lazy_safetensors.conversion.build_key_mapping` under the conversions and hub
    renames the family spec declares.

    The family is read through ``text_config``: a composite config keeps ``model_type`` on its text
    sub-config, and a wrapper-only read would return no renames, a whole module absent from the load.
    """
    model_type = getattr(text_config(model.config), "model_type", "")
    steps = resolve_conversion_steps(model, model_type)
    spec = find_moe_family_spec(model_type)
    return build_key_mapping(model, disk_keys, steps=steps, hub_renames=spec.hub_to_module_renames() if spec else ())


def loaded_key_mapping(model: nn.Module, disk_keys: list[str]) -> tuple[dict[str, str], dict[str, ConvertTargets]]:
    """:func:`~src.model_loading.lazy_safetensors.conversion.build_key_mapping` under the conversions an eager
    load of ``model`` recorded, with the family's hub renames."""
    model_type = getattr(text_config(model.config), "model_type", "")
    steps = replay_loaded_conversions(model)
    spec = find_moe_family_spec(model_type)
    return build_key_mapping(model, disk_keys, steps=steps, hub_renames=spec.hub_to_module_renames() if spec else ())


class WeightPlanner:
    """Classify each checkpoint key into a :class:`WeightAction`: a fused expert tensor is this rank's
    ``EXPERT_SHARD`` slice, a per-expert key ``REPLICATE`` when local and ``IGNORE`` otherwise, and
    everything else ``REPLICATE``. ``ep_config=None`` shards no experts: the whole-checkpoint plan.
    """

    def __init__(self, ep_config: EPConfig | None):
        # An unfinalized config would plan every expert key as a full replicate on a job that asked for EP.
        if ep_config is not None and ep_config.expert_start_idx is None:
            raise RuntimeError(
                "WeightPlanner received an EPConfig whose expert assignment is not finalized "
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
        """One plan per checkpoint key that aligns to a model tensor (a fan-out key, one per target)."""
        plans: list[WeightPlan] = []
        claimed_by: dict[str, str] = {}
        unexpected: list[str] = []
        fanout = fanout or {}
        # A fan-in's siblings are read by its first source's plan (the leading Concat), never on their own.
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
                if model_key in claimed_by:
                    raise RuntimeError(
                        f"Checkpoint keys {claimed_by[model_key]!r} and {disk_key!r} both align to "
                        f"model tensor {model_key!r}; only one of them can be the weight it holds. "
                        f"Check the checkpoint-to-model key mapping (hub renames / conversion entries)."
                    )

                # Classified on the canonical key: a converted family's disk spelling (Inkling's
                # ``w13_weight``) never matches the fused-expert pattern.
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

    def _classify(self, model_key: str) -> tuple[WeightAction, int, int, int]:
        if self.start is None or self.end is None:
            return WeightAction.REPLICATE, 0, 0, 0

        if FUSED_EXPERT_KEY_PATTERN.search(model_key):
            return WeightAction.EXPERT_SHARD, 0, self.start, self.end

        m = PER_EXPERT_KEY_PATTERN.search(model_key)
        if m:
            expert_idx = int(m.group(2))
            if self.start <= expert_idx < self.end:
                return WeightAction.REPLICATE, 0, 0, 0
            return WeightAction.IGNORE, 0, 0, 0

        return WeightAction.REPLICATE, 0, 0, 0

    def _is_expected_per_expert_key(self, disk_key: str) -> bool:
        """A per-expert key the fuser (or a per-expert live model) consumes, so not an unexpected one.

        Still reported: an undeclared projection suffix, which nothing reads, and an expert index at or
        beyond the config's expert count, which no rank owns.
        """
        m = PER_EXPERT_KEY_PATTERN.search(disk_key)
        if not m:
            return False
        if disk_key[m.end() :] not in per_expert_fusion_map():
            return False
        return self.num_experts is None or int(m.group(2)) < self.num_experts


@dataclass(frozen=True)
class LazyLoadPlan:
    """The meta shell and whole-checkpoint weight plan both lazy loaders start from."""

    model: nn.Module
    plans: list[WeightPlan]
    weight_map: dict[str, str]
    shard_files: list[str]
    missing_shard_files: list[str]
    expert_layout: ExpertKeyLayout
    disk_to_model: dict[str, str]
    model_keys: set[str]
    dtype: torch.dtype | None


def lazy_load_prologue(
    model_path: str,
    config,
    ep_config: EPConfig | None,
    *,
    dtype,
    meta_init_phase: str,
    build_meta_shell: Callable[[bool], nn.Module],
) -> LazyLoadPlan:
    """COLLECTIVE-EQUIVALENT (``meta_init_phase`` is a store rendezvous): what the EP and PP lazy loaders
    need before the first rank-local read.

    The expert assignment is finalized before any plan exists (the planner slices by this rank's range),
    and the meta shell before the keys are aligned against it. ``build_meta_shell`` receives whether the
    index names shard files this filesystem lacks, which a caller answers with a config-only build.
    """
    rank = get_global_rank()
    dtype = resolve_load_dtype(dtype, config)
    if ep_config is not None and ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(require_expert_count(config))

    checkpoint = resolve_checkpoint_weights(model_path).require_safetensors("checkpoint")
    weight_map, shard_files = checkpoint.weight_map, list(checkpoint.shard_files)
    expert_layout = ExpertKeyLayout.detect(weight_map)
    missing_shard_files = checkpoint.absent_shard_files()
    logger.info(f"[Rank {rank}] {len(weight_map)} keys, {len(shard_files)} shards, format={expert_layout.name}")

    logger.info(f"[Rank {rank}] Instantiating model on meta device...")
    with fs_aware_main_first(meta_init_phase):
        model = build_meta_shell(bool(missing_shard_files))

    model_keys = set(model.state_dict().keys())
    disk_to_model, conversion_fanout = family_key_mapping(model, list(weight_map.keys()))
    plans = WeightPlanner(ep_config).build(weight_map, disk_to_model, model_keys, fanout=conversion_fanout)
    return LazyLoadPlan(
        model=model,
        plans=plans,
        weight_map=weight_map,
        shard_files=shard_files,
        missing_shard_files=missing_shard_files,
        expert_layout=expert_layout,
        disk_to_model=disk_to_model,
        model_keys=model_keys,
        dtype=dtype,
    )
