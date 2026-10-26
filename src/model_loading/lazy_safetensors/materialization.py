"""Materializing a planned safetensors checkpoint onto a meta shell, key by key.

A :class:`WeightPlan` per checkpoint key says how it loads: this rank's expert slice, the whole tensor, or
nothing. How expert weights are laid out on disk is the EP planner's knowledge; it arrives here as plans.
"""

from __future__ import annotations

import gc
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial

import torch
import torch.nn as nn
from safetensors import safe_open

from src.log import KEY_PREVIEW_COUNT
from src.model_loading.checkpoint_coverage import verify_checkpoint_coverage
from src.model_loading.dtype import reject_fp8_tensor
from src.model_loading.lazy_safetensors.conversion import Concat
from src.models.param_roles import fp32_pinned_names

logger = logging.getLogger(__name__)


class WeightAction(Enum):
    """How a single checkpoint key should be loaded."""

    EXPERT_SHARD = "expert_shard"
    REPLICATE = "replicate"
    IGNORE = "ignore"


@dataclass
class WeightPlan:
    """Loading plan for a single checkpoint key."""

    action: WeightAction
    shard_file: str
    disk_key: str
    model_key: str
    shard_dim: int = 0
    shard_start: int = 0
    shard_end: int = 0
    # Hub-conversion ops applied after the ranged read; a leading ``Concat`` reads its siblings the same way.
    ops: tuple = ()
    # Full extent the model expects along ``shard_dim`` (the config's expert count); 0 is unknown.
    shard_total: int = 0

    @property
    def disk_keys(self) -> tuple[str, ...]:
        """Every checkpoint key this plan reads: its own, then a fan-in's siblings."""
        return (self.disk_key, *(sibling for op in self.ops if isinstance(op, Concat) for sibling in op.siblings))


def materialize_weight_plan(plan: WeightPlan, read: Callable[[str, WeightPlan], torch.Tensor]) -> torch.Tensor:
    """Read a plan and replay its conversion ops through the caller's whole/ranged tensor reader."""
    tensor = read(plan.disk_key, plan)
    for op in plan.ops:
        if isinstance(op, Concat):
            tensor = torch.cat([tensor, *(read(sibling, plan) for sibling in op.siblings)], dim=op.dim)
        else:
            tensor = op(tensor)
    return tensor


def _resolve_owner(model: nn.Module, key: str) -> tuple[nn.Module, str]:
    """The module holding ``key``'s tensor, and the tensor's attribute name on it."""
    owner, _, attr = key.rpartition(".")
    return model.get_submodule(owner), attr


def _target_tensor(model: nn.Module, key: str):
    """The live object currently at ``key`` on ``model``, or ``None`` when the path does not resolve."""
    try:
        owner, attr = _resolve_owner(model, key)
    except AttributeError:
        return None
    return getattr(owner, attr, None)


def verify_loaded_shape(
    model: nn.Module,
    model_key: str,
    source: str,
    tensor: torch.Tensor,
    shard_dim: int | None = None,
    shard_len: int | None = None,
) -> None:
    """Refuse a materialized tensor whose shape disagrees with the live target's.

    The lazy path installs the disk tensor as the new parameter, so without this gate the checkpoint's shape
    would win over the config's (``from_pretrained`` raises instead). ``shard_dim``/``shard_len`` replace an
    expert-sharded target's full expert count with this rank's slice, which also catches a checkpoint holding
    fewer experts than the config declares: a ranged read past the end returns fewer rows instead of raising.
    """
    live = _target_tensor(model, model_key)
    if not isinstance(live, torch.Tensor):
        raise RuntimeError(
            f"Lazy load: planned model key {model_key!r} (from {source}) resolves to no tensor on "
            f"the model shell — the plan and the model tree disagree about this key."
        )
    expected = list(live.shape)
    if shard_dim is not None and shard_len is not None:
        expected[shard_dim] = shard_len
    if list(tensor.shape) == expected:
        return
    sliced = "" if shard_dim is None else f" (dim {shard_dim} = this rank's {shard_len}-expert slice)"
    raise RuntimeError(
        f"Lazy load: tensor from {source} has shape {tuple(tensor.shape)} but model tensor "
        f"{model_key!r} expects {tuple(expected)}{sliced}. The checkpoint does not match the "
        f"config.json the model was built from (e.g. a patch_vocab-grown checkpoint paired with "
        f"the base config, or a changed intermediate_size / expert count)."
    )


def assign_tensor_to_model(model: nn.Module, key: str, tensor: torch.Tensor, *, preserve_parameter: bool = False):
    """Install a real tensor, optionally retaining a pre-wrapper plain parameter's tied aliases."""
    owner, attr = _resolve_owner(model, key)
    old = getattr(owner, attr, None)
    if isinstance(old, nn.Parameter):
        if preserve_parameter:
            if type(old) is not nn.Parameter or type(old.data) is not torch.Tensor or old.is_meta:
                raise TypeError(f"Cannot replace storage of {key!r}: expected a real, plain pre-wrapper parameter.")
            old.data = tensor
        else:
            setattr(owner, attr, nn.Parameter(tensor, requires_grad=old.requires_grad))
    elif attr in dict(owner.named_buffers(recurse=False)):
        owner.register_buffer(attr, tensor)
    else:
        setattr(owner, attr, tensor)


class SafetensorsWeightLoader:
    """Materializes weight plans out of a safetensors checkpoint whose index is ``weight_map``.

    The shard files the plans name are opened before any tensor is read, so a missing one fails first; a
    shard a plan reaches only through a fan-in sibling opens on first use.
    """

    def __init__(self, model_path: str, weight_map: dict[str, str], device: str = "cpu"):
        self._model_path = model_path
        self._weight_map = weight_map
        self._device = device
        self._handles: dict = {}

    def load_into_model(
        self,
        model: nn.Module,
        plans: list[WeightPlan],
        dtype: torch.dtype | None = None,
        keep_fp32: frozenset[str] = frozenset(),
    ):
        """Assign every non-``IGNORE`` plan's tensor, shape-checked against the live target.

        Every float parameter takes ``dtype`` (FSDP2 rejects mixed dtypes in one shard group), except the
        ``keep_fp32`` model keys, the fp32 masters. A float buffer keeps its stored dtype (Zaya's balancing
        biases are fp32 by design) unless the family pins it, which loads it fp32 as ``from_pretrained`` does.
        """
        pinned = fp32_pinned_names(model, include_buffers=True)
        live = [plan for plan in plans if plan.action != WeightAction.IGNORE]
        try:
            for shard_file in sorted({plan.shard_file for plan in live}):
                self._handle(shard_file)
            expert_sharded = 0
            for plan in live:
                # Ops act on non-shard dims (asserted at planning), so the ranged reads stay valid.
                tensor = materialize_weight_plan(plan, self._read)
                sharded = plan.action == WeightAction.EXPERT_SHARD
                verify_loaded_shape(
                    model,
                    plan.model_key,
                    f"checkpoint key(s) {plan.disk_keys}",
                    tensor,
                    shard_dim=plan.shard_dim if sharded else None,
                    shard_len=plan.shard_end - plan.shard_start if sharded else None,
                )
                if tensor.is_floating_point() and isinstance(_target_tensor(model, plan.model_key), nn.Parameter):
                    reject_fp8_tensor(plan.model_key, tensor, dtype)
                    if dtype is not None:
                        tensor = tensor.to(torch.float32 if plan.model_key in keep_fp32 else dtype)
                elif tensor.is_floating_point() and plan.model_key in pinned:
                    tensor = tensor.to(torch.float32)
                assign_tensor_to_model(model, plan.model_key, tensor)
                expert_sharded += sharded
            logger.info(f"Loaded {len(live)} parameters ({expert_sharded} expert-sharded) to device={self._device}")
        finally:
            self._handles.clear()

    def _handle(self, shard_file: str):
        handle = self._handles.get(shard_file)
        if handle is None:
            handle = self._handles[shard_file] = safe_open(
                os.path.join(self._model_path, shard_file), framework="pt", device=self._device
            )
        return handle

    def _read(self, disk_key: str, plan: WeightPlan) -> torch.Tensor:
        """``disk_key`` read the way ``plan`` reads its own key: whole, or this rank's expert slice."""
        shard_file = self._weight_map.get(disk_key)
        if shard_file is None:
            raise RuntimeError(
                f"Lazy load: planned checkpoint key {disk_key!r} (for model tensor {plan.model_key!r}) "
                f"is absent from the safetensors index at {self._model_path}."
            )
        s = self._handle(shard_file).get_slice(disk_key)
        if plan.action != WeightAction.EXPERT_SHARD:
            return s[...]

        shape = s.get_shape()
        if plan.shard_total and shape[plan.shard_dim] != plan.shard_total:
            # A longer axis satisfies the ranged reads without error and no rank would own the extra
            # experts; a shorter one returns fewer rows and is caught by the post-read shape check.
            raise RuntimeError(
                f"Lazy load: checkpoint key {disk_key!r} holds {shape[plan.shard_dim]} "
                f"experts on dim {plan.shard_dim}, but the config declares {plan.shard_total}. "
                f"The expert-parallel split covers only the config's count, so the extra experts "
                f"would be dropped in silence. Load this checkpoint with its own config.json."
            )
        idx = [slice(None)] * len(shape)
        idx[plan.shard_dim] = slice(plan.shard_start, plan.shard_end)
        return s[tuple(idx)]


def materialize_empty_like(model: nn.Module, key: str, device: str, dtype: torch.dtype | None) -> None:
    """Give ``key`` real (uninitialized) storage shaped like its current (meta) tensor.

    A float parameter follows the run's dtype as the loaded ones do; a buffer keeps its own, since an integer
    buffer (routing tables, position ids) would have its indices corrupted and a float one may be fp32 by design.
    """
    owner, attr = _resolve_owner(model, key)
    old = getattr(owner, attr)
    follows_run_dtype = dtype is not None and old.is_floating_point() and isinstance(old, nn.Parameter)
    target_dtype = dtype if follows_run_dtype else old.dtype
    assign_tensor_to_model(model, key, torch.empty_like(old, device=device, dtype=target_dtype))


def init_checkpoint_absent_modules(
    model: nn.Module,
    planned: set[str],
    device: str,
    context: str,
    *,
    dtype: torch.dtype | None = None,
) -> list[str]:
    """Random-init the live tensors the checkpoint does not carry (``from_pretrained``'s missing set), the
    same values on every rank; returns their keys. ``context`` names the caller in every message.

    Only modules whose state is entirely absent are initialized: ``_init_weights`` writes every parameter of
    the module it is handed, so a partially loaded one raises instead. An absence inside the pretrained
    backbone raises through :func:`verify_checkpoint_coverage`.
    """
    # keep_vars, so a tied tensor is one object under both names: the post-load tie_weights() restores the
    # shadow, and drawing it here would allocate a second vocab-sized tensor only to discard it.
    state = model.state_dict(keep_vars=True)
    planned_tensors = {id(state[key]) for key in planned if key in state}
    absent = sorted(key for key, value in state.items() if key not in planned and id(value) not in planned_tensors)
    if not absent:
        return []

    by_owner: dict[str, set[str]] = defaultdict(set)
    for key in absent:
        by_owner[key.rpartition(".")[0]].add(key)

    # Partial absence first: every partially absent module is also an unexpected one, so the coverage gate
    # would otherwise mask the more specific diagnosis.
    for owner, keys in sorted(by_owner.items()):
        owned = {f"{owner}.{name}" if owner else name for name in model.get_submodule(owner).state_dict()}
        if owned - set(absent):
            raise RuntimeError(
                f"{context}: module {owner!r} is only PARTIALLY absent from the checkpoint "
                f"({len(keys)} of {len(owned)} tensors missing), so initializing it would overwrite "
                f"the tensors that did load, and skipping it would leave the rest on the meta "
                f"device. Missing: {sorted(keys)[:KEY_PREVIEW_COUNT]}."
            )

    verify_checkpoint_coverage(model, absent, source=context)

    init_weights = getattr(model, "_init_weights", None)
    if not callable(init_weights):
        raise RuntimeError(
            f"{context}: {len(absent)} live tensor(s) are absent from the checkpoint and "
            f"{type(model).__name__} exposes no _init_weights to initialize them the way "
            f"from_pretrained would. First few: {absent[:KEY_PREVIEW_COUNT]}."
        )

    for owner, keys in sorted(by_owner.items()):
        module = model.get_submodule(owner)
        for key in sorted(keys):
            materialize_empty_like(model, key, device, dtype)
        module.apply(init_weights)

    logger.info(
        f"{context}: initialized {len(absent)} tensor(s) absent from the checkpoint: {absent[:KEY_PREVIEW_COUNT]}"
    )
    return absent


def stream_planned_weights(
    model: nn.Module,
    plans: list[WeightPlan],
    planned: set[str],
    *,
    model_path: str,
    weight_map: dict[str, str],
    device: str,
    dtype: torch.dtype | None,
    keep_fp32: frozenset[str],
    run: Callable[[Callable[[], object]], object],
    label: str,
) -> None:
    """A lazy load's tail: random-init what the checkpoint lacks (``planned`` is every key the load covers,
    fusions included), then stream ``plans``. Each step goes through ``run``, the caller's deferred-failure
    guard, since both raise rank-locally between the caller's collectives.
    """
    # Before the stream, with the planned set final: a task head the checkpoint lacks (``score`` on a CausalLM
    # base) is nobody else's to initialize, and the trainer's meta sweep would hand it per-rank garbage.
    run(partial(init_checkpoint_absent_modules, model, planned, device, label, dtype=dtype))
    loader = SafetensorsWeightLoader(model_path, weight_map, device=device)
    run(partial(loader.load_into_model, model, plans, dtype=dtype, keep_fp32=keep_fp32))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
