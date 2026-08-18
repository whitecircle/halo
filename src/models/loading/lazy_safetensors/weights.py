"""Reading a safetensors checkpoint key by key: index, gate, key alignment, plans, materialization.

:class:`SafetensorsWeightLoader` materializes one :class:`WeightPlan` per key — this rank's expert
slice for an ``EXPERT_SHARD`` plan, the whole tensor otherwise — and
:func:`init_checkpoint_absent_modules` draws what the checkpoint does not carry.

How expert weights are laid out on disk is not known here: the EP package owns the key patterns,
planner and fuser, and injects its verdict as a plan.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from safetensors import safe_open

from src.checkpoint.format import has_whole_model_weight_file, resolve_checkpoint_weights
from src.log import KEY_PREVIEW_COUNT
from src.models.loading.checkpoint_coverage import verify_checkpoint_coverage
from src.models.loading.lazy_safetensors.conversion import Concat, Convert, Rename, convert_disk_keys

logger = logging.getLogger(__name__)


def resolve_safetensors_index(model_path: str) -> tuple[dict[str, str], list[str]]:
    """``(weight_map, shard_files)``: ``{disk_key: shard_filename}`` and the deduplicated shard list.

    The lazy path is safetensors-only, so a directory holding just a legacy ``pytorch_model.bin``
    reads as absent and the caller falls back to an eager load.
    """
    layout = resolve_checkpoint_weights(model_path)
    if not layout.shard_files:
        raise FileNotFoundError(
            f"No safetensors checkpoint found at {model_path}. "
            f"Expected model.safetensors.index.json or model.safetensors"
        )
    return layout.weight_map, list(layout.shard_files)


def has_safetensors_checkpoint(model_path: str) -> bool:
    """Whether a model directory holds a safetensors checkpoint, the lazy path's entry question.

    The safetensors-only spelling of :func:`~src.checkpoint.format.has_whole_model_weight_file`,
    which defines the filenames that count; re-listing them here would drift from it.
    """
    return has_whole_model_weight_file(model_path, safetensors_only=True)


def resolve_run_dtype(dtype: torch.dtype | None, config) -> torch.dtype:
    """The run's dtype: the explicit override, else the checkpoint config's, else bf16 (the
    toolkit-wide default precision).

    The fallback keys on the value rather than on the attribute being absent: a config declaring no
    dtype carries ``dtype = None``, which each consumer would resolve differently (the meta shell to
    torch's fp32 default, the materializer to no cast at all), leaving a family's fp32-pinned modules
    mixed in with bf16 ones, which FSDP2 rejects inside one shard group.
    """
    return dtype or getattr(config, "dtype", None) or torch.bfloat16


def build_key_mapping(
    model: nn.Module,
    disk_keys: list[str],
    steps: tuple[Rename | Convert, ...] | None = None,
    hub_renames: tuple[tuple[str, str], ...] = (),
) -> tuple[dict[str, str], dict[str, tuple]]:
    """Build disk_key → model_key, plus the fan-out map of the caller-resolved hub conversion.

    The caller-resolved conversion ``steps`` (``None`` when already canonical) run first, rewriting
    vendor-namespace disk keys and attaching materialization ops; every produced target then goes
    through the same :func:`_align_key` cascade as a canonical key. Returns the 1:1 ``disk_to_model``
    map (first target) plus ``disk_key`` to ``((model_key, ops), ...)`` for every converted key.

    Key translation comes from the declared entries alone: an unmapped key is skipped by the planner,
    so an unapplied rename means a whole module is absent from the load.

    A converted key whose targets all miss the model's key space while the key itself resolves is
    kept unconverted: transformers' conversion sources are not vendor-anchored (5.16's deepseek_v4
    maps ``\\.norm\\.`` to ``.kv_norm.``, which also matches the canonical ``model.norm.weight``), and
    its own loader undoes such a rename too.
    """
    prefix = getattr(model, "base_model_prefix", "")
    _prefix = f"{prefix}." if prefix else ""
    model_keys = set(model.state_dict().keys())
    if steps is None:
        return {key: _align_key(key, model_keys, _prefix, hub_renames) for key in disk_keys}, {}

    converted = convert_disk_keys(disk_keys, steps, _prefix)

    disk_to_model: dict[str, str] = {}
    fanout: dict[str, tuple] = {}
    for key in disk_keys:
        plain = _align_key(key, model_keys, _prefix, hub_renames)
        aligned = None
        targets = converted.get(key)
        if targets is not None:
            aligned = tuple((_align_key(target, model_keys, _prefix, hub_renames), ops) for target, ops in targets)
            if plain in model_keys and all(target not in model_keys for target, _ in aligned):
                aligned = None  # already-canonical key caught by an unanchored source; keep it
        if aligned is None:
            disk_to_model[key] = plain
        else:
            disk_to_model[key] = aligned[0][0]
            fanout[key] = aligned
    return disk_to_model, fanout


def _align_key(key: str, model_keys: set[str], _prefix: str, hub_renames) -> str:
    """Align one checkpoint key with the model's state-dict spelling: the Mistral3 nested namespace,
    ``base_model_prefix`` added then stripped (a backbone checkpoint into a task wrapper, and the
    mirror case), the caller's hub → module renames, then a ``language_model.`` segment dropped for a
    VLM checkpoint loaded as a text-only causal LM. Each step runs only while the key is unresolved."""
    new_key = key
    # Public multimodal Mistral3 checkpoints expose the nested text model as
    # ``language_model.model.*`` while the wrapper holds it at ``model.language_model.*``; mapped
    # before the generic base-prefix handling below.
    if new_key not in model_keys and new_key.startswith("language_model.model."):
        candidate = f"{_prefix}language_model.{new_key[len('language_model.model.') :]}"
        if candidate in model_keys:
            new_key = candidate

    if new_key not in model_keys and _prefix and not new_key.startswith(_prefix):
        prefixed = _prefix + new_key
        if prefixed in model_keys:
            new_key = prefixed

    if new_key not in model_keys and _prefix and new_key.startswith(_prefix):
        stripped = new_key[len(_prefix) :]
        if stripped in model_keys:
            new_key = stripped

    if new_key not in model_keys:
        for hub_spelling, module_spelling in hub_renames:
            candidate = new_key.replace(hub_spelling, module_spelling, 1)
            if candidate in model_keys:
                new_key = candidate
                break

    if new_key not in model_keys and ".language_model." in new_key:
        stripped = new_key.replace(".language_model.", ".", 1)
        if stripped in model_keys:
            new_key = stripped
    if new_key not in model_keys and new_key.startswith("language_model."):
        stripped = new_key[len("language_model.") :]
        if stripped in model_keys:
            new_key = stripped

    return new_key


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
    # Hub-conversion ops applied at materialization, after the ranged read (see the op classes above);
    # a leading ``Concat`` reads its sibling keys through the same ranged slice first.
    ops: tuple = ()
    # Full extent the model expects along ``shard_dim`` (the config's expert count); 0 is unknown.
    shard_total: int = 0

    @property
    def disk_keys(self) -> tuple[str, ...]:
        """Every checkpoint key this plan reads: its own, then a fan-in's siblings."""
        return (self.disk_key, *(sibling for op in self.ops if isinstance(op, Concat) for sibling in op.siblings))


class SafetensorsWeightLoader:
    """Load weights from safetensors files according to a weight plan.

    ``shard_files`` are opened up front, so a missing file fails before any tensor is read; a shard a
    plan reaches only through a fan-in sibling opens on first use, resolved through the checkpoint's
    own index.
    """

    def __init__(self, model_path: str, shard_files: list[str], device: str = "cpu"):
        self._model_path = model_path
        self._shard_files = shard_files
        self._device = device
        self._weight_map, _ = resolve_safetensors_index(model_path)
        self._handles: dict = {}

    def load_into_model(
        self,
        model: nn.Module,
        plans: list[WeightPlan],
        dtype: torch.dtype | None = None,
    ):
        """Materialize weights and assign to the model's parameters.

        EXPERT_SHARD: only the local expert slice is read from disk.
        REPLICATE: full tensor is read.
        IGNORE: skipped.

        Every materialized tensor is shape-checked against the live target before assignment, since
        this path bypasses ``from_pretrained``'s own size-mismatch check.
        """
        self._open()
        try:
            loaded = expert_sharded = 0
            for plan in plans:
                if plan.action == WeightAction.IGNORE:
                    continue

                tensor = self._materialize(plan)
                sharded = plan.action == WeightAction.EXPERT_SHARD
                verify_loaded_shape(
                    model,
                    plan.model_key,
                    f"checkpoint key(s) {plan.disk_keys}",
                    tensor,
                    shard_dim=plan.shard_dim if sharded else None,
                    shard_len=plan.shard_end - plan.shard_start if sharded else None,
                )
                # Every float parameter, overriding the class's _keep_in_fp32_modules[_strict]:
                # FSDP2 rejects mixed dtypes in one shard group, and the EP from_pretrained fallback
                # re-casts to match. Parameters only: a float buffer may be fp32 by design (Zaya's
                # balancing biases).
                if (
                    dtype is not None
                    and tensor.is_floating_point()
                    and isinstance(_target_tensor(model, plan.model_key), nn.Parameter)
                ):
                    tensor = tensor.to(dtype)

                assign_tensor_to_model(model, plan.model_key, tensor)
                loaded += 1
                if sharded:
                    expert_sharded += 1

            logger.info(f"Loaded {loaded} parameters ({expert_sharded} expert-sharded) to device={self._device}")
        finally:
            self._close()

    def _open(self):
        for sf in self._shard_files:
            self._handle(sf)

    def _close(self):
        self._handles.clear()

    def _handle(self, shard_file: str):
        handle = self._handles.get(shard_file)
        if handle is None:
            handle = self._handles[shard_file] = safe_open(
                os.path.join(self._model_path, shard_file), framework="pt", device=self._device
            )
        return handle

    def _materialize(self, plan: WeightPlan) -> torch.Tensor:
        tensor = self._read(plan.disk_key, plan)
        # Ops act on non-shard dims (asserted at planning), so the ranged reads stay valid.
        for op in plan.ops:
            if isinstance(op, Concat):
                tensor = torch.cat([tensor, *(self._read(sibling, plan) for sibling in op.siblings)], dim=op.dim)
            else:
                tensor = op(tensor)
        return tensor

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
            # A ranged read is satisfiable by a longer axis without error: the per-rank slices would
            # cover only the config's expert count and no rank would own the rest. A shorter axis
            # returns fewer rows and is caught by the post-read shape check.
            raise RuntimeError(
                f"Lazy load: checkpoint key {disk_key!r} holds {shape[plan.shard_dim]} "
                f"experts on dim {plan.shard_dim}, but the config declares {plan.shard_total}. "
                f"The expert-parallel split covers only the config's count, so the extra experts "
                f"would be dropped in silence. Load this checkpoint with its own config.json."
            )
        idx = [slice(None)] * len(shape)
        idx[plan.shard_dim] = slice(plan.shard_start, plan.shard_end)
        return s[tuple(idx)]


def _target_tensor(model: nn.Module, key: str):
    """The live object currently at ``key`` on ``model``, or ``None`` when the path does not resolve."""
    parent = model
    for part in key.split(".")[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    return getattr(parent, key.rpartition(".")[2], None)


def verify_loaded_shape(
    model: nn.Module,
    model_key: str,
    source: str,
    tensor: torch.Tensor,
    shard_dim: int | None = None,
    shard_len: int | None = None,
) -> None:
    """Refuse a materialized tensor whose shape disagrees with the live target's.

    The lazy path installs the disk tensor as the new parameter/buffer, so without this gate the
    checkpoint's shape would win over the config's (``from_pretrained`` raises instead). For an
    expert-sharded target, ``shard_dim``/``shard_len`` replace the shell's full expert count with this
    rank's slice, which also catches a checkpoint holding fewer experts than the config declares,
    since a ranged read past the end returns fewer rows instead of raising.
    """
    live = _target_tensor(model, model_key)
    if not isinstance(live, torch.Tensor):
        # Every planned key comes from model.state_dict(), so this is unreachable in practice, but
        # the guard must not pass when its own resolution fails.
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
        f"config.json the model was built from (e.g. a patch_vocab-shrunk checkpoint paired with "
        f"the base config, or a changed intermediate_size / expert count)."
    )


def assign_tensor_to_model(model: nn.Module, key: str, tensor: torch.Tensor):
    """Replace a meta-device parameter or buffer with a real tensor."""
    parts = key.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    attr_name = parts[-1]
    old = getattr(parent, attr_name, None)

    if isinstance(old, nn.Parameter):
        setattr(parent, attr_name, nn.Parameter(tensor, requires_grad=old.requires_grad))
    elif attr_name in dict(parent.named_buffers(recurse=False)):
        parent.register_buffer(attr_name, tensor)
    else:
        setattr(parent, attr_name, tensor)


def materialize_empty_like(model: nn.Module, key: str, device: str, dtype: torch.dtype | None) -> None:
    """Give ``key`` real (uninitialized) storage shaped like its current (meta) tensor.

    A float parameter follows the run's dtype exactly as the loaded ones do; buffers keep their own,
    since an integer buffer (routing tables, position ids) would have its indices corrupted and a
    float one may be fp32 by design (Zaya's balancing biases).
    """
    parent = model.get_submodule(key.rpartition(".")[0]) if "." in key else model
    old = getattr(parent, key.rpartition(".")[2])
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
    """Random-init the live tensors the checkpoint does not carry: ``from_pretrained``'s missing set.

    A task head built on top of a base checkpoint has no tensor on disk, and these loaders bypass
    ``from_pretrained``, so nothing else initializes it off the meta device. Every rank draws the
    same values. ``context`` names the caller in every message; ``dtype`` is the run's, not the
    shell's.

    Only modules whose state is entirely absent are initialized: ``_init_weights`` writes every
    parameter of the module it is handed, so a partially loaded one would lose checkpoint weights and
    raises instead. An absence inside the pretrained backbone raises through
    :func:`verify_checkpoint_coverage`. Returns the initialized keys.
    """
    # keep_vars so tied state is the same object under both names (``state_dict()`` detaches, hiding
    # the sharing): the post-load ``tie_weights()`` restores the shadow, and drawing it here would
    # allocate a second vocab-sized tensor only to discard it.
    state = model.state_dict(keep_vars=True)
    planned_tensors = {id(state[key]) for key in planned if key in state}
    absent = sorted(key for key, value in state.items() if key not in planned and id(value) not in planned_tensors)
    if not absent:
        return []

    by_owner: dict[str, set[str]] = defaultdict(set)
    for key in absent:
        by_owner[key.rpartition(".")[0]].add(key)

    # Partial absence first: it is the more specific diagnosis, and every partially absent module is
    # also an unexpected one, so the coverage gate would otherwise mask it.
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
