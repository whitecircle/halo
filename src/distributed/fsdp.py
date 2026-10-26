"""FSDP2 (``fully_shard``) data-parallel gradient sync for every torchrun mode, plus the sharding-state
helpers its consumers need afterwards.

The DP entry point builds its mesh and the TP one takes the loader's ``(dp, tp)`` mesh; both run one
wrap body, so a policy knob cannot drift between them. ``reshard_after_forward``: False =
SHARD_GRAD_OP (faster), True = FULL_SHARD (lower peak memory).
"""

import contextlib
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers.modeling_utils import PreTrainedModel

from src.distributed.mesh import MeshDim, build_dp_mesh, build_dp_tp_mesh, mesh_dim_names
from src.models.module_tree import (
    DECODER_LAYER_LIST_ATTRS,
    backbone_with_layers,
    decoder_layers,
    input_embedding_backbone,
)
from src.models.param_roles import fp32_pinned_names
from src.models.unwrapping import base_transformers_model
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class IdentityParamSet:
    """id()-based membership: FSDP2's ``param not in ignored_params`` falls back to tensor ``__eq__``
    on a hash collision, which raises across shapes (EP's fused 3D experts vs 2D weights)."""

    def __init__(self, params=()):
        self._ids = {id(p) for p in params}

    def __contains__(self, item):
        return id(item) in self._ids

    def __bool__(self):
        return bool(self._ids)


def mixed_precision_policy(
    compute_dtype: torch.dtype, *, fp32_reduce: bool, cast_forward_inputs: bool
) -> MixedPrecisionPolicy:
    """FSDP2 policy computing in ``compute_dtype`` and reducing grads in fp32 under ``fp32_reduce``
    (a bf16 sum loses ~10^4x precision), else in ``compute_dtype``. An fp32 run gets an all-fp32
    policy: ``fully_shard`` requires a non-None one, and it is a functional no-op.
    """
    return MixedPrecisionPolicy(
        param_dtype=compute_dtype,
        reduce_dtype=torch.float32 if fp32_reduce else compute_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )


def _should_cast_forward_inputs(model: nn.Module) -> bool:
    """False when a layer class declares ``_fp32_interlayer_residual`` (Zaya): a bf16 cast at the layer
    boundary makes non-reentrant GC save bf16 where the recompute keeps fp32, which torch rejects."""
    return not any(getattr(type(module), "_fp32_interlayer_residual", False) for module in model.modules())


def _reject_unreachable_decoder_layers(model: nn.Module) -> None:
    """Raise when a generative decoder exposes no decoder-layer list :func:`decoder_layers` reaches:
    the whole model would be one shard group, all-gathered for the entire forward. The layer-less
    shapes this wrap serves (a SentenceTransformer, a classification backbone, a PP stage) do not
    generate."""
    inner = base_transformers_model(model)
    if not (isinstance(inner, PreTrainedModel) and inner.can_generate()):
        return
    raise RuntimeError(
        f"FSDP2 found no decoder-layer list on {type(inner).__name__}: none of "
        f"{DECODER_LAYER_LIST_ATTRS} is reachable from it, so the whole model would be wrapped as one "
        f"shard group and all-gathered for the entire forward instead of per layer. Add this family's "
        f"spelling to DECODER_LAYER_LIST_ATTRS, which every layer-list consumer reads."
    )


def _warn_fp32_pins_cast_by_policy(model: nn.Module, mp_policy: MixedPrecisionPolicy) -> None:
    """Warn when the policy will compute fp32-pinned params in low precision.

    They reach the wrap in fp32 only under fp32 master weights, and ``MixedPrecisionPolicy`` casts a
    whole group with no per-parameter dtype, so they compute at ``param_dtype`` where a single-GPU run
    keeps fp32.
    """
    if mp_policy.param_dtype in (None, torch.float32):
        return
    pinned = fp32_pinned_names(model)
    cast_params = [name for name, p in model.named_parameters() if p.dtype == torch.float32 and name in pinned]
    if not cast_params or not is_global_main_process():
        return
    pinned_modules = sorted({name.rsplit(".", 1)[0] for name in cast_params})
    logger.warning(
        f"  FSDP2 mixed precision (param_dtype={mp_policy.param_dtype}) will cast "
        f"{len(cast_params)} fp32-pinned parameter(s) — transformers "
        f"_keep_in_fp32_modules(_strict) on {type(model).__name__} — to {mp_policy.param_dtype} "
        f"for forward/backward compute (storage and optimizer state stay fp32; a single-GPU run "
        f"computes these in fp32). Affected modules: {pinned_modules[:20]}"
        f"{' …' if len(pinned_modules) > 20 else ''}"
    )


def _tied_parameters(model: nn.Module) -> "IdentityParamSet":
    """Parameters reachable under more than one name (``tie_word_embeddings``): ``fully_shard``
    rebinds a shared parameter only within one call, so a tie straddling two calls splits in two."""
    seen: set[int] = set()
    tied = []
    for _name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            tied.append(param)
        seen.add(id(param))
    return IdentityParamSet(tied)


def _uniform_dtype_split(layer: nn.Module, left_out: "IdentityParamSet") -> list[nn.Module]:
    """Submodules of ``layer`` that take a ``fully_shard`` group of their own, innermost first, so every
    group's trainable parameters share one dtype (FSDP2 asserts it; an ``fp32_router`` breaks it).

    A subtree of another dtype splits off at the smallest submodule holding all of it; a mixed subtree
    splits further. The layer keeps the dtype needing the fewest extra groups. ``left_out`` holds the
    parameters no group of the layer takes. Raises where one module directly owns two dtypes.
    """

    def trainable(module: nn.Module, *, recurse: bool) -> list[nn.Parameter]:
        return [p for p in module.parameters(recurse=recurse) if p.requires_grad and p not in left_out]

    def own_dtype(module: nn.Module) -> torch.dtype | None:
        dtypes = {p.dtype for p in trainable(module, recurse=False)}
        if len(dtypes) > 1:
            raise RuntimeError(
                f"FSDP2 needs one dtype per shard group, but {type(module).__name__} (in "
                f"{type(layer).__name__}) directly owns trainable parameters of dtypes "
                f"{sorted(map(str, dtypes))}, which no split into submodule groups can separate. Store "
                f"that module's trainable parameters in one dtype."
            )
        return next(iter(dtypes), None)

    def smallest_holder(module: nn.Module) -> nn.Module:
        while not trainable(module, recurse=False):
            holders = [child for child in module.children() if trainable(child, recurse=True)]
            if len(holders) != 1:
                break
            module = holders[0]
        return module

    def split_below(module: nn.Module, group_dtype: torch.dtype) -> list[nn.Module]:
        split = []
        for child in module.children():
            dtypes = {p.dtype for p in trainable(child, recurse=True)}
            if dtypes <= {group_dtype}:
                continue
            if len(dtypes) == 1:
                split.append(smallest_holder(child))
                continue
            own = own_dtype(child)
            if own is not None and own != group_dtype:
                split += [*split_below(child, own), child]
            else:
                split += split_below(child, group_dtype)
        return split

    dtypes = {p.dtype for p in trainable(layer, recurse=True)}
    if len(dtypes) < 2:
        return []
    own = own_dtype(layer)
    candidates = [own] if own is not None else sorted(dtypes, key=str)
    return min((split_below(layer, dtype) for dtype in candidates), key=len)


def _unshard_with_layer(layer: nn.Module, split: list[nn.Module]) -> None:
    """Unshard the ``split`` groups whenever ``layer`` runs forward: a family can read a submodule's
    parameters without calling it (``F.linear(x, self.gate.weight)``), skipping its own pre-hook. The
    root post-backward callback still reduces their gradients."""

    def _pre_forward(_module: nn.Module, _args: tuple) -> None:
        for module in split:
            module.unshard()

    layer.register_forward_pre_hook(_pre_forward)


def apply_fsdp2_per_layer(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    ignored_params: "IdentityParamSet",
) -> int:
    """Apply FSDP2 per decoder layer, then to the backbone and the model root; returns the group count
    (for logging). A layer mixing trainable dtypes first gets nested groups."""
    _warn_fp32_pins_cast_by_policy(model, mp_policy)
    underlying_model = backbone_with_layers(model) or model

    # Shard the module whose forward consumes ``embed_tokens``: a group on a nested backbone unshards
    # it after the composite parent above has already embedded the ids.
    embed_backbone = input_embedding_backbone(model) or underlying_model

    # A tied weight spans two `_shard` calls; reserve it for the root call or fully_shard silently
    # severs the tie, leaving each half with half the gradient.
    root_reserved = _tied_parameters(model) if model is not embed_backbone else IdentityParamSet()

    def _left_out(module: nn.Module, reserved: "IdentityParamSet") -> "IdentityParamSet":
        return IdentityParamSet(p for p in module.parameters() if p in ignored_params or p in reserved)

    def _shard(module: nn.Module, *, reserved: "IdentityParamSet") -> None:
        scoped = _left_out(module, reserved)
        fully_shard(
            module,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            **({"ignored_params": scoped} if scoped else {}),
        )

    layers = decoder_layers(underlying_model)
    if layers is None:
        _reject_unreachable_decoder_layers(model)

    sharded = 0
    split_off: list[str] = []
    if layers is not None:
        for layer in layers:
            split = _uniform_dtype_split(layer, _left_out(layer, root_reserved))
            split_off += [type(module).__name__ for module in split]
            for module in split:
                _shard(module, reserved=root_reserved)
            _shard(layer, reserved=root_reserved)
            if split:
                _unshard_with_layer(layer, split)
            sharded += 1 + len(split)
        _shard(embed_backbone, reserved=root_reserved)
        sharded += 1
        if split_off and is_global_main_process():
            logger.info(
                f"    - {len(split_off)} submodule(s) sharded as their own group for a uniform trainable "
                f"dtype per group: {sorted(set(split_off))}"
            )

    if model is not embed_backbone:
        _shard(model, reserved=IdentityParamSet())
        sharded += 1
    elif layers is None:
        # No reachable layer list: shard the root anyway, or nothing would sync its gradients.
        _shard(model, reserved=IdentityParamSet())
        sharded += 1

    return sharded


def _apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    *,
    compute_dtype: torch.dtype,
    fp32_reduce: bool,
    topology: str,
    ignored_params: list[nn.Parameter] | None = None,
    reshard_after_forward: bool = False,
    fp32_master_weights: bool = False,
) -> None:
    """Wrap ``model`` over ``dp_mesh`` and log it; ``topology`` names the mesh in the caller's terms."""
    mp_policy = mixed_precision_policy(
        compute_dtype,
        fp32_reduce=fp32_reduce or fp32_master_weights,
        cast_forward_inputs=_should_cast_forward_inputs(model),
    )
    label = "FULL_SHARD" if reshard_after_forward else "SHARD_GRAD_OP"

    if is_global_main_process():
        logger.info(f"  Applying FSDP2 for data parallelism ({label}, {topology}):")
        logger.info(f"    - DP group size: {dp_mesh.size()}")
        logger.info(f"    - Mixed precision: {mp_policy.param_dtype}")
        logger.info(f"    - reshard_after_forward: {reshard_after_forward}")
        if not mp_policy.cast_forward_inputs:
            logger.info("    - cast_forward_inputs: False (model maintains an fp32 inter-layer residual)")
        if ignored_params:
            logger.info(f"    - Ignored params: {len(ignored_params)} (EP-module params and frozen dtype exclusions)")

    sharded = apply_fsdp2_per_layer(
        model,
        dp_mesh,
        mp_policy,
        reshard_after_forward,
        IdentityParamSet(ignored_params or ()),
    )

    if is_global_main_process():
        logger.info(f"  ✓ FSDP2 ({label}) applied for data parallelism ({sharded} shard groups)")


def _require_single_device_params(model: nn.Module) -> None:
    """Raise unless every param is on one device. A raise, not a fallback: the check is rank-local and
    the mesh construction after it collective, so one rank skipping would hang the others."""
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        raise RuntimeError(
            f"FSDP2 requires single-device modules, but this rank's model has parameters on "
            f"{sorted(str(d) for d in devices)}. This comes from device_map='auto'/offload; use "
            f"device_map=None under torchrun and let FSDP place the shards."
        )


def apply_fsdp2_for_dp(
    model: nn.Module,
    dp_size: int,
    *,
    compute_dtype: torch.dtype,
    fp32_reduce: bool,
    ignored_params: list[nn.Parameter] | None = None,
    dp_group: dist.ProcessGroup | None = None,
    reshard_after_forward: bool = False,
    fp32_master_weights: bool = False,
    dp_replicate_size: int = 1,
) -> bool:
    """Apply per-layer FSDP2 for data-parallel grad sync in place; returns whether it ran.

    ``ignored_params`` (EP-module params, frozen params of a dtype no trainable one shares) stay out.
    ``dp_replicate_size > 1`` shards within an NVLink domain and replicates across domains (HSDP).
    """
    if dp_size <= 1:
        if is_global_main_process():
            logger.info("  Single GPU — skipping FSDP2")
        return False

    _require_single_device_params(model)

    _apply_fsdp2(
        model,
        build_dp_mesh(dp_size, dp_replicate_size=dp_replicate_size, dp_group=dp_group),
        compute_dtype=compute_dtype,
        fp32_reduce=fp32_reduce,
        topology=(
            f"HSDP {dp_replicate_size}×{dp_size // dp_replicate_size} (replicate×shard)"
            if dp_replicate_size > 1
            else "1D full-shard"
        ),
        ignored_params=ignored_params,
        reshard_after_forward=reshard_after_forward,
        fp32_master_weights=fp32_master_weights,
    )
    return True


def apply_fsdp2_for_tp(
    model: nn.Module,
    tp_size: int,
    *,
    compute_dtype: torch.dtype,
    fp32_reduce: bool,
    dp_size: int,
    ignored_params: list[nn.Parameter] | None = None,
    fp32_master_weights: bool = False,
    reshard_after_forward: bool = False,
) -> tuple[bool, DeviceMesh | None]:
    """Apply FSDP2 over the DP dimension of a TP run; returns ``(fsdp_wrapped, device_mesh)``.

    ``dp_size`` is the caller's ``data_parallel_size``, never ``world // tp_size``, which agrees only
    while every other DP-dividing axis is rejected beside TP. ``ignored_params`` are an EP+TP run's EP
    modules, which sync their own gradients.
    """
    if dp_size <= 1:
        logger.info("  DP=1 under TP: no FSDP2 wrap (DTensor and EP sync carry the gradients)")
        return False, None

    _require_single_device_params(model)

    # A wrongly shaped loader mesh raises rather than being rebuilt: the params stay sharded on it.
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None:
        try:
            device_mesh = build_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"Could not create the (dp={dp_size}, tp={tp_size}) device mesh required for FSDP2 DP grad sync: {e}"
            ) from e
    elif MeshDim.DP not in mesh_dim_names(device_mesh):
        raise RuntimeError(
            f"TP+DP (dp_size={dp_size}) requires a device mesh with a {MeshDim.DP!r} dimension, but the "
            f"model's mesh has {mesh_dim_names(device_mesh)}. Rebuild the mesh via "
            f"build_dp_tp_mesh — training without DP grad sync silently diverges."
        )

    _apply_fsdp2(
        model,
        device_mesh[MeshDim.DP],
        compute_dtype=compute_dtype,
        fp32_reduce=fp32_reduce,
        topology=f"DP×TP {dp_size}×{tp_size}",
        ignored_params=ignored_params,
        reshard_after_forward=reshard_after_forward,
        fp32_master_weights=fp32_master_weights,
    )
    return True, device_mesh


def fsdp2_modules(model: nn.Module) -> list[FSDPModule]:
    """Every ``fully_shard``-wrapped module under ``model``, in ``modules()`` order (a list, so the
    per-window toggle can cache it instead of re-walking the tree every backward)."""
    return [module for module in model.modules() if isinstance(module, FSDPModule)]


def reshard_fsdp2_modules(model: nn.Module) -> None:
    """Re-register every FSDP2 module's sharded DTensor params before a state-dict API call.

    An eval forward leaves the transient UNSHARDED params registered (``reshard_after_forward=False``
    skips the post-forward reshard; the root skips it even at True). ``get_optimizer_state_dict`` maps
    params by identity while the optimizer holds the sharded DTensors (``KeyError: 0``), and
    ``set_model_state_dict`` would write into buffers the next unshard discards. Per-rank; a no-op when
    already sharded.
    """
    for module in fsdp2_modules(model):
        module.reshard()


def install_fsdp2_safe_disable_adapter(peft_model: nn.Module, fsdp_root: nn.Module) -> None:
    """Make ``peft_model.disable_adapter()`` restore ``requires_grad`` on the FSDP2 sharded params.

    peft clears the flag on the adapter params registered at entry and restores it on those registered
    at exit. A reference pass that is the first forward after a reshard enters on the sharded params and
    exits on the unsharded ones that forward left registered, so the sharded adapters would stay frozen.
    When the params in the trainable slots changed, every FSDP2 module under ``fsdp_root`` is resharded
    before peft's exit; a pass that enters and exits on the same params reshards nothing. The slots are
    read once: FSDP2 swaps what a slot holds, never the slot. Per-rank, idempotent, a no-op without FSDP2.
    """
    if getattr(peft_model, "_fsdp2_safe_disable_adapter", False):
        return
    disable_adapter = peft_model.disable_adapter
    trainable_slots = [
        (module, name)
        for module in fsdp_root.modules()
        for name, param in module.named_parameters(recurse=False)
        if param.requires_grad
    ]

    @contextlib.contextmanager
    def _fsdp2_safe_disable_adapter():
        registered_at_entry = [getattr(module, name) for module, name in trainable_slots]
        with disable_adapter():
            try:
                yield
            finally:
                if any(
                    getattr(module, name) is not param
                    for (module, name), param in zip(trainable_slots, registered_at_entry, strict=True)
                ):
                    reshard_fsdp2_modules(fsdp_root)

    peft_model.disable_adapter = _fsdp2_safe_disable_adapter
    peft_model._fsdp2_safe_disable_adapter = True
