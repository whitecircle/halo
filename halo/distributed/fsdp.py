"""FSDP2 (``fully_shard``) data-parallel gradient sync for every torchrun mode, plus sharding-state helpers. A
multi-layer unit's hooks and :func:`complete_fsdp2_post_backward` drive FSDP2 internals of the installed torch."""

import contextlib
import functools
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState, _cast_fp_tensor, is_bw
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import disable_if_config_true
from torch.distributed.tensor import DTensor
from torch.distributed.utils import _apply_to_tensors
from transformers.modeling_utils import PreTrainedModel

from halo.distributed.mesh import MeshDim, build_dp_mesh, build_dp_tp_mesh, mesh_dim_names
from halo.models.module_tree import (
    DECODER_LAYER_LIST_ATTRS,
    backbone_with_layers,
    decoder_layers,
    input_embedding_backbone,
)
from halo.models.param_roles import FP32_INTERLAYER_RESIDUAL_ATTR, fp32_pinned_names
from halo.models.unwrapping import base_transformers_model
from halo.runtime.collectives import nccl_safe_broadcast
from halo.runtime.dtensor import to_local
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Decoder layers per FSDP2 shard group under ZeRO-2; a group all-gathers and reduce-scatters once for all of them.
# ZeRO-2 keeps every layer gathered from forward to backward anyway; under ZeRO-3 FSDP2 holds the running group and its
# prefetch, so units would multiply the gathered window ZeRO-3 exists to bound.
LAYERS_PER_UNIT = 2


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
    """FSDP2 policy computing in ``compute_dtype``, reducing in fp32 under ``fp32_reduce`` (a bf16 sum loses ~10^4x
    precision). ``fully_shard`` requires a non-None policy, so an fp32 run gets a no-op all-fp32 one."""
    return MixedPrecisionPolicy(
        param_dtype=compute_dtype,
        reduce_dtype=torch.float32 if fp32_reduce else compute_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )


def _should_cast_forward_inputs(model: nn.Module) -> bool:
    """False when a layer class declares :data:`FP32_INTERLAYER_RESIDUAL_ATTR` (Zaya): a bf16 cast at the layer
    boundary makes non-reentrant GC save bf16 where the recompute keeps fp32, which torch rejects."""
    return not any(getattr(type(module), FP32_INTERLAYER_RESIDUAL_ATTR, False) for module in model.modules())


def _reject_unreachable_decoder_layers(model: nn.Module) -> None:
    """Raise when a generative decoder exposes no decoder-layer list, since the whole model would be one shard group,
    all-gathered for the entire forward. Layer-less shapes this wrap serves (a SentenceTransformer, a classification
    backbone, a PP stage) do not generate."""
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
    """Warn when the policy will compute fp32-pinned params in low precision: ``MixedPrecisionPolicy`` casts a whole
    group with no per-parameter dtype, where a single-GPU run keeps fp32."""
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
    """Submodules of ``layer`` needing their own ``fully_shard`` group, innermost first, so every group's trainable
    params share one dtype (FSDP2 asserts it; an ``fp32_router`` breaks it). The layer keeps the dtype needing fewest
    extra groups; ``left_out`` holds params no group takes. Raises where one module directly owns two dtypes."""

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
    """Unshard the ``split`` groups whenever ``layer`` runs forward, since a family can read a submodule's parameters
    without calling it (``F.linear(x, self.gate.weight)``), skipping its own pre-hook. The root post-backward callback
    still reduces their gradients."""

    def _pre_forward(_module: nn.Module, _args: tuple) -> None:
        for module in split:
            module.unshard()

    layer.register_forward_pre_hook(_pre_forward)


def _layer_units(
    layers: nn.ModuleList,
    splits: list[list[nn.Module]],
    left_out: list["IdentityParamSet"],
    layers_per_unit: int,
) -> list[list[int]]:
    """Indices of consecutive decoder layers, up to ``layers_per_unit`` per unit. A layer whose trainable dtype
    (outside ``splits`` and ``left_out``) differs from its unit's opens the next, since FSDP2 asserts one per group."""
    units: list[list[int]] = []
    unit_dtypes: set[torch.dtype] = set()
    for index, layer in enumerate(layers):
        in_split = IdentityParamSet(p for module in splits[index] for p in module.parameters())
        dtypes = {
            p.dtype for p in layer.parameters() if p.requires_grad and p not in left_out[index] and p not in in_split
        }
        if units and len(units[-1]) < layers_per_unit and len(unit_dtypes | dtypes) <= 1:
            units[-1].append(index)
            unit_dtypes |= dtypes
        else:
            units.append([index])
            unit_dtypes = dtypes
    return units


def layers_per_shard_group(reshard_after_forward: bool) -> int:
    return 1 if reshard_after_forward else LAYERS_PER_UNIT


def _install_unit_hooks(unit: list[nn.Module]) -> None:
    """Replace torch's forward hooks on a multi-layer ``fully_shard`` group so its layers train as in groups of their
    own, per-layer GC included. torch starts a group at whichever module runs first, in a reentrant recompute its last
    layer, whose post-backward would reduce and reshard the group before the layers below ran backward. Here the first
    layer's pre-forward and the last's post-forward bracket the forward, and the first layer's recompute registers the
    backward hooks a reentrant forward under ``no_grad`` could not. Each later layer casts its inputs as its own group
    would and, in a recompute, re-gathers what a reshard after forward freed."""
    state = unit[0]._get_fsdp_state()
    group = state._fsdp_param_group
    if group is None:
        return
    # One handle covers torch's pre- and post-forward hooks on every module of the group.
    state._pre_forward_hook_handle.remove()
    first, last = unit[0], unit[-1]
    policy = state._mp_policy
    cast = (
        functools.partial(_cast_fp_tensor, policy.param_dtype)
        if policy.cast_forward_inputs and policy.param_dtype
        else None
    )

    @disable_if_config_true
    def first_pre_forward(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        args, kwargs = state._pre_forward(module, args, kwargs)
        if is_bw() and group._training_state == TrainingState.PRE_BACKWARD:
            # PRE_BACKWARD here means a reentrant checkpoint around this layer alone, whose forward registered nothing.
            # A non-reentrant recompute builds a graph that never runs backward, so the hook is inert there.
            args, kwargs = group._register_post_backward_hook(args, kwargs)
        return args, kwargs

    @disable_if_config_true
    def first_post_forward(module: nn.Module, args: tuple, output):
        return state._post_forward(module, args, output) if is_bw() else output

    @disable_if_config_true
    def later_pre_forward(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        # FSDP2's pre-forward casts unless the group is in PRE_BACKWARD, as in a non-reentrant recompute.
        if cast is not None and state._training_state != TrainingState.PRE_BACKWARD:
            args, kwargs = _apply_to_tensors(cast, args), _apply_to_tensors(cast, kwargs)
        if is_bw() and not group.is_unsharded:
            group.unshard()
            group.wait_for_unshard()
        return args, kwargs

    @disable_if_config_true
    def last_post_forward(module: nn.Module, args: tuple, output):
        return output if is_bw() else state._post_forward(module, args, output)

    first.register_forward_pre_hook(first_pre_forward, prepend=True, with_kwargs=True)
    first.register_forward_hook(first_post_forward)
    for layer in unit[1:]:
        layer.register_forward_pre_hook(later_pre_forward, prepend=True, with_kwargs=True)
    last.register_forward_hook(last_post_forward)


def apply_fsdp2_per_layer(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    ignored_params: "IdentityParamSet",
    *,
    layers_per_unit: int | None = None,
) -> int:
    """Apply FSDP2 per unit of consecutive decoder layers, then to the backbone and the root; returns the group count.
    A layer mixing trainable dtypes first gets nested groups."""
    _warn_fp32_pins_cast_by_policy(model, mp_policy)
    underlying_model = backbone_with_layers(model) or model

    # Shard the module whose forward consumes ``embed_tokens``: a group on a nested backbone unshards
    # it after the composite parent above has already embedded the ids.
    embed_backbone = input_embedding_backbone(model) or underlying_model

    # A tied weight spans two `_shard` calls; reserve it for the root call or fully_shard silently
    # severs the tie, leaving each half with half the gradient.
    root_reserved = _tied_parameters(model) if model is not embed_backbone else IdentityParamSet()

    def _left_out(modules: tuple[nn.Module, ...], reserved: "IdentityParamSet") -> "IdentityParamSet":
        return IdentityParamSet(
            p for module in modules for p in module.parameters() if p in ignored_params or p in reserved
        )

    def _shard(*modules: nn.Module, reserved: "IdentityParamSet") -> None:
        scoped = _left_out(modules, reserved)
        fully_shard(
            modules[0] if len(modules) == 1 else list(modules),
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
        left_out = [_left_out((layer,), root_reserved) for layer in layers]
        splits = [_uniform_dtype_split(layer, scoped) for layer, scoped in zip(layers, left_out, strict=True)]
        per_unit = layers_per_shard_group(reshard_after_forward) if layers_per_unit is None else layers_per_unit
        for unit in _layer_units(layers, splits, left_out, per_unit):
            unit_layers = [layers[index] for index in unit]
            for index in unit:
                split_off += [type(module).__name__ for module in splits[index]]
                for module in splits[index]:
                    _shard(module, reserved=root_reserved)
            _shard(*unit_layers, reserved=root_reserved)
            if len(unit_layers) > 1:
                _install_unit_hooks(unit_layers)
            for index in unit:
                if splits[index]:
                    _unshard_with_layer(layers[index], splits[index])
            sharded += 1 + sum(len(splits[index]) for index in unit)
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
        logger.info(f"    - Decoder layers per shard group: {layers_per_shard_group(reshard_after_forward)}")
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
    """Raise unless every param is on one device. The check is rank-local and the mesh construction after it
    collective, so one rank skipping instead would hang the others."""
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        raise RuntimeError(
            f"FSDP2 requires single-device modules, but this rank's model has parameters on "
            f"{sorted(str(d) for d in devices)}. This comes from device_map='auto'/offload; use "
            f"device_map=None under torchrun and let FSDP place the shards."
        )


def _broadcast_from_first_coordinate(tensor: torch.Tensor, mesh: DeviceMesh, dims: list[int]) -> None:
    """COLLECTIVE over ``mesh``'s groups along ``dims``, in order: ``tensor`` takes each group's first copy."""
    wire = tensor if tensor.is_contiguous() else tensor.contiguous()
    for dim in dims:
        group = mesh.get_group(dim)
        nccl_safe_broadcast(wire, src=dist.get_global_rank(group, 0), group=group)
    if wire is not tensor:
        tensor.copy_(wire)


def _start_replicas_identical(model: nn.Module, dp_mesh: DeviceMesh, replicated: IdentityParamSet) -> int:
    """COLLECTIVE over ``dp_mesh``: every parameter more than one DP rank holds a copy of takes the first copy; returns
    how many this rank broadcast. FSDP2, unlike DDP, broadcasts no module state and those copies only average
    gradients, so a tensor drawn per rank (a LoRA adapter, a new head row) stays different all run."""
    broadcast = 0
    with torch.no_grad():
        for param in model.parameters():
            data = param.data
            if isinstance(data, DTensor) and data.device_mesh == dp_mesh:
                tensor = to_local(data)
                dims = [dim for dim, placement in enumerate(data.placements) if placement.is_replicate()]
            elif param in replicated:
                tensor, dims = data, list(range(dp_mesh.ndim))
            else:
                continue
            # A group shares its shard index, so an empty shard (dim 0 narrower than the shard group) is empty on each.
            if dims and tensor.numel():
                _broadcast_from_first_coordinate(tensor, dp_mesh, dims)
                broadcast += 1
    return broadcast


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
    ignored_params_replicated: bool = False,
) -> bool:
    """Wrap ``model`` per unit of decoder layers for data-parallel grad sync in place; returns whether it ran.
    ``ignored_params`` (EP modules, frozen params of a dtype no trainable one shares) stay out;
    ``ignored_params_replicated`` declares each a whole copy on every DP rank. ``dp_replicate_size > 1`` shards within
    a domain, replicates across (HSDP)."""
    if dp_size <= 1:
        if is_global_main_process():
            logger.info("  Single GPU — skipping FSDP2")
        return False

    _require_single_device_params(model)

    dp_mesh = build_dp_mesh(dp_size, dp_replicate_size=dp_replicate_size, dp_group=dp_group)
    _apply_fsdp2(
        model,
        dp_mesh,
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
    replicated = IdentityParamSet((ignored_params or ()) if ignored_params_replicated else ())
    broadcast = _start_replicas_identical(model, dp_mesh, replicated)
    if broadcast and is_global_main_process():
        logger.info(f"  ✓ {broadcast} replicated parameters start from their first copy")
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
    """Wrap the DP dimension of a TP run; returns ``(fsdp_wrapped, device_mesh)``. ``dp_size`` is the caller's
    ``data_parallel_size``, never ``world // tp_size``. ``ignored_params`` are EP modules, which sync themselves."""
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
    """Every ``fully_shard``-wrapped module under ``model``, as a list so the per-window toggle can cache it."""
    return [module for module in model.modules() if isinstance(module, FSDPModule)]


def reshard_fsdp2_modules(model: nn.Module) -> None:
    """Re-register every FSDP2 module's sharded DTensor params before a state-dict API call (per-rank). An eval forward
    leaves the transient unsharded params registered: ``get_optimizer_state_dict`` maps params by identity
    (``KeyError: 0``) and ``set_model_state_dict`` would write into buffers the next unshard discards."""
    for module in fsdp2_modules(model):
        module.reshard()


def fsdp2_param_groups(model: nn.Module) -> list[FSDPParamGroup]:
    """Each ``fully_shard`` module's parameter group, once each (a multi-layer unit's layers share one)."""
    groups = (module._get_fsdp_state()._fsdp_param_group for module in fsdp2_modules(model))
    return list({id(group): group for group in groups if group is not None}.values())


def _holds_unsharded_grad(group: FSDPParamGroup) -> bool:
    """Whether a parameter of ``group`` holds a gradient FSDP2 has not reduce-scattered onto its shard: autograd's
    on the unsharded parameter, or the accumulator a ``set_requires_gradient_sync(False)`` microstep keeps."""
    return any(
        hasattr(fsdp_param, "_unsharded_param")
        and (fsdp_param.unsharded_param.grad is not None or fsdp_param.unsharded_accumulated_grad is not None)
        for fsdp_param in group.fsdp_params
    )


def complete_fsdp2_post_backward(groups: list[FSDPParamGroup]) -> int:
    """COLLECTIVE where a group reduces: run FSDP2's post-backward on every group a finished backward left gathered or
    holding an unsharded gradient, then make the current stream wait on the reductions; returns how many
    reduce-scattered. Which groups reduce follows rank-local gradient state, so every rank must run each layer equally
    often per backward. A one-module group that reentrant checkpointing recomputes K>1 times in one backward gets
    FSDP2's post-backward hook in only one recompute. The others find it PRE_BACKWARD, re-gather it and accumulate into
    the unsharded gradient, and the root's final callback skips it as POST_BACKWARD: K-1 contributions never reach the
    sharded gradient, and the gathered params outlive the optimizer step. A multi-layer unit's group is finished by the
    callback's later visit. Since ``post_backward`` is idempotent and honours the window flags, this is a no-op
    wherever every recompute was hooked."""
    pending = [group for group in groups if group.is_unsharded or _holds_unsharded_grad(group)]
    busy = [
        f"{group!r}: {group._training_state.name}" for group in pending if group._training_state != TrainingState.IDLE
    ]
    if busy:
        raise RuntimeError(
            f"complete_fsdp2_post_backward ran while FSDP2 groups were mid-pass ({busy[:4]}): it finishes a "
            f"backward whose root final callback has already run, never one in flight."
        )
    reducing = [group for group in pending if group.reduce_grads and _holds_unsharded_grad(group)]
    for group in pending:
        with group.use_training_state(TrainingState.IDLE):
            group.post_backward()
    # The root final callback's tail: each group's reduce and HSDP all-reduce, then the last reduce-scatter,
    # whose input buffer stays referenced until the stream has consumed it.
    for group in pending:
        group.finalize_backward()
    for comm_ctx in {id(group.comm_ctx): group.comm_ctx for group in pending}.values():
        reduce_scatter = comm_ctx.reduce_scatter_state
        if reduce_scatter is None:
            continue
        if reduce_scatter.event is not None:
            comm_ctx.device_handle.current_stream().wait_event(reduce_scatter.event)
        comm_ctx.reduce_scatter_state = None
    return len(reducing)


def reject_unfinished_fsdp2_backward(groups: list[FSDPParamGroup]) -> None:
    """Raise if a group's post-backward is undone after :func:`complete_fsdp2_post_backward`: an unreduced gradient is
    missing from what the clip and optimizer read, and a group left gathered feeds the next forward params the step
    never reached. Rank-local, so a leftover on some ranks raises there while peers wait in the next collective."""
    unreduced = [repr(group) for group in groups if group.reduce_grads and _holds_unsharded_grad(group)]
    gathered = [repr(group) for group in groups if group.reshard_after_backward and group.is_unsharded]
    if not unreduced and not gathered:
        return
    raise RuntimeError(
        f"FSDP2 post-backward left undone after complete_fsdp2_post_backward: {len(unreduced)} group(s) still "
        f"hold an unreduced gradient {unreduced[:4]} and {len(gathered)} armed to reshard are still gathered "
        f"{gathered[:4]}. Training on would step sharded gradients missing those contributions. Either a "
        f"backward ran outside the trainer's training_step, or the FSDP2 internals the completion drives "
        f"(FSDPParamGroup.post_backward and the unsharded gradients it reduces) changed in torch "
        f"{torch.__version__}."
    )


def install_fsdp2_safe_disable_adapter(peft_model: nn.Module, fsdp_root: nn.Module) -> None:
    """Make ``peft_model.disable_adapter()`` restore ``requires_grad`` on the FSDP2 sharded params. peft clears the
    flag on the params registered at entry and restores it on those registered at exit; a reference pass that is the
    first forward after a reshard enters on sharded and exits on unsharded params, leaving the sharded adapters frozen.
    So when the trainable slots' params changed, every FSDP2 module is resharded before peft's exit. The slots are read
    once, since FSDP2 swaps what a slot holds, never the slot. Per-rank and idempotent."""
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
