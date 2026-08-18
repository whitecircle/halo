"""FSDP v2 (fully_shard) data-parallel gradient sync for all torchrun modes, plus the sharding-state
mechanics its consumers need afterwards (:func:`fsdp2_modules`, :func:`reshard_fsdp2_modules`).

:func:`setup_fsdp2_for_dp` / :func:`setup_fsdp2_for_tp` build the mesh themselves; a caller that
already holds one (the trainer's EP+TP wrap reuses the loader's 2D mesh) composes
:func:`create_mixed_precision_policy_v2` with :func:`apply_fsdp2_per_layer` instead.

``reshard_after_forward``: False (default) = SHARD_GRAD_OP (faster, higher peak memory);
True = FULL_SHARD (lower peak memory, slightly slower).
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import is_peft_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers.modeling_utils import PreTrainedModel

from src.distributed.mesh import MeshDim, create_dp_mesh, create_dp_tp_mesh, mesh_dim_names
from src.distributed.runtime import is_global_main_process
from src.models.structure import (
    DECODER_LAYER_LIST_ATTRS,
    backbone_with_layers,
    decoder_layers,
    fp32_pinned_param_names,
    unwrap_model,
)

logger = logging.getLogger(__name__)


def reshard_label(reshard_after_forward: bool) -> str:
    """FSDP2 sharding strategy under this ``reshard_after_forward``, as the setup logs name it."""
    return "FULL_SHARD" if reshard_after_forward else "SHARD_GRAD_OP"


class IdentityParamSet:
    """id()-based membership set, avoiding tensor ``__eq__`` shape-comparison errors.

    FSDP2's ``param not in ignored_params`` triggers tensor ``__eq__`` on hash collisions,
    raising for different-shaped tensors (EP fused 3D expert weights vs standard 2D).
    """

    def __init__(self, params=()):
        self._ids = {id(p) for p in params}

    def __contains__(self, item):
        return id(item) in self._ids

    def __bool__(self):
        return bool(self._ids)


def _require_single_device_params(model: nn.Module) -> None:
    """Raise unless every param is on one device (FSDP requires single-device modules).

    A raise, not a fallback: the predicate is RANK-LOCAL while the mesh construction that follows is
    collective, so one rank bailing out both hangs the others in ``split_group`` and leaves itself
    unsharded with no gradient sync, under a caller that still reports success.
    """
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        raise RuntimeError(
            f"FSDP2 requires single-device modules, but this rank's model has parameters on "
            f"{sorted(str(d) for d in devices)}. This comes from device_map='auto'/offload; use "
            f"device_map=None under torchrun and let FSDP place the shards."
        )


def _get_underlying_model(model: nn.Module) -> nn.Module:
    """The transformer backbone to shard, falling back to ``model`` itself when none is reachable.

    Layout knowledge lives in :func:`backbone_with_layers`; the fallback is local because FSDP must
    still wrap something when a model exposes no recognizable layer list.
    """
    return backbone_with_layers(model) or model


def _reject_unreachable_decoder_layers(model: nn.Module) -> None:
    """Raise when a generative decoder exposes no decoder-layer list :func:`decoder_layers` reaches.

    Without one the whole model becomes a SINGLE shard group, all-gathered for the entire forward —
    the per-DP-rank memory ceiling FSDP2 exists to remove — and the wrap still reports success. The
    gate is the class hierarchy: the layer-less shapes this wrap legitimately serves (a
    SentenceTransformer, a classification backbone, a PP stage) are not generative decoders.
    """
    inner = unwrap_model(model)
    if is_peft_model(inner):
        inner = unwrap_model(inner.get_base_model())
    if not (isinstance(inner, PreTrainedModel) and inner.can_generate()):
        return
    raise RuntimeError(
        f"FSDP2 found no decoder-layer list on {type(inner).__name__}: none of "
        f"{DECODER_LAYER_LIST_ATTRS} is reachable from it, so the whole model would be wrapped as one "
        f"shard group and all-gathered for the entire forward instead of per layer. Add this family's "
        f"spelling to DECODER_LAYER_LIST_ATTRS (src/models/structure.py), which every "
        f"layer-list consumer reads."
    )


def setup_fsdp2_for_dp(
    model: nn.Module,
    dp_size: int,
    args,
    ignored_params: list[nn.Parameter] | None = None,
    dp_group: dist.ProcessGroup | None = None,
    reshard_after_forward: bool = False,
    fp32_master_weights: bool = False,
    dp_replicate_size: int = 1,
) -> bool:
    """Apply per-layer FSDP v2 for data-parallel grad sync (in-place); returns whether it ran.

    Params shard across DP ranks; ``ignored_params`` (EP modules) handle their own sync.
    ``dp_replicate_size > 1`` builds a 2D HSDP mesh (shard within an NVLink domain, replicate
    across domains — one inter-domain grad all-reduce/step); 1 keeps the 1D full-shard mesh.
    """
    if dp_size <= 1:
        if is_global_main_process():
            logger.info("  Single GPU — skipping FSDP2")
        return False

    _require_single_device_params(model)

    _apply_fsdp2(
        model,
        create_dp_mesh(dp_size, dp_replicate_size=dp_replicate_size, dp_group=dp_group),
        args,
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


def _apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    args,
    *,
    topology: str,
    ignored_params: list[nn.Parameter] | None = None,
    reshard_after_forward: bool = False,
    fp32_master_weights: bool = False,
) -> None:
    """Wrap ``model`` over ``dp_mesh`` and log what it did — the one body both entry points share.

    ``topology`` names the mesh in the caller's own terms (full-shard/HSDP, or the DP×TP grid);
    everything else is identical, and a second copy is how the two paths drift on a policy knob.
    """
    mp_policy = create_mixed_precision_policy_v2(
        args,
        fp32_master_weights=fp32_master_weights,
        cast_forward_inputs=_should_cast_forward_inputs(model),
    )
    label = reshard_label(reshard_after_forward)

    if is_global_main_process():
        logger.info(f"  Applying FSDP2 for data parallelism ({label}, {topology}):")
        logger.info(f"    - DP group size: {dp_mesh.size()}")
        mp_label = "bf16" if getattr(args, "bf16", False) else ("fp16" if getattr(args, "fp16", False) else "fp32")
        logger.info(f"    - Mixed precision: {mp_label}")
        logger.info(f"    - reshard_after_forward: {reshard_after_forward}")
        if mp_policy is not None and not mp_policy.cast_forward_inputs:
            logger.info("    - cast_forward_inputs: False (model maintains an fp32 inter-layer residual)")
        if ignored_params:
            logger.info(f"    - Ignored params: {len(ignored_params)} (EP modules)")

    sharded = apply_fsdp2_per_layer(
        model,
        dp_mesh,
        mp_policy,
        reshard_after_forward,
        IdentityParamSet(ignored_params or ()),
    )

    if is_global_main_process():
        logger.info(f"  ✓ FSDP2 ({label}) applied for data parallelism ({sharded} shard groups)")


def _warn_fp32_pins_cast_by_policy(model: nn.Module, mp_policy: MixedPrecisionPolicy | None) -> None:
    """Warn (once, at wrap time) when the FSDP2 policy will compute fp32-pinned params in low precision.

    ``MixedPrecisionPolicy`` casts per fully_shard group with no per-parameter dtype mechanism, so
    fp32-pinned params run forward/backward at ``param_dtype`` under multi-GPU where a single-GPU run
    honors fp32 (storage and optimizer state stay fp32 either way). The pins come from the
    transformers class attributes, so a family without them — or an EP load that materializes uniform
    bf16 — warns about nothing.
    """
    if mp_policy is None or mp_policy.param_dtype in (None, torch.float32):
        return
    pinned = fp32_pinned_param_names(model)
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
    """Parameters reachable under more than one name (``tie_word_embeddings``).

    ``fully_shard`` rebinds a shared parameter only among the modules of the *same* call, so a tie
    whose two names straddle two calls is silently split into two independent parameters that each
    receive only their own half of the true gradient and diverge from step 1.
    """
    seen: set[int] = set()
    tied = []
    for _name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            tied.append(param)
        seen.add(id(param))
    return IdentityParamSet(tied)


def apply_fsdp2_per_layer(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    reshard_after_forward: bool,
    ignored_params: "IdentityParamSet",
) -> int:
    """Apply FSDP v2 per transformer layer, then to the model root. Returns the shard-group count.

    EP module params pass as ``ignored_params``. Every branch below wraps at least the root, so the
    count is a log line rather than a guard — what must be caught is a decoder whose layer list this
    probe cannot reach, which is :func:`_reject_unreachable_decoder_layers`.
    """
    _warn_fp32_pins_cast_by_policy(model, mp_policy)
    underlying_model = _get_underlying_model(model)

    # Shard the module whose forward CONSUMES ``embed_tokens``: wrappers that build ``inputs_embeds``
    # in the parent while decoder layers sit in a nested ``language_model`` unshard it too late.
    if hasattr(model, "model") and getattr(model.model, "language_model", None) is underlying_model:
        embed_backbone = model.model
    else:
        embed_backbone = underlying_model

    # A tied weight spans two `_shard` calls; reserve it for the root call or fully_shard silently
    # severs the tie, leaving each half with half the gradient.
    root_reserved = _tied_parameters(model) if model is not embed_backbone else IdentityParamSet()

    def _shard(module: nn.Module, *, reserved: "IdentityParamSet") -> None:
        scoped = IdentityParamSet(p for p in module.parameters() if p in ignored_params or p in reserved)
        fully_shard(
            module,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            **({"ignored_params": scoped} if scoped else {}),
        )

    # Through decoder_layers, so a backbone spelling reaches this wrap as soon as it is registered.
    layers = decoder_layers(underlying_model)
    if layers is None:
        _reject_unreachable_decoder_layers(model)

    sharded = 0
    if layers is not None:
        for layer in layers:
            _shard(layer, reserved=root_reserved)
            sharded += 1
        _shard(embed_backbone, reserved=root_reserved)
        sharded += 1

    if model is not embed_backbone:
        _shard(model, reserved=IdentityParamSet())
        sharded += 1
    elif layers is None:
        # No reachable layer list (a SentenceTransformer, a classification backbone): shard the root
        # anyway. The caller reports success and suppresses the DDP fallback, so leaving it unsharded
        # is not a missed optimization but no gradient sync at all.
        _shard(model, reserved=IdentityParamSet())
        sharded += 1

    return sharded


def setup_fsdp2_for_tp(
    model: nn.Module,
    tp_size: int,
    args,
    *,
    dp_size: int,
    fp32_master_weights: bool = False,
    reshard_after_forward: bool = False,
) -> tuple[bool, DeviceMesh | None]:
    """Apply FSDP v2 over the DP dimension of a TP run (when ``dp_size > 1``).

    ``dp_size`` is the caller's ``ParallelismConfig.data_parallel_size``, never ``world // tp_size``:
    the two agree only while every other axis that divides DP is rejected alongside TP, so deriving
    it here would silently double the DP mesh the day such a combination is allowed and shard params
    over ranks holding a different batch.

    Returns ``(fsdp_wrapped, device_mesh)``. ``fp32_master_weights`` reduces DP grads in fp32.
    """
    if dp_size <= 1:
        logger.info("  TP mode: Pure TP (no DP), skipping FSDP2")
        return False, None

    # Rank-local, so it has to run before the collective mesh construction below.
    _require_single_device_params(model)

    # The loader's mesh is taken as-is, or built here when there is none; a wrongly shaped one RAISES
    # rather than being rebuilt around, since the params stay sharded on the mesh that made them.
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None:
        try:
            device_mesh = create_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"Could not create the (dp={dp_size}, tp={tp_size}) device mesh required for FSDP2 DP grad sync: {e}"
            ) from e
    elif MeshDim.DP not in mesh_dim_names(device_mesh):
        raise RuntimeError(
            f"TP+DP (dp_size={dp_size}) requires a device mesh with a {MeshDim.DP!r} dimension, but the "
            f"model's mesh has {mesh_dim_names(device_mesh)}. Rebuild the mesh via "
            f"create_dp_tp_mesh — training without DP grad sync silently diverges."
        )

    _apply_fsdp2(
        model,
        device_mesh[MeshDim.DP],
        args,
        topology=f"DP×TP {dp_size}×{tp_size}",
        reshard_after_forward=reshard_after_forward,
        fp32_master_weights=fp32_master_weights,
    )
    if is_global_main_process():
        logger.info("✓ TP mode configured with FSDP2 for DP")

    return True, device_mesh


def create_mixed_precision_policy_v2(
    args,
    fp32_master_weights: bool = False,
    cast_forward_inputs: bool = True,
) -> MixedPrecisionPolicy | None:
    """Create FSDP v2 MixedPrecisionPolicy from training args.

    ``fp32_master_weights=True``: params stored fp32, cast to compute dtype for forward/backward,
    grads reduced in fp32. ``cast_forward_inputs=False`` for models carrying an fp32 activation
    across FSDP layers (see :func:`_should_cast_forward_inputs`).
    """
    if getattr(args, "bf16", False):
        compute_dtype = torch.bfloat16
    elif getattr(args, "fp16", False):
        compute_dtype = torch.float16
    else:
        # torch 2.11's fully_shard REQUIRES a non-None policy; an all-fp32 one is a functional no-op.
        return MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            cast_forward_inputs=cast_forward_inputs,
        )

    # Summing many bf16 grads loses ~10^4x precision vs fp32 (TorchTitan/Megatron/DeepSpeed default).
    fp32_grad_reduce = getattr(args, "fp32_grad_reduce", False)
    reduce_dtype = torch.float32 if (fp32_master_weights or fp32_grad_reduce) else compute_dtype
    return MixedPrecisionPolicy(
        param_dtype=compute_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )


def _should_cast_forward_inputs(model: nn.Module) -> bool:
    """Return False for models that pass an fp32 activation between FSDP layers.

    ``cast_forward_inputs=True`` casts layer-boundary activations to bf16, which breaks GC with
    ``use_reentrant=False`` where a model carries an fp32 tensor across layers (Zaya's residual):
    forward saves the bf16 view, recompute keeps fp32, torch rejects it.

    Read off the module tree, from the ``_fp32_interlayer_residual`` class attribute the owning layer
    declares — a model_type list would go stale, and its failure is a torch error deep inside a
    recompute rather than a message about this policy.
    """
    return not any(getattr(type(module), "_fp32_interlayer_residual", False) for module in model.modules())


def fsdp2_modules(model: nn.Module) -> list[FSDPModule]:
    """Every ``fully_shard``-wrapped module under ``model``, in ``modules()`` order.

    Materialized rather than yielded: the per-microstep ``set_reshard_after_backward`` toggle
    (:meth:`~src.trainers.mixins.grad_sync.GradientSyncMixin._set_backward_reshard`) caches this
    once instead of re-walking a 400B module tree every backward.
    """
    return [module for module in model.modules() if isinstance(module, FSDPModule)]


def reshard_fsdp2_modules(model: nn.Module) -> None:
    """Re-register every FSDP2 module's sharded DTensor params before a state-dict API call.

    An eval-only forward leaves the transient UNSHARDED plain params registered (no backward to
    reshard, and the toolkit's ``reshard_after_forward=False`` skips the post-forward one; the ROOT
    module skips it even at ``True``), and HF evaluates immediately before the end-of-training save.
    ``get_optimizer_state_dict`` then maps params to FQNs by IDENTITY against ``named_parameters()``
    while the optimizer holds the sharded DTensors, so every FSDP2 param goes unmapped
    (``KeyError: 0``), and ``set_model_state_dict`` would write into buffers the next unshard
    discards. ``reshard()`` is per-rank and a no-op when already sharded.
    """
    for module in fsdp2_modules(model):
        module.reshard()
