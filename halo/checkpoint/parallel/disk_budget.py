"""The disk a training save needs against the free space of its output volume, judged before the first step."""

import copy
import logging

import torch
from transformers import PreTrainedModel
from transformers.trainer_utils import sort_checkpoints

from halo.checkpoint.parallel.context import CheckpointContext
from halo.checkpoint.parallel.save import saves_adapters_only
from halo.hardware import free_disk_bytes
from halo.model_loading.lazy_safetensors.meta_shell import MetaShellBuildError, config_only_parameter_bytes
from halo.models.introspection.unwrapping import base_transformers_model
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.dtensor import local_numel
from halo.runtime.filesystem import is_output_shared_filesystem
from halo.runtime.ranks import get_num_nodes

logger = logging.getLogger(__name__)

# Free space under this multiple of the estimate warns; under 1.0 it refuses.
_WARN_MARGIN = 1.1
# Optimizer state per trainable parameter, in parameter-sized elements (two moments).
_OPTIMIZER_STATE_ELEMENTS = 2


def estimate_model_bytes(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """Bytes of the whole model at ``dtype``. A transformers model is sized from a config-only meta shell, since the
    live parameters are shards (FSDP2, EP, PP stages) no rank can sum alone; any other model is held whole. Raises
    :class:`MetaShellBuildError` for a model the shell cannot build."""
    base = base_transformers_model(model)
    if isinstance(base, PreTrainedModel):
        return config_only_parameter_bytes(type(base), copy.deepcopy(base.config), dtype, trust_remote_code=False)
    return sum(p.numel() for p in model.parameters()) * torch.empty((), dtype=dtype).element_size()


def estimate_checkpoint_bytes(
    ctx: CheckpointContext, dtype: torch.dtype, *, save_optimizer: bool, writes_optimizer_shards: bool
) -> int:
    """Bytes one training checkpoint puts on this rank's output filesystem: the weights (the adapters alone where a
    PEFT save writes only those), plus two moments per trainable parameter when the optimizer is saved. Per-rank
    optimizer shards divide over the nodes of a node-local filesystem. An upper bound for a sharded EP save there,
    which also splits the weights."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    params = list(ctx.model.parameters())
    if saves_adapters_only(ctx):
        weights = trainable = sum(p.numel() for p in params if p.requires_grad) * itemsize
    else:
        weights = estimate_model_bytes(ctx.model, dtype)
        live_total = sum(local_numel(p) for p in params)
        live_trainable = sum(local_numel(p) for p in params if p.requires_grad)
        trainable = weights * live_trainable // max(live_total, 1)
    optimizer = _OPTIMIZER_STATE_ELEMENTS * trainable if save_optimizer else 0
    if writes_optimizer_shards and not is_output_shared_filesystem():
        optimizer //= get_num_nodes()
    return weights + optimizer


def verify_checkpoint_disk_space(
    ctx: CheckpointContext,
    output_dir: str,
    dtype: torch.dtype,
    *,
    save_total_limit: int | None,
    save_optimizer: bool,
    writes_optimizer_shards: bool,
) -> None:
    """COLLECTIVE. Refuse, on every rank, a run whose output volume cannot hold its checkpoints; warn when it holds
    them with under 10% to spare. Each filesystem's save rank judges its own volume. Rotation deletes the oldest only
    after the new save completes, so ``save_total_limit`` + 1 checkpoints coexist at the peak; the ``checkpoint-N``
    directories already in ``output_dir`` (a resume) hold their share of that peak, so at least the next save is
    still billed. With no limit nothing is deleted and only the next save is checked."""
    reason = None
    if ctx.is_save_rank:
        try:
            per_checkpoint = estimate_checkpoint_bytes(
                ctx, dtype, save_optimizer=save_optimizer, writes_optimizer_shards=writes_optimizer_shards
            )
        except MetaShellBuildError as exc:  # an estimate must not take down a run whose model it cannot size
            logger.warning(f"Could not size a checkpoint to check the free disk space of {output_dir}: {exc!r}")
            per_checkpoint = None
        if per_checkpoint is not None:
            if save_total_limit is None:
                copies = 1
            else:
                copies = max(1, save_total_limit + 1 - len(sort_checkpoints(output_dir)))
            needed, free = copies * per_checkpoint, free_disk_bytes(output_dir)
            summary = (
                f"{copies} more checkpoint(s) of ~{per_checkpoint / 1e9:.0f} GB need ~{needed / 1e9:.0f} GB at "
                f"the rotation peak, and the volume of {output_dir} has ~{free / 1e9:.0f} GB free"
            )
            if free < needed:
                reason = (
                    f"{summary}: the save would fail with ENOSPC. Lower save_total_limit, set save_only_model: "
                    f"true to drop the optimizer state, or write to a larger volume."
                )
            elif free < _WARN_MARGIN * needed:
                logger.warning(f"{summary}, under a {_WARN_MARGIN - 1:.0%} margin.")
    reject_across_ranks(reason, "Checking the free disk space for checkpoints", exc_type=ValueError)
