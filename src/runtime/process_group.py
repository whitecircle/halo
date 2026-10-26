"""Default process-group construction for a torchrun / SLURM launch, with the NCCL watchdog timeout
pinned for every later subgroup."""

import datetime
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from src.env import nccl_timeout
from src.runtime.filesystem import resolve_shared_filesystem_consensus
from src.runtime.ranks import (
    get_local_rank,
    has_process_group,
    is_global_main_process,
    launcher_global_rank,
    launcher_global_world_size,
)

logger = logging.getLogger(__name__)


def _apply_default_pg_timeout(timeout: datetime.timedelta) -> None:
    """Make ``timeout`` the NCCL watchdog of every later ``new_group`` and extend the world group's.

    ``new_group`` otherwise falls back to PyTorch's 10 minutes, and ``init_device_mesh`` takes no
    timeout at all. A torch without ``default_pg_nccl_timeout`` raises rather than silently leaving
    every subgroup on that default.
    """
    if not hasattr(c10d, "default_pg_nccl_timeout"):
        raise RuntimeError(
            "torch.distributed.distributed_c10d.default_pg_nccl_timeout is absent on this PyTorch "
            f"build ({torch.__version__}), so DIST_NCCL_TIMEOUT_MINUTES cannot reach the subgroups: "
            "every device-mesh (DP/HSDP/TP) and EP/CP group would fall back to the 10 min NCCL "
            "watchdog and abort long cross-node collectives (a gathered save at 100B+). Point "
            "_apply_default_pg_timeout at the renamed symbol."
        )
    c10d.default_pg_nccl_timeout = timeout

    # Extend-only: shortening the world group would resurface the straggler aborts.
    if not has_process_group():
        return
    try:
        current = dist.group.WORLD._get_backend(torch.device("cuda")).options._timeout
    except Exception:  # introspection only: torch's internals may hide it
        current = None
    if current is None:
        # Setting blindly could SHORTEN an external watchdog (accelerate ddp_timeout).
        if is_global_main_process():
            logger.warning(
                "Could not read the world group's current NCCL timeout; skipping the retroactive "
                f"timeout update to {timeout} (future subgroups still inherit it)."
            )
    elif current >= timeout:
        if is_global_main_process():
            logger.info(f"World process group already has a >= timeout ({current}); not shortening to {timeout}.")
    else:
        try:
            c10d._set_pg_timeout(timeout)
        except Exception:
            logger.warning("Could not extend the world group's NCCL timeout (non-fatal)", exc_info=True)


def _require_rendezvous_env() -> None:
    """Refuse a multi-rank launch without an ``env://`` rendezvous (bare ``srun --ntasks-per-node``),
    naming the fix c10d's own error does not."""
    missing = [var for var in ("MASTER_ADDR", "MASTER_PORT") if not os.environ.get(var)]
    if not missing:
        return
    raise RuntimeError(
        f"The launcher declares world_size={launcher_global_world_size()} but {missing} is unset, so "
        f"no process group can be built. Bare 'srun --ntasks-per-node=N' provides no env:// "
        f"rendezvous: launch one torchrun per node instead — 'srun --ntasks-per-node=1 torchrun "
        f"--nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=N "
        f"--master_addr=<head> --master_port=<port> <script> <config>' — or export MASTER_ADDR / "
        f"MASTER_PORT identically on every task (see agent-docs/parallelism/launch-recipes.md)."
    )


def init_distributed(backend: str = "nccl") -> bool:
    """COLLECTIVE: initialize the default process group with ``nccl_timeout()``, bind this rank's CUDA
    device and agree the shared-filesystem flags; return whether it initialized.

    A single process with no launcher vars is a no-op; an already-initialized group (``accelerate
    launch``) still gets the timeout pin and the consensus.
    """
    timeout = nccl_timeout()
    # A multi-rank world counts even without RANK: that is exactly the bare-srun shape, where
    # skipping init leaves every rank at world 1, "main", and writing the same output_dir.
    world = launcher_global_world_size()
    if ("RANK" not in os.environ and world <= 1) or has_process_group():
        if has_process_group():
            _apply_default_pg_timeout(timeout)
            resolve_shared_filesystem_consensus()
        return False
    if world > 1:
        _require_rendezvous_env()

    kwargs = {"backend": backend, "timeout": timeout}
    if "RANK" not in os.environ:
        # Bare srun declares the world through SLURM_* alone, while c10d's env:// handler reads
        # RANK / WORLD_SIZE itself and raises on their absence.
        kwargs.update(rank=launcher_global_rank(), world_size=world)
    if backend == "nccl" and torch.cuda.is_available():
        # Launcher's rank, not is_global_main_process(): pre-init the latter is 0 on every rank.
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1" and launcher_global_rank() == 0:
            logger.warning(
                "CUDA_DEVICE_MAX_CONNECTIONS is %r; Expert Parallelism is validated with 1, the image "
                "default. Set it in the process environment: the driver latches it at initialization, "
                "so a Python-level override here is too late.",
                os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
            )
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        kwargs["device_id"] = torch.device("cuda", local_rank)

    dist.init_process_group(**kwargs)
    _apply_default_pg_timeout(timeout)
    # The flag picks the coordination scope, so it must be one value for the whole job.
    resolve_shared_filesystem_consensus()
    if is_global_main_process():
        logger.info("Initialized distributed: backend=%s timeout=%s world_size=%d", backend, kwargs["timeout"], world)
    return True
