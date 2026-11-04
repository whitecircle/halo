"""Distributed setup and teardown for torchrun-based tests."""

import contextlib
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import datasets.config
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore as TorchFakeStore
from transformers import AutoConfig, AutoTokenizer

from halo.distributed.expert_parallel.deepep import dispatcher as ep_dispatcher
from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env
from halo.runtime import filesystem, process_group
from halo.runtime.collectives import barrier, broadcast_from_rank0
from halo.runtime.ranks import get_global_rank, get_global_world_size, get_local_rank
from tests.common.numerics import max_or_nan
from tests.common.scratch import SCRATCH_DIR_TAG


class FakeStore:
    """In-memory c10d store for single-process coordination tests; a wait on an absent key raises instead of blocking.
    ``set_keys`` and ``get_keys`` / ``multi_get_keys`` record calls, so a test can pin how many round-trips a join
    costs."""

    def __init__(self):
        self.kv: dict[str, str | int] = {}
        self.set_keys: list[str] = []
        self.get_keys: list[str] = []
        self.multi_get_keys: list[list[str]] = []

    def wait(self, keys, timeout=None):
        del timeout  # The real store blocks; absence is the interesting case here.
        for key in keys:
            if key not in self.kv:
                raise TimeoutError(f"would block on {key}")

    def set(self, key, value):
        self.kv[key] = value
        self.set_keys.append(key)

    def get(self, key):
        self.get_keys.append(key)
        return self.kv[key]

    def multi_get(self, keys):
        keys = list(keys)
        self.multi_get_keys.append(keys)
        return [self.kv[key] for key in keys]

    def add(self, key, amount):
        self.kv[key] = int(self.kv.get(key, 0)) + amount
        return self.kv[key]

    def delete_key(self, key):
        return self.kv.pop(key, None) is not None


def reset_shared_filesystem_consensus() -> None:
    """Drop the process's agreed shared-filesystem flags so they re-read the environment: a run fixes its scope once, a
    test varies it."""
    filesystem._shared_filesystem_consensus = None


@contextlib.contextmanager
def fake_process_group(rank: int, world_size: int):
    """torch's fake process group as the default group, impersonating ``rank``; collectives complete at once and move
    no data. A leftover group from an earlier test is torn down first, since the fake backend needs the default
    slot."""
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", store=TorchFakeStore(), rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()


@contextlib.contextmanager
def fake_process_group_mesh(rank: int, world_size: int):
    """A 1-D CPU device mesh over torch's fake group; ``distribute_tensor(..., src_data_rank=None)`` then slices any
    rank's shard with no collective."""
    with fake_process_group(rank, world_size):
        yield init_device_mesh("cpu", (world_size,))


def init_distributed() -> tuple[int, int, int]:
    """COLLECTIVE: production process-group init; returns ``(rank, world_size, local_rank)``. Anchors the FA4 kernel
    cache first: a test loading via ``from_pretrained`` would otherwise re-JIT every kernel every run, past any suite
    timeout."""
    ensure_fa4_kernel_cache_env()
    process_group.init_distributed()
    return get_global_rank(), get_global_world_size(), get_local_rank()


def pin_deterministic_ep_dispatch() -> None:
    """Build every DeepEP ``ElasticBuffer`` in deterministic mode, placing received tokens by source rank and index.
    Atomic claim order changes run to run, and with it the rounding of expert weight gradients; an exact replay cannot
    have that."""
    ep_dispatcher._deterministic_dispatch = lambda: True


def setup_cache_dirs(prefix: str, rank: int) -> tuple[str, str]:
    """Create isolated output and HF cache dirs; returns (output_dir, cache_dir). Names start with
    :data:`SCRATCH_DIR_TAG` for conftest's crash sweep."""
    cache_dir = tempfile.mkdtemp(prefix=f"{SCRATCH_DIR_TAG}{prefix}_cache_r{rank}_")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    # datasets latches the env at import, so the env write alone does not isolate direct load_dataset callers.
    datasets.config.HF_DATASETS_CACHE = Path(cache_dir)
    output_dir = tempfile.mkdtemp(prefix=f"{SCRATCH_DIR_TAG}{prefix}_out_")
    return output_dir, cache_dir


def shared_scratch_dir(name: str) -> str:
    """A scratch dir under the launcher's ``TMPDIR``, identical on every rank; a pure function of the environment.
    ``MASTER_PORT`` separates concurrent runs sharing one ``TMPDIR``."""
    path = os.path.join(tempfile.gettempdir(), f"{name}_{os.environ.get('MASTER_PORT', 'standalone')}")
    os.makedirs(path, exist_ok=True)
    return path


def shared_output_dir(ctx) -> str:
    """Rank 0's output dir, on every rank; a per-rank dir would leave every peer with no checkpoint to resume from."""
    return broadcast_from_rank0(ctx.output_dir if ctx.rank == 0 else None)


def cleanup_dirs(*dirs: str):
    for d in dirs:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def ensure_model_downloaded(model_name: str, rank: int, revision: str | None = None):
    """Download model config and tokenizer on rank 0, then barrier."""
    if rank == 0:
        AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    barrier()


def _world_reduced(value: float, op: dist.ReduceOp, device) -> torch.Tensor:
    local = torch.tensor(float(value), device=device or torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(local, op=op)
    return local


def world_mean(value: float, device=None) -> float:
    """DP-average a per-rank scalar, matching what HF logs for a loss. Collective."""
    return float(_world_reduced(value, dist.ReduceOp.AVG, device))


def world_any(flag: bool, device=None) -> bool:
    """True if ``flag`` holds on any rank, for a precondition on a quantity compared world-wide. Collective."""
    return bool(_world_reduced(bool(flag), dist.ReduceOp.MAX, device).item())


def world_all(flag: bool, device=None) -> bool:
    """True if ``flag`` holds on every rank, for a property each rank checks on its own shard. Collective."""
    return bool(_world_reduced(bool(flag), dist.ReduceOp.MIN, device).item())


def world_min(value: float, device=None) -> float:
    """The smallest ``value`` across all ranks, for a precondition that must hold on every shard. Collective."""
    return float(_world_reduced(value, dist.ReduceOp.MIN, device))


def world_max(value: float, device=None) -> float:
    """The largest ``value`` across all ranks, for an error every rank's shard must keep under a bar; ``inf`` when any
    is non-finite, since NCCL's reduce is free to drop a NaN. Collective."""
    return float(_world_reduced(value if math.isfinite(value) else math.inf, dist.ReduceOp.MAX, device))


def world_spread(value: float, device=None) -> float:
    """``max - min`` of ``value`` across ranks; ``inf`` when any is non-finite, since NaN would read as agreement.
    Collective."""
    local = torch.tensor(
        [float(value)], dtype=torch.float64, device=device or torch.device("cuda", torch.cuda.current_device())
    )
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = torch.cat(gathered)
    if not torch.isfinite(values).all():
        return math.inf
    return float(values.max() - values.min())


def group_max_abs_diff(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> float:
    """Largest elementwise ``|Δ|`` between any rank of ``group`` and its first rank. Collective over ``group``
    (default: the world). ``0.0`` means bit-identical. NaN on any rank returns NaN, so the reduction stays in torch,
    whose ``max`` propagates it. An empty FSDP2 shard reads ``0.0``."""
    local = tensor.detach().contiguous()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    if local.numel() == 0:
        return 0.0
    return torch.stack([(g - gathered[0]).abs().max() for g in gathered]).max().item()


def max_abs_diff_across(tensors: Iterable[torch.Tensor], group: dist.ProcessGroup | None = None) -> float:
    """The largest :func:`group_max_abs_diff` of ``tensors``; NaN when empty or any difference is NaN. Collective, one
    gather per tensor in the same order."""
    return max_or_nan((group_max_abs_diff(tensor, group) for tensor in tensors), default=math.nan)


def snapshot_full_weights(model) -> dict[str, torch.Tensor]:
    """CPU copy of ``model``'s state dict as full tensors, so a reference runs on the exact weights. Collective with
    any DTensor."""
    snapshot = {}
    for name, tensor in model.state_dict().items():
        full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        snapshot[name] = full.detach().to("cpu", copy=True)
    return snapshot


def _dp_step_losses(step_losses: list[float], dp_rank: int, dp_size: int) -> list[list[float]]:
    """Per step, each DP replica's loss in DP-rank order: the chunks a metric gather keeps. Stages and TP siblings hold
    the same value, so which one supplies a replica does not matter; an absent replica raises. Collective, same step
    count on all ranks."""
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    local = torch.tensor(
        [float(dp_rank), *(float(value) for value in step_losses)], dtype=torch.float64, device=device
    )
    peers = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, local)

    by_dp: dict[int, list[float]] = {}
    for peer in peers:
        by_dp.setdefault(int(peer[0].item()), peer[1:].tolist())
    absent = [rank for rank in range(dp_size) if rank not in by_dp]
    if absent:
        raise AssertionError(f"data-parallel ranks {absent} contributed no eval losses; the gather scope is wrong")
    return [[by_dp[rank][step] for rank in range(dp_size)] for step in range(len(step_losses))]


def hf_eval_loss_reference(
    step_losses: list[float], batch_size: int, remainder: int, *, dp_rank: int, dp_size: int
) -> float:
    """Reproduce HF ``evaluation_loop``'s eval_loss from per-rank per-step reference losses. Collective. The vector is
    one chunk per DP rank, and ``gather_for_metrics`` truncates the final step to ``remainder`` entries (0 = no trim).
    Cross-check against :func:`eval_loss_over_examples`."""
    kept: list[float] = []
    per_step = _dp_step_losses(step_losses, dp_rank, dp_size)
    for step, replicas in enumerate(per_step):
        vector = [value for value in replicas for _ in range(batch_size)]
        if step == len(per_step) - 1 and remainder > 0:
            vector = vector[:remainder]
        kept.extend(vector)
    return sum(kept) / len(kept)


def eval_loss_over_examples(
    step_losses: list[float], batch_size: int, dataset_rows: int, *, dp_rank: int, dp_size: int
) -> float:
    """The eval_loss an example-by-example walk gives: an independent oracle that models neither the dedup nor
    accelerate's trim. It disagrees with :func:`hf_eval_loss_reference` if either is wrong, including a gather
    repeating each replica per pipeline stage."""
    entries = [
        value
        for replicas in _dp_step_losses(step_losses, dp_rank, dp_size)
        for value in replicas
        for _ in range(batch_size)
    ]
    if len(entries) < dataset_rows:
        raise AssertionError(f"{len(entries)} gathered entries cannot cover {dataset_rows} eval examples")
    return sum(entries[:dataset_rows]) / dataset_rows


def teardown_distributed():
    if dist.is_initialized():
        # Free DeepEP ElasticBuffers before the group (see destroy_all_dispatchers for why order matters).
        ep_dispatcher.destroy_all_dispatchers()
        dist.destroy_process_group()
