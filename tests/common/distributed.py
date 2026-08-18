"""Distributed setup/teardown for torchrun-based tests.

One home for process-group init, scratch/cache dir allocation under the launcher's ``TMPDIR``,
rank-0-then-barrier model download, world-wide scalar reductions, and teardown.
"""

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import datasets.config
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore as TorchFakeStore
from transformers import AutoConfig, AutoTokenizer

from src.distributed.runtime import barrier
from src.models.patches.attention import ensure_fa4_kernel_cache_env


class FakeStore:
    """In-memory stand-in for the c10d key-value store, for single-process store-coordination tests.

    A wait on an absent key raises instead of blocking, so a test that would deadlock against the
    real store fails immediately and names the key. ``set_keys`` records writes in order, and
    ``get_keys`` / ``multi_get_keys`` record reads so a test can pin how many round-trips a join
    costs (the real store serves them one request at a time, from one server).
    """

    def __init__(self):
        self.kv: dict[str, str | int] = {}
        self.set_keys: list[str] = []
        self.get_keys: list[str] = []
        self.multi_get_keys: list[list[str]] = []

    def wait(self, keys, timeout=None):
        del timeout  # the real store blocks; absence is the interesting case here
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


@contextlib.contextmanager
def fake_process_group_mesh(rank: int, world_size: int):
    """A 1-D CPU device mesh over torch's fake process group, impersonating ``rank``.

    ``distribute_tensor(..., src_data_rank=None)`` — what transformers' TP ``shard_param`` runs —
    slices the local shard with no collective, so a single CPU process can obtain ANY rank's shard
    from the real sharding machinery by re-entering with a different ``rank``. A leftover process
    group from an earlier test in the same worker is torn down first: the fake backend needs the
    default group slot.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", store=TorchFakeStore(), rank=rank, world_size=world_size)
    try:
        yield init_device_mesh("cpu", (world_size,))
    finally:
        dist.destroy_process_group()


def init_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group and return (rank, world_size, local_rank).

    Also sets the CUDA device for the local rank, and anchors the FA4 kernel cache the way
    production does.
    """
    # Before any model load — and therefore before the first ``import flash_attn.cute``, which
    # reads this env at import. Production gets it from ``setup_training_environment``; a GPU test
    # that loads through ``from_pretrained`` directly would otherwise leave the CuTe DSL with the
    # cache ENABLED but no directory, and re-JIT every kernel on every run (measured: 18 min for
    # one varlen backward, against a 900 s budget). The one call every GPU entry point makes.
    ensure_fa4_kernel_cache_env()
    # CUDA_DEVICE_MAX_CONNECTIONS=1 (serialize device work onto one hardware queue) is what
    # keeps multi-group Expert Parallelism (e.g. ep_size=4 on 8 GPUs) from deadlocking the
    # DeepEP intra-node combine barrier against FSDP2's NCCL collectives. It must already be
    # in the process environment (the image ENV / Makefile DOCKER_RUN set it): the driver
    # latches it at cuInit, which the DeepEP/NVSHMEM imports trigger before this runs, so a
    # write here would be a no-op. Warn (rank 0) if a launch is missing it.
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1" and os.environ.get("RANK", "0") == "0":
        print(
            "[init_distributed] WARNING: CUDA_DEVICE_MAX_CONNECTIONS != 1 "
            f"({os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')!r}); multi-group EP may deadlock.",
            flush=True,
        )
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if not dist.is_initialized():
        kwargs = {"backend": "nccl"}
        if torch.cuda.is_available():
            # Bind this rank's device BEFORE init so the NCCL backend does not "guess" the
            # device from global rank (which warns and can hang on heterogeneous mappings).
            # Mirrors src.distributed.runtime.init_distributed.
            torch.cuda.set_device(local_rank)
            kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**kwargs)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def setup_cache_dirs(prefix: str, rank: int) -> tuple[str, str]:
    """Create isolated output and HF cache directories.

    Returns (output_dir, cache_dir). Sets HF_DATASETS_CACHE env var.
    """
    cache_dir = tempfile.mkdtemp(prefix=f"{prefix}_cache_r{rank}_")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    # datasets froze the env at import (datasets.config latches at module load), so the env write
    # alone doesn't isolate direct load_dataset callers — update the live config too (mirrors
    # src/training/environment.py).
    datasets.config.HF_DATASETS_CACHE = Path(cache_dir)
    output_dir = tempfile.mkdtemp(prefix=f"{prefix}_out_")
    return output_dir, cache_dir


def shared_scratch_dir(name: str) -> str:
    """A scratch dir under the launcher's per-run ``TMPDIR``, identical on every rank.

    Rank 0 builds a tiny synthetic checkpoint here and the peers read it back, so the path has to
    be a pure function of the environment — ``mkdtemp`` would hand each rank a different dir.
    ``MASTER_PORT`` (uniform across a job's ranks, distinct across concurrent jobs) separates two
    runs that share one ``TMPDIR``; ``tests/gpu/conftest.py`` points ``TMPDIR`` at pytest's
    basetemp, so the dir is reclaimed with the run instead of accumulating on the scratch volume.
    """
    path = os.path.join(tempfile.gettempdir(), f"{name}_{os.environ.get('MASTER_PORT', 'standalone')}")
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_dirs(*dirs: str):
    """Remove temp directories."""
    for d in dirs:
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def ensure_model_downloaded(model_name: str, rank: int, revision: str | None = None):
    """Download model config/tokenizer on rank 0, then barrier.

    Prevents race conditions when multiple ranks try to download simultaneously.
    """
    if rank == 0:
        AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    barrier()


def world_mean(value: float, device=None) -> float:
    """DP-average a per-rank scalar — what HF logs for a loss (mean of per-replica batch means).

    Collective: every rank must call it with its own value.
    """
    local = torch.tensor(float(value), device=device or torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(local, op=dist.ReduceOp.AVG)
    return float(local)


def world_any(flag: bool, device=None) -> bool:
    """True if ``flag`` holds on ANY rank. Collective.

    For preconditions on a quantity that is compared world-wide (a DP-averaged loss): the property
    has to hold somewhere in the world for the comparison to be meaningful, not on every shard.
    """
    local = torch.tensor(float(bool(flag)), device=device or torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return bool(local.item())


def world_min(value: float, device=None) -> float:
    """The smallest ``value`` across all ranks. Collective.

    For a precondition that must hold on EVERY shard (unlike :func:`world_any`): the worst rank's
    value is what decides whether a comparison is sound.
    """
    local = torch.tensor(float(value), device=device or torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return float(local)


def snapshot_full_weights(model) -> dict[str, torch.Tensor]:
    """CPU copy of ``model``'s state dict as FULL tensors, all-gathering FSDP2/TP DTensor shards.

    Lets a test rebuild an independent copy of the exact weights a trainer held at a chosen step, so
    an objective reference runs on those weights rather than on a re-initialised approximation of
    them. Collective when any parameter is a DTensor — every rank must call it at the same point.
    """
    snapshot = {}
    for name, tensor in model.state_dict().items():
        full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        snapshot[name] = full.detach().to("cpu", copy=True)
    return snapshot


def _dp_step_losses(step_losses: list[float], dp_rank: int, dp_size: int) -> list[list[float]]:
    """Per step, each DATA-PARALLEL replica's loss in DP-rank order — the chunks a metric gather keeps.

    The eval gather is scoped to the DP replicas, so a pipeline chain's stages (and a TP group's
    siblings) contribute their shared batch ONCE. They hold the same value, so which holder supplies
    a replica's entry cannot change the result — an absent replica can, and raises.

    Collective: every rank all-gathers its own vector, so all ranks must call with the same number
    of steps (accelerate's ``even_batches`` guarantees it).
    """
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
    """Reproduce HF ``evaluation_loop``'s eval_loss from per-rank per-step reference losses.

    HF repeats each step's scalar loss ``eval_batch_size`` times and gathers; the toolkit scopes
    that gather to the DP replicas (``_install_dp_metric_gather``), so the vector is one chunk per
    DP rank in DP-rank order rather than one per GLOBAL rank — the same batch is not counted once
    per pipeline stage. At the dataloader's final step accelerate's ``gather_for_metrics`` then
    truncates THAT vector to ``remainder`` entries (``len(eval_dataset) % (eval_batch_size *
    dp_size)``, the tail trim; 0 = no trim). The final metric is the mean over every kept entry.

    Collective — every rank must call with its own ``step_losses``. Cross-check any result against
    :func:`eval_loss_over_examples`, which derives the same number without modelling the gather.
    """
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
    """The eval_loss an example-by-example walk of the eval dataset gives — an INDEPENDENT oracle.

    Derived from what the metric means rather than from how the gather is built: every eval example
    contributes its own batch's loss exactly once, so the metric is the mean over ``dataset_rows``
    entries laid out in the order the loader produced them (per step, DP rank 0's batch, then rank
    1's, ...). Nothing here models the dedup or accelerate's trim, so it disagrees with
    :func:`hf_eval_loss_reference` if either is wrong — including the case that motivates both: a
    gather that repeats each replica once per pipeline stage keeps a prefix of the DUPLICATES on a
    partial final batch, which is a different set of examples, not merely a re-weighted one.
    """
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
    """Clean up distributed process group."""
    if dist.is_initialized():
        # Free DeepEP ElasticBuffers before the group (see destroy_all_dispatchers for why order
        # matters). No-op without EP; the lazy import keeps this helper usable where deep_ep isn't
        # installed (CPU-only / IDE envs).
        with contextlib.suppress(ImportError):
            from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers

            destroy_all_dispatchers()
        dist.destroy_process_group()
