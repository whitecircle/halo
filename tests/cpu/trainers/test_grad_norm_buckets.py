#!/usr/bin/env python
"""The one bucketed grad-norm accumulation the EP, TP and PP clip paths share.

``bucketed_grad_norm_sq`` / ``local_grad_norm_sq`` (``src/trainers/mixins/grad_clip.py``) replaced
three per-path accumulations, so two properties have to hold or a clip threshold silently drifts
between topologies:

  * the local value is the fp32 sum of squared per-tensor norms — accumulating in a bf16 shard's own
    dtype, or widening an already-rounded value with a trailing ``.float()``, moves the norm;
  * it is ADDITIVE across ranks — every caller sums its buckets with an ``all_reduce`` and takes the
    square root, so the summed local squares must equal the squared norm of the union of the shards.

Empty buckets are exercised on purpose: the reduces that follow are structural (issued whether or
not this rank's parameters carried a gradient), so a missing or ``None`` bucket is a hang at scale.

    python tests/cpu/trainers/test_grad_norm_buckets.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.trainers.mixins.grad_clip import bucketed_grad_norm_sq, local_grad_norm_sq
from tests.common.ports import free_port

WORLD_SIZE = 2

# The bucket dicts the three clip paths build, each keyed exactly as its call site keys it. The EP
# path routes expert / FSDP-full / TP-mesh / replicated grads apart because each reduces over a
# different group; the TP path splits the same mesh buckets from the replicated remainder; the PP
# path has one bucket, the whole stage.
TOPOLOGY_BUCKETS = {
    "ep": ("expert", "fsdp_full", "tp1d", "tp2d", "dp", "other"),
    "tp": ("tp1d", "tp2d", "dp", "replicated"),
    "pp": ("stage",),
}


def _shards(rank: int, bucket: str, index: int) -> list[torch.Tensor]:
    """Per-(rank, bucket) shards: assorted shapes, mixed dtypes, one non-contiguous, one empty bucket.

    A bf16 shard is the realistic case (bf16 training) and the one an accumulation in the shard's own
    dtype gets wrong; the empty bucket is what a rank holding no parameter of that kind produces.
    """
    if bucket in ("tp2d", "fsdp_full"):
        return []
    g = torch.manual_seed(100 * index + rank)
    return [
        torch.randn(4, 5, generator=g),
        torch.randn(3, 3, generator=g, dtype=torch.bfloat16),
        torch.randn(2, 6, generator=g, dtype=torch.bfloat16).t(),
        torch.randn(7, generator=g) * 1e3,
    ]


def _reference_norm_sq(shards: list[torch.Tensor]) -> torch.Tensor:
    """The pre-fold accumulation: per-tensor fp32 norm, squared and summed."""
    total = torch.zeros((), dtype=torch.float32)
    for shard in shards:
        total = total + shard.norm(dtype=torch.float32) ** 2
    return total


@pytest.mark.parametrize("topology", sorted(TOPOLOGY_BUCKETS))
def test_local_value_matches_the_accumulation_it_replaced(topology):
    buckets = {name: _shards(0, name, i) for i, name in enumerate(TOPOLOGY_BUCKETS[topology])}

    got = bucketed_grad_norm_sq(buckets, device="cpu")

    assert sorted(got) == sorted(buckets), "every declared bucket must come back — the reduces are structural"
    for name, shards in buckets.items():
        assert got[name].dtype is torch.float32 and got[name].shape == ()
        torch.testing.assert_close(got[name], _reference_norm_sq(shards), rtol=1e-6, atol=0)
    for empty in (name for name, shards in buckets.items() if not shards):
        assert got[empty].item() == 0.0


def test_single_bucket_helper_agrees_with_the_bucketed_one():
    shards = _shards(0, "stage", 0)
    torch.testing.assert_close(
        local_grad_norm_sq(shards, device="cpu"),
        bucketed_grad_norm_sq({"stage": shards}, device="cpu")["stage"],
        rtol=0,
        atol=0,
    )


def _worker(rank: int, out: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        failures = []
        for topology, names in TOPOLOGY_BUCKETS.items():
            buckets = {name: _shards(rank, name, i) for i, name in enumerate(names)}
            norm_sq = bucketed_grad_norm_sq(buckets, device="cpu")
            for name in names:
                summed = norm_sq[name].clone()
                dist.all_reduce(summed, op=dist.ReduceOp.SUM)
                # Ground truth: the shards of BOTH ranks form one tensor set, whose squared norm is
                # what the clip threshold is judged against.
                union = [s for r in range(WORLD_SIZE) for s in _shards(r, name, names.index(name))]
                want = _reference_norm_sq(union)
                if not torch.allclose(summed, want, rtol=1e-6, atol=0):
                    failures.append(f"{topology}/{name}: summed={summed.item()} want={want.item()}")
        result = "PASS" if not failures else "FAILURES: " + "; ".join(failures)
        if rank == 0:
            with open(out, "w") as fh:
                fh.write(result)
    finally:
        dist.destroy_process_group()


def test_buckets_are_additive_across_ranks(tmp_path):
    """Every caller sums its buckets over a process group, so the local squares must add up."""
    out = str(tmp_path / "result.txt")
    mp.start_processes(_worker, args=(out, free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    with open(out) as fh:
        assert fh.read() == "PASS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
