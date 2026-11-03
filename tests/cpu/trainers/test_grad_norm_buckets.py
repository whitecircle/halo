#!/usr/bin/env python
"""The one grad-norm accumulation the mixin's EP/TP norm and the PP clip share: an fp32 sum, additive across ranks.
Empty buckets are exercised: the reduces that follow are structural, so a missing or ``None`` bucket hangs at scale."""

import pytest
import torch
import torch.distributed as dist

from halo.trainers.parallelism.grad_clipping import local_grad_norm_sq
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# Keyed as each call site keys them. Each EP/TP bucket reduces over its own group; the PP clip has one, the stage.
TOPOLOGY_BUCKETS = {
    "ep_tp": ("expert", "fsdp_full", "tp1d", "tp2d", "dp", "other"),
    "pp": ("stage",),
}


def _shards(rank: int, bucket: str, index: int) -> list[torch.Tensor]:
    """bf16 shards are what an own-dtype accumulation gets wrong; an empty bucket is a rank with no such parameter."""
    if bucket in ("tp2d", "fsdp_full"):
        return []
    g = torch.manual_seed(100 * index + rank)
    return [
        torch.randn(4, 5, generator=g),
        torch.randn(3, 3, generator=g, dtype=torch.bfloat16),
        torch.randn(2, 6, generator=g, dtype=torch.bfloat16).t(),
        torch.randn(7, generator=g) * 1e3,
    ]


def _bucketed_norm_sq(buckets: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: local_grad_norm_sq(shards, device="cpu") for name, shards in buckets.items()}


def _reference_norm_sq(shards: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32)
    for shard in shards:
        total = total + shard.norm(dtype=torch.float32) ** 2
    return total


@pytest.mark.parametrize("topology", sorted(TOPOLOGY_BUCKETS))
def test_local_value_matches_the_per_tensor_reference(topology):
    buckets = {name: _shards(0, name, i) for i, name in enumerate(TOPOLOGY_BUCKETS[topology])}

    got = _bucketed_norm_sq(buckets)

    for name, shards in buckets.items():
        assert got[name].dtype is torch.float32 and got[name].shape == ()
        torch.testing.assert_close(got[name], _reference_norm_sq(shards), rtol=1e-6, atol=0)
    for empty in (name for name, shards in buckets.items() if not shards):
        assert got[empty].item() == 0.0


def _worker(rank: int, out: str) -> None:
    failures = []
    for topology, names in TOPOLOGY_BUCKETS.items():
        buckets = {name: _shards(rank, name, i) for i, name in enumerate(names)}
        norm_sq = _bucketed_norm_sq(buckets)
        for name in names:
            summed = norm_sq[name].clone()
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            # The shards of both ranks form one tensor set, whose squared norm the clip threshold is judged against.
            union = [s for r in range(WORLD_SIZE) for s in _shards(r, name, names.index(name))]
            want = _reference_norm_sq(union)
            if not torch.allclose(summed, want, rtol=1e-6, atol=0):
                failures.append(f"{topology}/{name}: summed={summed.item()} want={want.item()}")
    result = "PASS" if not failures else "FAILURES: " + "; ".join(failures)
    if rank == 0:
        with open(out, "w") as fh:
            fh.write(result)


def test_buckets_are_additive_across_ranks(tmp_path):
    out = str(tmp_path / "result.txt")
    run_gloo_ranks(_worker, WORLD_SIZE, out)
    with open(out) as fh:
        assert fh.read() == "PASS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
