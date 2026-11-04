"""``gather_on_rank0`` and ``combine_across_ranks`` on a real gloo group. NCCL runs a gather, scatter or all-to-all as
a send per peer, keeping ~0.3 GiB per NVLink peer, so every other collective raises here on any backend."""

import datetime
import json
import math
import os
import pickle
import re

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed import distributed_c10d as c10d

from halo.runtime import collectives
from halo.runtime.collectives import combine_across_ranks, gather_on_rank0
from tests.common.collective_recorder import collectives_except, rebind_everywhere
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 3
PG_TIMEOUT = datetime.timedelta(seconds=90)
# All run over the ring every communicator holds: the size all-gather, payload reduce and verdict broadcast.
_RING_COLLECTIVES = (c10d.all_gather_into_tensor, c10d.reduce, c10d.broadcast, c10d.broadcast_object_list)
# Below every payload's pickle, so each rank's span crosses window boundaries.
_SMALL_WINDOW = 7

_unpickled_origins: list[int] = []


def _unpickle_tattle(origin: int) -> "_Tattle":
    _unpickled_origins.append(origin)
    return _Tattle(origin)


class _Tattle:
    """Records, in the process that unpickles it, the rank it came from."""

    def __init__(self, origin: int) -> None:
        self.origin = origin

    def __reduce__(self):
        return _unpickle_tattle, (self.origin,)


def _refusal(name: str):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{name} reached; the rank-0 gather may issue only ring collectives")

    return refuse


def _ring_only_guard():
    return rebind_everywhere(
        {function: _refusal(name) for name, function in collectives_except(_RING_COLLECTIVES).items()}
    )


def _payload(rank: int) -> bytes:
    """Distinct per rank in length and content, zero bytes included."""
    return bytes(range(256)) * (rank + 1) + bytes([rank]) * (13 * rank)


def _ring_only_worker(rank: int) -> None:
    with _ring_only_guard():
        gathered = gather_on_rank0({"rank": rank})
        merged = combine_across_ranks(rank, sum, what="the ranks")
    assert gathered == ([{"rank": peer} for peer in range(WORLD_SIZE)] if rank == 0 else None), gathered
    assert merged == sum(range(WORLD_SIZE)), merged


def _guard_control_worker(rank: int) -> None:
    to_rank0 = [1] + [0] * (WORLD_SIZE - 1)
    point_to_point = {
        "torch.distributed.gather_object": lambda: dist.gather_object(
            rank, [None] * WORLD_SIZE if rank == 0 else None, dst=0
        ),
        "torch.distributed.all_to_all_single": lambda: dist.all_to_all_single(
            torch.empty(WORLD_SIZE if rank == 0 else 0), torch.ones(1), input_split_sizes=to_rank0
        ),
        f"{funcol.__name__}.all_to_all_single": lambda: funcol.all_to_all_single(
            torch.ones(1), None, to_rank0, dist.group.WORLD
        ),
    }
    with _ring_only_guard():
        for name, issue in point_to_point.items():
            with pytest.raises(AssertionError, match=re.escape(f"{name} reached")):
                issue()


def _rank_order_worker(rank: int, window: int | None) -> None:
    rounds: list[int] = []
    reduce = c10d.reduce

    def recording_reduce(tensor, *args, **kwargs):
        rounds.append(tensor.numel())
        return reduce(tensor, *args, **kwargs)

    if window is not None:
        collectives._GATHER_WINDOW_BYTES = window
    with rebind_everywhere({reduce: recording_reduce}):
        gathered = gather_on_rank0(_payload(rank))
    assert gathered == ([_payload(peer) for peer in range(WORLD_SIZE)] if rank == 0 else None)
    total = sum(len(pickle.dumps(_payload(peer))) for peer in range(WORLD_SIZE))
    bound = collectives._GATHER_WINDOW_BYTES
    assert len(rounds) == math.ceil(total / bound) and max(rounds) <= bound, (rounds, total, bound)


def _unpickle_worker(rank: int) -> None:
    gathered = gather_on_rank0(_Tattle(rank))
    if rank == 0:
        assert [tattle.origin for tattle in gathered] == list(range(WORLD_SIZE))
        assert _unpickled_origins == list(range(WORLD_SIZE)), _unpickled_origins
    else:
        assert gathered is None
        assert _unpickled_origins == [], f"rank {rank} unpickled the payloads of ranks {_unpickled_origins}"


def _combine_worker(rank: int, tmp_dir: str) -> None:
    def outcome(payload, combine) -> object:
        try:
            return combine_across_ranks(payload, combine, what="the fragments")
        except RuntimeError as failure:
            return f"raised: {failure}"

    def malformed(gathered):
        raise TypeError("fragment is not a pair")

    def collision(gathered):
        raise RuntimeError("two writers claimed one key")

    outcomes = {
        # Rank 1 contributes nothing, as a non-writer does.
        "order": outcome(None if rank == 1 else f"r{rank}", list),
        "malformed": outcome(rank, malformed),
        "collision": outcome(rank, collision),
    }
    with open(os.path.join(tmp_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(outcomes, handle)


def test_the_gather_issues_only_ring_collectives():
    run_gloo_ranks(_ring_only_worker, WORLD_SIZE, pg_timeout=PG_TIMEOUT)


def test_the_guard_refuses_point_to_point_collectives():
    run_gloo_ranks(_guard_control_worker, WORLD_SIZE, pg_timeout=PG_TIMEOUT)


@pytest.mark.parametrize("window", [None, _SMALL_WINDOW], ids=["one-window", "spans-cross-windows"])
def test_rank0_gets_every_payload_exact_and_in_rank_order(window):
    run_gloo_ranks(_rank_order_worker, WORLD_SIZE, window, pg_timeout=PG_TIMEOUT)


def test_no_peer_unpickles_another_ranks_payload():
    run_gloo_ranks(_unpickle_worker, WORLD_SIZE, pg_timeout=PG_TIMEOUT)


def test_the_combine_and_its_failure_reach_every_rank(tmp_path):
    run_gloo_ranks(_combine_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    for rank in range(WORLD_SIZE):
        outcomes = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert outcomes["order"] == ["r0", None, "r2"], f"rank {rank}: {outcomes['order']}"
        assert outcomes["malformed"] == (
            "raised: merging the fragments failed on rank 0: TypeError: fragment is not a pair"
        ), f"rank {rank}: {outcomes['malformed']}"
        assert outcomes["collision"] == "raised: two writers claimed one key", f"rank {rank}: {outcomes['collision']}"


def test_without_a_process_group_both_stay_local():
    assert not dist.is_initialized()
    assert gather_on_rank0({"a": 1}) == [{"a": 1}]
    assert combine_across_ranks(5, list, what="the fragments") == [5]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
