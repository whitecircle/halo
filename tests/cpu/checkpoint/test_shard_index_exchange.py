#!/usr/bin/env python
"""The checkpoint index is merged on rank 0 alone, and its verdict reaches every rank.

``exchange_shard_index`` replaces an ``all_gather_object`` of the whole weight map. That all-gather
left EVERY rank holding one fragment per rank of a map only the index writer ever reads — at 512
ranks, 512 copies of the entire key space on each of 512 ranks, for a file one of them writes.

Gathering to rank 0 moves the merge off the peers, which puts the collision check somewhere only one
rank can see it. So the merged result AND the merge's verdict both have to come back over the
broadcast: a collision that raised on rank 0 alone would strand every peer in the barrier that
follows the exchange — a job-wide hang whose traceback names the barrier, not the duplicated key.

What these tests pin: non-zero ranks never allocate the gather list, the merged index is identical
on every rank, a planted collision raises on ALL of them with the diagnostic naming the key, and a
non-contributing rank still joins both collectives.

Run: pytest tests/cpu/checkpoint/test_shard_index_exchange.py
"""

import pytest
import torch.distributed as dist

from src.distributed.checkpoint import write as checkpoint_write
from src.distributed.checkpoint.write import exchange_shard_index

WORLD_SIZE = 4


class _FakeWorld:
    """One process standing in for ``WORLD_SIZE`` ranks running ``exchange_shard_index``.

    Every rank's payload is known up front, so ``gather_object`` can serve rank 0 the whole table
    while the peers only announce that they joined. ``broadcast_object_list`` routes through one box
    the way the real collective routes through the wire, which is what lets the peers observe exactly
    what rank 0 published — including a raise.

    ``gather_lists`` records the ``object_gather_list`` each rank passed, the direct evidence that a
    peer allocated nothing: torch requires it to be ``None`` anywhere but the destination.
    """

    def __init__(self, payloads: list[tuple[dict[str, str], int] | None]):
        self.payloads = payloads
        self.rank = 0
        self.box: list[object] = [None]
        self.gather_lists: dict[int, list | None] = {}

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: WORLD_SIZE)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: self.rank)
        monkeypatch.setattr(dist, "gather_object", self._gather_object)
        monkeypatch.setattr(dist, "broadcast_object_list", self._broadcast_object_list)
        # The regression this whole module exists for: a revert to the all-gather form must fail
        # here rather than quietly pass because the merged map still comes out right.
        monkeypatch.setattr(dist, "all_gather_object", self._banned_all_gather)

    def _banned_all_gather(self, *args, **kwargs):
        raise AssertionError("exchange_shard_index must gather to rank 0, not all-gather to everyone")

    def _gather_object(self, obj, object_gather_list=None, dst=0, group=None):
        del group
        assert dst == 0, f"the index writer is rank 0, but the gather targeted {dst}"
        assert obj == self.payloads[self.rank], "a rank contributed something other than its own fragment"
        self.gather_lists[self.rank] = object_gather_list
        if self.rank == 0:
            assert object_gather_list is not None, "the destination rank must supply the output list"
            object_gather_list[:] = list(self.payloads)
        return None

    def _broadcast_object_list(self, object_list, src=0, group=None):
        del group
        assert src == 0, f"the merged index is published by rank 0, but the broadcast source was {src}"
        if self.rank == 0:
            self.box = list(object_list)
        else:
            object_list[:] = list(self.box)
        return None

    def run_all(self) -> list[object]:
        """Each rank's outcome, rank 0 first — the broadcast source must publish before its peers read.

        A raise is captured rather than propagated so a test can assert the verdict is WORLD-UNIFORM
        instead of stopping at whichever rank happened to run first.
        """
        outcomes: list[object] = []
        for rank in range(WORLD_SIZE):
            self.rank = rank
            payload = self.payloads[rank]
            try:
                outcomes.append(
                    exchange_shard_index(*(payload if payload else ({}, 0)), contribute=payload is not None)
                )
            except RuntimeError as exc:
                outcomes.append(exc)
        return outcomes


def _stage_payloads() -> list[tuple[dict[str, str], int] | None]:
    """A 4-rank PP save: two stage writers with disjoint globals, two non-writers."""
    return [
        ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100),
        None,
        ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250),
        None,
    ]


def test_every_rank_returns_the_same_merged_index(monkeypatch):
    """The exchange's whole product: one map and one byte count, identical world-wide.

    The index writer is rank 0, but a resume reads the map on every rank, so a peer that came back
    with a partial map would resume against keys no shard claims.
    """
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)

    outcomes = world.run_all()

    expected = (
        {
            "model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors",
            "model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors",
        },
        350,
    )
    for rank, outcome in enumerate(outcomes):
        assert outcome == expected, f"rank {rank} returned {outcome!r}, not the merged index"


def test_only_rank_zero_allocates_the_gather_list(monkeypatch):
    """The memory the switch exists to save: a peer must hold no per-rank copy of the map.

    ``object_gather_list`` is the allocation — one slot per rank, each filled with a whole fragment.
    Rank 0 needs it to merge; anywhere else it is N copies of a map that rank never reads.
    """
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)

    world.run_all()

    assert world.gather_lists[0] is not None, "rank 0 must collect the fragments it merges"
    assert len(world.gather_lists[0]) == WORLD_SIZE, "the destination list must be sized to the world"
    for rank in range(1, WORLD_SIZE):
        assert world.gather_lists[rank] is None, f"rank {rank} allocated a {WORLD_SIZE}-slot gather list"


def test_a_non_writer_still_joins_both_collectives(monkeypatch):
    """``contribute=False`` is a rank that wrote nothing, not a rank that sits the exchange out.

    PP non-writers outnumber writers by ``stage_world_size - 1``. One skipping the gather or the
    broadcast desynchronizes the world for the rest of the run.
    """
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank in (1, 3):
        assert rank in world.gather_lists, f"non-writer rank {rank} never entered the gather"
        assert isinstance(outcomes[rank], tuple), f"non-writer rank {rank} never received the merged index"
        assert outcomes[rank][0], "a non-writer must still learn the keys its peers wrote"


def test_any_merge_failure_reaches_every_rank_not_just_rank_zero(monkeypatch):
    """The verdict must carry EVERY failure of the merge, not only the collision it is written to
    raise. A malformed fragment (a ``TypeError``) escaping on rank 0 leaves the peers in the
    broadcast and then the barrier until the watchdog fires, which is the exact failure the move to
    a single merging rank introduced and this exchange exists to keep impossible."""
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)

    def _malformed(_gathered):
        raise TypeError("fragment is not a (weight_map, bytes) pair")

    monkeypatch.setattr(checkpoint_write, "merge_shard_index", _malformed)

    outcomes = world.run_all()

    for rank, outcome in enumerate(outcomes):
        assert isinstance(outcome, RuntimeError), f"rank {rank} did not raise ({outcome!r})"
        assert "TypeError" in str(outcome) and "fragment is not" in str(outcome), str(outcome)


def test_a_planted_collision_raises_on_every_rank_with_the_key_named(monkeypatch):
    """Two writers claiming one key must stop the whole world, naming the key.

    Only rank 0 can see the collision once the merge moves there. If its raise did not ride the
    broadcast, the peers would walk into the post-exchange barrier and hang until the watchdog fired
    — and the checkpoint would carry one writer's tensor under the other's name.
    """
    collided = "model.layers.0.weight"
    world = _FakeWorld(
        [
            ({collided: "model-pp00000-of-00002-00001.safetensors"}, 100),
            None,
            ({collided: "model-pp00001-of-00002-00001.safetensors"}, 100),
            None,
        ]
    )
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank, outcome in enumerate(outcomes):
        assert isinstance(outcome, RuntimeError), f"rank {rank} returned {outcome!r} instead of raising"
        assert "claimed by both" in str(outcome), f"rank {rank}'s diagnostic does not name the conflict"
        assert collided in str(outcome), f"rank {rank}'s diagnostic does not name the duplicated key"


def test_per_node_writers_of_one_stage_are_counted_once(monkeypatch):
    """Per-node storage puts one writer per NODE, all emitting their stage's shard under one name.

    Folding every fragment's bytes would multiply ``total_size`` by the nodes per stage while the
    map itself still round-trips, so nothing else in the checkpoint would show it.
    """
    stage0 = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    stage1 = ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250)
    world = _FakeWorld([stage0, stage0, stage1, stage1])
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank, outcome in enumerate(outcomes):
        assert outcome[1] == 350, f"rank {rank} counted {outcome[1]} bytes; each shard file contributes once"


def test_without_a_process_group_the_lone_fragment_is_the_index(monkeypatch):
    """A single-process save (no dist) must still get a merged index, not a raw fragment list."""
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    assert exchange_shard_index({"a": "f0.safetensors"}, 100) == ({"a": "f0.safetensors"}, 100)


def test_a_lost_broadcast_raises_instead_of_returning_an_empty_index(monkeypatch):
    """The verdict never arriving must fail loud, not return ``None`` into the index writer.

    ``None`` would flow into ``json.dump`` as a ``weight_map`` of nothing, producing an index that
    names no key — a checkpoint that loads with every weight missing.
    """
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)
    monkeypatch.setattr(checkpoint_write.dist, "broadcast_object_list", lambda object_list, src=0, group=None: None)

    world.rank = 1
    with pytest.raises(RuntimeError, match="never reached this rank"):
        exchange_shard_index({}, 0, contribute=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
