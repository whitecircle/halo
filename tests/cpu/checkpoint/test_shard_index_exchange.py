#!/usr/bin/env python
"""The checkpoint index is merged on rank 0 alone, and its verdict reaches every rank. A collision raising on rank 0
alone strands peers in the next barrier, a hang naming the barrier instead of the key."""

import pytest
import torch.distributed as dist

from halo.checkpoint.writing import merge_shard_index
from halo.runtime import collectives
from halo.runtime.collectives import combine_across_ranks

WORLD_SIZE = 4


def _exchange(payload, combine=merge_shard_index):
    return combine_across_ranks(payload, combine, what="the checkpoint index")


class _FakeWorld:
    """``gathered_by`` records the ranks that entered the gather; the broadcast box lets peers observe what rank 0
    published, including a raise."""

    def __init__(self, payloads: list[tuple[dict[str, str], int] | None], combine=merge_shard_index):
        self.payloads = payloads
        self.combine = combine
        self.rank = 0
        self.box: list[object] = [None]
        self.gathered_by: list[int] = []

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: WORLD_SIZE)
        monkeypatch.setattr(dist, "get_rank", lambda group=None: self.rank)
        monkeypatch.setattr(collectives, "gather_on_rank0", self._gather_on_rank0)
        monkeypatch.setattr(dist, "broadcast_object_list", self._broadcast_object_list)
        # An all-gather form must fail here even though the merged map still comes out right.
        monkeypatch.setattr(dist, "all_gather_object", self._banned_all_gather)

    def _banned_all_gather(self, *args, **kwargs):
        raise AssertionError("combine_across_ranks must gather to rank 0, not all-gather to everyone")

    def _gather_on_rank0(self, payload):
        assert payload == self.payloads[self.rank], "a rank contributed something other than its own fragment"
        self.gathered_by.append(self.rank)
        return list(self.payloads) if self.rank == 0 else None

    def _broadcast_object_list(self, object_list, src=0, group=None):
        del group
        assert src == 0, f"the merged index is published by rank 0, but the broadcast source was {src}"
        if self.rank == 0:
            self.box = list(object_list)
        else:
            object_list[:] = list(self.box)
        return None

    def run_all(self) -> list[object]:
        """A raise is captured so a test can assert the verdict is world-uniform."""
        outcomes: list[object] = []
        for rank in range(WORLD_SIZE):
            self.rank = rank
            try:
                outcomes.append(_exchange(self.payloads[rank], self.combine))
            except RuntimeError as exc:
                outcomes.append(exc)
        return outcomes


def _stage_payloads() -> list[tuple[dict[str, str], int] | None]:
    """Two stage writers with disjoint globals, two non-writers."""
    return [
        ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100),
        None,
        ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250),
        None,
    ]


def test_every_rank_returns_the_same_merged_index(monkeypatch):
    """A resume reads the map on every rank, so a partial map resumes against keys no shard claims."""
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


def test_a_non_writer_still_joins_both_collectives(monkeypatch):
    """One skipping the gather or broadcast desynchronizes the world for the rest of the run."""
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank in (1, 3):
        assert rank in world.gathered_by, f"non-writer rank {rank} never entered the gather"
        assert isinstance(outcomes[rank], tuple), f"non-writer rank {rank} never received the merged index"
        assert outcomes[rank][0], "a non-writer must still learn the keys its peers wrote"


def test_any_merge_failure_reaches_every_rank_not_just_rank_zero(monkeypatch):
    """A ``TypeError`` from a malformed fragment on rank 0 would leave peers in the broadcast until the watchdog."""

    def _malformed(_gathered):
        raise TypeError("fragment is not a (weight_map, bytes) pair")

    world = _FakeWorld(_stage_payloads(), combine=_malformed)
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank, outcome in enumerate(outcomes):
        assert isinstance(outcome, RuntimeError), f"rank {rank} did not raise ({outcome!r})"
        assert "TypeError" in str(outcome) and "fragment is not" in str(outcome), str(outcome)


def test_a_planted_collision_raises_on_every_rank_with_the_key_named(monkeypatch):
    """Otherwise peers hang until the watchdog and the checkpoint carries one writer's tensor under the other's
    name."""
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
    """Folding every fragment's bytes multiplies ``total_size`` while the map still round-trips."""
    stage0 = ({"model.layers.0.weight": "model-pp00000-of-00002-00001.safetensors"}, 100)
    stage1 = ({"model.layers.9.weight": "model-pp00001-of-00002-00001.safetensors"}, 250)
    world = _FakeWorld([stage0, stage0, stage1, stage1])
    world.install(monkeypatch)

    outcomes = world.run_all()

    for rank, outcome in enumerate(outcomes):
        assert outcome[1] == 350, f"rank {rank} counted {outcome[1]} bytes; each shard file contributes once"


def test_without_a_process_group_the_lone_fragment_is_the_index(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    assert _exchange(({"a": "f0.safetensors"}, 100)) == ({"a": "f0.safetensors"}, 100)


def test_a_lost_broadcast_raises_instead_of_returning_an_empty_index(monkeypatch):
    """``None`` would produce an index naming no key, a checkpoint that loads with every weight missing."""
    world = _FakeWorld(_stage_payloads())
    world.install(monkeypatch)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda object_list, src=0, group=None: None)

    world.rank = 1
    with pytest.raises(RuntimeError, match="never reached this rank"):
        _exchange(None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
