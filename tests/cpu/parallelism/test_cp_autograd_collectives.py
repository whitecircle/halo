#!/usr/bin/env python
"""Ulysses' all-to-all moves real data — pin what it moves, on CPU.

The collective in ``src/distributed/context_parallel/all_to_all.py`` cannot run for real here: gloo
implements no alltoall at all, and the CP GPU suites only ever compare an end-to-end loss, where a
transposed head/sequence axis or a non-inverted backward is one contribution among many.

So it is driven against a simulated world instead. ``_CollectiveWorld`` runs every rank's
computation TWICE — once recording what each rank contributes to a collective, once serving each
rank from that record — which reproduces the real collective's semantics while assuming nothing
about how the code under test splits its input. That is the point: the reference here is the
GLOBAL tensor the shards come from, not a second copy of the implementation's narrow/cat.

Every case runs at ``cp_size`` 2 and 4. Two ranks make ``scatter``/``gather`` chunk indices coincide
often enough that a rank-ordering error can survive, and 4 is the shape a cross-domain job actually
runs, so neither width alone is a sufficient gate.

What these tests pin:

* ``UlyssesAllToAll`` forward turns a sequence-sharded tensor into a head-sharded one — the actual
  Ulysses contract, not merely "some permutation".
* Its backward is the INVERSE, so a head-sharded gradient comes back sequence-sharded. The
  sign-flip check below is what keeps that assertion from passing on an all-zero gradient.

Run: pytest tests/cpu/parallelism/test_cp_autograd_collectives.py
"""

import pytest
import torch
import torch.distributed as dist

from src.distributed.context_parallel.all_to_all import UlyssesAllToAll, heads_to_sequence, sequence_to_heads

CP_SIZES = (2, 4)
BATCH = 2
SEQ = 8
HEADS = 4
HEAD_DIM = 3
GROUP = object()  # never dereferenced: the functions forward the group straight to the collective


class _CollectiveWorld:
    """A single-process stand-in for a ``cp_size``-rank group, for one collective at a time.

    A real all-to-all hands rank ``r``, for every peer ``i``, the chunk peer ``i`` placed at index
    ``r``. That needs every rank's contribution before any rank's output exists, which one process
    cannot produce in one pass — so :meth:`run` makes two, recording contributions in the first and
    serving them in the second.

    Recording what the code under test actually passed (rather than re-deriving it) is what keeps
    this from re-implementing the narrow/cat it is meant to check.
    """

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        self.rank = 0
        self.recording = True
        self.transcript: dict[int, list[list[torch.Tensor]]] = {}
        self._call = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: self.cp_size)
        monkeypatch.setattr(dist, "all_to_all", self._all_to_all)

    def _contribute(self, tensors: list[torch.Tensor]) -> list[list[torch.Tensor]] | None:
        """Record this rank's contribution, or return the whole world's for this call index."""
        if self.recording:
            self.transcript.setdefault(self.rank, []).append([t.detach().clone() for t in tensors])
            return None
        # Ranks run the same program, so call ``n`` on every rank is the same collective.
        return [self.transcript[peer][self._call] for peer in range(self.cp_size)]

    def _all_to_all(self, output_list, input_list, group=None):
        del group
        peers = self._contribute(input_list)
        if peers is not None:
            for peer, out in enumerate(output_list):
                out.copy_(peers[peer][self.rank])
        self._call += 1

    def run(self, per_rank):
        """``per_rank(rank)`` for every rank, under a faithful collective; returns each rank's result.

        The recording pass discards its results: the outputs it produced were never filled in.
        """
        for pass_is_recording in (True, False):
            self.recording = pass_is_recording
            results = []
            for rank in range(self.cp_size):
                self.rank = rank
                self._call = 0
                results.append(per_rank(rank))
        return results


def _seq_shard(tensor: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    """Rank's slice of ``[batch, seq, ...]`` split along the sequence — CP's input layout."""
    chunk = SEQ // cp_size
    return tensor[:, rank * chunk : (rank + 1) * chunk].contiguous()


def _head_shard(tensor: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    """Rank's slice of ``[batch, seq, heads, dim]`` split along the heads — Ulysses' output layout."""
    chunk = HEADS // cp_size
    return tensor[:, :, rank * chunk : (rank + 1) * chunk].contiguous()


def _scatter_heads(world: _CollectiveWorld, full: torch.Tensor) -> list[torch.Tensor]:
    """Every rank's forward all-to-all: sequence-sharded in, head-sharded out."""
    return world.run(lambda rank: UlyssesAllToAll.apply(_seq_shard(full, rank, world.cp_size), GROUP, 2, 1))


def _grads_from_head_sharded_upstream(world: _CollectiveWorld, full, upstream) -> list[torch.Tensor]:
    """Every rank's input gradient for a head-sharded upstream gradient."""

    def backward_on(rank: int) -> torch.Tensor:
        local = _seq_shard(full, rank, world.cp_size).requires_grad_(True)
        UlyssesAllToAll.apply(local, GROUP, 2, 1).backward(_head_shard(upstream, rank, world.cp_size))
        return local.grad

    return world.run(backward_on)


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_all_to_all_turns_a_sequence_shard_into_a_head_shard(monkeypatch, cp_size):
    """The Ulysses contract itself: ``[b, seq/CP, heads, d]`` in, ``[b, seq, heads/CP, d]`` out.

    Attention runs on the head shard, so every rank must come out holding its OWN heads over the
    WHOLE sequence. Checking against slices of one global tensor is what makes a transposed or
    rank-rotated result fail here — a shape-only check passes on all three.
    """
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    full = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)

    outputs = _scatter_heads(world, full)

    for rank, out in enumerate(outputs):
        assert out.shape == (BATCH, SEQ, HEADS // cp_size, HEAD_DIM), f"rank {rank} got {tuple(out.shape)}"
        torch.testing.assert_close(out, _head_shard(full, rank, cp_size))


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_all_to_all_backward_is_the_inverse_scatter(monkeypatch, cp_size):
    """A head-sharded gradient must return sequence-sharded, or every rank trains on another's tokens.

    The backward re-applies the same collective with ``scatter_dim`` and ``gather_dim`` SWAPPED.
    Dropping that swap is invisible at the loss — the gradient still has a plausible norm — and
    silently attributes each rank's gradient to the wrong slice of the sequence.
    """
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    full = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)
    upstream = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)

    grads = _grads_from_head_sharded_upstream(world, full, upstream)

    for rank, grad in enumerate(grads):
        assert grad is not None, f"rank {rank} received no input gradient at all"
        expected = _seq_shard(upstream, rank, cp_size)
        assert grad.shape == expected.shape, f"rank {rank} got {tuple(grad.shape)}, want {tuple(expected.shape)}"
        torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_the_backward_assertion_would_catch_a_corrupted_gradient(monkeypatch, cp_size):
    """Mutation check: the comparison above must discriminate, not accept anything.

    A backward that produced zeros — the failure mode of a dropped collective — matches a
    sign-flipped reference just as well as the true one, so the test above would certify nothing.
    Negating the expectation has to make it fail.
    """
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    full = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)
    upstream = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)

    grads = _grads_from_head_sharded_upstream(world, full, upstream)

    for rank, grad in enumerate(grads):
        assert grad.abs().max() > 0, f"rank {rank}'s gradient is all zeros; the comparison is vacuous"
        with pytest.raises(AssertionError):
            torch.testing.assert_close(grad, -_seq_shard(upstream, rank, cp_size))


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_all_to_all_round_trips_back_to_the_sequence_shard(monkeypatch, cp_size):
    """Ulysses attention scatters in and gathers back out; the pair must be the identity.

    ``sequence_to_heads`` scatters dim 2 and gathers dim 1; ``heads_to_sequence`` reverses it.
    Any rank-ordering error that the two directions do not share survives the round trip.
    """
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    full = torch.randn(BATCH, SEQ, HEADS, HEAD_DIM)

    scattered = _scatter_heads(world, full)
    world.transcript.clear()
    gathered = world.run(lambda rank: UlyssesAllToAll.apply(scattered[rank], GROUP, 1, 2))

    for rank, out in enumerate(gathered):
        torch.testing.assert_close(out, _seq_shard(full, rank, cp_size))


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_the_attention_wrappers_swap_into_heads_and_back(monkeypatch, cp_size):
    """What the CP attention forward calls: Q/K/V go sequence-sharded → head-sharded (K and V fused
    into one collective and split back exactly), and the attention output returns head-sharded →
    sequence-sharded."""
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    q, k, v = (torch.randn(BATCH, SEQ, HEADS, HEAD_DIM) for _ in range(3))

    swapped = world.run(lambda rank: sequence_to_heads(*(_seq_shard(t, rank, cp_size) for t in (q, k, v)), GROUP))
    for rank, outputs in enumerate(swapped):
        for name, full, out in zip("qkv", (q, k, v), outputs, strict=True):
            assert torch.equal(out, _head_shard(full, rank, cp_size)), (rank, name)

    # A fresh world: each one serves the call indices of a single program.
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    restored = world.run(lambda rank: heads_to_sequence(_head_shard(q, rank, cp_size), GROUP))
    for rank, out in enumerate(restored):
        assert torch.equal(out, _seq_shard(q, rank, cp_size)), rank


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
