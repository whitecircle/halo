#!/usr/bin/env python
"""Ulysses' two collectives move real data — pin what they move, on CPU.

``src/distributed/context_parallel/autograd.py`` had no CPU coverage beyond a divisibility guard,
because neither collective can run for real here: gloo implements no alltoall at all, and the CP GPU
suites only ever compare an end-to-end loss, where a transposed head/sequence axis or a
non-inverted backward is one contribution among many.

So both are driven against a simulated world instead. ``_CollectiveWorld`` runs every rank's
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
* ``gather_pos_embeddings`` reassembles the full sequence in RANK-MAJOR order and keeps cos and sin
  on their own halves of the fused buffer.

Run: pytest tests/cpu/parallelism/test_cp_autograd_collectives.py
"""

import pytest
import torch
import torch.distributed as dist

from src.distributed.context_parallel.autograd import UlyssesAllToAll, gather_pos_embeddings

CP_SIZES = (2, 4)
BATCH = 2
SEQ = 8
HEADS = 4
HEAD_DIM = 3
GROUP = object()  # never dereferenced: both functions forward the group straight to the collective


class _CollectiveWorld:
    """A single-process stand-in for a ``cp_size``-rank group, for one collective at a time.

    A real all-to-all hands rank ``r``, for every peer ``i``, the chunk peer ``i`` placed at index
    ``r``; a real all-gather concatenates every rank's buffer in rank order. Both need every rank's
    contribution before any rank's output exists, which one process cannot produce in one pass — so
    :meth:`run` makes two, recording contributions in the first and serving them in the second.

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
        monkeypatch.setattr(dist, "all_gather_into_tensor", self._all_gather_into_tensor)

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

    def _all_gather_into_tensor(self, output, input, group=None):  # noqa: A002 — torch's parameter name
        del group
        peers = self._contribute([input])
        if peers is not None:
            # Rank-major, exactly as the real collective lays it out.
            output.copy_(torch.cat([peers[peer][0] for peer in range(self.cp_size)]))
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

    ``base_layer`` follows ``(scatter_dim=2, gather_dim=1)`` with ``(scatter_dim=1, gather_dim=2)``.
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
def test_gather_pos_embeddings_rebuilds_the_full_sequence_in_rank_order(monkeypatch, cp_size):
    """cos/sin must come back as the unsharded originals, in rank-major sequence order.

    The legacy CP path applies RoPE after the all-to-all, so each rank needs the position embeddings
    for the WHOLE sequence. The function all-gathers a flat fused buffer and rebuilds the sequence
    axis with a permute; without it the result is batch-interleaved — same shape, same values, wrong
    positions, and RoPE then rotates every token to the wrong angle.
    """
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    # Distinct ranges so a cos/sin mix-up in the fused split cannot pass as noise.
    cos_full = torch.rand(BATCH, SEQ, HEAD_DIM)
    sin_full = torch.rand(BATCH, SEQ, HEAD_DIM) + 10.0

    outputs = world.run(
        lambda rank: gather_pos_embeddings(
            _seq_shard(cos_full, rank, cp_size), _seq_shard(sin_full, rank, cp_size), GROUP, cp_size
        )
    )

    for rank, (cos, sin) in enumerate(outputs):
        assert cos.shape == (BATCH, SEQ, HEAD_DIM), f"rank {rank} got {tuple(cos.shape)}"
        torch.testing.assert_close(cos, cos_full, msg=f"rank {rank} rebuilt cos in the wrong order")
        torch.testing.assert_close(sin, sin_full, msg=f"rank {rank} rebuilt sin in the wrong order")


def test_gather_pos_embeddings_batch_interleaving_would_be_caught(monkeypatch):
    """Mutation check for the permute, whose absence preserves both shape and value multiset.

    Reassembling the flat all-gather WITHOUT the rank-major → sequence-major permute yields a tensor
    of the right shape holding exactly the right numbers in the wrong positions. If the assertion
    above could not separate those two, it would certify only the shape.
    """
    cp_size = 4
    world = _CollectiveWorld(cp_size)
    world.install(monkeypatch)
    cos_full = torch.rand(BATCH, SEQ, HEAD_DIM)
    sin_full = torch.rand(BATCH, SEQ, HEAD_DIM)

    outputs = world.run(
        lambda rank: gather_pos_embeddings(
            _seq_shard(cos_full, rank, cp_size), _seq_shard(sin_full, rank, cp_size), GROUP, cp_size
        )
    )

    chunk = SEQ // cp_size
    # What dropping the permute produces: the flat buffer read as [batch, seq, dim] directly.
    unpermuted = torch.cat([_seq_shard(cos_full, rank, cp_size).reshape(-1) for rank in range(cp_size)]).view(
        BATCH, SEQ, HEAD_DIM
    )
    assert not torch.equal(unpermuted, cos_full), (
        f"the {BATCH}x{chunk}x{HEAD_DIM} shard layout makes the permute a no-op here, so this "
        f"fixture cannot detect its absence"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(unpermuted, outputs[0][0])


def test_gather_pos_embeddings_is_a_no_op_without_context_parallelism():
    """``cp_size == 1`` must skip the collective entirely — no group is even constructed there."""
    cos = torch.rand(BATCH, SEQ, HEAD_DIM)
    sin = torch.rand(BATCH, SEQ, HEAD_DIM)

    out_cos, out_sin = gather_pos_embeddings(cos, sin, GROUP, 1)

    assert out_cos is cos and out_sin is sin


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
