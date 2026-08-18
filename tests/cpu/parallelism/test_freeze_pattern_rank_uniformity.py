#!/usr/bin/env python
"""A freeze/unfreeze pattern that misses on ONE stage must abort every rank — on a real group.

``freeze_modules_by_patterns`` / ``unfreeze_modules_by_patterns`` route their verdict through
:func:`~src.distributed.runtime.reject_across_ranks` because the patterns are matched
against the LIVE module tree, which under pipeline parallelism is this rank's stage only. A pattern
naming a layer that lives on stage 0 misses on stage 1 — so the verdict is genuinely rank-local, and
a bare ``raise`` there takes one rank out while its peers block in the next collective until the
watchdog fires, with the real diagnostic printed nowhere.

``tests/cpu/models/test_freeze_patterns.py`` pins the verdicts (which patterns raise, what the
message says), but every one of its cases takes ``reject_across_ranks``'s ``world_size <= 1`` early
return, which degenerates to a plain local raise. Replacing both call sites with
``if reason: raise ValueError(reason)`` passes all of them. This is the test that fails for that
mutation: two ranks, real gloo, the pattern matching on rank 0 only.

The process-group timeout is deliberately short so a bare local raise FAILS instead of stalling the suite.

Run: python tests/cpu/parallelism/test_freeze_pattern_rank_uniformity.py  (or pytest -m cpu)
"""

from __future__ import annotations

import contextlib
import datetime
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from src.distributed.loading.peft_setup import freeze_modules_by_patterns, unfreeze_modules_by_patterns

WORLD_SIZE = 2
# One port per test — a reused port collides with the previous run's lingering TCPStore.
FREEZE_MISS_PORT = "29771"
UNFREEZE_MISS_PORT = "29772"
UNIFORM_HIT_PORT = "29773"
# Far below any plausible real wait: with a stage-local raise the surviving rank can only end in a
# gloo timeout at the modelled next collective, and the suite must not sit on it.
PG_TIMEOUT_SEC = 15


class _Stage(nn.Module):
    """A two-stage pipeline split: each rank holds the block named for it, never both."""

    def __init__(self, rank: int):
        super().__init__()
        self.add_module(f"block_{rank}", nn.Linear(4, 4))


def _init(rank: int, port: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )


def _worker(rank: int, tmp_dir: str, which: str, pattern: str, port: str) -> None:
    """Apply the pattern to this rank's stage, then enter the collective a real load would next hit."""
    _init(rank, port)
    model = _Stage(rank)
    reached_next_collective = False
    try:
        if which == "freeze":
            freeze_modules_by_patterns(model, [pattern])
        else:
            unfreeze_modules_by_patterns(model, [pattern])
        # Stands in for the next collective the loading path enters (an FSDP shard, a barrier): the
        # rank that did NOT raise arrives here alone under a stage-local raise.
        dist.barrier()
        reached_next_collective = True
        outcome = "NO RAISE"
    except BaseException as e:  # noqa: BLE001 — the outcome is the assertion subject
        outcome = f"{type(e).__name__}: {e}"

    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(f"{reached_next_collective}|{outcome}")
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _results(tmp_path, which: str, pattern: str, port: str) -> list[tuple[bool, str]]:
    mp.start_processes(
        _worker, args=(str(tmp_path), which, pattern, port), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            reached, outcome = fh.read().split("|", 1)
        out.append((reached == "True", outcome))
    return out


def test_a_freeze_pattern_that_misses_one_stage_takes_every_rank_down(tmp_path):
    """``block_0.*`` names a parameter only rank 0 holds. Rank 0's own verdict is clean, so it is the
    rank a stage-local raise would strand — it must raise too, with rank 1's real reason."""
    results = _results(tmp_path, which="freeze", pattern="block_0.*", port=FREEZE_MISS_PORT)

    for rank, (reached, outcome) in enumerate(results):
        assert outcome != "NO RAISE", f"rank {rank} sailed past a pattern that trains the wrong parameters"
        assert not reached, f"rank {rank} entered the next collective after a rejected freeze"
        assert "Timeout" not in outcome and "timed out" not in outcome, (
            f"rank {rank} only saw a timeout — the verdict never crossed the ranks: {outcome}"
        )
        assert "freeze_layers_patterns" in outcome, f"rank {rank} was not told which knob failed: {outcome}"
        assert "matched no parameter" in outcome, f"rank {rank} was not told the real cause: {outcome}"

    assert "1 of 2 rank(s)" in results[0][1], f"the verdict did not identify the failing rank: {results[0][1]}"


def test_an_unfreeze_pattern_that_misses_one_stage_takes_every_rank_down(tmp_path):
    """Its counterpart: ``block_0`` leaves rank 1 with nothing trainable, and that rank's stage —
    not the world — is what the predicate can see."""
    results = _results(tmp_path, which="unfreeze", pattern="block_0", port=UNFREEZE_MISS_PORT)

    for rank, (reached, outcome) in enumerate(results):
        assert outcome != "NO RAISE", f"rank {rank} sailed past a stage that would train nothing"
        assert not reached, f"rank {rank} entered the next collective after a rejected unfreeze"
        assert "Timeout" not in outcome and "timed out" not in outcome, f"rank {rank} only saw a timeout: {outcome}"
        assert "unfreeze_layers_patterns" in outcome, f"rank {rank} was not told which knob failed: {outcome}"

    assert "1 of 2 rank(s)" in results[0][1], f"the verdict did not identify the failing rank: {results[0][1]}"


def test_a_pattern_every_stage_carries_raises_on_nobody(tmp_path):
    """Anti-vacuity: the seam must not be a raise-always. ``block_*`` matches on both stages, so both
    ranks proceed through the next collective."""
    results = _results(tmp_path, which="freeze", pattern="block_*", port=UNIFORM_HIT_PORT)
    for rank, (reached, outcome) in enumerate(results):
        assert outcome == "NO RAISE", f"rank {rank} raised on a pattern its stage carries: {outcome}"
        assert reached, f"rank {rank} did not reach the next collective"


def test_the_local_verdict_is_unchanged_by_the_routing():
    """The seam must not have moved the decision: single-process, a hit still freezes and a miss
    still raises ``ValueError`` (the type every caller and config gate matches on)."""
    model = _Stage(0)
    freeze_modules_by_patterns(model, ["block_0.*"])
    assert not any(p.requires_grad for p in model.parameters())

    with pytest.raises(ValueError, match="matched no parameter"):
        freeze_modules_by_patterns(_Stage(0), ["block_9.*"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
