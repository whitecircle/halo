#!/usr/bin/env python
"""A freeze/unfreeze pattern missing on one stage aborts every rank, on real gloo.
``tests/cpu/models/test_freeze_patterns.py`` takes the single-rank early return, so a bare raise passes there."""

from __future__ import annotations

import datetime
import os

import pytest
import torch.distributed as dist
from torch import nn

from halo.model_loading.parallel.freeze_patterns import freeze_parameters_by_patterns, unfreeze_modules_by_patterns
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
# With a stage-local raise the surviving rank ends in a gloo timeout at the modelled next collective.
PG_TIMEOUT_SEC = 15


class _Stage(nn.Module):
    """Each rank holds the block named for it."""

    def __init__(self, rank: int):
        super().__init__()
        self.add_module(f"block_{rank}", nn.Linear(4, 4))


def _worker(rank: int, tmp_dir: str, which: str, pattern: str) -> None:
    model = _Stage(rank)
    reached_next_collective = False
    try:
        if which == "freeze":
            freeze_parameters_by_patterns(model, [pattern])
        else:
            unfreeze_modules_by_patterns(model, [pattern])
        # The rank that did not raise arrives here alone under a stage-local raise.
        dist.barrier()
        reached_next_collective = True
        outcome = "NO RAISE"
    except BaseException as e:  # noqa: BLE001 — the outcome is the assertion subject
        outcome = f"{type(e).__name__}: {e}"

    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(f"{reached_next_collective}|{outcome}")


def _results(tmp_path, which: str, pattern: str) -> list[tuple[bool, str]]:
    run_gloo_ranks(
        _worker, WORLD_SIZE, str(tmp_path), which, pattern, pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            reached, outcome = fh.read().split("|", 1)
        out.append((reached == "True", outcome))
    return out


def test_a_freeze_pattern_that_misses_one_stage_takes_every_rank_down(tmp_path):
    """Rank 0's own verdict is clean, so it is the rank a stage-local raise would strand."""
    results = _results(tmp_path, which="freeze", pattern="block_0.*")

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
    """``block_0`` leaves rank 1 with nothing trainable, which its stage alone can see."""
    results = _results(tmp_path, which="unfreeze", pattern="block_0")

    for rank, (reached, outcome) in enumerate(results):
        assert outcome != "NO RAISE", f"rank {rank} sailed past a stage that would train nothing"
        assert not reached, f"rank {rank} entered the next collective after a rejected unfreeze"
        assert "Timeout" not in outcome and "timed out" not in outcome, f"rank {rank} only saw a timeout: {outcome}"
        assert "unfreeze_layers_patterns" in outcome, f"rank {rank} was not told which knob failed: {outcome}"

    assert "1 of 2 rank(s)" in results[0][1], f"the verdict did not identify the failing rank: {results[0][1]}"


def test_a_pattern_every_stage_carries_raises_on_nobody(tmp_path):
    """Anti-vacuity: the seam must not raise always."""
    results = _results(tmp_path, which="freeze", pattern="block_*")
    for rank, (reached, outcome) in enumerate(results):
        assert outcome == "NO RAISE", f"rank {rank} raised on a pattern its stage carries: {outcome}"
        assert reached, f"rank {rank} did not reach the next collective"


def test_the_local_verdict_is_unchanged_by_the_routing():
    """``ValueError`` is the type every caller and config gate matches on."""
    model = _Stage(0)
    freeze_parameters_by_patterns(model, ["block_0.*"])
    assert not any(p.requires_grad for p in model.parameters())

    with pytest.raises(ValueError, match="matched no parameter"):
        freeze_parameters_by_patterns(_Stage(0), ["block_9.*"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
