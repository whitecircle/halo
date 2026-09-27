"""``assert_consistent`` on a real 2-rank gloo group: agreeing ranks pass, a divergent value raises
on every rank under ``strict`` and names the probe's label.

Run: python tests/cpu/diagnostics/test_assert_consistent.py
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist

from src.diagnostics.debugging import assert_consistent, assert_tensor_shape_consistent
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=90)


def _worker(rank: int, tmp_dir: str) -> None:
    group = dist.group.WORLD
    assert_consistent({"step": 3}, group=group, label="agree", strict=True)
    assert_tensor_shape_consistent(torch.zeros(2, 4), group=group, label="same_shape", strict=True)
    try:
        assert_tensor_shape_consistent(torch.zeros(2, 4 + rank), group=group, label="attn_in", strict=True)
        verdict = "NO RAISE"
    except RuntimeError as error:
        verdict = str(error)
    with open(os.path.join(tmp_dir, f"rank{rank}.txt"), "w") as handle:
        handle.write(verdict)


def test_a_divergent_shape_raises_on_every_rank_and_agreement_passes(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    for rank in range(WORLD_SIZE):
        verdict = (tmp_path / f"rank{rank}.txt").read_text()
        assert "[assert_consistent:attn_in] divergent values across ranks" in verdict, f"rank {rank}: {verdict}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
