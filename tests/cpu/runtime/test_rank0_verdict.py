"""``rank0_verdict`` on a real 2-rank gloo group: rank 0's result reaches every rank, and only rank 0
probes; a raise in rank 0's probe raises the caller's exception type on every rank, naming the cause.

Run: python tests/cpu/runtime/test_rank0_verdict.py
"""

import pytest

from src.runtime.consensus import rank0_verdict
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2


def _verdict_worker(rank: int) -> None:
    probed: list[int] = []

    def probe() -> dict[str, str]:
        probed.append(rank)
        return {"format": "huggingface"}

    assert rank0_verdict(probe, "format probe") == {"format": "huggingface"}
    assert probed == ([0] if rank == 0 else []), f"rank {rank} probed {probed}"


def _failure_worker(rank: int) -> None:
    def probe() -> str:
        raise OSError("torn index")

    with pytest.raises(
        ValueError, match=r"format probe failed on 1 of 2 rank\(s\) \[0\]\. First \(rank 0\): OSError: torn index"
    ):
        rank0_verdict(probe, "format probe", ValueError)


def test_rank0s_verdict_reaches_every_rank_and_only_rank0_probes():
    run_gloo_ranks(_verdict_worker, WORLD_SIZE)


def test_a_rank0_probe_failure_raises_on_every_rank():
    run_gloo_ranks(_failure_worker, WORLD_SIZE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
