#!/usr/bin/env python
"""The head's load on real gloo ranks: every rank enters the same collectives, and a tail one rank cannot read raises on
all of them with the reason instead of leaving a peer in the next collective."""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from tests.common.gloo import run_gloo_ranks
from tests.common.mtp_heads import MTP_TINY_FAMILIES, write_checkpoint_with_tail
from tests.common.tiny_models import tiny_family_model

_WORLD = 2
# A rank stuck in a collective its peer left behind reaches this; a rank-uniform raise never does.
_COLLECTIVE_TIMEOUT = datetime.timedelta(seconds=30)
_NAME = "qwen3_5_moe_text"


def _load_on_rank(rank: int, directory: str, source_of_rank: list[str]) -> None:
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[_NAME]).to(torch.bfloat16)
    path = source_of_rank[rank]
    try:
        head = attach_mtp_head(
            model,
            MtpHeadRequest(MtpObjective(weight=1.0)),
            pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: None),
            source=ModelSource(path, None, path),
            dtype=torch.bfloat16,
            wraps_experts=False,
            init_from_scratch=False,
            preserve_checkpoint_precision=False,
        )
        outcome = {"sum": float(head.fc.weight.float().abs().sum())}
    except RuntimeError as error:
        outcome = {"refused": str(error)}
    (Path(directory) / f"rank_{rank}.json").write_text(json.dumps(outcome))


def _outcomes(tmp_path, sources: list[str]) -> list[dict]:
    run_gloo_ranks(_load_on_rank, _WORLD, str(tmp_path), sources, pg_timeout=_COLLECTIVE_TIMEOUT)
    return [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(_WORLD)]


def test_every_rank_loads_one_head(tmp_path):
    tail = write_checkpoint_with_tail(_NAME, tmp_path / "ckpt", per_expert=True)
    first, second = _outcomes(tmp_path, [str(tmp_path / "ckpt")] * _WORLD)
    assert first == second
    assert first["sum"] == pytest.approx(float(tail["mtp.fc.weight"].float().abs().sum()))


@pytest.mark.parametrize("failing_rank", range(_WORLD))
def test_a_tail_one_rank_cannot_read_refuses_on_every_rank(tmp_path, failing_rank):
    write_checkpoint_with_tail(_NAME, tmp_path / "good", per_expert=True)
    write_checkpoint_with_tail(
        _NAME, tmp_path / "bad", per_expert=True, edit=lambda stored: stored.pop("mtp.fc.weight")
    )
    sources = [str(tmp_path / ("bad" if rank == failing_rank else "good")) for rank in range(_WORLD)]
    for outcome in _outcomes(tmp_path, sources):
        assert "lacks 1 tensor" in outcome.get("refused", ""), outcome


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
