"""The checkpoint modality probe is agreed across ranks, and a split verdict fails every rank.

A rank that cannot read the checkpoint config (its node lacks the source) decides text-vs-multimodal
off the name heuristic, which can disagree with the config its peers read. The ranks would then build
different model classes and enter different store phases, which surfaces only as a coordination
timeout. The probe joins its verdict instead and raises on every rank, naming the cause.

    python tests/cpu/models/test_modality_probe_agreement.py
"""

import json
import os

import pytest
from transformers import AutoConfig, Qwen3Config

import src.models.modality as modality
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG

_RANKS = 2


def _probe_on_rank(rank: int, checkpoint: str, unreadable_rank: int | None, outcome_dir: str) -> None:
    if rank == unreadable_rank:
        # This rank's node does not hold the source: the config read fails, the name decides.
        def missing(*_args, **_kwargs):
            raise OSError(f"{checkpoint} is not on this node")

        AutoConfig.from_pretrained = missing
    try:
        _config, verdict = modality.probe_checkpoint(checkpoint)
        outcome = {"verdict": verdict}
    except RuntimeError as exc:
        outcome = {"error": str(exc)}
    with open(os.path.join(outcome_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(outcome, handle)


def _run(checkpoint: str, unreadable_rank: int | None, outcome_dir: str) -> list[dict]:
    run_gloo_ranks(_probe_on_rank, _RANKS, checkpoint, unreadable_rank, outcome_dir)
    outcomes = []
    for rank in range(_RANKS):
        with open(os.path.join(outcome_dir, f"rank{rank}.json")) as handle:
            outcomes.append(json.load(handle))
    return outcomes


@pytest.fixture
def text_only_checkpoint_with_a_vlm_name(tmp_path):
    """A registered text-only config under a name the heuristic reads as multimodal."""
    checkpoint = tmp_path / "tiny-qwen3-vl"
    Qwen3Config(**TINY_QWEN3_CONFIG).save_pretrained(checkpoint)
    return str(checkpoint)


def test_every_rank_reading_the_config_agrees(text_only_checkpoint_with_a_vlm_name, tmp_path):
    """Anti-vacuity: the join passes a uniform verdict through, here the config's text-only answer
    over the name's multimodal hint."""
    outcomes = _run(text_only_checkpoint_with_a_vlm_name, None, str(tmp_path))
    assert outcomes == [{"verdict": False}] * _RANKS


def test_a_rank_that_cannot_read_the_source_fails_the_probe_on_every_rank(
    text_only_checkpoint_with_a_vlm_name, tmp_path
):
    outcomes = _run(text_only_checkpoint_with_a_vlm_name, 1, str(tmp_path))
    assert all("multimodal on some ranks and text-only on others" in outcome.get("error", "") for outcome in outcomes)
    assert "could not read its config" in outcomes[1]["error"]
    assert "read its config: text-only" in outcomes[0]["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
