#!/usr/bin/env python
"""CPU test: per-step trainer metrics survive the pipeline.

Under PP only the LAST stage runs the loss closure, so a trainer's per-step metrics exist on that
stage alone — while HF logs from global rank 0, which sits on the FIRST stage. The mixin pins the
metric names at setup and broadcasts the values down the chain every step; these tests drive that
against a fake two-stage chain and fail if the metrics stop arriving, arrive with the wrong values,
or stop being rank-uniform.

    python tests/cpu/parallelism/test_pp_step_metrics.py
"""

import pytest
import torch

from src.distributed.pipeline_parallel.losses import PPLossAdapter, causal_lm_token_loss
from src.trainers.mixins.pipeline import PipelineTrainerMixin

LAST_STAGE_RANK = 1


class _FakeChain:
    """The two collectives ``_pp_share_step_metrics`` / ``_pp_pin_metric_keys`` use.

    ``broadcast`` copies the last stage's buffer into whichever rank is currently calling, which is
    what makes "every stage ends up with the last stage's values" testable in one process.
    """

    def __init__(self):
        self.source_values = None
        self.source_keys = None
        self.broadcasts = 0
        self.object_broadcasts = 0

    def broadcast(self, tensor, src, group):
        self.broadcasts += 1
        if self.source_values is None:
            self.source_values = tensor.clone()
        else:
            tensor.copy_(self.source_values)

    def broadcast_object_list(self, objects, src, group):
        self.object_broadcasts += 1
        if self.source_keys is None:
            self.source_keys = objects[0]
        else:
            objects[0] = self.source_keys


class _Stage(PipelineTrainerMixin):
    """One pipeline rank: the mixin's metric plumbing over a fake adapter and a metric store."""

    def __init__(self, metrics_fn, is_last: bool):
        self._pp_adapter = PPLossAdapter(token_loss_fn=causal_lm_token_loss, metrics_fn=metrics_fn)
        self._is_last = is_last
        self._pp_chain_group = "chain"
        self.stored: list[tuple[dict, str]] = []

    @property
    def _pp_last_stage_rank(self) -> int:
        return LAST_STAGE_RANK

    def store_metrics(self, metrics, train_eval="train"):
        self.stored.append(({key: float(value) for key, value in metrics.items()}, train_eval))


@pytest.fixture
def chain(monkeypatch):
    fake = _FakeChain()
    monkeypatch.setattr("src.trainers.mixins.pipeline.dist", fake)
    monkeypatch.setattr("src.trainers.mixins.pipeline.torch.cuda.current_device", lambda: "cpu")
    monkeypatch.setattr(
        "src.trainers.mixins.pipeline.reject_across_ranks",
        lambda reason, what, exc_type=RuntimeError: None
        if reason is None
        else (_ for _ in ()).throw(exc_type(reason)),
    )
    return fake


def _metrics(**values):
    return lambda: {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}


def test_every_stage_records_the_last_stages_values(chain):
    """The non-last stage has no loss closure, so its own reading is zeros; it must log the last
    stage's numbers instead — otherwise rank 0's log line carries silent zeros."""
    last = _Stage(_metrics(**{"rewards/chosen": 1.5, "sft_loss/chosen": 0.25}), is_last=True)
    first = _Stage(_metrics(**{"rewards/chosen": 0.0, "sft_loss/chosen": 0.0}), is_last=False)
    for stage in (last, first):
        stage._pp_pin_metric_keys()

    # The pin probes the store with an empty dict, so a trainer with nowhere to put metrics fails
    # at setup rather than on the first step.
    assert first.stored == [({}, "train")]

    last._pp_share_step_metrics("train")  # the source rank fills the fake's buffer
    first._pp_share_step_metrics("train")

    expected = ({"rewards/chosen": 1.5, "sft_loss/chosen": 0.25}, "train")
    assert first.stored[-1] == expected
    assert last.stored[-1] == expected


def test_one_values_only_collective_per_step_not_an_object_hop(chain):
    """The per-step path must stay a fixed-size tensor broadcast: the object hop (a pickle plus a
    device round trip) is what the setup-time key pin exists to keep off it."""
    stage = _Stage(_metrics(a=1.0, b=2.0), is_last=True)
    stage._pp_pin_metric_keys()
    objects_after_setup = chain.object_broadcasts

    for _ in range(3):
        stage._pp_share_step_metrics("train")

    assert chain.broadcasts == 3, "one values broadcast per step"
    assert chain.object_broadcasts == objects_after_setup, "no object hop on the per-step path"


def test_metric_names_that_disagree_across_the_chain_are_refused(chain):
    """A data-derived key set would pair one stage's names with another's numbers, silently."""
    last = _Stage(_metrics(a=1.0, b=2.0), is_last=True)
    last._pp_pin_metric_keys()
    drifted = _Stage(_metrics(a=0.0, c=0.0), is_last=False)

    with pytest.raises(ValueError, match="differ from the last stage"):
        drifted._pp_pin_metric_keys()


def test_dropping_a_pinned_metric_mid_run_raises(chain):
    """The broadcast carries values positionally; a metrics_fn that stops reporting one name would
    otherwise shift every later value onto the wrong series."""
    reported = {"a": torch.ones(()), "b": torch.zeros(())}
    stage = _Stage(lambda: dict(reported), is_last=True)
    stage._pp_pin_metric_keys()

    reported.pop("b")
    with pytest.raises(KeyError, match="dropped the pinned"):
        stage._pp_share_step_metrics("train")


def test_no_metrics_fn_costs_no_collective(chain):
    """The causal-LM contract declares none; the per-step path must then not broadcast at all."""
    stage = _Stage(None, is_last=True)
    stage._pp_pin_metric_keys()
    stage._pp_share_step_metrics("train")

    assert chain.broadcasts == 0 and chain.object_broadcasts == 0
    assert stage.stored == [], "no key pin, so not even the setup probe"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
