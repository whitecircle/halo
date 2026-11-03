#!/usr/bin/env python
"""Per-step trainer metrics survive the pipeline: they exist on the last stage while HF logs from rank 0 on the first.
The mixin pins the names at setup and broadcasts values down the chain every step."""

from types import SimpleNamespace

import pytest
import torch

from halo.distributed.pipeline_parallel.stage_losses import causal_lm_token_loss
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin

LAST_STAGE_RANK = 1


class _FakeChain:
    """``broadcast`` copies the last stage's buffer into the caller, making the outcome testable in one process."""

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
    FORWARD_ROWS_PER_EXAMPLE = 1

    def __init__(self, metrics_fn, is_last: bool):
        self._pp_adapter = PipelineLossAdapter(token_loss_fn=causal_lm_token_loss, metrics_fn=metrics_fn)
        self._is_last = is_last
        self._pp_chain_group = "chain"
        self.stored: list[tuple[dict, str]] = []

    @property
    def _pp_last_stage_rank(self) -> int:
        return LAST_STAGE_RANK

    def store_metrics(self, metrics, train_eval="train", rows=1):
        self.stored.append(({key: float(value) for key, value in metrics.items()}, train_eval))
        self.stored_rows = rows


@pytest.fixture
def chain(monkeypatch):
    fake = _FakeChain()
    monkeypatch.setattr("halo.trainers.parallelism.pipeline_trainer.dist", fake)
    monkeypatch.setattr("halo.trainers.parallelism.pipeline_trainer.torch.cuda.current_device", lambda: "cpu")
    monkeypatch.setattr(
        "halo.trainers.parallelism.pipeline_trainer.reject_across_ranks",
        lambda reason, what, exc_type=RuntimeError: None
        if reason is None
        else (_ for _ in ()).throw(exc_type(reason)),
    )
    return fake


def _metrics(**values):
    return lambda: {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}


def test_every_stage_records_the_last_stages_values(chain):
    """Otherwise rank 0's log line carries silent zeros."""
    last = _Stage(_metrics(**{"rewards/chosen": 1.5, "sft_loss/chosen": 0.25}), is_last=True)
    first = _Stage(_metrics(**{"rewards/chosen": 0.0, "sft_loss/chosen": 0.0}), is_last=False)
    for stage in (last, first):
        stage._pp_pin_metric_keys()

    # A trainer with nowhere to put metrics fails at setup rather than on the first step.
    assert first.stored == [({}, "train")]

    last._pp_share_step_metrics("train")  # The source rank fills the fake's buffer.
    first._pp_share_step_metrics("train")

    expected = ({"rewards/chosen": 1.5, "sft_loss/chosen": 0.25}, "train")
    assert first.stored[-1] == expected
    assert last.stored[-1] == expected


def test_one_values_only_collective_per_step_not_an_object_hop(chain):
    """An object hop (a pickle plus a device round trip) is what the key pin keeps off this path."""
    stage = _Stage(_metrics(a=1.0, b=2.0), is_last=True)
    stage._pp_pin_metric_keys()
    objects_after_setup = chain.object_broadcasts

    for _ in range(3):
        stage._pp_share_step_metrics("train")

    assert chain.broadcasts == 3, "one values broadcast per step"
    assert chain.object_broadcasts == objects_after_setup, "no object hop on the per-step path"


def test_metric_names_that_disagree_across_the_chain_are_refused(chain):
    """A data-derived key set would pair one stage's names with another's numbers."""
    last = _Stage(_metrics(a=1.0, b=2.0), is_last=True)
    last._pp_pin_metric_keys()
    drifted = _Stage(_metrics(a=0.0, c=0.0), is_last=False)

    with pytest.raises(ValueError, match="differ from the last stage"):
        drifted._pp_pin_metric_keys()


def test_dropping_a_pinned_metric_mid_run_raises(chain):
    """The broadcast is positional, so a dropped name shifts every later value onto the wrong series."""
    reported = {"a": torch.ones(()), "b": torch.zeros(())}
    stage = _Stage(lambda: dict(reported), is_last=True)
    stage._pp_pin_metric_keys()

    reported.pop("b")
    with pytest.raises(KeyError, match="dropped the pinned"):
        stage._pp_share_step_metrics("train")


def test_no_metrics_fn_costs_no_collective(chain):
    stage = _Stage(None, is_last=True)
    stage._pp_pin_metric_keys()
    stage._pp_share_step_metrics("train")

    assert chain.broadcasts == 0 and chain.object_broadcasts == 0
    assert stage.stored == [], "no key pin, so not even the setup probe"


class _Runtime:
    """The loss-only schedule records the batch and normalizer; the forward-only one returns per-token logits."""

    def eval_loss(self, input_ids, labels, *, attention_mask, position_ids, num_items_in_batch, extra_targets):
        self.input_ids, self.labels, self.count = input_ids, labels, num_items_in_batch
        return torch.zeros(())

    def forward_only(self, input_ids, *, attention_mask, position_ids):
        self.input_ids = input_ids
        return torch.zeros(*input_ids.shape, 5)


class _PairedStage(_Stage):
    FORWARD_ROWS_PER_EXAMPLE = 2


def _eval_stage(real_examples: int, compute_metrics=None, pairs: int = 2) -> _Stage:
    seen = {}
    stage = _PairedStage(_metrics(m=1.0), is_last=True)
    stage._pp_adapter = PipelineLossAdapter(
        token_loss_fn=causal_lm_token_loss,
        step_state_fn=lambda inputs: seen.update(state_rows=inputs["input_ids"].size(0)),
        eval_normalizer=lambda inputs: seen.setdefault("normalizer_rows", inputs["labels"].size(0)),
        metrics_fn=_metrics(m=1.0),
    )
    stage.seen = seen
    stage._pp_runner = _Runtime()
    stage.args = SimpleNamespace(per_device_train_batch_size=pairs)
    stage.data_collator = SimpleNamespace(pad_values={"input_ids": 0, "labels": LABEL_IGNORE_INDEX})
    stage.compute_metrics = compute_metrics
    stage._prepare_inputs = lambda inputs: inputs
    stage._pp_broadcast_loss_from_last_stage = lambda loss: loss
    stage._pp_broadcast_output_from_last_stage = lambda tensor: tensor
    stage.eval_split_rows = lambda num_rows: real_examples
    return stage


def test_an_eval_split_s_padding_examples_leave_the_step_as_inert_rows(chain):
    """Pair 1 pads the final round: state and normalizer see pair 0's rows alone, the schedule scores pair 1 as
    filler."""
    stage = _eval_stage(real_examples=1)
    stage._pp_pin_metric_keys()
    labels = torch.arange(1, 13).view(4, 3)
    batch = {"input_ids": torch.arange(12).view(4, 3), "attention_mask": torch.ones(4, 3), "labels": labels}

    stage._pp_prediction_step(batch, prediction_loss_only=True)

    assert stage.seen == {"state_rows": 2, "normalizer_rows": 2}
    runtime = stage._pp_runner
    assert runtime.input_ids.shape == (4, 3), "the frozen batch shape is kept"
    torch.testing.assert_close(runtime.labels[:2], labels[:2])
    assert (runtime.labels[2:] == LABEL_IGNORE_INDEX).all(), "the padding pair must score as inert rows"
    assert stage.stored[-1] == ({"m": 1.0}, "eval") and stage.stored_rows == 1


def test_the_metrics_path_weighs_the_real_examples_too(chain):
    """Predictions keep the batch's rows for the gather to cut."""
    stage = _eval_stage(real_examples=2, compute_metrics=lambda eval_pred: {}, pairs=3)
    stage._pp_pin_metric_keys()
    labels = torch.arange(1, 19).view(6, 3) % 5

    _, predictions, returned_labels = stage._pp_prediction_step(
        {"input_ids": torch.zeros(6, 3, dtype=torch.long), "labels": labels}, prediction_loss_only=False
    )

    assert stage.stored[-1] == ({"m": 1.0}, "eval") and stage.stored_rows == 2
    assert stage.seen == {"state_rows": 4, "normalizer_rows": 4}, "the step reads the two real pairs' rows"
    assert predictions.shape[0] == returned_labels.shape[0] == 6


def test_a_step_of_padding_alone_divides_by_one(chain):
    """Never a pair or row count of 0, which DPO's and KTO's normalizers count."""
    stage = _eval_stage(real_examples=0)
    stage._pp_pin_metric_keys()

    stage._pp_prediction_step(
        {"input_ids": torch.zeros(4, 3), "labels": torch.arange(1, 13).view(4, 3)}, prediction_loss_only=True
    )

    assert stage._pp_runner.count == 1.0
    assert (stage._pp_runner.labels == LABEL_IGNORE_INDEX).all()
    assert stage.stored_rows == 0


def test_a_round_without_padding_reaches_the_schedule_whole(chain):
    stage = _eval_stage(real_examples=2)
    stage._pp_pin_metric_keys()
    labels = torch.arange(1, 13).view(4, 3)

    stage._pp_prediction_step({"input_ids": torch.zeros(4, 3), "labels": labels}, prediction_loss_only=True)

    assert stage.seen == {"state_rows": 4, "normalizer_rows": 4}
    torch.testing.assert_close(stage._pp_runner.labels, labels)
    assert stage.stored_rows == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
