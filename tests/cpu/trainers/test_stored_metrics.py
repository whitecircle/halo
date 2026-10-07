#!/usr/bin/env python
"""StoredMetricsMixin: accumulation + log-time drain (floats and detached 0-dim tensors).

The mixin buffers per-microbatch metrics and folds their row-weighted mean into ``logs`` once per
``log`` call. Values may be plain floats or detached 0-dim tensors (the buffered-drain path that
avoids a host sync per microbatch — teacher distillation stores tensors); both must produce
identical logged values and the buffer must clear after the flush.

On a multi-rank run the bucket is rank-local, so ``log`` reduces it over the world: every rank logs
the world mean, never rank 0's own (a sampling bias, and a split ``should_save`` verdict when
``metric_for_best_model`` names one of them), and ranks storing different key sets raise together
instead of pairing one rank's sums with another's names.

    python tests/cpu/trainers/test_stored_metrics.py
"""

import datetime
import json
import math

import pytest
import torch

from src.trainers.mixins.stored_metrics import StoredMetricsMixin
from tests.common.gloo import run_gloo_ranks


class _Recorder:
    """MRO terminal capturing what the mixin passes up."""

    def log(self, logs, start_time=None):
        self.last_logs = dict(logs)
        return logs


class _Trainer(StoredMetricsMixin, _Recorder):
    pass


def test_float_metrics_mean_flushed_and_cleared():
    t = _Trainer()
    t.store_metrics({"sft_loss": 1.0, "kd": 3.0})
    t.store_metrics({"sft_loss": 2.0, "kd": 5.0})
    logs = {"loss": 0.1}
    t.log(logs)
    assert logs["sft_loss"] == pytest.approx(1.5)
    assert logs["kd"] == pytest.approx(4.0)
    # Buffer cleared: a second log window starts fresh (no carry-over mean).
    t.store_metrics({"sft_loss": 4.0})
    logs2 = {"loss": 0.1}
    t.log(logs2)
    assert logs2["sft_loss"] == pytest.approx(4.0)


def test_tensor_metrics_drain_identically_to_item():
    # Detached 0-dim tensors (incl. bf16, as the losses come off a bf16 forward) must log the same
    # value the eager `.item()` path produced.
    t = _Trainer()
    values = [torch.tensor(1.25, dtype=torch.bfloat16), torch.tensor(2.75, dtype=torch.bfloat16)]
    for v in values:
        t.store_metrics({"distillation_loss": v})
    logs = {"loss": 0.1}
    t.log(logs)
    expected = torch.tensor([v.item() for v in values]).mean().item()
    assert logs["distillation_loss"] == pytest.approx(expected)


def test_train_eval_buffers_are_separate():
    t = _Trainer()
    t.store_metrics({"m": 1.0}, train_eval="train")
    t.store_metrics({"m": 9.0}, train_eval="eval")
    train_logs = {"loss": 0.1}  # "loss" key routes to the train buffer
    t.log(train_logs)
    assert train_logs["m"] == pytest.approx(1.0)
    eval_logs = {"eval_loss": 0.1}
    t.log(eval_logs)
    # Eval flushes gain the eval_ prefix (TRL convention) so they never land on the train series.
    assert eval_logs["eval_m"] == pytest.approx(9.0)
    assert "m" not in eval_logs


def test_rows_weight_the_mean_and_an_empty_entry_adds_nothing():
    """A value is a mean over ``rows`` real rows; a batch of eval padding alone passes ``rows=0``,
    and its value (NaN, a mean over nothing) must not reach the logged number."""
    t = _Trainer()
    t.store_metrics({"m": 1.0}, rows=3)
    t.store_metrics({"m": torch.tensor(5.0)}, rows=torch.tensor(1))
    t.store_metrics({"m": math.nan}, rows=0)
    logs = {"loss": 0.1}
    t.log(logs)
    assert logs["m"] == pytest.approx(2.0)


def test_a_key_with_no_rows_is_not_logged():
    t = _Trainer()
    t.store_metrics({"m": 1.0, "padding_only": math.nan}, rows=0)
    t.store_metrics({"m": 4.0})
    logs = {"loss": 0.1}
    t.log(logs)
    assert logs["m"] == pytest.approx(4.0)
    assert "padding_only" not in logs


@pytest.mark.parametrize(
    ("metrics", "rows"),
    [({"m": torch.tensor([1.0, 2.0])}, 1), ({"m": 1.0}, torch.tensor([1, 1]))],
    ids=["per-row value", "per-row rows"],
)
def test_a_non_scalar_entry_is_refused_where_it_is_stored(metrics, rows):
    """Weighted, a per-row tensor would broadcast against its rows and log a sum, not a mean."""
    with pytest.raises(ValueError, match="store_metrics"):
        _Trainer().store_metrics(metrics, rows=rows)


def test_a_one_element_tensor_drains_beside_0_dim_ones():
    t = _Trainer()
    t.store_metrics({"m": torch.tensor([3.0])})
    t.store_metrics({"m": torch.tensor(5.0)})
    logs = {"loss": 0.1}
    t.log(logs)
    assert logs["m"] == pytest.approx(4.0)


# Per rank: (value, rows) entries for "m"; rank 0 alone would log 1.0.
_RANK_ENTRIES = {0: [(1.0, 1), (1.0, 1)], 1: [(5.0, 2), (math.nan, 0)]}
_WORLD_MEAN = (1.0 + 1.0 + 5.0 * 2) / (1 + 1 + 2)


def _world_mean_worker(rank: int, out: str) -> None:
    t = _Trainer()
    for value, rows in _RANK_ENTRIES[rank]:
        t.store_metrics({"m": value, "eval_n": float(rank)}, train_eval="eval", rows=rows)
    logs = {"eval_loss": 0.1}
    t.log(logs)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(logs, fh)


def test_every_rank_logs_the_world_mean_not_rank_zeros(tmp_path):
    out = str(tmp_path / "logs")
    run_gloo_ranks(_world_mean_worker, 2, out)
    for rank in range(2):
        with open(f"{out}.{rank}") as fh:
            logs = json.load(fh)
        assert logs["eval_m"] == pytest.approx(_WORLD_MEAN), f"rank {rank} logged {logs['eval_m']}"
        assert logs["eval_n"] == pytest.approx(1 * 2 / 4), f"rank {rank}: an eval_ key must not gain a second prefix"


def _mismatch_worker(rank: int, out: str) -> None:
    t = _Trainer()
    t.store_metrics({"shared": 1.0, **({"only_on_rank_1": 2.0} if rank == 1 else {})})
    try:
        t.log({"loss": 0.1})
        outcome = "logged"
    except RuntimeError as exc:
        outcome = str(exc)
    with open(f"{out}.{rank}", "w") as fh:
        fh.write(outcome)


def test_a_key_set_mismatch_raises_on_every_rank_instead_of_hanging(tmp_path):
    out = str(tmp_path / "outcome")
    run_gloo_ranks(_mismatch_worker, 2, out, pg_timeout=datetime.timedelta(seconds=60))
    for rank in range(2):
        with open(f"{out}.{rank}") as fh:
            outcome = fh.read()
        assert "differ across ranks" in outcome, f"rank {rank}: {outcome}"
        assert "only_on_rank_1" in outcome, f"rank {rank}: {outcome}"


def _empty_rank_worker(rank: int, out: str, entries: dict[int, tuple[float, int]]) -> None:
    """A rank absent from ``entries`` ran no eval batch (more ranks than batches with even_batches
    off): it stores nothing and must learn the keys from a rank that did."""
    t = _Trainer()
    if rank in entries:
        value, rows = entries[rank]
        t.store_metrics({"m": value}, train_eval="eval", rows=rows)
    logs = {"eval_loss": 0.1}
    t.log(logs)
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(logs, fh)


@pytest.mark.parametrize(
    ("world", "entries", "mean"),
    [
        pytest.param(2, {0: (3.0, 2)}, 3.0, id="rank 1 empty"),
        # The keys come from the lowest rank holding them, never from rank 0 by default.
        pytest.param(3, {1: (3.0, 2), 2: (6.0, 1)}, 4.0, id="rank 0 empty"),
    ],
)
def test_a_rank_that_ran_no_batch_weighs_nothing_instead_of_raising(tmp_path, world, entries, mean):
    out = str(tmp_path / "logs")
    run_gloo_ranks(_empty_rank_worker, world, out, entries, pg_timeout=datetime.timedelta(seconds=60))
    for rank in range(world):
        with open(f"{out}.{rank}") as fh:
            assert json.load(fh).get("eval_m") == pytest.approx(mean), f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
