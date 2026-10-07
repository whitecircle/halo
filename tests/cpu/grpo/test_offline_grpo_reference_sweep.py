"""Reference sweeps keep DP row order and discard equalization replays across CP/PP siblings."""

import datetime
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from torch import nn

from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.grpo.reference_cache import ReferenceScoreCache
from tests.common.gloo import run_gloo_ranks

PartialState()

_ROWS = 33
_BATCH_SIZE = 2


def _collate_rows(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {"row_id": torch.tensor([row["row_id"] for row in rows], dtype=torch.int64)}


def _expected_row(row: int) -> torch.Tensor:
    return -(torch.arange(row % 3 + 1, dtype=torch.float32) + 13 * row + 0.75)


def _assert_row_order(values: list[torch.Tensor]) -> None:
    assert len(values) == _ROWS, "equalization replay rows leaked into the reference table"
    for row, got in enumerate(values):
        torch.testing.assert_close(got, _expected_row(row), rtol=0, atol=0)


def _ranked_sweep(rank: int, siblings: int, mode: str, root: str, shared: bool) -> None:
    dp_size = 2
    rank_map = [
        global_rank % dp_size if mode == "pp" else global_rank // siblings
        for global_rank in range(dist.get_world_size())
    ]
    dp_rank = rank_map[rank]
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.ref_model = None
    trainer._pp_runtime = None
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=mode == "cp", is_pp_mode=mode == "pp")
    trainer.cp_config = SimpleNamespace(cp_size=siblings)
    if not shared:
        os.environ.update(LOCAL_WORLD_SIZE="1", LOCAL_RANK="0")
    trainer.args = SimpleNamespace(
        per_device_train_batch_size=_BATCH_SIZE, output_dir=root if shared else os.path.join(root, f"rank-{rank}")
    )
    trainer.data_collator = _collate_rows
    trainer.dp_shard_geometry = lambda: (dp_size, dp_rank)
    trainer._data_parallel_rank_by_global_rank = lambda: rank_map
    trainer._prepare_inputs = lambda batch: batch
    calls: list[list[int]] = []

    def collective_hidden_state(model, ids, mask, width):
        forward = torch.ones(1)
        dist.all_reduce(forward)
        assert forward.item() == dist.get_world_size()
        return torch.zeros(1, width, 1)

    trainer._get_last_hidden_state = collective_hidden_state

    def score_batch(batch):
        row_ids = batch["row_id"].tolist()
        calls.append(row_ids)
        if mode == "native":
            # Exercise the production per-row backbone loop, as FA4/FSDP/EP does. Equalizing
            # only batch count strands one peer when the final actual batch has one row.
            ids = torch.ones(len(row_ids), 4, dtype=torch.int64)
            mask = torch.tensor([[1] + [1] * (row % 3 + 1) + [0] * (2 - row % 3) for row in row_ids])
            trainer._dense_last_hidden_state(trainer.model, ids, mask, 3)
        return [_expected_row(row) for row in row_ids]

    trainer._score_reference_batch = score_batch
    dataset = Dataset.from_dict(
        {"row_id": list(range(_ROWS)), "completion_input_ids": [list(range(row % 3 + 1)) for row in range(_ROWS)]}
    )
    values = trainer._sweep_reference_logps(dataset, "training")
    _assert_row_order(
        [
            values.values[int(values.offsets[index]) : int(values.offsets[index + 1])]
            for index in range(values.lengths.numel())
        ]
    )
    assert trainer.model.training, "the reference sweep did not restore the model's training mode"

    start, end = dp_rank * _ROWS // dp_size, (dp_rank + 1) * _ROWS // dp_size
    actual_batches = math.ceil((end - start) / _BATCH_SIZE)
    required_batches = max(
        math.ceil(((shard + 1) * _ROWS // dp_size - shard * _ROWS // dp_size) / _BATCH_SIZE)
        for shard in range(dp_size)
    )
    assert len(calls) == required_batches
    assert all(len(batch) == _BATCH_SIZE for batch in calls), "reference forward row counts differ across DP peers"
    real_rows = [
        row
        for index, batch in enumerate(calls[:actual_batches])
        for row in batch[: min(_BATCH_SIZE, end - start - index * _BATCH_SIZE)]
    ]
    assert real_rows == list(range(start, end))
    if actual_batches < required_batches:
        assert calls[-1] == calls[actual_batches - 1], "the shorter shard did not replay its final batch"

    # Reverse the ownership labels at the actual writer seam; neither row count nor finite
    # scores can expose the bug, but the row-length validator and numeric oracle must.
    original_collect = ReferenceScoreCache.collect_batch

    def reverse_dp_labels(cache, rows, representatives):
        original_collect(cache, rows, {dp_size - 1 - shard: source for shard, source in representatives.items()})

    ReferenceScoreCache.collect_batch = reverse_dp_labels
    try:
        with pytest.raises(ValueError, match="lengths do not match"):
            trainer._sweep_reference_logps(dataset, "training")
    finally:
        ReferenceScoreCache.collect_batch = original_collect


@pytest.mark.parametrize("siblings,mode", [(1, "native"), (2, "cp"), (2, "pp")])
def test_reference_sweep_dp_order_replays_and_sibling_deduplication(tmp_path, siblings, mode):
    run_gloo_ranks(
        _ranked_sweep, 2 * siblings, siblings, mode, str(tmp_path), True, pg_timeout=datetime.timedelta(seconds=30)
    )


def test_node_local_cache_writers_receive_the_same_ordered_scores(tmp_path):
    run_gloo_ranks(
        _ranked_sweep,
        2,
        1,
        "native",
        str(tmp_path),
        False,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
        pg_timeout=datetime.timedelta(seconds=30),
    )


class _ScoringFailure(RuntimeError):
    """An original forward failure must escape, not become a deferred local-write verdict."""


def _failure_trainer(output_dir):
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer.model = nn.Linear(1, 1).train()
    trainer.ref_model = None
    trainer._pp_runtime = None
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=False, is_pp_mode=False)
    trainer.args = SimpleNamespace(per_device_train_batch_size=1, output_dir=output_dir)
    trainer.dp_shard_geometry = lambda: (1, 0)
    trainer._data_parallel_rank_by_global_rank = lambda: [0] * dist.get_world_size() if dist.is_initialized() else [0]
    trainer.data_collator = _collate_rows
    trainer._prepare_inputs = lambda batch: batch
    return trainer


def test_collective_forward_failure_preserves_the_original_exception_and_training_mode(tmp_path):
    trainer = _failure_trainer(str(tmp_path))

    def fail(batch):
        raise _ScoringFailure("injected reference forward OOM")

    trainer._score_reference_batch = fail
    dataset = Dataset.from_dict({"row_id": [0], "completion_input_ids": [[1]]})
    with pytest.raises(_ScoringFailure, match="injected reference forward OOM"):
        trainer._sweep_reference_logps(dataset, "training")
    assert trainer.model.training
    # The launch's lock stays held beside its scratch for the run; no score file may.
    scratch = (tmp_path / "_reference_cache").rglob("*")
    assert not [path for path in scratch if path.is_file() and path.suffix != ".lock"]


def test_reference_sweep_logs_progress_before_the_first_update(tmp_path, caplog):
    trainer = _failure_trainer(str(tmp_path))
    trainer._score_reference_batch = lambda batch: [_expected_row(row) for row in batch["row_id"].tolist()]
    dataset = Dataset.from_dict({"row_id": [0, 1], "completion_input_ids": [[1], [1, 2]]})
    with caplog.at_level("INFO"):
        scores = trainer._sweep_reference_logps(dataset, "evaluation")
    assert scores.lengths.numel() == 2
    assert "Preparing run-start KL reference for 'evaluation': 2 rows, 2 batches before training" in caplog.text
    assert "Run-start KL reference 'evaluation': batch 2/2" in caplog.text


def test_cleanup_failure_does_not_mask_the_collective_forward_exception(tmp_path, monkeypatch):
    trainer = _failure_trainer(str(tmp_path))

    def fail(batch):
        raise _ScoringFailure("injected reference forward OOM")

    def failed_cleanup(cache):
        raise OSError("cache unlink denied")

    trainer._score_reference_batch = fail
    monkeypatch.setattr(ReferenceScoreCache, "discard", failed_cleanup)
    dataset = Dataset.from_dict({"row_id": [0], "completion_input_ids": [[1]]})
    with pytest.raises(_ScoringFailure, match="injected reference forward OOM") as error:
        trainer._sweep_reference_logps(dataset, "training")
    assert error.value.__notes__ == ["Reference cache cleanup also failed: cache unlink denied"]
    assert trainer.model.training


def _ranked_forward_failure(rank, root):
    """Rank 0's reference forward fails before its collective; rank 1's forward waits in that collective."""
    trainer = _failure_trainer(root)

    def fail_or_collect(batch):
        if rank == 0:
            raise _ScoringFailure("injected reference forward OOM")
        value = torch.ones(1)
        dist.all_reduce(value)
        return [value]

    trainer._score_reference_batch = fail_or_collect
    dataset = Dataset.from_dict({"row_id": [0], "completion_input_ids": [[1]]})
    try:
        trainer._sweep_reference_logps(dataset, "training")
        outcome = "NO RAISE"
    except Exception as error:
        outcome = f"{type(error).__name__}: {error}"
    with open(os.path.join(root, f"forward_failure_{rank}.txt"), "w") as output:
        output.write(outcome)


def test_a_rank_failing_before_forward_collectives_stops_peers_without_the_pg_timeout(tmp_path):
    """The failing rank raises its own forward error at once, entering no consensus collective first, so
    its exit (not the 30-second watchdog) is what releases the peer waiting in the forward's collective."""
    run_gloo_ranks(_ranked_forward_failure, 2, str(tmp_path), pg_timeout=datetime.timedelta(seconds=30))
    failed, peer = ((tmp_path / f"forward_failure_{rank}.txt").read_text() for rank in range(2))
    assert failed == "_ScoringFailure: injected reference forward OOM", failed
    # A closed connection, not the gloo watchdog's "Timed out waiting ... ms".
    assert peer.startswith("RuntimeError: ") and "timed out" not in peer.lower(), peer


def _empty_replica_sweep(rank: int, root: str) -> None:
    """One replica of the sweep over a one-row split: rank 0's shard is empty, rank 1's holds the row."""
    trainer = _failure_trainer(os.path.join(root, f"rank-{rank}"))
    trainer.dp_shard_geometry = lambda: (2, rank)
    dataset = Dataset.from_dict({"row_id": [0], "completion_input_ids": [[1]]})
    try:
        trainer._sweep_reference_logps(dataset, "training")
        result = "NO RAISE"
    except Exception as error:
        result = f"{type(error).__name__}: {error}"
    with open(os.path.join(root, f"result_{rank}.txt"), "w") as output:
        output.write(result)


def test_a_replica_with_no_rows_refuses_the_sweep_on_every_rank(tmp_path):
    """Rank 0 has nothing to score while rank 1 heads into the batch-count all-reduce; a raise on rank 0
    alone parks rank 1 there until the group timeout."""
    run_gloo_ranks(_empty_replica_sweep, 2, str(tmp_path), pg_timeout=datetime.timedelta(seconds=90))
    for rank in range(2):
        path = tmp_path / f"result_{rank}.txt"
        result = path.read_text() if path.exists() else "NO RESULT (the rank never returned)"
        assert result.startswith("ValueError:"), f"rank {rank}: {result}"
        assert "rank 0 no rows to score the KL reference" in result, f"rank {rank}: {result}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
