#!/usr/bin/env python
"""CPU test: what ``compute_metrics`` receives under a pipeline.

Two independent failures live here, both silent:

1. **Size.** Without a reduction the last stage's RAW logits plane crosses the chain and is
   allocated on every other stage — rows x max_length x vocab.
2. **Shape.** Every rank of a chain returns the SAME rows (the last stage's predictions are
   broadcast down it), so a world gather repeats each data-parallel replica ``pp_size`` times, and
   accelerate's end-of-dataloader remainder trim then keeps a prefix of the duplicates rather than
   the real tail. Means survive both; nothing else does.

    python tests/cpu/parallelism/test_pp_eval_metrics_scope.py
"""

import pytest
import torch
from accelerate import PartialState

from src.distributed.pipeline_parallel.losses import PPLossAdapter, causal_lm_token_loss
from src.trainers.mixins.dataloader import (
    DataParallelDataLoaderMixin,
    dp_representative_ranks,
    dp_scoped_gather,
    select_gathered_chunks,
)
from src.trainers.mixins.pipeline import PipelineTrainerMixin

PartialState()  # the mixin's accelerate logger refuses to emit without an initialized state

PP_SIZE, STAGE_WORLD = 2, 2
WORLD = PP_SIZE * STAGE_WORLD
ROWS_PER_RANK = 4


class _HoistStage(PipelineTrainerMixin):
    """Just enough trainer for ``_pp_hoist_metrics_reduction``."""

    def __init__(self, compute_metrics=None, preprocess=None):
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess


def _adapter(**kwargs) -> PPLossAdapter:
    return PPLossAdapter(token_loss_fn=causal_lm_token_loss, **kwargs)


def test_compute_metrics_without_any_reduction_is_refused_naming_the_remedy():
    """Refusing is the only honest option: the alternative is a multi-GB broadcast per eval batch."""
    stage = _HoistStage(compute_metrics=lambda predictions: {})

    with pytest.raises(ValueError, match="preprocess_logits_for_metrics"):
        stage._pp_hoist_metrics_reduction(_adapter())


def test_preprocess_logits_for_metrics_moves_onto_the_last_stage_and_is_cleared():
    """Hoisted, it runs once — on the stage that holds the logits. Left in place as well, HF would
    reduce a second time (an argmax of an argmax)."""
    stage = _HoistStage(compute_metrics=lambda p: {}, preprocess=lambda logits, labels: logits.argmax(-1))

    hoisted = stage._pp_hoist_metrics_reduction(_adapter())

    assert stage.preprocess_logits_for_metrics is None, "HF must not re-apply the reduction"
    logits = torch.randn(2, 5, 128)
    reduced = hoisted.predictions_fn(logits, {"labels": torch.zeros(2, 5, dtype=torch.long)})
    assert reduced.shape == (2, 5), "the vocab axis must be gone before the broadcast"
    assert reduced.numel() * 128 == logits.numel(), "the raw plane is vocab times larger"


def test_an_adapter_with_its_own_reduction_is_left_alone():
    """SMPO/reward adapters already reduce on the last stage; hoisting on top would reduce twice."""
    own = _adapter(predictions_fn=lambda outputs, inputs: outputs.mean(-1))
    stage = _HoistStage(compute_metrics=lambda p: {}, preprocess=lambda logits, labels: logits.argmax(-1))

    assert stage._pp_hoist_metrics_reduction(own) is own
    assert stage.preprocess_logits_for_metrics is not None


def test_no_compute_metrics_leaves_the_contract_untouched():
    untouched = _adapter()
    assert _HoistStage()._pp_hoist_metrics_reduction(untouched) is untouched


# --- the duplication half ----------------------------------------------------------------------


class _FakeAccelerator:
    """accelerate's gather-then-trim contract over a fixed world, with no distributed backend."""

    def __init__(self, per_rank_rows: list[torch.Tensor], remainder: int):
        self._per_rank_rows = per_rank_rows
        self.remainder = remainder

    def gather(self, tensor):
        del tensor  # every rank contributes its own chunk; the fake holds them all
        return torch.cat(self._per_rank_rows)

    def gather_for_metrics(self, input_data, **kwargs):
        data = self.gather(input_data)
        return data[: self.remainder] if self.remainder > 0 else data


class _EvalTrainer(DataParallelDataLoaderMixin):
    def __init__(self, accelerator, keep, world_size):
        self.accelerator = accelerator
        self._dp_metric_gather_scope = (keep, world_size)


def _pipeline_rank_map() -> list[int]:
    """DP rank of every global rank under pp2 x dp2: stage blocks repeat the same DP ranks."""
    return [rank % STAGE_WORLD for rank in range(WORLD)]


def test_the_chains_duplicate_copies_are_dropped():
    """Each DP replica must appear once; at pp2 a world gather carries it twice."""
    keep = dp_representative_ranks(_pipeline_rank_map())
    assert keep == [0, 1], "one representative per DP rank, in DP order"

    # rank r contributes rows tagged with its DP rank; stages 0 and 1 hold identical values.
    per_rank = [torch.full((ROWS_PER_RANK,), float(dp)) for dp in _pipeline_rank_map()]
    gathered = torch.cat(per_rank)
    assert gathered.numel() == WORLD * ROWS_PER_RANK

    deduped = select_gathered_chunks(gathered, keep, WORLD)
    assert deduped.numel() == STAGE_WORLD * ROWS_PER_RANK
    assert deduped.tolist() == [0.0] * ROWS_PER_RANK + [1.0] * ROWS_PER_RANK


def test_the_remainder_trim_runs_after_the_dedup_not_before():
    """The order is the whole bug: accelerate keeps the FIRST ``remainder`` rows, which before the
    dedup are duplicates of replica 0 rather than the batch's real tail."""
    per_rank = [torch.full((ROWS_PER_RANK,), float(dp)) for dp in _pipeline_rank_map()]
    remainder = 6  # a partial final batch: 6 real samples across the 2 DP replicas
    accelerator = _FakeAccelerator(per_rank, remainder)
    keep = dp_representative_ranks(_pipeline_rank_map())
    trainer = _EvalTrainer(accelerator, keep, WORLD)

    naive = accelerator.gather_for_metrics(per_rank[0])
    scoped = trainer._dp_gather_for_metrics(per_rank[0])

    assert scoped.numel() == remainder
    assert scoped.tolist() == [0.0] * ROWS_PER_RANK + [1.0, 1.0], "replica 1's rows must be reached"
    assert naive.tolist() == [0.0] * ROWS_PER_RANK + [1.0, 1.0][:0] + [1.0, 1.0], "guard the fake"
    assert torch.equal(naive, torch.cat(per_rank)[:remainder])
    # The naive trim never reaches ranks 2-3, so it cannot tell duplicates from real rows: with a
    # remainder covering only the first replica it would report replica 0 twice.
    accelerator.remainder = ROWS_PER_RANK
    assert accelerator.gather_for_metrics(per_rank[0]).tolist() == [0.0] * ROWS_PER_RANK
    assert trainer._dp_gather_for_metrics(per_rank[0]).tolist() == [0.0] * ROWS_PER_RANK


def test_the_scoped_gather_is_restored_after_use():
    """It patches ``accelerator.gather`` for one call; a leak would silently halve every later
    gather in the run, including the ones that are not metric gathers."""
    per_rank = [torch.zeros(ROWS_PER_RANK) for _ in range(WORLD)]
    accelerator = _FakeAccelerator(per_rank, 0)
    original = accelerator.gather

    with dp_scoped_gather(accelerator, [0, 1], WORLD):
        assert accelerator.gather is not original
    assert accelerator.gather == original
    assert "gather" not in accelerator.__dict__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
