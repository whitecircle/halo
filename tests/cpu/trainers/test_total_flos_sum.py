#!/usr/bin/env python
"""``state.total_flos`` is the world sum of every rank's FLOPs estimate, with no collective of its own on a logged
step.

HF's ``store_flos`` all-gathers ``current_flos`` and reads it back each time it is called, and a logged step calls
it right before the log, whose own all-reduce sums the token counters anyway. The trainer folds the count into that
all-reduce on a logged step and keeps the standalone sum for a save and the end of training, which write
``trainer_state.json`` and the train metrics. Two gloo ranks train a tiny SFT model with a save off the logging
cadence, recording every rank's per-micro-batch estimate and the collectives each ``store_flos`` call issues.

    python tests/cpu/trainers/test_total_flos_sum.py
"""

import datetime
import json
import os
import pickle
from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig

from halo.distributed import fsdp, mesh
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.token_metrics import TokenMetricsMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo import OFFLINE_VOCAB, make_offline_tokenizer
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

WORLD = 2
PG_TIMEOUT = datetime.timedelta(seconds=120)
BATCH = 2
GRAD_ACCUM = 2
MAX_STEPS = 7
LOGGING_STEPS = 2
# A save off the logging cadence: its trainer state must carry every FLOP since the last log.
SAVE_STEP = 5
LOGGED_STEPS = frozenset(range(LOGGING_STEPS, MAX_STEPS + 1, LOGGING_STEPS))
_COLLECTIVES = ("all_reduce", "all_gather", "all_gather_into_tensor", "all_gather_object", "broadcast")


def _count_collectives_inside(trainer: DistributedSFTTrainer) -> list[tuple[int, int]]:
    """``(global step, collectives issued)`` for every ``store_flos`` call of ``trainer``."""
    calls: list[tuple[int, int]] = []
    inside = []

    def counting(original):
        def collective(*args, **kwargs):
            if inside:
                inside[-1] += 1
            return original(*args, **kwargs)

        return collective

    for name in _COLLECTIVES:
        setattr(dist, name, counting(getattr(dist, name)))
    store_flos = trainer.store_flos

    def recording_store_flos():
        inside.append(0)
        try:
            store_flos()
        finally:
            calls.append((trainer.state.global_step, inside.pop()))

    trainer.store_flos = recording_store_flos
    return calls


class _FlosLedger(TrainerCallback):
    """``state.total_flos`` as each training log line sees it."""

    def __init__(self):
        self.at_log: dict[int, float] = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in (logs or {}):
            self.at_log[state.global_step] = state.total_flos


def _record_flops(seen: list[float]) -> None:
    """This rank's estimate per micro-batch, the reference the world sums are held to."""
    estimate = TokenMetricsMixin.floating_point_ops

    def recording(self, inputs):
        flops = estimate(self, inputs)
        seen.append(float(flops))
        return flops

    TokenMetricsMixin.floating_point_ops = recording


def _trainer(rank: int, output_dir: str, ledger: _FlosLedger) -> DistributedSFTTrainer:
    torch.manual_seed(0)
    model = tiny_family_model(
        TINY_DENSE_FAMILY, overrides={"num_hidden_layers": 2, "vocab_size": len(OFFLINE_VOCAB), "pad_token_id": 0}
    )
    vocab = len(OFFLINE_VOCAB)
    # Lengths that differ within a batch and between the ranks' shards, so the estimates do too.
    rows = [[3 + (i + j) % (vocab - 3) for j in range(6 + (5 * i) % 11)] for i in range(128)]
    args = SFTConfig(
        output_dir=output_dir,
        use_cpu=True,
        bf16=False,
        max_length=32,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEP,
        save_only_model=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=True,
        seed=0,
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": rows}),
        processing_class=make_offline_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=WORLD, gpus_per_node=WORLD, rank=rank),
        callbacks=[ledger],
    )


def _worker(rank: int, out: str) -> None:
    seen: list[float] = []
    ledger = _FlosLedger()
    _record_flops(seen)
    # The data-parallel FSDP2 wrap of a two-rank run, on the CPU mesh gloo runs.
    with patch.object(fsdp, "build_dp_mesh", partial(mesh.build_dp_mesh, device_type="cpu")):
        trainer = _trainer(rank, os.path.join(out, "run"), ledger)
    calls = _count_collectives_inside(trainer)
    trainer.train()
    every_rank_seen = [None] * WORLD
    dist.all_gather_object(every_rank_seen, seen)
    with open(os.path.join(out, "run", f"checkpoint-{SAVE_STEP}", "trainer_state.json")) as fh:
        saved = json.load(fh)["total_flos"]
    result = {
        "calls": calls,
        "at_log": ledger.at_log,
        "seen": every_rank_seen,
        "saved": saved,
        "final": trainer.state.total_flos,
        "train_metrics": trainer.state.log_history[-1].get("total_flos"),
    }
    with open(os.path.join(out, f"result.{rank}"), "wb") as fh:
        pickle.dump(result, fh)


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> list[dict]:
    out = str(tmp_path_factory.mktemp("flos"))
    run_gloo_ranks(_worker, WORLD, out, pg_timeout=PG_TIMEOUT)
    loaded = []
    for rank in range(WORLD):
        with open(os.path.join(out, f"result.{rank}"), "rb") as fh:
            loaded.append(pickle.load(fh))
    return loaded


def _world_flops_through(step: int, seen: list[list[float]]) -> float:
    return sum(sum(rank_seen[: step * GRAD_ACCUM]) for rank_seen in seen)


def test_a_logged_step_issues_no_collective_of_its_own(results):
    for rank, result in enumerate(results):
        logged = [count for step, count in result["calls"] if step in LOGGED_STEPS]
        assert len(logged) == len(LOGGED_STEPS), f"rank {rank}: premise — HF sums the FLOPs at every log"
        assert logged == [0] * len(LOGGED_STEPS), f"rank {rank}: {result['calls']}"


def test_a_save_and_the_end_of_training_sum_the_flops_once_each(results):
    for rank, result in enumerate(results):
        unlogged = [(step, count) for step, count in result["calls"] if step not in LOGGED_STEPS]
        assert (SAVE_STEP, 1) in unlogged, f"rank {rank}: the off-cadence save must sum its FLOPs: {unlogged}"
        assert all(count == 1 for _, count in unlogged), f"rank {rank}: {unlogged}"


def test_total_flos_is_the_world_sum_wherever_it_is_read(results):
    """Each log, the trainer state saved off the logging cadence, the final state and the train metrics."""
    seen = results[0]["seen"]
    assert seen[0] != seen[1], "premise: the ranks' estimates differ"
    for rank, result in enumerate(results):
        assert result["at_log"] == {step: _world_flops_through(step, seen) for step in LOGGED_STEPS}, f"rank {rank}"
        assert result["saved"] == _world_flops_through(SAVE_STEP, seen), f"rank {rank}"
        assert result["final"] == _world_flops_through(MAX_STEPS, seen), f"rank {rank}"
        assert result["train_metrics"] == result["final"], f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
