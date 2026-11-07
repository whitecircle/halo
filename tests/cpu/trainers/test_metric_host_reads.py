#!/usr/bin/env python
"""Between logs a step reads metrics back only for the efficiency callback; a log reads world metrics and tokens once.
On a host-bound step every device-to-host read drains the stream, turning the GPU's queue into idle time."""

import datetime
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig

from halo.callbacks.efficiency import EfficiencyCallback
from halo.distributed import fsdp, mesh
from halo.trainers.metrics.token_metrics import TokenMetricsMixin
from halo.trainers.sft import DistributedSFTTrainer
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
# A save off the logging cadence: its trainer state must carry the input tokens of the steps since the last log.
SAVE_STEP = 5
# The steps counted: one plain, one logged, past the first log, whose flush agrees the metric keys once.
PLAIN_STEP, LOGGED_STEP = 3, 4
# The metric path's modules and TRL's, never a dependency's read or the non-finite step guard's per-step verdict.
METRIC_SOURCES = (
    "halo/trainers/metrics/world_metrics.py",
    "halo/trainers/metrics/token_metrics.py",
    "halo/trainers/metrics/stored_metrics.py",
    "halo/trainers/sft.py",
    "halo/callbacks/efficiency.py",
    "halo/runtime/consensus.py",
    "trl/",
)
_READS = ("item", "tolist", "__bool__", "__float__", "__int__", "cpu", "numpy")


def _source(filename: str) -> str:
    """``filename`` from its package root on (``halo/trainers/sft.py``), or the bare package of a dependency."""
    parts = filename.split(os.sep)
    for package in ("halo", "trl", "transformers", "accelerate", "torch", "tests"):
        if package in parts:
            index = len(parts) - 1 - parts[::-1].index(package)
            return "/".join(parts[index:]) if package in ("halo", "trl") else package
    return "other"


class _StepWindows(TrainerCallback):
    """Files host reads under the optimizer step they fall in, up to the next step's begin, so a log and save count."""

    def __init__(self):
        self.step = None
        self.reads: dict[int, Counter] = defaultdict(Counter)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step = state.global_step + 1

    def on_train_end(self, args, state, control, **kwargs):
        self.step = None


class _TokenLedger(TrainerCallback):
    def __init__(self):
        self.at_step_end: dict[int, int] = {}

    def on_step_end(self, args, state, control, **kwargs):
        self.at_step_end[state.global_step] = state.num_input_tokens_seen


def _count_reads(windows: _StepWindows) -> None:
    def counting(name, original):
        def read(self, *args, **kwargs):
            if windows.step is not None:
                windows.reads[windows.step][(name, _source(sys._getframe(1).f_code.co_filename))] += 1
            return original(self, *args, **kwargs)

        return read

    for name in _READS:
        setattr(torch.Tensor, name, counting(name, getattr(torch.Tensor, name)))


def _record_batch_tokens(seen: list[int]) -> None:
    """This rank's attended tokens per micro-batch, off the batches: the reference the trainer's world sums must match."""
    track = TokenMetricsMixin._track_num_input_tokens

    def recording(self, inputs):
        seen.append(int(inputs["attention_mask"].sum()))
        return track(self, inputs)

    TokenMetricsMixin._track_num_input_tokens = recording


def _trainer(rank: int, output_dir: str, callbacks: list) -> DistributedSFTTrainer:
    torch.manual_seed(0)
    tokenizer = make_offline_tokenizer()
    model = tiny_family_model(
        TINY_DENSE_FAMILY, overrides={"num_hidden_layers": 2, "vocab_size": len(OFFLINE_VOCAB), "pad_token_id": 0}
    )
    vocab = len(OFFLINE_VOCAB)
    # Lengths that differ within a batch and between the ranks' shards, so the padded batches do too.
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
        include_num_input_tokens_seen="all",
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=True,
        seed=0,
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": rows}),
        processing_class=tokenizer,
        parallelism_config=make_parallelism_config(world_size=WORLD, gpus_per_node=WORLD, rank=rank),
        callbacks=callbacks,
    )


def _worker(rank: int, out: str, efficiency: bool) -> None:
    windows, ledger, seen = _StepWindows(), _TokenLedger(), []
    efficiency_callback = None
    if efficiency:
        efficiency_callback = EfficiencyCallback(make_parallelism_config(world_size=WORLD, gpus_per_node=WORLD), 0)
    callbacks = [windows, *([efficiency_callback] if efficiency else []), ledger]
    _record_batch_tokens(seen)
    # The data-parallel FSDP2 wrap of a two-rank run, on the CPU mesh gloo runs.
    with patch.object(fsdp, "build_dp_mesh", partial(mesh.build_dp_mesh, device_type="cpu")):
        trainer = _trainer(rank, os.path.join(out, "run"), callbacks)
    _count_reads(windows)
    trainer.train()
    every_rank_seen = [None] * WORLD
    dist.all_gather_object(every_rank_seen, seen)
    with open(os.path.join(out, "run", f"checkpoint-{SAVE_STEP}", "trainer_state.json")) as fh:
        saved = json.load(fh)["num_input_tokens_seen"]
    result = {
        "reads": {step: dict(counts) for step, counts in windows.reads.items()},
        "seen": every_rank_seen,
        "logged": {entry["step"]: entry["num_input_tokens_seen"] for entry in trainer.state.log_history},
        "at_step_end": ledger.at_step_end,
        "saved": saved,
        "final": trainer.state.num_input_tokens_seen,
        "cluster_tokens": None if efficiency_callback is None else efficiency_callback._run.elapsed_cluster_tokens,
    }
    with open(os.path.join(out, f"result.{rank}"), "wb") as fh:
        pickle.dump(result, fh)


@pytest.fixture(scope="module", params=[False, True], ids=["efficiency_off", "efficiency_on"])
def runs(request, tmp_path_factory) -> tuple[bool, list[dict]]:
    out = str(tmp_path_factory.mktemp("reads"))
    run_gloo_ranks(_worker, WORLD, out, request.param, pg_timeout=PG_TIMEOUT)
    results = []
    for rank in range(WORLD):
        with open(os.path.join(out, f"result.{rank}"), "rb") as fh:
            results.append(pickle.load(fh))
    return request.param, results


def _world_tokens_through(step: int, seen: list[list[int]]) -> int:
    return sum(sum(rank_seen[: step * GRAD_ACCUM]) for rank_seen in seen)


def _metric_reads(counts: dict) -> dict:
    return {key: count for key, count in counts.items() if key[1].startswith(METRIC_SOURCES)}


# The callback's step end: the input-token world sum and the step's attention work.
_MEASURED_STEP_READS = {
    ("tolist", "halo/trainers/metrics/world_metrics.py"): 1,
    ("tolist", "halo/trainers/metrics/token_metrics.py"): 1,
}


def test_a_plain_step_reads_its_metrics_back_only_for_the_efficiency_callback(runs):
    """The callback reads the input-token sum and attention work at the boundary its step clock already waits on."""
    efficiency, results = runs
    expected = _MEASURED_STEP_READS if efficiency else {}
    for rank, result in enumerate(results):
        reads = result["reads"][PLAIN_STEP]
        assert _metric_reads(reads) == expected, f"rank {rank}: {reads}"


def test_a_logged_step_reads_the_world_metrics_and_the_token_counts_once_each(runs):
    """The loss metrics' flush and the token counters' sum, beside a measured step's own."""
    efficiency, results = runs
    logged = {("tolist", "halo/trainers/metrics/world_metrics.py"): 2}
    expected = Counter(logged) + Counter(_MEASURED_STEP_READS) if efficiency else logged
    for rank, result in enumerate(results):
        reads = result["reads"][LOGGED_STEP]
        assert _metric_reads(reads) == dict(expected), f"rank {rank}: {reads}"


def test_the_input_token_count_is_exact_wherever_it_is_read(runs):
    efficiency, results = runs
    seen = results[0]["seen"]
    assert len({len(rank_seen) for rank_seen in seen}) == 1 and seen[0] != seen[1], "premise: unequal shards"
    for rank, result in enumerate(results):
        assert result["seen"] == seen, f"rank {rank}"
        logged = {step: count for step, count in result["logged"].items() if step % LOGGING_STEPS == 0}
        assert logged == {step: _world_tokens_through(step, seen) for step in range(2, MAX_STEPS, 2)}, f"rank {rank}"
        assert result["saved"] == _world_tokens_through(SAVE_STEP, seen), f"rank {rank}"
        assert result["final"] == _world_tokens_through(MAX_STEPS, seen), f"rank {rank}"
        if efficiency:
            through = [_world_tokens_through(step, seen) for step in range(MAX_STEPS + 1)]
            assert result["at_step_end"] == {step: through[step] for step in range(1, MAX_STEPS + 1)}, f"rank {rank}"
            # The callback's cluster tokens: each step's world count, floored to whole tokens per rank.
            deltas = [through[step] - through[step - 1] for step in range(1, MAX_STEPS + 1)]
            assert result["cluster_tokens"] == sum(delta // WORLD * WORLD for delta in deltas), f"rank {rank}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
