#!/usr/bin/env python
"""A checkpoint save stopped partway must never be what resume detection picks, on any filesystem.

``trainer_state.json`` is the file resume detection takes a checkpoint by, and the base Trainer
writes it before the toolkit's sidecars and per-rank optimizer shards. The mixin's save therefore
removes an earlier save's copy before writing anything (a resumed run re-saving a step rewrites that
step's directory in place) and publishes the base save's copy only once every other file is on
disk. Two ranks on a gloo group, as one shared output filesystem (one writer) and as two nodes with
their own local disks (one writer each), pin, for a save stopped while it writes its weights (before
the base save's trainer state) or its optimizer shards (after it):

1. **A fresh step.** The stopped ``checkpoint-2`` holds no ``trainer_state.json`` on any node, and
   ``resume_from_checkpoint: true`` resumes every rank from the complete ``checkpoint-1``.
2. **A rewritten step.** ``checkpoint-2`` already held an abandoned run's complete save; stopped
   midway, the rewrite leaves no trainer state vouching for the mix of old and new files, and
   detection again resumes from ``checkpoint-1``.
3. **An explicit path** to the stopped step is refused on every rank, naming it.

    python tests/cpu/checkpoint/test_interrupted_save_resume.py
"""

import datetime
import json
import os
from types import SimpleNamespace

import pytest
from accelerate import PartialState
from transformers import TrainerState

from src.distributed import runtime
from src.distributed.runtime import fs_aware_save_rank, get_global_rank
from src.trainers.mixins.base import DistributedTrainerMixin
from src.training.environment import detect_resume_checkpoint
from tests.common.base_save import BaseTrainerSave
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=60)
# Spelled literally: importing the writer's constants would make a rename invisible here.
TRAINER_STATE = "trainer_state.json"
PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
SHARED_ENV = {"DIST_OUTPUT_SHARED_FILESYSTEM": "1"}


class _Preempted(RuntimeError):
    pass


class _StateWriteThenStop(TrainerState):
    """A trainer state whose write is the last thing the save does before it is stopped."""

    def save_to_json(self, json_path):
        super().save_to_json(json_path)
        raise _Preempted("stopped right after the trainer state was written (simulated preemption)")


class _ShardSaveStopped:
    """Stands in for ``OptimizerShardStore.save``, stopping the way a preemption would."""

    def save(self, output_dir):
        raise _Preempted("optimizer shard save stopped (simulated preemption)")


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """The mixin's save over HF's: the weights, the rank-0 optimizer view and the trainer state, written
    by the ``should_save`` rank(s) — or stopped on every rank mid-weights, or on the writer(s) right
    after the trainer state."""

    def __init__(self, run_dir, stop: str):
        self.run_dir = run_dir
        self.stop = stop
        self.args = SimpleNamespace(
            save_total_limit=2, save_only_model=False, should_save=fs_aware_save_rank(), push_to_hub=False
        )
        self.state = (_StateWriteThenStop if stop == "state" else TrainerState)(global_step=2)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            with open(os.path.join(output_dir, "model.safetensors"), "wb") as fh:
                fh.write(b"new run")
        if self.stop == "weights":
            raise _Preempted("weights write stopped (simulated preemption)")
        self._mark_model_save_collectives_done()

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.args.should_save:
            with open(os.path.join(output_dir, "optimizer.pt"), "wb") as fh:
                fh.write(b"new run")

    def _optimizer_store(self):
        return _ShardSaveStopped()

    def _persist_trainer_sidecars(self, checkpoint_dir):
        pass


def _plant_complete_checkpoint(run_dir: str, step: int, run: str) -> None:
    checkpoint = os.path.join(run_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint, exist_ok=True)
    for name in ("model.safetensors", "optimizer_shard_00000.pt", "optimizer_meta.pt"):
        with open(os.path.join(checkpoint, name), "wb") as fh:
            fh.write(run.encode())
    with open(os.path.join(checkpoint, TRAINER_STATE), "w") as fh:
        json.dump({"global_step": step, "run": run}, fh)


def _worker(rank: int, root: str, per_node: bool, rewrite: bool, stop: str) -> None:
    outcome = {}
    try:
        PartialState()
        runtime.resolve_shared_filesystem_consensus()
        run_dir = os.path.join(root, f"node_{rank}" if per_node else "shared", "out")
        if fs_aware_save_rank():
            _plant_complete_checkpoint(run_dir, 1, "abandoned" if rewrite else "new")
            if rewrite:
                _plant_complete_checkpoint(run_dir, 2, "abandoned")
        runtime.barrier()

        try:
            _Trainer(run_dir, stop)._save_checkpoint(model=None, trial=None)
            outcome["save"] = "completed"
        except (_Preempted, RuntimeError) as e:
            outcome["save"] = "stopped" if "simulated preemption" in str(e) else f"failed: {e}"
        stopped = os.path.join(run_dir, "checkpoint-2")
        outcome["trainer_state_in_stopped_step"] = os.path.exists(os.path.join(stopped, TRAINER_STATE))

        # The explicit path first: an accepted resume moves the stopped step out of the namespace.
        explicit = SimpleNamespace(output_dir=run_dir, resume_from_checkpoint=stopped, overwrite_output_dir=False)
        try:
            detect_resume_checkpoint(explicit)
            outcome["explicit"] = "accepted"
        except RuntimeError as e:
            outcome["explicit"] = str(e)
        auto = SimpleNamespace(output_dir=run_dir, resume_from_checkpoint=True, overwrite_output_dir=False)
        outcome["auto_resume"] = os.path.basename(detect_resume_checkpoint(auto))
    except Exception as e:  # the setup failing must still leave a verdict file for the assertions
        outcome["setup"] = f"{type(e).__name__}: {e}"
    finally:
        runtime.reset_shared_filesystem_consensus()
    with open(os.path.join(root, f"result_{get_global_rank()}.json"), "w") as fh:
        json.dump(outcome, fh)


@pytest.mark.parametrize(
    "stop",
    ["weights", "state", "shards"],
    ids=["stopped-in-weights", "stopped-after-trainer-state", "stopped-in-shards"],
)
@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh-step", "rewritten-step"])
@pytest.mark.parametrize("per_node", [False, True], ids=["shared-output", "per-node-output"])
def test_a_stopped_save_is_passed_over_by_resume_detection(tmp_path, per_node, rewrite, stop):
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        per_node,
        rewrite,
        stop,
        pg_timeout=PG_TIMEOUT,
        env=PER_NODE_ENV if per_node else SHARED_ENV,
    )
    for rank in range(WORLD_SIZE):
        outcome = json.loads((tmp_path / f"result_{rank}.json").read_text())
        assert "setup" not in outcome, f"rank {rank}: {outcome['setup']}"
        assert outcome["save"] == "stopped", f"rank {rank}: premise — the save must stop partway"
        assert not outcome["trainer_state_in_stopped_step"], (
            f"rank {rank}: the stopped checkpoint-2 holds a trainer state that vouches for it"
        )
        assert outcome["auto_resume"] == "checkpoint-1", f"rank {rank} resumed from {outcome['auto_resume']}"
        assert "checkpoint-2" in outcome["explicit"] and "incomplete" in outcome["explicit"], (
            f"rank {rank} accepted an explicit resume from the stopped step: {outcome['explicit']}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
