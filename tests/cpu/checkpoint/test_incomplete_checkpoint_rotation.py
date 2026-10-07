#!/usr/bin/env python
"""An incomplete checkpoint must never take a rotation slot, on any filesystem.

A resume passes over a newer ``checkpoint-<N>`` whose save never completed and continues from an older
one, so the run's next saves are numbered below that directory. Rotation orders by mtime, and on a
mount whose mtimes it distrusts (a fuse/S3 mount reporting one value for every directory) by step —
where the torn directory is the newest and the one it protects. Under ``save_total_limit: 1`` the next
save then deletes the checkpoint the run resumed from and itself. The resume therefore moves every
incomplete step directory into ``_incomplete_checkpoints/`` on each save rank, data kept.

Two gloo ranks, as one shared output filesystem (one writer) and as two nodes with their own local
disks (one writer each). On per-node storage ``checkpoint-200`` is complete on node 0 but torn on
node 1, and ``checkpoint-250`` exists on node 1 alone: both are incomplete world-wide, and node 0's
rotation must not keep its own complete-looking copy. Every mtime reads the same, as on such a mount;
the resumed run saves step 150 under ``save_total_limit: 1``.

    python tests/cpu/checkpoint/test_incomplete_checkpoint_rotation.py
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
# Spelled literally: importing the writers' constants would make a rename invisible here.
TRAINER_STATE = "trainer_state.json"
HOLDING_DIR = "_incomplete_checkpoints"
PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
SHARED_ENV = {"DIST_OUTPUT_SHARED_FILESYSTEM": "1"}
# What a fuse mount reports for every directory: rotation sees no spread and orders by step.
FAKE_MTIME = 1_700_000_000.0


class _ShardStore:
    def save(self, output_dir):
        with open(os.path.join(output_dir, f"optimizer_shard_{get_global_rank():05d}.pt"), "wb") as fh:
            fh.write(b"shard")


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """The mixin's save over HF's, rotation left to the mixin."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.args = SimpleNamespace(
            save_total_limit=1, save_only_model=False, should_save=fs_aware_save_rank(), push_to_hub=False
        )
        self.state = TrainerState(global_step=150)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            with open(os.path.join(output_dir, "model.safetensors"), "wb") as fh:
                fh.write(b"step 150")
        self._mark_model_save_collectives_done()

    def _optimizer_store(self):
        return _ShardStore()

    def _persist_trainer_sidecars(self, checkpoint_dir):
        pass


def _plant(run_dir: str, step: int, *, complete: bool) -> None:
    checkpoint = os.path.join(run_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint, exist_ok=True)
    with open(os.path.join(checkpoint, "model.safetensors"), "wb") as fh:
        fh.write(f"step {step}".encode())
    if complete:
        with open(os.path.join(checkpoint, TRAINER_STATE), "w") as fh:
            json.dump({"global_step": step}, fh)


def _worker(rank: int, root: str, per_node: bool) -> None:
    outcome = {}
    try:
        PartialState()
        runtime.resolve_shared_filesystem_consensus()
        os.path.getmtime = lambda path: FAKE_MTIME
        run_dir = os.path.join(root, f"node_{rank}" if per_node else "shared", "out")
        if fs_aware_save_rank():
            _plant(run_dir, 100, complete=True)
            _plant(run_dir, 200, complete=per_node and rank == 0)
            if per_node and rank == 1:
                _plant(run_dir, 250, complete=True)
        runtime.barrier()

        config = SimpleNamespace(output_dir=run_dir, resume_from_checkpoint=True, overwrite_output_dir=False)
        outcome["resumed"] = os.path.basename(detect_resume_checkpoint(config))
        _Trainer(run_dir)._save_checkpoint(model=None, trial=None)
        runtime.barrier()
        outcome["step_dirs"] = sorted(name for name in os.listdir(run_dir) if name.startswith("checkpoint-"))
        holding = os.path.join(run_dir, HOLDING_DIR)
        outcome["held"] = sorted(os.listdir(holding)) if os.path.isdir(holding) else []
        outcome["next_resume"] = os.path.basename(detect_resume_checkpoint(config))
    except Exception as e:  # the setup failing must still leave a verdict file for the assertions
        outcome["setup"] = f"{type(e).__name__}: {e}"
    finally:
        runtime.reset_shared_filesystem_consensus()
    with open(os.path.join(root, f"result_{get_global_rank()}.json"), "w") as fh:
        json.dump(outcome, fh)


@pytest.mark.parametrize("per_node", [False, True], ids=["shared-output", "per-node-output"])
def test_a_resume_moves_incomplete_checkpoints_out_of_rotation(tmp_path, per_node):
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        per_node,
        pg_timeout=PG_TIMEOUT,
        env=PER_NODE_ENV if per_node else SHARED_ENV,
    )
    for rank in range(WORLD_SIZE):
        outcome = json.loads((tmp_path / f"result_{rank}.json").read_text())
        assert "setup" not in outcome, f"rank {rank}: {outcome['setup']}"
        assert outcome["resumed"] == "checkpoint-100", f"rank {rank} resumed from {outcome['resumed']}"
        assert outcome["step_dirs"] == ["checkpoint-150"], (
            f"rank {rank}: save_total_limit 1 kept {outcome['step_dirs']} — an incomplete checkpoint took the slot"
        )
        assert outcome["next_resume"] == "checkpoint-150", f"rank {rank} would resume {outcome['next_resume']}"
    held = {node: json.loads((tmp_path / f"result_{node}.json").read_text())["held"] for node in range(WORLD_SIZE)}
    if per_node:
        assert held == {0: ["checkpoint-200"], 1: ["checkpoint-200", "checkpoint-250"]}, held
        node_copy = tmp_path / "node_0" / "out" / HOLDING_DIR / "checkpoint-200"
        assert (node_copy / TRAINER_STATE).is_file(), "the move lost node 0's copy of the torn step"
    else:
        assert held[0] == ["checkpoint-200"], held
        assert (tmp_path / "shared" / "out" / HOLDING_DIR / "checkpoint-200" / "model.safetensors").read_bytes() == (
            b"step 200"
        ), "the torn checkpoint's data did not survive the move"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
