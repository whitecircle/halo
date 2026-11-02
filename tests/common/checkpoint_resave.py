"""A save that rewrites an existing ``checkpoint-N`` in place, as a resumed run saving a step it saved before does:
the mixin's real ``_save_checkpoint`` over :class:`~tests.common.base_save.BaseTrainerSave`, with HF's own RNG-state
writer and the real per-rank optimizer shard store over a stepped optimizer, and no weights."""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import Trainer, TrainerState
from transformers.training_args import ParallelMode

from halo.checkpoint.format import CHECKPOINT_STEP_DIR_RE
from halo.checkpoint.parallel.context import CheckpointLoadContext
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_rank
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_io import load_context


def stepped_store_context(*, fsdp_wrapped: bool = True) -> CheckpointLoadContext:
    """A load context over a live model and an optimizer that holds state to save."""
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()
    return load_context(model, optimizer=optimizer, fsdp_wrapped=fsdp_wrapped)


class _ResavingTrainer(DistributedTrainerMixin, BaseTrainerSave):
    """The mixin's save over HF's, writing this rank's optimizer shard and RNG state and no weights."""

    _save_rng_state = Trainer._save_rng_state

    def __init__(self, run_dir: str, step: int, world_size: int):
        self.run_dir = run_dir
        self.args = SimpleNamespace(
            save_total_limit=None,
            save_only_model=False,
            should_save=is_fs_save_rank(),
            push_to_hub=False,
            include_num_input_tokens_seen="no",
            world_size=world_size,
            process_index=get_global_rank(),
            parallel_mode=ParallelMode.DISTRIBUTED,
        )
        self.state = TrainerState(global_step=step)
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, merge_expert_lora_on_save=False)
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def save_model(self, output_dir=None, _internal_call=False):
        self._mark_model_save_collectives_done()

    def _optimizer_store(self) -> OptimizerShardStore:
        return OptimizerShardStore(stepped_store_context())


def resave_in_place(checkpoint_dir: Path, *, world_size: int) -> None:
    """COLLECTIVE. Save the step of ``checkpoint_dir``, a ``checkpoint-N`` directory, into it again, as this
    process's global rank of a ``world_size``-rank run (above one, so the RNG file is the rank's own)."""
    step = CHECKPOINT_STEP_DIR_RE.match(checkpoint_dir.name)
    if step is None:
        raise ValueError(f"{checkpoint_dir} is no checkpoint-N directory, the only kind a save rewrites")
    _ResavingTrainer(str(checkpoint_dir.parent), int(step[1]), world_size)._save_checkpoint(model=None, trial=None)
