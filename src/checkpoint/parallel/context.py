"""Checkpoint contexts: the trainer state a saver or the resume loader reads instead of the trainer.

Built fresh per save/load call, so collective and rank invariants stay explicit and the object
references (model, optimizer, scheduler) are current.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from src.runtime.filesystem import is_fs_save_rank

if TYPE_CHECKING:
    from src.distributed.parallelism_config import ParallelismConfig


@dataclass
class CheckpointContext:
    """Everything a checkpoint saver needs, captured from the trainer at call time.

    Every field but ``is_save_rank`` is rank-uniform, so every rank picks the same saver.
    """

    # The unwrapped module the save gathers iterate.
    model: nn.Module
    parallelism_config: ParallelismConfig
    has_ep_layers: bool
    # Mixin-managed FSDP2; accelerate-managed FSDP saves through the base Trainer.
    toolkit_fsdp2: bool
    accelerate_manages_fsdp: bool
    is_save_rank: bool
    max_shard_size: str
    save_sharded_ep: bool
    # Native grouped-LoRA on EP experts.
    has_expert_lora: bool
    cp_wrapper: Any
    tokenizer: Any
    # PP only: tensors no stage holds (a multimodal wrapper's vision tower), by global name, that the
    # save rank re-emits so the checkpoint keeps the wrapper layout. Empty for a plain causal LM.
    pp_wrapper_state: dict[str, Any] | None = None
    # A training checkpoint rather than an export: every tensor keeps its live dtype, so fp32 masters
    # are written unrounded; an export casts to the save dtype (``save_dtype_caster``).
    training_checkpoint: bool = False

    @classmethod
    def capture(cls, trainer) -> CheckpointContext:
        """Snapshot ``trainer`` for a save: object references plus the resolved write gate and shard size."""
        model = trainer._top_level_model()
        return cls(
            model=model,
            parallelism_config=trainer.parallelism_config,
            has_ep_layers=trainer._has_ep_layers,
            # A FullyShardedDataParallel (v1) model only comes from an accelerate FSDP launch, which
            # rejects every custom parallelism, so accelerate manages it.
            toolkit_fsdp2=trainer._fsdp_wrapped and not trainer._accelerate_manages_fsdp,
            accelerate_manages_fsdp=trainer._accelerate_manages_fsdp,
            is_save_rank=is_fs_save_rank(),
            max_shard_size=trainer.save_max_shard_size or DEFAULT_MAX_SHARD_SIZE,
            save_sharded_ep=trainer.save_sharded_ep,
            has_expert_lora=model_has_expert_lora(model),
            cp_wrapper=trainer._find_cp_wrapper(),
            tokenizer=getattr(trainer, "processing_class", None),
            pp_wrapper_state=trainer._pp_wrapper_state,
            training_checkpoint=trainer._writing_training_checkpoint,
        )

    @property
    def merges_expert_lora(self) -> bool:
        """Fold the native expert-LoRA deltas into a merged base on save instead of writing an adapter."""
        return self.has_expert_lora and self.parallelism_config.merge_expert_lora_on_save


@dataclass
class CheckpointLoadContext:
    """Everything a resume reads (:func:`~src.checkpoint.parallel.resume_weights.load_resume_weights`,
    :class:`~src.checkpoint.parallel.optimizer.OptimizerShardStore`), captured per call so the
    ``optimizer`` / ``lr_scheduler`` references are current."""

    # The FSDP2/parallel wrapper, NOT the unwrapped module the save gathers iterate.
    model: nn.Module
    optimizer: Any
    lr_scheduler: Any
    parallelism_config: ParallelismConfig
    has_ep_layers: bool
    # The bare trainer._fsdp_wrapped (mixin-managed FSDP2 only), so accelerate-managed FSDP takes the base
    # Trainer's own loaders.
    fsdp_wrapped: bool
    tp_rank: int
    # The base Trainer's own optimizer + scheduler load, reached without holding the trainer.
    super_load_optimizer_and_scheduler: Callable[..., None]
    # A shard restore that fails under a matching topology, or an interrupted save's shard set,
    # warm-restarts instead of failing the resume.
    allow_optimizer_warm_restart: bool = False
    # The base Trainer's own loader owns an unsharded run's whole-weight reload: accelerate-managed FSDP
    # reads its own sharded format, and sentence-transformers rebuilds its whole pipeline through
    # ``from_pretrained``. Otherwise the base loader would read by raw key and the loader streams it.
    base_owns_whole_weight_load: bool = False
