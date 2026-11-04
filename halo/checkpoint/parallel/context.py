"""The trainer state a saver or the resume loader reads, built fresh per call so object references stay current."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from halo.distributed.parallelism_config import ParallelismConfig


@dataclass
class CheckpointContext:
    """Everything a checkpoint saver needs; every field but ``is_save_rank`` is rank-uniform, so all pick one saver."""

    # The unwrapped module the save gathers iterate.
    model: nn.Module
    parallelism_config: ParallelismConfig
    has_ep_layers: bool
    # Mixin-managed FSDP2. An FSDP v1 model only comes from an accelerate launch, which rejects every custom
    # parallelism and saves through the base Trainer.
    toolkit_fsdp2: bool
    accelerate_manages_fsdp: bool
    is_save_rank: bool
    max_shard_size: str
    save_sharded_ep: bool
    has_expert_lora: bool
    cp_wrapper: Any
    tokenizer: Any
    # PP only: tensors no stage holds (a multimodal vision tower), by global name, that the save rank re-emits so the
    # checkpoint keeps the wrapper layout.
    pp_wrapper_state: dict[str, Any] | None = None
    # Every tensor keeps its live dtype, so fp32 masters are written unrounded; an export casts to the save dtype.
    training_checkpoint: bool = False

    @property
    def merges_expert_lora(self) -> bool:
        """Fold the native expert-LoRA deltas into a merged base on save instead of writing an adapter."""
        return self.has_expert_lora and self.parallelism_config.merge_expert_lora_on_save


@dataclass
class CheckpointLoadContext:
    """Everything a resume reads, captured per call so the ``optimizer`` / ``lr_scheduler`` references are current."""

    # The FSDP2/parallel wrapper, NOT the unwrapped module the save gathers iterate.
    model: nn.Module
    optimizer: Any
    lr_scheduler: Any
    parallelism_config: ParallelismConfig
    has_ep_layers: bool
    # The bare trainer._fsdp_wrapped (mixin FSDP2 only), so accelerate-managed FSDP takes the base Trainer's loaders.
    fsdp_wrapped: bool
    tp_rank: int
    # Reached without holding the trainer.
    super_load_optimizer_and_scheduler: Callable[..., None]
    # A shard restore that fails under a matching topology, or an interrupted save's shard set,
    # warm-restarts instead of failing the resume.
    allow_optimizer_warm_restart: bool = False
    # The base Trainer's loader owns an unsharded run's whole-weight reload: accelerate-managed FSDP reads its own
    # sharded format, and sentence-transformers rebuilds its pipeline through ``from_pretrained``.
    base_owns_whole_weight_load: bool = False
