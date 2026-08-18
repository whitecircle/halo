"""Checkpoint context — the trainer-state snapshot threaded into every saver.

Built once per save/load call by the trainer's ``_checkpoint_context()`` factory. The savers read
only this object (never the trainer), keeping collective/rank invariants explicit. Carries object
references plus bound ``super()`` callables for the base-Trainer fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class CheckpointContext:
    """Everything a checkpoint saver needs, captured from the trainer at call time."""

    # The unwrapped module the save gathers iterate.
    model: nn.Module
    parallelism_config: Any

    # Mode flags (rank-uniform).
    is_pp_mode: bool
    is_cp_mode: bool
    is_tp_mode: bool
    is_ep_tp_mode: bool
    has_ep_layers: bool
    fsdp_wrapped: bool
    accelerate_manages_fsdp: bool

    is_save_rank: bool

    max_shard_size: str
    save_sharded_ep: bool

    # Native grouped-LoRA on EP experts: merge into the base on save, or write a standalone adapter.
    has_expert_lora: bool
    merge_expert_lora_on_save: bool

    cp_wrapper: Any
    tokenizer: Any

    # PP only: tensors no stage holds (a multimodal wrapper's vision tower), by global name, that the
    # save rank re-emits so the checkpoint keeps the wrapper layout. Empty for a plain causal LM.
    pp_wrapper_state: dict[str, Any] | None = None


@dataclass
class CheckpointLoadContext:
    """Everything :class:`~src.distributed.checkpoint.loader.CheckpointLoader` needs on resume.

    Separate from :class:`CheckpointContext`: the load paths gate on the bare ``trainer._fsdp_wrapped``
    (mixin-managed FSDP2 only), whereas the save context widens ``fsdp_wrapped`` with
    ``isinstance(model, FSDP)`` — folding the two would route accelerate-FSDP into the FSDP2 load
    path. Built fresh per call so the ``optimizer`` / ``lr_scheduler`` references are current.
    """

    # model = the FSDP2/parallel wrapper (NOT the unwrapped module the save gathers iterate).
    model: nn.Module
    optimizer: Any
    lr_scheduler: Any
    parallelism_config: Any

    is_pp_mode: bool
    is_cp_mode: bool
    is_tp_mode: bool
    has_ep_layers: bool
    fsdp_wrapped: bool  # bare trainer._fsdp_wrapped — mixin-managed FSDP2 only

    tp_rank: int
    tp_size: int

    # Bound base-Trainer fallbacks — reach super() without holding the trainer.
    super_load_from_checkpoint: Callable[..., None]
    super_load_optimizer_and_scheduler: Callable[..., None]
