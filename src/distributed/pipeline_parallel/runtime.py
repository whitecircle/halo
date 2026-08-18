"""The pipeline runtime: the stage, the schedule and the microbatched step.

Wraps ``torch.distributed.pipelining``. The schedule replaces the inherited HF training step and
drives every microbatch's forward and backward, so gradient accumulation happens here and the caller
must not call ``.backward()``.

The schedule engine is not yet available in this release: ``parallelism_config_from_args``
(``src/training/parallelism_args.py``) rejects ``pipeline_parallel_size > 1`` at config time, and
constructing :class:`PipelineRuntime` raises. The batch contract (:data:`PP_BATCH_PAD_VALUES`) and
the public signatures below are what the trainer mixin builds against — see
``agent-docs/parallelism/pipeline-parallelism.md``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import causal_lm_token_loss
from src.distributed.pipeline_parallel.stage import PipelineStageModule

_NOT_AVAILABLE = (
    "Pipeline parallelism is not yet available in this release: the schedule engine behind "
    "PipelineRuntime is not shipped. Set pipeline_parallel_size=1 (the default); see "
    "agent-docs/parallelism/pipeline-parallelism.md."
)

# Each key the pipeline consumes → its pad value (None = tokenizer pad id). The trainer's
# dataset-column pin and its fixed-shape padding both derive from this.
PP_BATCH_PAD_VALUES: dict[str, int | None] = {
    "input_ids": None,
    "labels": LABEL_IGNORE_INDEX,
    "attention_mask": 0,
    "position_ids": 0,
}


class PipelineRuntime:
    """Drives one optimizer step's worth of microbatches through this rank's pipeline stage."""

    def __init__(
        self,
        stage_module: PipelineStageModule,
        config: ParallelismConfig,
        pp_group: dist.ProcessGroup,
        device: torch.device,
        n_microbatches: int,
        token_loss_fn: Callable[[torch.Tensor, torch.Tensor | dict], torch.Tensor] = causal_lm_token_loss,
        paired_examples: bool = False,
        fused_head_loss: bool = False,
    ):
        raise NotImplementedError(_NOT_AVAILABLE)

    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        extra_targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Run one optimizer step's microbatches. Returns the summed loss on the last stage, else None."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def eval_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        extra_targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Loss-only counterpart to :meth:`step`, with backward suppressed."""
        raise NotImplementedError(_NOT_AVAILABLE)

    def forward_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Drive a forward-only pass across the pipeline; returns the last stage's output, else None."""
        raise NotImplementedError(_NOT_AVAILABLE)
