"""The pipeline runtime: the stage, the schedule and the microbatched step.

Wraps ``torch.distributed.pipelining``. The schedule replaces the inherited HF training step and
drives every microbatch's forward and backward, so gradient accumulation happens here and the caller
must not call ``.backward()``.

The trainer supplies what the pipelining package does not: ``optimizer.zero_grad()``, a grad norm
reduced across the pipeline chain (each stage sees only its own parameters), the last stage's loss
broadcast for logging, and ``gradient_accumulation_steps=1``. ``n_microbatches`` is the accumulation,
so HF's own loop would multiply both the tokens per step and the LR schedule's step count.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.pipelining.schedules import PipelineScheduleSingle

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import PP_SCHEDULES, ParallelismConfig
from src.distributed.pipeline_parallel.losses import (
    causal_lm_token_loss,
    fused_causal_lm_token_loss,
    loss_token_count,
)
from src.distributed.pipeline_parallel.stage import PipelineStageModule

_SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}
if set(_SCHEDULES) != set(PP_SCHEDULES):
    raise RuntimeError(
        f"Pipeline schedule table {sorted(_SCHEDULES)} drifted from PP_SCHEDULES {sorted(PP_SCHEDULES)}"
    )

# Each key the pipeline consumes → its pad value (None = tokenizer pad id). The trainer's
# dataset-column pin and its fixed-shape padding both derive from this.
PP_BATCH_PAD_VALUES: dict[str, int | None] = {
    "input_ids": None,
    "labels": LABEL_IGNORE_INDEX,
    "attention_mask": 0,
    "position_ids": 0,
}


# Private FSDP2 state members torch's pipelining grad reduction calls; without them grads are never
# reduced over the DP mesh.
_REQUIRED_FSDP_INTERNALS = ("_state_ctx", "_root_post_backward_final_callback")

# Private schedule and stage members this runtime reads or overrides.
_REQUIRED_SCHEDULE_INTERNALS = (
    "_stage",
    "_loss_fn",
    "_has_backward",
    "_n_microbatches",
    "_split_inputs",
    "_step_microbatches",
    "_initialize_stage",
    "_stage_forward_initialized",
    "_stage_backward_initialized",
)
_REQUIRED_STAGE_INTERNALS = (
    "has_backward",
    "clear_runtime_states",
    "perform_reduce_grad",
    "_validate_fwd_outputs",
)


def _accept_any_outputs(outputs: tuple[torch.Tensor, ...]) -> None:
    """Stand-in for ``PipelineStage._validate_fwd_outputs`` while a fused last stage returns logits."""
    del outputs


def _skip_reduce_grad(grad_scale_factor: int) -> None:
    """Stand-in for ``PipelineStage.perform_reduce_grad`` on a drive that runs no backward."""
    del grad_scale_factor


def _assert_pipelining_fsdp_contract(stage_module: nn.Module) -> None:
    """Verify the private FSDP2 hooks torch's pipelining grad reduction depends on still exist."""
    if not isinstance(stage_module, FSDPModule):
        raise RuntimeError(
            "The pipeline stage module is not an FSDPModule: fully_shard must be applied to the "
            "stage ROOT, not only to its inner blocks. Without it torch's schedule skips its FSDP "
            "grad-reduction path and reduce-scatters once per microbatch instead of once per step."
        )
    state = fully_shard.state(stage_module)
    missing = [attr for attr in _REQUIRED_FSDP_INTERNALS if not hasattr(state, attr)]
    if missing:
        raise RuntimeError(
            f"torch {torch.__version__}'s FSDP2 state is missing {missing}, which "
            f"torch.distributed.pipelining uses to reduce gradients once per pipeline step. "
            f"Pipeline parallelism must be re-validated against this torch version before use."
        )


def _assert_pipelining_schedule_contract(schedule: PipelineScheduleSingle) -> None:
    """Verify the private schedule and stage members the runtime steers still exist."""
    missing = [attr for attr in _REQUIRED_SCHEDULE_INTERNALS if not hasattr(schedule, attr)]
    # The stage is probed only once ``_stage`` itself is known to exist.
    if not missing:
        missing = [f"stage.{attr}" for attr in _REQUIRED_STAGE_INTERNALS if not hasattr(schedule._stage, attr)]
    if missing:
        raise RuntimeError(
            f"torch {torch.__version__}'s {type(schedule).__name__} is missing {missing}, which the "
            f"pipeline runtime reads or overrides. Pipeline parallelism must be re-validated against "
            f"this torch version before use."
        )


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
        _assert_pipelining_fsdp_contract(stage_module)
        self.paired_examples = paired_examples

        self.stage_module = stage_module
        self.config = config
        self.n_microbatches = n_microbatches
        self.is_last = config.is_last_pp_stage
        self.is_first = config.is_first_pp_stage
        self._token_loss_fn = token_loss_fn

        # A fused last stage returns the summed loss from its forward, so the [mb, S, V] logits plane
        # and its gradient are never built. Only the causal-LM CE decomposes over its token chunks;
        # any other objective fused here would pass through _loss_fn unexamined.
        if fused_head_loss and token_loss_fn is not causal_lm_token_loss:
            raise ValueError(
                f"fused_head_loss=True requires the causal-LM token loss, but this runtime was given "
                f"{getattr(token_loss_fn, '__name__', token_loss_fn)}. Only a per-token cross-entropy "
                f"over the head output decomposes exactly over the token chunks the fused stage sums."
            )
        self._fused_head_loss = fused_head_loss
        if fused_head_loss and self.is_last:
            stage_module.fused_loss_fn = fused_causal_lm_token_loss

        self.stage = PipelineStage(stage_module, config.pp_rank, config.pp_size, device, group=pp_group)
        # Every stage gets loss_fn: torch derives ``has_backward`` from it, so a None silently drops that
        # rank's backward. scale_grads=False because _loss_fn already normalizes by the step.
        self.schedule = _SCHEDULES[config.pp_schedule](
            self.stage,
            n_microbatches=n_microbatches,
            loss_fn=self._loss_fn,
            scale_grads=False,
        )
        _assert_pipelining_schedule_contract(self.schedule)
        self._num_items_in_batch: torch.Tensor | int = 1

    def _loss_fn(self, output: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Per-microbatch loss, divided by the whole step's normalizer.

        Dividing by the step's count rather than the microbatch's makes the per-microbatch losses sum
        to exactly the full-batch loss, which keeps PP gradient-identical to the unsplit model when
        microbatches hold unequal token counts. ``target`` is the labels tensor, or a
        ``{"labels": ..., **extras}`` dict when the step ran with ``extra_targets``. A fused last
        stage has already returned the summed loss from its forward, so only the division is left.
        """
        if self._fused_head_loss:
            return output / self._num_items_in_batch
        return self._token_loss_fn(output, target) / self._num_items_in_batch

    def _validate_pair_alignment(self, batch_size: int) -> None:
        """Require every (chosen, rejected) pair to land in one microbatch.

        In the interleaved layout a pair occupies two adjacent rows, so an even microbatch size is
        exactly the condition for ``tensor_split(dim=0)`` to split no pair. An odd size would compare
        unrelated examples and train the wrong objective without any error.
        """
        if batch_size % (2 * self.n_microbatches) != 0:
            raise ValueError(
                f"A paired (preference) batch of {batch_size} cannot be split into "
                f"{self.n_microbatches} microbatches without separating a chosen/rejected pair: "
                f"batch_size must be a multiple of 2 * pipeline_microbatches "
                f"({2 * self.n_microbatches}). Adjust per_device_train_batch_size or "
                f"pipeline_microbatches."
            )

    def _step_with_dict_targets(
        self,
        args: tuple,
        kwargs: dict,
        labels: torch.Tensor,
        extra_targets: dict[str, torch.Tensor],
        losses: list,
        *,
        backward: bool,
    ) -> None:
        """Drive the schedule with per-microbatch dict targets ``{"labels": ..., **extras}``.

        torch's ``step`` tensor-splits only a single target tensor, but ``_step_microbatches`` takes a
        pre-split target list and passes each entry to ``loss_fn(output, target)`` uninspected.
        Splitting the extras with the same ``torch.tensor_split`` torch applies to the args keeps
        every side tensor aligned with its rows, uneven splits included. The surrounding housekeeping
        is ``step``'s own preamble, and the ``_has_backward`` toggle is ``eval``'s wrapper around it.
        """
        schedule = self.schedule
        if backward and not torch.is_grad_enabled():
            raise RuntimeError("step() with backward requires gradients enabled; use eval_loss() instead.")
        original_has_backward = schedule._has_backward
        try:
            schedule._has_backward = backward
            schedule._stage.has_backward = backward
            schedule._stage.clear_runtime_states()
            arg_mbs, kwarg_mbs = schedule._split_inputs(args, kwargs)
            n = schedule._n_microbatches
            label_mbs = torch.tensor_split(labels, n)
            extra_mbs = {key: torch.tensor_split(value, n) for key, value in extra_targets.items()}
            target_mbs = [{"labels": label_mbs[i], **{k: v[i] for k, v in extra_mbs.items()}} for i in range(n)]
            schedule._step_microbatches(arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs=False)
        finally:
            schedule._has_backward = original_has_backward

    def _schedule_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor | None = None,
    ) -> tuple[tuple, dict]:
        """One batch's ``(args, kwargs)`` for the schedule.

        Only stage 0 consumes ``input_ids``; the others receive activations over P2P. The side inputs
        reach every stage as kwargs, which torch microbatches like the positional args. A fused last
        stage computes the loss inside its forward, so its labels ride along as a kwarg too.
        """
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if labels is not None and self._fused_head_loss:
            kwargs["labels"] = labels
        return (input_ids,) if self.is_first else (), kwargs

    def _validate_extra_targets(self, labels: torch.Tensor, extra_targets: dict[str, torch.Tensor]) -> None:
        for key, value in extra_targets.items():
            if value.size(0) != labels.size(0):
                raise ValueError(
                    f"extra target '{key}' has leading dim {value.size(0)} but labels has "
                    f"{labels.size(0)}; per-example side tensors must align row-for-row with the "
                    f"batch or the microbatch split pairs them with the wrong examples."
                )

    @contextlib.contextmanager
    def _reduce_grad_suppressed(self, suppress: bool) -> Generator[None]:
        """Drop the schedule's trailing gradient reduction for a drive that runs no backward.

        Both schedules end ``_step_microbatches`` with ``stage.perform_reduce_grad``, even under
        ``eval``. On an FSDP2 stage it reshards with no gradients to reduce, so each forward-only batch
        would pay an all-gather to unshard again. The stage stays gathered instead, with its transient
        unsharded params registered, so anything that captures parameter identity calls
        :func:`~src.distributed.fsdp.reshard_fsdp2_modules` first. The previous override is restored on
        exit, so a nested scope cannot re-arm the reduction inside an outer one.
        """
        if not suppress:
            yield
            return
        had_override = "perform_reduce_grad" in self.stage.__dict__
        previous = self.stage.__dict__.get("perform_reduce_grad")
        self.stage.__dict__["perform_reduce_grad"] = _skip_reduce_grad
        try:
            yield
        finally:
            if had_override:
                self.stage.__dict__["perform_reduce_grad"] = previous
            else:
                self.stage.__dict__.pop("perform_reduce_grad", None)

    def _drive(
        self,
        *,
        backward: bool,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        num_items_in_batch: torch.Tensor | int | None,
        extra_targets: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        """Drive one batch's microbatches through the schedule, with or without backward.

        Returns the summed loss on the last stage, else None.
        """
        self._num_items_in_batch = (
            num_items_in_batch if num_items_in_batch is not None else loss_token_count(labels).clamp(min=1)
        )
        if backward and self.paired_examples:
            self._validate_pair_alignment(input_ids.size(0))
        if extra_targets:
            self._validate_extra_targets(labels, extra_targets)

        args, kwargs = self._schedule_inputs(input_ids, attention_mask, position_ids, labels)
        # Only the last stage holds the head and the loss.
        target = labels if self.is_last else None

        self._ensure_stage_initialized(args, kwargs)

        # return_outputs=False: torch's default keeps every microbatch's logits plus a merged copy the
        # loss path never reads.
        losses: list[torch.Tensor] = []
        with self._reduce_grad_suppressed(not backward):
            if self.is_last and extra_targets:
                self._step_with_dict_targets(args, kwargs, labels, extra_targets, losses, backward=backward)
            else:
                run = self.schedule.step if backward else self.schedule.eval
                run(*args, target=target, losses=losses, return_outputs=False, **kwargs)

        if not self.is_last:
            return None
        return torch.stack(losses).sum()

    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        extra_targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Run one optimizer step's microbatches. Returns the summed loss on the last stage, else None.

        Every rank of a chain receives the same batch: stage 0 consumes ``input_ids``, the last stage
        consumes ``labels``, and the side inputs reach every stage as kwargs.

        Args:
            num_items_in_batch: loss normalizer. Defaults to this rank's non-ignored label count;
                callers that normalize across data-parallel ranks pass the reduced count so the
                loss keeps matching the unsplit model.
            extra_targets: per-example side tensors for the loss (advantages, ref log-probs, pool
                positions), each with leading dim == batch. The last stage's ``loss_fn`` then
                receives a dict target ``{"labels": ..., **extras}`` per microbatch, split in
                lockstep with the rows.
        """
        return self._drive(
            backward=True,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_items_in_batch=num_items_in_batch,
            extra_targets=extra_targets,
        )

    @torch.no_grad()
    def eval_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        extra_targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Loss-only counterpart to :meth:`step`. Returns the summed loss on the last stage, else None.

        ``schedule.eval`` suppresses backward but keeps the loss path, and ``return_outputs=False``
        never materializes the merged ``[B, S, V]`` logits. Collective over the chain: every rank
        calls it with the same batch, as for :meth:`step`.
        """
        return self._drive(
            backward=False,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_items_in_batch=num_items_in_batch,
            extra_targets=extra_targets,
        )

    def _ensure_stage_initialized(self, args: tuple, kwargs: dict) -> None:
        """Initialize the stage's I/O in its backward-capable form, whichever drive runs first.

        torch defers shape inference and recv-buffer allocation to the schedule's first call and keeps
        the result: a non-first stage's activation recv buffers get ``requires_grad_(True)`` only if
        ``stage.has_backward`` is set at that moment. A forward-only drive clears that flag, so
        initializing from one would leave those buffers non-differentiable, and every later
        :meth:`step` would fail sending None input gradients upstream. The last stage's output
        metadata is kept the same way, so ``kwargs`` must carry the labels a fused stage needs.
        Idempotent, and collective over the chain (P2P handshake and shape-inference exchange).
        """
        if self.schedule._stage_forward_initialized and self.schedule._stage_backward_initialized:
            return
        # torch initializes from one microbatch's inputs, not the whole batch.
        arg_mbs, kwarg_mbs = self.schedule._split_inputs(args, kwargs)
        self.stage.has_backward = True
        self.schedule._initialize_stage(arg_mbs[0], kwarg_mbs[0])

    @contextlib.contextmanager
    def _unfused_last_stage(self) -> Generator[None]:
        """Run a fused last stage in its logits-returning form for one pass.

        Shape inference recorded the fused scalar as this stage's output metadata, and torch checks
        every forward against it. The last stage never sends its output (its ``act_send_info`` is
        empty), so that check guards nothing here and is bypassed for the pass. Non-last stages are
        left alone: their metadata does guard a send.
        """
        if not (self._fused_head_loss and self.is_last):
            yield
            return
        # Restore the previous values, not defaults, so a nested scope cannot re-arm fusion early.
        previous_loss_fn = self.stage_module.fused_loss_fn
        had_validator = "_validate_fwd_outputs" in self.stage.__dict__
        previous_validator = self.stage.__dict__.get("_validate_fwd_outputs")
        self.stage_module.fused_loss_fn = None
        self.stage.__dict__["_validate_fwd_outputs"] = _accept_any_outputs
        try:
            yield
        finally:
            if had_validator:
                self.stage.__dict__["_validate_fwd_outputs"] = previous_validator
            else:
                self.stage.__dict__.pop("_validate_fwd_outputs", None)
            self.stage_module.fused_loss_fn = previous_loss_fn

    @torch.no_grad()
    def forward_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Drive a forward-only pass across the pipeline; returns the last stage's output, else None.

        No rank holds the whole model, so model outputs needed outside the backward are scored here.
        Every stage takes part, and the last stage returns the merged logits over the full batch, a
        fused stage included (:meth:`_unfused_last_stage`).
        """
        args, kwargs = self._schedule_inputs(input_ids, attention_mask, position_ids)

        # Shape inference must see the fused form whichever drive runs first; a zero placeholder of the
        # labels' shape suffices, since only the shape of the resulting scalar is recorded.
        init_kwargs = {**kwargs, "labels": torch.zeros_like(input_ids)} if self._fused_head_loss else kwargs
        self._ensure_stage_initialized(args, init_kwargs)

        # torch's ``eval`` keeps the loss whenever ``loss_fn`` is set, and would index a None target.
        loss_fn, self.schedule._loss_fn = self.schedule._loss_fn, None
        try:
            with self._reduce_grad_suppressed(True), self._unfused_last_stage():
                outputs = self.schedule.eval(*args, **kwargs)
        finally:
            self.schedule._loss_fn = loss_fn
        return outputs if self.is_last else None
