"""The pipeline runner: this rank's stage, the schedule, and one optimizer step's microbatches.

Wraps ``torch.distributed.pipelining``, whose schedule runs every microbatch's forward and backward, so
the caller never calls ``.backward()`` and runs ``gradient_accumulation_steps=1``. The trainer supplies
``zero_grad``, a grad norm reduced across the chain and the last stage's loss broadcast for logging.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
from torch.distributed.pipelining.schedules import PipelineScheduleSingle

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import PPSchedule
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from halo.distributed.pipeline_parallel.stage_losses import FusedHeadLoss, TokenLoss, fused_causal_lm_token_loss
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, loss_token_counts_per_row
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_SCHEDULES: dict[PPSchedule, type[PipelineScheduleSingle]] = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}

# Each key the pipeline consumes → its pad value (None = the tokenizer's pad id). The trainer's
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
# Private schedule and stage members this runner reads or overrides.
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

# 1F1B bubble fraction = (pp - 1) / (microbatches + pp - 1): _BUBBLE_TARGET_FACTOR·(pp-1) microbatches
# put it at 1/10, and below _BUBBLE_WARN_FACTOR·pp it grows large enough to warn.
_BUBBLE_WARN_FACTOR = 4
_BUBBLE_TARGET_FACTOR = 9


def resolve_batch_pad_values(pad_token_id: int, extra: Mapping[str, int] | None = None) -> dict[str, int]:
    """:data:`PP_BATCH_PAD_VALUES` with this run's tokenizer pad id, plus an adapter's seq-shaped extras."""
    resolved = {key: pad_token_id if value is None else value for key, value in PP_BATCH_PAD_VALUES.items()}
    resolved.update(extra or {})
    return resolved


def resolve_microbatch_count(
    config: ParallelismConfig, gradient_accumulation_steps: int, per_device_train_batch_size: int
) -> int:
    """The pipeline's microbatch count: ``pipeline_microbatches``, else the user's accumulation.

    The schedule runs every microbatch in one optimizer step, so HF's accumulation loop left on would
    square the effective batch; the caller then sets ``gradient_accumulation_steps=1``.
    """
    if config.pp_microbatches and gradient_accumulation_steps > 1:
        raise ValueError(
            f"pipeline_microbatches={config.pp_microbatches} and gradient_accumulation_steps="
            f"{gradient_accumulation_steps} are both set: under PP the microbatches ARE the "
            f"accumulation, so the accumulation would be silently discarded. Set exactly one of them."
        )
    requested = config.pp_microbatches or gradient_accumulation_steps
    n_microbatches = config.pp_microbatch_count(gradient_accumulation_steps)
    if n_microbatches != requested and is_global_main_process():
        logger.warning(
            "pipeline_microbatches=%d is below pipeline_parallel_size=%d (Schedule1F1B needs a full "
            "pipeline); raising to %d.",
            requested,
            config.pp_size,
            n_microbatches,
        )
    if n_microbatches < _BUBBLE_WARN_FACTOR * config.pp_size and is_global_main_process():
        logger.warning(
            "pipeline_microbatches=%d gives a large pipeline bubble at pipeline_parallel_size=%d (idle "
            "fraction (pp-1)/(m+pp-1): 1/5 of the step at m=2*pp on 2 stages). For a 10%% "
            "bubble use at least %d microbatches (rule: %d*(pipeline_parallel_size-1)).",
            n_microbatches,
            config.pp_size,
            _BUBBLE_TARGET_FACTOR * (config.pp_size - 1),
            _BUBBLE_TARGET_FACTOR,
        )
    if per_device_train_batch_size % n_microbatches != 0:
        raise ValueError(
            f"per_device_train_batch_size ({per_device_train_batch_size}) must be "
            f"divisible by the pipeline microbatch count ({n_microbatches})."
        )
    return n_microbatches


def _accept_any_outputs(outputs: tuple[torch.Tensor, ...]) -> None:
    """Stand-in for ``PipelineStage._validate_fwd_outputs`` while a fused last stage returns logits."""
    del outputs


def _skip_reduce_grad(grad_scale_factor: int) -> None:
    """Stand-in for ``PipelineStage.perform_reduce_grad`` on a drive that runs no backward.

    Both schedules end every drive with it, ``eval`` included; on an FSDP2 stage it reshards with no
    gradient to reduce, so each forward-only batch would pay an all-gather to unshard again. The stage
    stays gathered instead, so anything that captures parameter identity calls
    :func:`~halo.distributed.fsdp.reshard_fsdp2_modules` first.
    """
    del grad_scale_factor


def _assert_pipelining_contract(stage_module: nn.Module, schedule: PipelineScheduleSingle) -> None:
    """Verify the private FSDP2, schedule and stage members this runner depends on still exist."""
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
    missing = [attr for attr in _REQUIRED_SCHEDULE_INTERNALS if not hasattr(schedule, attr)]
    # The stage is probed only once ``_stage`` itself is known to exist.
    if not missing:
        missing = [f"stage.{attr}" for attr in _REQUIRED_STAGE_INTERNALS if not hasattr(schedule._stage, attr)]
    if missing:
        raise RuntimeError(
            f"torch {torch.__version__}'s {type(schedule).__name__} is missing {missing}, which the "
            f"pipeline runner reads or overrides. Pipeline parallelism must be re-validated against "
            f"this torch version before use."
        )


class PipelineRunner:
    """Drives one optimizer step's worth of microbatches through this rank's pipeline stage.

    ``loss`` is the last stage's loss, and every stage needs it: under :class:`FusedHeadLoss` the labels
    ride as a kwarg to every stage and shape inference must see the fused scalar output.
    ``rows_per_example`` is 2 for interleaved ``(chosen, rejected)`` batches, whose pairs must not be
    split across microbatches.
    """

    def __init__(
        self,
        stage_module: PipelineStageModule,
        config: ParallelismConfig,
        pp_group: dist.ProcessGroup,
        device: torch.device,
        n_microbatches: int,
        loss: TokenLoss | FusedHeadLoss,
        *,
        rows_per_example: int = 1,
    ):
        if not isinstance(loss, TokenLoss | FusedHeadLoss):
            raise TypeError(
                f"PipelineRunner takes loss=TokenLoss(fn) or loss=FusedHeadLoss(), got {type(loss).__name__}; "
                f"wrap a bare token loss function as TokenLoss(fn)."
            )
        self.stage_module = stage_module
        self.config = config
        self.n_microbatches = n_microbatches
        self.loss = loss
        self.rows_per_example = rows_per_example
        self.is_first = config.is_first_pp_stage
        self.is_last = config.is_last_pp_stage
        # The backbone kwarg this stage re-reads the microbatch's token ids under, ``None`` where only its
        # positional activation carries them.
        self.token_ids_kwarg = stage_module.token_ids_kwarg
        if isinstance(loss, FusedHeadLoss) and self.is_last:
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
        _assert_pipelining_contract(stage_module, self.schedule)
        self._num_items_in_batch: torch.Tensor | int = 1

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

        COLLECTIVE over the chain: every rank passes the same batch; stage 0 consumes ``input_ids``,
        the last stage ``labels``, and the side inputs reach every stage as kwargs.

        Args:
            num_items_in_batch: loss normalizer; defaults to this rank's next-token target count.
                Callers that normalize across data-parallel ranks pass the reduced count.
            extra_targets: per-example side tensors for the loss, leading dim == batch. The last
                stage's loss then receives ``{"labels": ..., **extras}`` per microbatch, split in
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
        """Loss-only counterpart to :meth:`step`; COLLECTIVE over the chain like it.

        ``schedule.eval`` suppresses backward but keeps the loss path, and ``return_outputs=False``
        never materializes the merged ``[B, S, V]`` logits.
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

    @torch.no_grad()
    def forward_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """COLLECTIVE over the chain: a forward-only pass; the last stage returns the merged logits
        over the full batch, a fused stage included (:meth:`_unfused_last_stage`), others None."""
        args, kwargs = self._schedule_inputs(input_ids, attention_mask, position_ids)
        # Shape inference must see the fused form whichever drive runs first; a zero placeholder of the
        # labels' shape suffices, since only the shape of the resulting scalar is recorded.
        fused = isinstance(self.loss, FusedHeadLoss)
        self._ensure_stage_initialized(args, {**kwargs, "labels": torch.zeros_like(input_ids)} if fused else kwargs)

        # torch's ``eval`` keeps the loss whenever ``loss_fn`` is set, and would index a None target.
        loss_fn, self.schedule._loss_fn = self.schedule._loss_fn, None
        try:
            with self._override_stage_attr("perform_reduce_grad", _skip_reduce_grad), self._unfused_last_stage():
                outputs = self.schedule.eval(*args, **kwargs)
        finally:
            self.schedule._loss_fn = loss_fn
        return outputs if self.is_last else None

    def _loss_fn(self, output: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Per-microbatch loss, divided by the whole step's normalizer.

        Dividing by the step's count rather than the microbatch's makes the per-microbatch losses sum
        to exactly the full-batch loss, so PP stays gradient-identical to the unsplit model when
        microbatches hold unequal token counts. A fused last stage already returned the summed loss.
        """
        if isinstance(self.loss, FusedHeadLoss):
            return output / self._num_items_in_batch
        return self.loss.fn(output, target) / self._num_items_in_batch

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
        """Drive one batch's microbatches through the schedule, with or without backward; the summed
        loss on the last stage, else None."""
        self._num_items_in_batch = (
            num_items_in_batch
            if num_items_in_batch is not None
            else loss_token_counts_per_row(labels).sum().clamp(min=1)
        )
        # An even microbatch keeps every interleaved (chosen, rejected) pair whole; an odd one would
        # silently compare unrelated examples.
        if backward and self.rows_per_example == 2 and input_ids.size(0) % (2 * self.n_microbatches) != 0:
            raise ValueError(
                f"A paired (preference) batch of {input_ids.size(0)} cannot be split into "
                f"{self.n_microbatches} microbatches without separating a chosen/rejected pair: "
                f"batch_size must be a multiple of 2 * pipeline_microbatches "
                f"({2 * self.n_microbatches}). Adjust per_device_train_batch_size or "
                f"pipeline_microbatches."
            )
        for key, value in (extra_targets or {}).items():
            if value.size(0) != labels.size(0):
                raise ValueError(
                    f"extra target '{key}' has leading dim {value.size(0)} but labels has "
                    f"{labels.size(0)}; per-example side tensors must align row-for-row with the "
                    f"batch or the microbatch split pairs them with the wrong examples."
                )

        args, kwargs = self._schedule_inputs(input_ids, attention_mask, position_ids, labels)
        self._ensure_stage_initialized(args, kwargs)

        # return_outputs=False: torch's default keeps every microbatch's logits plus a merged copy the
        # loss path never reads.
        losses: list[torch.Tensor] = []
        reduce_grad = (
            contextlib.nullcontext()
            if backward
            else self._override_stage_attr("perform_reduce_grad", _skip_reduce_grad)
        )
        with reduce_grad:
            if self.is_last and extra_targets:
                self._step_with_dict_targets(args, kwargs, labels, extra_targets, losses, backward=backward)
            else:
                run = self.schedule.step if backward else self.schedule.eval
                # Only the last stage holds the head and the loss.
                run(*args, target=labels if self.is_last else None, losses=losses, return_outputs=False, **kwargs)

        if not self.is_last:
            return None
        return torch.stack(losses).sum()

    def _schedule_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor | None = None,
    ) -> tuple[tuple, dict]:
        """One batch's ``(args, kwargs)`` for the schedule.

        Only stage 0 consumes ``input_ids`` as its activation; the others receive activations over P2P, and
        take the ids as a ``token_ids`` kwarg where their family's layers re-read them
        (:attr:`~halo.distributed.pipeline_parallel.stage.PipelineStageModule.token_ids_kwarg`). The side
        inputs reach every stage as kwargs, which torch microbatches like the positional args, and so do the
        labels a fused last stage scores inside its forward.
        """
        kwargs = {}
        if self.token_ids_kwarg is not None:
            kwargs["token_ids"] = input_ids
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if labels is not None and isinstance(self.loss, FusedHeadLoss):
            kwargs["labels"] = labels
        return (input_ids,) if self.is_first else (), kwargs

    def _ensure_stage_initialized(self, args: tuple, kwargs: dict) -> None:
        """COLLECTIVE over the chain (P2P handshake and shape inference): initialize the stage's I/O in
        its backward-capable form, whichever drive runs first.

        torch infers shapes and allocates recv buffers on the schedule's first call and keeps them: a
        non-first stage's buffers require grad only if ``stage.has_backward`` is set then, which a
        forward-only drive clears, leaving every later :meth:`step` sending None input gradients. The
        last stage's output metadata is latched too, so ``kwargs`` must carry a fused stage's labels.
        """
        if self.schedule._stage_forward_initialized and self.schedule._stage_backward_initialized:
            return
        # torch initializes from one microbatch's inputs, not the whole batch.
        arg_mbs, kwarg_mbs = self.schedule._split_inputs(args, kwargs)
        self.stage.has_backward = True
        self.schedule._initialize_stage(arg_mbs[0], kwarg_mbs[0])

    @contextlib.contextmanager
    def _override_stage_attr(self, name: str, value) -> Generator[None]:
        """Shadow the torch stage's ``name`` in its instance dict for the scope.

        The previous shadow is restored on exit, not the class member, so a nested scope cannot re-arm
        the original inside an outer one.
        """
        had_override = name in self.stage.__dict__
        previous = self.stage.__dict__.get(name)
        self.stage.__dict__[name] = value
        try:
            yield
        finally:
            if had_override:
                self.stage.__dict__[name] = previous
            else:
                self.stage.__dict__.pop(name, None)

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

        torch's ``step`` splits only a single target tensor, but ``_step_microbatches`` takes a
        pre-split target list and hands each entry to the loss uninspected. The extras split with the
        same ``tensor_split`` torch applies to the args, so every side tensor stays with its rows,
        uneven splits included. The housekeeping around it is ``step``'s own preamble, and the
        ``_has_backward`` toggle is ``eval``'s wrapper.
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

    @contextlib.contextmanager
    def _unfused_last_stage(self) -> Generator[None]:
        """Run a fused last stage in its logits-returning form for one pass.

        Shape inference recorded the fused scalar as this stage's output metadata, and torch checks
        every forward against it; the last stage sends nothing, so the check guards nothing and is
        bypassed. The previous loss function is restored, so a nested scope cannot re-arm fusion early.
        """
        if not (isinstance(self.loss, FusedHeadLoss) and self.is_last):
            yield
            return
        previous_loss_fn = self.stage_module.fused_loss_fn
        self.stage_module.fused_loss_fn = None
        try:
            with self._override_stage_attr("_validate_fwd_outputs", _accept_any_outputs):
                yield
        finally:
            self.stage_module.fused_loss_fn = previous_loss_fn
