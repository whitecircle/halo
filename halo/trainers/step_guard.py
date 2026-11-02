"""Non-finite optimizer steps are never applied.

One NaN or inf anywhere in a step's gradients reaches every parameter the optimizer touches, and from the next
forward on the whole model is NaN. The global gradient norm transformers computes right before
``optimizer.step()`` is all-reduced by every clip path, so every rank reads the same verdict off it. A non-finite
one discards the step's gradients and its loss instead of stepping: weights, optimizer moments and the run's loss
accumulator stay as they were, while the step counter, the LR schedule and the data move on.

Two cases raise on every rank instead: weights that are already non-finite (the corruption predates the step's
gradients, and a skip cannot undo it), and more than ``max_consecutive_nonfinite_steps`` non-finite steps in a row
(a run whose gradients stay non-finite is broken, not unlucky). The verdict costs one device-to-host read per
optimizer step.
"""

import functools
import logging
import math
import types
import weakref

import torch
from accelerate.optimizer import AcceleratedOptimizer

from halo.configs.script_arguments import DEFAULT_MAX_CONSECUTIVE_NONFINITE_STEPS
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.log import KEY_PREVIEW_COUNT
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.dtensor import materialize_dtensor, to_local
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class NonFiniteStepGuardMixin:
    """Skips the optimizer step whose global gradient norm is NaN or inf, for :class:`DistributedTrainerMixin`."""

    _max_consecutive_nonfinite_steps = DEFAULT_MAX_CONSECUTIVE_NONFINITE_STEPS
    _skip_next_optimizer_step = False
    _nonfinite_streak = 0
    _nonfinite_steps_skipped = 0
    _unapplied_window = False

    def _install_nonfinite_step_skip(self) -> None:
        """Route ``self.optimizer.step`` through the verdict :meth:`_get_grad_norm` leaves for it.

        Installed over whatever ``step`` is at that point, holding the trainer weakly as torch's LR scheduler holds
        the optimizer. Before a scheduler exists it is a method bound to the optimizer, the shape the scheduler
        re-wraps (it reads ``step.__func__``); after one built ahead of ``create_optimizer`` (a caller's
        ``optimizers=(optimizer, scheduler)``) it wraps the scheduler's step and keeps its marker. A skipped step
        discards the gradients through the optimizer's own ``zero_grad``, which also drops state an optimizer
        deferred for them (AdamWBF16's clip scale), and returns transformers' loss accumulator to its value
        before the step's micro-batches (:meth:`get_batch_samples`). The zero-LR step that materializes state
        during an optimizer restore always steps.
        """
        optimizer = self.optimizer
        if getattr(optimizer, "_nonfinite_step_skip_installed", False):
            return
        current = optimizer.step
        trainer_ref = weakref.ref(self)

        def skips(opt) -> bool:
            trainer = trainer_ref()
            if (
                trainer is None
                or not trainer._skip_next_optimizer_step
                or getattr(trainer, "_restoring_optimizer_state", False)
            ):
                return False
            trainer._skip_next_optimizer_step = False
            opt.zero_grad(set_to_none=True)
            trainer._tr_loss.copy_(trainer._tr_loss_before_step)
            return True

        if isinstance(current, types.MethodType):
            stock_step = current.__func__

            def step(opt, *args, **kwargs):
                return None if skips(opt) else stock_step(opt, *args, **kwargs)

            optimizer.step = types.MethodType(step, optimizer)
        else:
            optimizer_ref = weakref.ref(optimizer)

            @functools.wraps(current)
            def step(*args, **kwargs):
                opt = optimizer_ref()
                if not skips(opt):
                    return current(*args, **kwargs)
                # The scheduler's wrapper, bypassed here, marks the step taken; unmarked, its next step warns.
                opt._opt_called = True
                return None

            optimizer.step = step
        optimizer._nonfinite_step_skip_installed = True

    def _get_grad_norm(self, model, grad_norm=None):
        """HF's pre-step gradient norm, plus the verdict the optimizer step reads."""
        grad_norm = super()._get_grad_norm(model, grad_norm=grad_norm)
        self._skip_next_optimizer_step = not self._step_is_finite(grad_norm)
        return grad_norm

    def _step_is_finite(self, grad_norm) -> bool:
        """Whether the step about to be taken may be applied; counts, warns about or raises on one that may not.

        Rank-uniform: every clip path all-reduces the norm. A loss scaler stepping accelerate's optimizer wrapper
        (fp16) owns overflow steps: it skips them itself and lowers its scale.
        """
        if isinstance(self.optimizer, AcceleratedOptimizer) and self.optimizer.scaler is not None:
            return True
        norm = float(materialize_dtensor(grad_norm)) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        if math.isfinite(norm):
            self._nonfinite_streak = 0
            return True
        step = self.state.global_step + 1
        self._raise_on_nonfinite_weights(step)
        self._nonfinite_streak += 1
        self._nonfinite_steps_skipped += 1
        limit = self._max_consecutive_nonfinite_steps
        if self._nonfinite_streak > limit:
            first = step - self._nonfinite_streak + 1
            raise RuntimeError(
                f"Optimizer steps {first}-{step}: {self._nonfinite_streak} consecutive steps had a non-finite "
                f"global gradient norm (last: {norm}), past max_consecutive_nonfinite_steps={limit}. None was "
                f"applied, so the weights are those after step {first - 1}. Gradients that stay non-finite point "
                f"at the model or the data, not at one batch: find the cause and resume from the last checkpoint."
            )
        if is_global_main_process():
            logger.warning(
                f"Optimizer step {step}: the global gradient norm is {norm}; the update is skipped (weights and "
                f"optimizer moments unchanged, the step's gradients discarded). Non-finite steps: "
                f"{self._nonfinite_streak} in a row (limit {limit}), {self._nonfinite_steps_skipped} since launch."
            )
        return False

    def _raise_on_nonfinite_weights(self, step: int) -> None:
        """COLLECTIVE: raise on every rank when any rank holds a non-finite parameter value.

        Runs only on a non-finite step: one isfinite reduction per local parameter shard, read back once.
        """
        reshard_fsdp2_modules(self.model)
        named = [(name, to_local(param.detach())) for name, param in self.model.named_parameters()]
        finite = torch.stack([torch.isfinite(shard).all() for _, shard in named]).tolist() if named else []
        bad = [name for (name, _), ok in zip(named, finite, strict=True) if not ok]
        reason = None
        if bad:
            shown = ", ".join(bad[:KEY_PREVIEW_COUNT])
            more = f" and {len(bad) - KEY_PREVIEW_COUNT} more" if len(bad) > KEY_PREVIEW_COUNT else ""
            reason = (
                f"{len(bad)} parameter(s) already hold NaN/inf before the update: {shown}{more}. Skipping the "
                f"step cannot repair them, since they went bad before this step's gradients. Resume from the last "
                f"checkpoint."
            )
        reject_across_ranks(reason, f"The finite-weights check at optimizer step {step}")

    def get_batch_samples(self, *args, **kwargs):
        """The batches of the optimizer step about to start, fetched ahead of its micro-batches: the loss
        accumulator's value here is the one a skip of this step restores."""
        self._tr_loss_before_step = self._tr_loss.clone()
        return super().get_batch_samples(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        """A logging window none of whose steps was applied logs a NaN loss, not the 0 its restored accumulator holds."""
        window = self.state.global_step - self._globalstep_last_logged
        self._unapplied_window = 0 < window <= self._nonfinite_streak
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)
        self._unapplied_window = False

    def log(self, logs: dict[str, float], start_time: float | None = None):
        if "loss" in logs:
            if self._unapplied_window:
                logs["loss"] = math.nan
            if self._nonfinite_steps_skipped:
                logs["nonfinite_steps_skipped"] = self._nonfinite_steps_skipped
        super().log(logs, start_time)
