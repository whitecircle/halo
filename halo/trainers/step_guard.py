"""Non-finite optimizer steps are never applied, since one NaN or inf in a step's gradients reaches every weight.

The verdict is read off the global gradient norm every clip path all-reduces, so every rank reaches the same one. A
toolkit optimizer skips the step on the device under a flag whose norm the host reads a step late; any other is
skipped on the host. Weights already non-finite, or too many non-finite steps in a row, raise on every rank.
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
from halo.optimizers.step_skip import DeviceStepSkipMixin
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.dtensor import materialize_dtensor, to_local
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def _stepped_optimizer(optimizer):
    """The optimizer whose ``step`` runs: the one inside accelerate's ``AcceleratedOptimizer``, which steps it."""
    return getattr(optimizer, "optimizer", optimizer)


class NonFiniteStepGuardMixin:
    """Skips the optimizer step whose global gradient norm is NaN or inf, for :class:`DistributedTrainerMixin`."""

    _max_consecutive_nonfinite_steps = DEFAULT_MAX_CONSECUTIVE_NONFINITE_STEPS
    # The verdict on the step about to be taken: a host skip, or (step, device flag) for a toolkit optimizer.
    _skip_next_optimizer_step = False
    _device_skip: tuple[int, torch.Tensor] | None = None
    # (step, optimizer) of the step an optimizer took under a device flag the host has not read yet.
    _unread_verdict: tuple[int, DeviceStepSkipMixin] | None = None
    # That step's norm in host memory (pinned on CUDA), and the event its copy completes at.
    _verdict_norm: torch.Tensor | None = None
    _verdict_copied: torch.cuda.Event | None = None
    _nonfinite_streak = 0
    _nonfinite_steps_skipped = 0
    # Skipped steps of the training run and of its logging window: transformers divides the run's and a window's
    # loss by every step, while only the applied ones add to it.
    _run_steps_skipped = 0
    _window_steps_skipped = 0
    # What the logged window's loss is multiplied by to divide it by its applied steps alone.
    _window_loss_scale = 1.0

    def _install_nonfinite_step_skip(self) -> None:
        """Route ``self.optimizer.step`` through the verdict :meth:`_get_grad_norm` leaves for it.

        Installed once, on the optimizer accelerate's wrapper steps (a later ``train()`` finds ``self.optimizer``
        wrapped), over whatever ``step`` is at that point, holding the trainer weakly as torch's LR scheduler holds
        the optimizer. Before a scheduler exists it is a method bound to the optimizer, the shape the scheduler
        re-wraps (it reads ``step.__func__``); after one built ahead of ``create_optimizer`` (a caller's
        ``optimizers=(optimizer, scheduler)``) it wraps the scheduler's step and keeps its marker.
        """
        optimizer = _stepped_optimizer(self.optimizer)
        if getattr(optimizer, "_nonfinite_step_skip_installed", False):
            return
        current = optimizer.step
        trainer_ref = weakref.ref(self)

        def skips(opt) -> bool:
            trainer = trainer_ref()
            return trainer is not None and trainer._hand_over_verdict(opt)

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

    def _hand_over_verdict(self, optimizer) -> bool:
        """Give the step about to run on ``optimizer`` its verdict; True when the host skips it instead of stepping.

        A non-finite step returns transformers' loss accumulator to its value before the step's micro-batches
        (:meth:`get_batch_samples`), on the device under a flag. A host skip discards the gradients through the
        optimizer's own ``zero_grad``, which also drops state it deferred for them (AdamWBF16's clip scale). The
        zero-LR step that materializes state during an optimizer restore takes neither, and always steps.
        """
        if getattr(self, "_restoring_optimizer_state", False):
            return False
        if self._device_skip is not None:
            (step, flag), self._device_skip = self._device_skip, None
            optimizer.skip_next_step_if(flag)
            torch.where(flag, self._tr_loss_before_step, self._tr_loss, out=self._tr_loss)
            self._unread_verdict = (step, optimizer)
            return False
        if not self._skip_next_optimizer_step:
            return False
        self._skip_next_optimizer_step = False
        optimizer.zero_grad(set_to_none=True)
        self._tr_loss.copy_(self._tr_loss_before_step)
        return True

    def _get_grad_norm(self, model, grad_norm=None):
        """HF's pre-step gradient norm, plus the verdict the optimizer step reads, once the previous step's device
        verdict is settled. A loss scaler stepping accelerate's optimizer wrapper (fp16) owns overflow steps: it
        skips them itself and lowers its scale, so no verdict is taken."""
        grad_norm = super()._get_grad_norm(model, grad_norm=grad_norm)
        self._read_device_verdict()
        if isinstance(self.optimizer, AcceleratedOptimizer) and self.optimizer.scaler is not None:
            return grad_norm
        step = self.state.global_step + 1
        if isinstance(grad_norm, torch.Tensor) and isinstance(_stepped_optimizer(self.optimizer), DeviceStepSkipMixin):
            self._device_skip = (step, self._copy_norm_to_host(grad_norm))
        else:
            norm = float(materialize_dtensor(grad_norm)) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            self._skip_next_optimizer_step = not self._record_verdict(step, norm)
        return grad_norm

    def _copy_norm_to_host(self, grad_norm: torch.Tensor) -> torch.Tensor:
        """Start the non-blocking copy of ``grad_norm`` to host memory; return the device flag of a non-finite one."""
        norm = materialize_dtensor(grad_norm).detach().float()
        if self._verdict_norm is None:
            self._verdict_norm = torch.empty((), dtype=torch.float32, pin_memory=norm.is_cuda)
            self._verdict_copied = torch.cuda.Event() if norm.is_cuda else None
        self._verdict_norm.copy_(norm, non_blocking=True)
        if self._verdict_copied is not None:
            self._verdict_copied.record(torch.cuda.current_stream(norm.device))
        return torch.isfinite(norm).logical_not_()

    def _read_device_verdict(self) -> None:
        """COLLECTIVE on a non-finite norm (:meth:`_record_verdict`). Read the norm of the last step an optimizer
        took under a device flag, if still unread, and settle it: the optimizer settles the step (its bookkeeping
        rolled back where the flag skipped it), then the guard counts, warns or raises. Its copy finished with its
        own step, so the wait returns at once unless the host runs a whole step ahead of the device."""
        if self._unread_verdict is None:
            return
        (step, optimizer), self._unread_verdict = self._unread_verdict, None
        if self._verdict_copied is not None:
            self._verdict_copied.synchronize()
        norm = float(self._verdict_norm)
        optimizer.settle_step(skipped=not math.isfinite(norm))
        self._record_verdict(step, norm)

    def _record_verdict(self, step: int, norm: float) -> bool:
        """COLLECTIVE on a non-finite ``norm``. Whether optimizer step ``step``, whose global gradient norm is
        ``norm``, may be applied; counts, warns about or raises on one that may not."""
        if math.isfinite(norm):
            self._nonfinite_streak = 0
            return True
        self._raise_on_nonfinite_weights(step)
        self._nonfinite_streak += 1
        self._nonfinite_steps_skipped += 1
        self._run_steps_skipped += 1
        self._window_steps_skipped += 1
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

    def _init_training_state(self, *args, **kwargs):
        """A training run starts with no unread verdict, settled ahead of any optimizer-state load (one an earlier
        run left when it raised), and counts its skipped steps from zero, as transformers restarts its state and
        loss accumulators."""
        self._read_device_verdict()
        self._run_steps_skipped = self._window_steps_skipped = 0
        return super()._init_training_state(*args, **kwargs)

    def get_batch_samples(self, *args, **kwargs):
        """The batches of the optimizer step about to start, fetched ahead of its micro-batches: the loss
        accumulator's value here is the one a skip of this step restores."""
        self._tr_loss_before_step = self._tr_loss.clone()
        return super().get_batch_samples(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        """A step that logs, evaluates or saves settles the unread verdict first, so a log reports it and a
        checkpoint's optimizer state counts applied steps only (an evaluation decides a ``save_strategy: best``
        save). A logged window's loss is over its applied steps, NaN where it applied none."""
        control = self.control
        if control.should_log or control.should_evaluate or control.should_save:
            self._read_device_verdict()
        window = self.state.global_step - self._globalstep_last_logged
        applied = window - self._window_steps_skipped
        self._window_loss_scale = window / applied if applied else math.nan
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)
        self._window_loss_scale = 1.0
        if self._globalstep_last_logged == self.state.global_step:
            self._window_steps_skipped = 0

    def log(self, logs: dict[str, float], start_time: float | None = None):
        if "loss" in logs:
            logs["loss"] *= self._window_loss_scale
            if self._nonfinite_steps_skipped:
                logs["nonfinite_steps_skipped"] = self._nonfinite_steps_skipped
        super().log(logs, start_time)

    def _finalize_training(self, *args, **kwargs):
        """Read the last step's verdict before the run ends, so a final non-finite step is reported; the run's
        ``train_loss`` over the steps it applied, NaN where it applied none."""
        self._read_device_verdict()
        if self._run_steps_skipped:
            steps = self.state.global_step - self._initial_global_step
            applied = steps - self._run_steps_skipped
            scale = steps / applied if applied else math.nan
            self._tr_loss.mul_(scale)
            self._total_loss_scalar *= scale
        return super()._finalize_training(*args, **kwargs)
