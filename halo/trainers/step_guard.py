"""Non-finite optimizer steps are never applied, since one NaN or inf reaches every weight. The verdict comes off the
all-reduced global gradient norm, so every rank reaches the same one. Non-finite weights or too long a streak raise
everywhere."""

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
    """The optimizer inside accelerate's ``AcceleratedOptimizer``, which steps it."""
    return getattr(optimizer, "optimizer", optimizer)


class NonFiniteStepGuardMixin:
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
        """Installed once, over whatever ``step`` is then: a method bound to the optimizer, the shape the LR scheduler
        re-wraps (it reads ``step.__func__``), or the step of a scheduler built ahead of ``create_optimizer``, keeping
        its marker."""
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
        """True when the host skips the step. A non-finite step returns HF's loss accumulator to its pre-step value; a
        host skip also drops state the optimizer deferred (AdamWBF16's clip scale). The zero-LR step that materializes
        state always steps."""
        if getattr(self, "_materializing_optimizer_state", False):
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
        """HF's pre-step gradient norm, plus the verdict the step reads. A loss scaler on accelerate's wrapper (fp16)
        owns overflow steps, skipping them and lowering its scale, so no verdict is taken."""
        grad_norm = super()._get_grad_norm(model, grad_norm=grad_norm)
        self._settle_device_verdict()
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
        # torch's clip returns a CPU zero on a step where no parameter holds a gradient: a mixed image run's text batch.
        if not norm.is_cuda:
            norm = norm.to(self._tr_loss.device)
        if self._verdict_norm is None:
            self._verdict_norm = torch.empty((), dtype=torch.float32, pin_memory=norm.is_cuda)
            self._verdict_copied = torch.cuda.Event() if norm.is_cuda else None
        self._verdict_norm.copy_(norm, non_blocking=True)
        if self._verdict_copied is not None:
            self._verdict_copied.record(torch.cuda.current_stream(norm.device))
        return torch.isfinite(norm).logical_not_()

    def _settle_device_verdict(self) -> None:
        """COLLECTIVE on a non-finite norm. The optimizer rolls back its bookkeeping where the flag skipped the step,
        then the guard counts, warns or raises."""
        if self._unread_verdict is None:
            return
        (step, optimizer), self._unread_verdict = self._unread_verdict, None
        if self._verdict_copied is not None:
            self._verdict_copied.synchronize()
        norm = float(self._verdict_norm)
        optimizer.settle_step(skipped=not math.isfinite(norm))
        self._record_verdict(step, norm)

    def _record_verdict(self, step: int, norm: float) -> bool:
        """COLLECTIVE on a non-finite ``norm``. Whether ``step`` may be applied; counts, warns or raises on one that
        may not."""
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
        """COLLECTIVE. Raise on every rank when any rank holds a non-finite parameter; runs only on a non-finite
        step."""
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
        """Settle a verdict an earlier run left when it raised, ahead of any optimizer-state load, and count skipped
        steps from zero, as transformers restarts its accumulators."""
        self._settle_device_verdict()
        self._run_steps_skipped = self._window_steps_skipped = 0
        return super()._init_training_state(*args, **kwargs)

    def get_batch_samples(self, *args, **kwargs):
        """Fetched ahead of the step's micro-batches, so the loss accumulator here is the value a skip restores."""
        self._tr_loss_before_step = self._tr_loss.clone()
        return super().get_batch_samples(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        """Settle the unread verdict first, so a log reports it and a checkpoint's optimizer state counts applied steps
        only. A window's loss is over its applied steps, NaN at none."""
        control = self.control
        if control.should_log or control.should_evaluate or control.should_save:
            self._settle_device_verdict()
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
        """Read the last step's verdict, so a final non-finite step is reported; ``train_loss`` is over applied steps,
        NaN at none."""
        self._settle_device_verdict()
        if self._run_steps_skipped:
            steps = self.state.global_step - self._initial_global_step
            applied = steps - self._run_steps_skipped
            scale = steps / applied if applied else math.nan
            self._tr_loss.mul_(scale)
            self._total_loss_scalar *= scale
        return super()._finalize_training(*args, **kwargs)
