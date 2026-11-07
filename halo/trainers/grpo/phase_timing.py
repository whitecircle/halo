"""The on-policy GRPO trainers' phase timers: the spans around the step loop's phases and their report at each log."""

import torch

from halo.callbacks.phase_timer_callback import MICRO_STEP_END, PhaseTimerCallback
from halo.diagnostics.phase_timers import Phase, PhaseTimers
from halo.trainers.world_metrics import WorldMetrics

PHASE_TIME_PREFIX = "phase_time/"


class PhaseTimedMixin:
    """Mixed in ahead of ``ProtectedTokenEntropyMixin`` (the chunked scorer behind it never calls ``super()``) and of
    ``DistributedTrainerMixin`` and ``DecoupledCompletionsLogMixin``, so its ``training_step`` and ``log`` wrap theirs.
    The host provides ``_log_phase_times`` before :meth:`_configure_phase_timers` runs."""

    _log_phase_times = False
    # Shared while disabled: a disabled timer holds no state.
    _phase_timers = PhaseTimers(enabled=False)

    def _configure_phase_timers(self) -> None:
        """Idempotent. Needs the callback handler, so it runs after TRL's constructor."""
        if not self._log_phase_times or self._phase_timers.enabled:
            return
        self._phase_timers = PhaseTimers(enabled=True)
        self._phase_metrics = WorldMetrics()
        self.add_callback(PhaseTimerCallback(self._phase_timers))

    def training_step(self, model, inputs, num_items_in_batch=None):
        """The span holds the generation round a window's first micro-step triggers, whose own spans nest in it, so
        what it reports is the backward and the rest of the step's body."""
        with self._phase_timers.span(Phase.MICRO_STEP):
            loss = super().training_step(model, inputs, num_items_in_batch)
        self._phase_timers.checkpoint(MICRO_STEP_END)
        return loss

    def evaluate(self, *args, **kwargs):
        """COLLECTIVE. An eval round's spans are not the train step's: its wall time lands in ``unaccounted``."""
        with self._phase_timers.suspended():
            return super().evaluate(*args, **kwargs)

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        """A no-grad scoring pass is timed as ``no_grad`` where nothing around it already times one: TRL's old and
        reference passes run inside its round build, while the environmental trainer wraps its own."""
        if torch.is_grad_enabled() or self._phase_timers.is_open(Phase.NO_GRAD, Phase.NO_GRAD_REF):
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)
        with self._phase_timers.span(Phase.NO_GRAD):
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        with self._phase_timers.span(Phase.LOSS_FORWARD):
            return super()._compute_loss(model, inputs)

    def _finish_fsdp2_backward(self, model) -> None:
        with self._phase_timers.span(Phase.POST_BACKWARD_FINISH):
            super()._finish_fsdp2_backward(model)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """COLLECTIVE in train mode with the timers on: the phases since the last log are reduced over the ranks."""
        if self._phase_timers.enabled and self.model.training:
            logs.update(self._reduce_phase_times())
        with self._phase_timers.span(Phase.LOG_EMIT, "host"):
            super().log(logs, start_time)

    def _reduce_phase_times(self) -> dict[str, float]:
        for phase, seconds in self._phase_timers.drain().items():
            self._phase_metrics.mean(f"{PHASE_TIME_PREFIX}{phase}_s", seconds)
            self._phase_metrics.maximum(f"{PHASE_TIME_PREFIX}{phase}_max_s", seconds)
        return self._phase_metrics.flush(
            key_mismatch_error=lambda keys_by_rank: RuntimeError(
                f"Phase-time keys differ across ranks: {keys_by_rank}. Every rank must run each timed span, since log "
                f"reduces each phase over the world."
            )
        )
