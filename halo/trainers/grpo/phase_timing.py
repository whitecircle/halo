"""The on-policy GRPO trainers' phase timers: the spans around the step loop's phases and their report at each log."""

from halo.callbacks.phase_timers import MICRO_STEP_END, PhaseTimerCallback
from halo.diagnostics.phase_timers import PhaseTimers
from halo.trainers.world_metrics import WorldMetrics

PHASE_TIME_PREFIX = "phase_time/"


class PhaseTimedMixin:
    """Mixed in ahead of ``DistributedTrainerMixin`` and ``DecoupledCompletionsLogMixin``, so its ``training_step``
    and ``log`` wrap theirs. The host provides ``_log_phase_times`` before :meth:`_configure_phase_timers` runs."""

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
        with self._phase_timers.span("micro_step"):
            loss = super().training_step(model, inputs, num_items_in_batch)
        self._phase_timers.checkpoint(MICRO_STEP_END)
        return loss

    def _compute_loss(self, model, inputs):
        with self._phase_timers.span("loss_forward"):
            return super()._compute_loss(model, inputs)

    def _finish_fsdp2_backward(self, model) -> None:
        with self._phase_timers.span("post_backward_finish"):
            super()._finish_fsdp2_backward(model)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """COLLECTIVE in train mode with the timers on: the phases since the last log are reduced over the ranks."""
        if self._phase_timers.enabled and self.model.training:
            logs.update(self._reduce_phase_times())
        with self._phase_timers.span("log_emit", "host"):
            super().log(logs, start_time)

    def _reduce_phase_times(self) -> dict[str, float]:
        for phase, seconds in self._phase_timers.drain().items():
            self._phase_metrics.mean(f"{PHASE_TIME_PREFIX}{phase}_s", seconds)
            self._phase_metrics.maximum(f"{PHASE_TIME_PREFIX}{phase}_max_s", seconds)
        return self._phase_metrics.flush()
