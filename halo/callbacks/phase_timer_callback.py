"""The optimizer-side phases of :class:`~halo.diagnostics.phase_timers.PhaseTimers`: the clip and reductions between the
last micro-step and the optimizer, the optimizer itself, and the step boundary the spans must add up to."""

from transformers import TrainerCallback

from halo.diagnostics.phase_timers import Phase, PhaseTimers

MICRO_STEP_END = "micro_step_end"
_PRE_OPTIMIZER = "pre_optimizer"


class PhaseTimerCallback(TrainerCallback):
    """Registered by a trainer that logs phase times; the trainer marks :data:`MICRO_STEP_END` after each micro-step."""

    def __init__(self, timers: PhaseTimers) -> None:
        self._timers = timers

    def on_train_begin(self, args, state, control, **kwargs):
        self._timers.begin_run()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._timers.since(MICRO_STEP_END, Phase.REDUCE)
        self._timers.checkpoint(_PRE_OPTIMIZER)

    def on_optimizer_step(self, args, state, control, **kwargs):
        self._timers.since(_PRE_OPTIMIZER, Phase.OPTIMIZER)

    def on_step_end(self, args, state, control, **kwargs):
        self._timers.finish_step()
