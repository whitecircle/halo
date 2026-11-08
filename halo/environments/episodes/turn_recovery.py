"""Recovery from an unproductive turn: one the engine cut short, or one the model ended on nothing, nudged and
retried within the episode's turn and recovery budgets."""

from halo.config_validation import present, require_int
from halo.environments.episodes.messages import Message, StepOutcome, Trajectory

# The trajectory-info counters of recovered turns; a turn cut inside a tool call counts under both cut keys.
LENGTH_CUTOFF_TURNS_KEY = "length_cutoff_turns"
LENGTH_CUTOFF_IN_CALL_TURNS_KEY = "length_cutoff_in_call_turns"
EMPTY_TURNS_KEY = "empty_turns"


class TurnRecoveryMixin:
    # What a protocol feeds back after a cut turn, a turn cut inside a tool call (``None`` = the plain cut nudge)
    # and an empty turn, asking for its own next move; ``None`` = no recovery path for that kind of turn.
    LENGTH_CUTOFF_NUDGE: str | None = None
    LENGTH_CUTOFF_IN_CALL_NUDGE: str | None = None
    EMPTY_TURN_NUDGE: str | None = None

    def _init_turn_recovery(self, max_length_cutoff_recoveries: int | None) -> None:
        require_int(type(self).__name__, ge=0, **present(max_length_cutoff_recoveries=max_length_cutoff_recoveries))
        self.max_length_cutoff_recoveries = max_length_cutoff_recoveries

    def _recover_length_cutoff(
        self, trajectory: Trajectory, in_tool_call: bool = False, *, last_turn: bool = False
    ) -> StepOutcome:
        """Never graded, since the fragment would read as a natural termination. A turn cut ``in_tool_call`` is counted
        under :data:`LENGTH_CUTOFF_IN_CALL_TURNS_KEY` too and told so."""
        nudge = "LENGTH_CUTOFF_NUDGE"
        if in_tool_call:
            trajectory.info[LENGTH_CUTOFF_IN_CALL_TURNS_KEY] = (
                trajectory.info.get(LENGTH_CUTOFF_IN_CALL_TURNS_KEY, 0) + 1
            )
            if self.LENGTH_CUTOFF_IN_CALL_NUDGE is not None:
                nudge = "LENGTH_CUTOFF_IN_CALL_NUDGE"
        return self._recover_unproductive_turn(
            trajectory, "length_cutoff", LENGTH_CUTOFF_TURNS_KEY, nudge, last_turn=last_turn
        )

    def _recover_empty_turn(self, trajectory: Trajectory, *, last_turn: bool = False) -> StepOutcome:
        """Flagged so no tokenization path rewards it; graded, it would end the episode empty."""
        trajectory.last_assistant_message().empty = True
        return self._recover_unproductive_turn(
            trajectory, "empty", EMPTY_TURNS_KEY, "EMPTY_TURN_NUDGE", last_turn=last_turn
        )

    def _recover_unproductive_turn(
        self, trajectory: Trajectory, kind: str, counter_key: str, nudge_attr: str, *, last_turn: bool
    ) -> StepOutcome:
        """Retried within ``max_turns`` and ``max_length_cutoff_recoveries``; past them, or on the ``last_turn``, the
        episode ends truncated."""
        nudge = self._unproductive_turn_nudge(nudge_attr)
        if nudge is None:
            raise NotImplementedError(
                f"{type(self).__name__} routed an unproductive turn to recovery without declaring "
                f"{nudge_attr} — the episode would continue with no message telling the model what happened."
            )
        trajectory.info[counter_key] = trajectory.info.get(counter_key, 0) + 1
        cap = self.max_length_cutoff_recoveries
        past_cap = cap is not None and self._unproductive_turns(trajectory) > cap
        ended = past_cap or last_turn or trajectory.num_turns >= self.max_turns
        if not ended:
            trajectory.add_message(Message.user(nudge))
        info = {kind: True, "unrecovered_turn": True} if ended else {kind: True}
        return StepOutcome(trajectory=trajectory, reward=0.0, done=ended, truncated=ended, info=info)

    def _unproductive_turn_nudge(self, nudge_attr: str) -> str | None:
        """A protocol whose wording depends on the instance (a tool-less registry) overrides this."""
        return getattr(self, nudge_attr)

    @staticmethod
    def _unproductive_turns(trajectory: Trajectory) -> int:
        return trajectory.info.get(LENGTH_CUTOFF_TURNS_KEY, 0) + trajectory.info.get(EMPTY_TURNS_KEY, 0)
