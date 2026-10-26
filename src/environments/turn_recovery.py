"""Recovery from an unproductive turn: one the engine cut short, or one the model ended on nothing, nudged and
retried within the episode's turn and recovery budgets."""

from typing import Any

from src.environments.messages import Message, Trajectory


class TurnRecoveryMixin:
    """The nudge-and-retry path for an unproductive turn, counted the same way under every protocol; the
    wording is each protocol's."""

    # What a protocol feeds back after a cut turn, a turn cut inside a tool call (``None`` = the plain cut nudge)
    # and an empty turn, asking for its own next move; ``None`` = no recovery path for that kind of turn.
    LENGTH_CUTOFF_NUDGE: str | None = None
    LENGTH_CUTOFF_IN_CALL_NUDGE: str | None = None
    EMPTY_TURN_NUDGE: str | None = None

    def _init_turn_recovery(self, max_length_cutoff_recoveries: int | None) -> None:
        if max_length_cutoff_recoveries is not None and max_length_cutoff_recoveries < 0:
            raise ValueError(f"max_length_cutoff_recoveries must be >= 0 or None, got {max_length_cutoff_recoveries}")
        self.max_length_cutoff_recoveries = max_length_cutoff_recoveries

    def _handle_length_cutoff(
        self, trajectory: Trajectory, in_tool_call: bool = False, *, last_turn: bool = False
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Recover a turn the engine cut short (at its token cap, or by aborting it) before a finished call or
        answer: never graded, since the fragment would read as a natural termination. A turn cut ``in_tool_call``
        is counted under ``length_cutoff_in_call_turns`` too and told so (:data:`LENGTH_CUTOFF_IN_CALL_NUDGE`)."""
        nudge = "LENGTH_CUTOFF_NUDGE"
        if in_tool_call:
            trajectory.info["length_cutoff_in_call_turns"] = trajectory.info.get("length_cutoff_in_call_turns", 0) + 1
            if self.LENGTH_CUTOFF_IN_CALL_NUDGE is not None:
                nudge = "LENGTH_CUTOFF_IN_CALL_NUDGE"
        return self._recover_unproductive_turn(trajectory, "length_cutoff", nudge, last_turn=last_turn)

    def _handle_empty_turn(
        self, trajectory: Trajectory, *, last_turn: bool = False
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Recover a turn the model ended with neither visible content nor a tool call, flagged
        (:attr:`Message.empty`) so no tokenization path rewards it; graded, it would end the episode empty."""
        trajectory.last_assistant_message().empty = True
        return self._recover_unproductive_turn(trajectory, "empty", "EMPTY_TURN_NUDGE", last_turn=last_turn)

    def _recover_unproductive_turn(
        self, trajectory: Trajectory, kind: str, nudge_attr: str, *, last_turn: bool
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Nudge and retry a turn that produced nothing, counted under ``<kind>_turns``, within ``max_turns`` and
        ``max_length_cutoff_recoveries``; past them, or on the ``last_turn``, the episode ends truncated."""
        nudge = self._unproductive_turn_nudge(nudge_attr)
        if nudge is None:
            raise NotImplementedError(
                f"{type(self).__name__} routed an unproductive turn to recovery without declaring "
                f"{nudge_attr} — the episode would continue with no message telling the model what happened."
            )
        counter = f"{kind}_turns"
        trajectory.info[counter] = trajectory.info.get(counter, 0) + 1
        cap = self.max_length_cutoff_recoveries
        past_cap = cap is not None and self._unproductive_turns(trajectory) > cap
        if past_cap or last_turn or trajectory.num_turns >= self.max_turns:
            return trajectory, 0.0, True, True, {kind: True, "unrecovered_turn": True}
        trajectory.add_message(Message.user(nudge))
        return trajectory, 0.0, False, False, {kind: True}

    def _unproductive_turn_nudge(self, nudge_attr: str) -> str | None:
        """The text sent after an unproductive turn: the class attribute ``nudge_attr`` names. A protocol
        whose wording depends on the instance (a tool-less tool registry) overrides this."""
        return getattr(self, nudge_attr)

    @staticmethod
    def _unproductive_turns(trajectory: Trajectory) -> int:
        """Turns that produced nothing — cut by the engine or ended by the model on nothing."""
        return trajectory.info.get("length_cutoff_turns", 0) + trajectory.info.get("empty_turns", 0)
