"""The context, step-context and ``info`` keys more than one environments module reads, and the solve verdict read
off an episode's metrics."""

from collections.abc import Mapping

# Context key (and dataset column) of an episode's expected answer; a ``requires_answer`` env grades against it.
ANSWER_KEY = "answer"
# ``info`` keys of the row context an episode was reset with, and of whether it ended on the model's own answer.
CONTEXT_KEY = "context"
COMPLETED_KEY = "completed"
# The episode's reward carries no learning signal, and why: the trainer leaves it out of the GRPO group baseline.
EPISODE_INVALID_KEY = "episode_invalid"
EPISODE_INVALID_REASON_KEY = "episode_invalid_reason"
# The cause on an episode its driver lost, stamped before the close so the turn-overflow price stays off it.
EPISODE_ERROR_KEY = "error"
# The served window refused the episode's next turn: it closed on what it earned, past which its whole render may run.
CONTEXT_OVERFLOW_KEY = "context_overflow"
# Whether an episode output budget no longer held a turn's answer room when the episode ended.
OUTPUT_BUDGET_EXHAUSTED_KEY = "output_budget_exhausted"
# Step-context flag on the last turn the output budget affords: an unproductive turn there cannot be retried.
LAST_TURN_KEY = "last_turn"
# The class of the sandbox fault that ended an episode: an infra one also invalidates it, an agent one is priced.
SANDBOX_FAULT_KEY = "sandbox_fault"
SANDBOX_FAULT_INFRA = "infra"
SANDBOX_FAULT_AGENT = "agent"
# Every term of the episode reward, ``reward/<name>`` → contribution, and the external scorers' accounts by term.
REWARD_COMPONENTS_KEY = "reward_components"
REWARD_DETAILS_KEY = "reward_details"
# The per-episode solve flag the group solve rate averages; a misspelling drops the metric rather than raising.
SOLVE_RATE_KEY = "outcome/solve_rate"
# Categorical facts an env stamps (``{"language": "cpp"}``); the trainer slices its rollout metrics by each.
EPISODE_SLICES_KEY = "slices"


def solve_verdict(metrics: Mapping[str, float]) -> bool | None:
    """``None`` where the environment reports no verdict."""
    verdict = metrics.get(SOLVE_RATE_KEY)
    return None if verdict is None else verdict >= 1.0
