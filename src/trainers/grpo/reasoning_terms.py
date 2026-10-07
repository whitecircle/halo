"""The reasoning length terms the environmental GRPO trainer charges: a per-level price on an episode's
reasoning tokens, an under-use floor against the per-turn thinking cap it ran under, and a charge on a
turn that runs into a cap — its reasoning into its thinking cap, or the whole turn into the turn cap.
Each reads what the rollout recorded on the trajectory."""

from src.environments.base import Trajectory

# The share of a cap the policy is expected to use up to: the floor asks an episode for it of the thinking
# cap, summed over its turns, and the overlong charge ramps from it to the cap, thinking cap and turn cap
# alike. Under 1, so one turn can clear the floor without running into the cap the engine enforces.
REASONING_TARGET_SHARE = 0.75


def reasoning_target(cap: int) -> int:
    """What a cap asks for: :data:`REASONING_TARGET_SHARE` of it, the floor's reference and the overlong
    ramps' start alike."""
    return round(cap * REASONING_TARGET_SHARE)


def _ramp(count: int | None, cap: int | None, penalty: float) -> float:
    """``-penalty`` times how far ``count`` sits into the last quarter under ``cap``: nothing until the
    count passes the target, the whole penalty at the cap; 0 without a count or a cap."""
    if count is None or cap is None:
        return 0.0
    # A cap too small to hold a target below it still ramps over its last token.
    start = min(reasoning_target(cap), cap - 1)
    return -penalty * min(max((count - start) / (cap - start), 0.0), 1.0)


def reasoning_token_counts(tokenizer, trajectory: Trajectory | None) -> list[int]:
    """Per-assistant-turn reasoning token counts under ``tokenizer``, what the reasoning terms price and
    the per-effort metrics average. Every assistant turn counts, a thinking-free one as 0, so the sum is
    the episode's reasoning and the length its turn count; an episode without a trajectory has none."""
    if trajectory is None:
        return []
    return [
        len(tokenizer(m.thinking, add_special_tokens=False)["input_ids"]) if m.thinking else 0
        for m in trajectory.messages
        if m.role == "assistant"
    ]


def reasoning_price_term(reasoning_tokens: list[int], price_per_1k: float, cap: float) -> float:
    """The capped reasoning price in ``[-cap, 0]``: ``-min(cap, price_per_1k * sum(reasoning_tokens) / 1000)``.

    Priced per effort level by the caller, so the same trace costs most where little reasoning was
    asked; the cap keeps a long trace from outweighing the task reward, which an uncapped per-token
    price does. Prices reasoning tokens only, summed over the trajectory's turns."""
    tokens = sum(reasoning_tokens)
    if tokens <= 0:
        return 0.0
    return -min(cap, price_per_1k * tokens / 1000)


def reasoning_floor_term(reasoning_tokens: list[int], thinking_cap: int, weight: float) -> float:
    """Under-use floor in ``[-weight, 0]``: ``-weight * shortfall / target`` while the episode's
    reasoning tokens fall short of its target, :data:`REASONING_TARGET_SHARE` of the per-turn
    ``thinking_cap`` it ran under.

    The price only ever pays for less reasoning; this is the term that resists reasoning shrinking
    toward nothing. Summed over the episode, never averaged per turn: a short repair turn after a
    verdict is not under-use, and an extra tool turn cannot lower the score. An episode with no
    assistant turn is a lost one, not under-use, and pays nothing; turns that carry no reasoning at
    all pay the whole weight."""
    target = reasoning_target(thinking_cap)
    shortfall = target - sum(reasoning_tokens)
    if not reasoning_tokens or target <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / target


def turn_overlong_term(
    trajectory: Trajectory | None, *, penalty: float, turn_cap: int | None = None
) -> tuple[float, int, int]:
    """The episode's overlong term in ``[-penalty, 0]``, the assistant turns it charged, and its assistant turns.

    A turn pays the larger of two ramps, each ``-penalty * clamp((count - start) / (cap - start), 0, 1)``
    with ``start`` at :data:`REASONING_TARGET_SHARE` of the cap: the reasoning it sampled against the
    thinking cap its level set, both as the rollout recorded them, and every token it sampled against
    ``turn_cap`` (the run's ``rollout_max_tokens``, the wall the engine cuts a turn at). Nothing until the
    count passes the target, the whole penalty at the cap, where the engine forces the close or the cut
    and the turn otherwise pays nothing for running into it — reasoning carried past the close into the
    call included. A turn whose reasoning was not counted, or that ran uncapped, pays no reasoning ramp;
    one without sampled ids, or under no ``turn_cap``, no turn ramp. The episode pays its most-charged
    turn, once: the pressure is against running a turn into a cap at all, and a sum would grow with the
    turns a run allows, leaving the term no bound below the prices the environment sets. An episode
    without a trajectory pays nothing.
    """
    charges = []
    for message in trajectory.messages if trajectory is not None else ():
        if message.role != "assistant":
            continue
        charge = _ramp(message.reasoning_tokens, message.thinking_cap, penalty)
        if turn_cap is not None and message.token_ids is not None:
            charge = min(charge, _ramp(len(message.token_ids), turn_cap, penalty))
        charges.append(charge)
    return min(charges, default=0.0), sum(charge < 0 for charge in charges), len(charges)
