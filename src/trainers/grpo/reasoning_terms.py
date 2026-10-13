"""The reasoning length terms the environmental GRPO trainer charges: a per-level price on an episode's
reasoning tokens (all of them, or what its turns reasoned past a soft budget) and an under-use floor against
the per-turn thinking budget it ran under, both read off what the rollout recorded on the trajectory."""

from src.environments.base import Trajectory

# The share of the per-turn thinking budget the floor asks an episode for, summed over its turns. Under 1, so
# one turn can clear the floor without running into an enforced budget.
REASONING_TARGET_SHARE = 0.75


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


def reasoning_price_term(
    reasoning_tokens: list[int], price_per_1k: float, cap: float, budget: int | None = None
) -> float:
    """The capped reasoning price in ``[-cap, 0]``: ``-min(cap, price_per_1k * priced / 1000)``, where ``priced``
    sums every turn's reasoning tokens, or with a per-turn ``budget`` only what each turn reasoned past it.

    Priced per effort level by the caller, so the same trace costs most where little reasoning was
    asked; the cap keeps a long trace from outweighing the task reward, which an uncapped per-token
    price does. Past a budget it is a soft budget's ceiling: reasoning within the budget is free, so the
    sibling that reasons less wins nothing unless the other ran past it, and per turn, so one long turn is
    not excused by short ones around it. Prices reasoning tokens only."""
    priced = sum(reasoning_tokens) if budget is None else sum(max(0, tokens - budget) for tokens in reasoning_tokens)
    if priced <= 0:
        return 0.0
    return -min(cap, price_per_1k * priced / 1000)


def reasoning_floor_term(reasoning_tokens: list[int], budget: int, weight: float) -> float:
    """Under-use floor in ``[-weight, 0]``: ``-weight * shortfall / target`` while the episode's
    reasoning tokens fall short of its target, :data:`REASONING_TARGET_SHARE` of the per-turn thinking
    ``budget`` it ran under, enforced or soft.

    The price only ever pays for less reasoning; this is the term that resists reasoning shrinking
    toward nothing. Summed over the episode, never averaged per turn: a short repair turn after a
    verdict is not under-use, and an extra tool turn cannot lower the score. An episode with no
    assistant turn is a lost one, not under-use, and pays nothing; turns that carry no reasoning at
    all pay the whole weight."""
    target = round(budget * REASONING_TARGET_SHARE)
    shortfall = target - sum(reasoning_tokens)
    if not reasoning_tokens or target <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / target
