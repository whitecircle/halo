"""The reasoning length terms the environmental GRPO trainer charges: a per-level price on an episode's
reasoning tokens and an under-use floor against the per-turn thinking cap it ran under, both read off what
the rollout recorded on the trajectory."""

from src.environments.base import Trajectory

# The share of the per-turn thinking cap the floor asks an episode for, summed over its turns. Under 1, so
# one turn can clear the floor without running into the cap the engine enforces.
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
    target = round(thinking_cap * REASONING_TARGET_SHARE)
    shortfall = target - sum(reasoning_tokens)
    if not reasoning_tokens or target <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / target
