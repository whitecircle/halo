"""MoE config spellings and the routing shape a run measures and balances."""

from typing import NamedTuple

from src.models.moe_balancing import detect_moe_experts_topk, has_discard_expert_slot


class MoERouting(NamedTuple):
    """How a model routes: between more than one expert, ``top_k`` picks per token, and whether the
    trailing slot is a discard every load statistic leaves out."""

    routes_between_experts: bool
    top_k: int
    exclude_last_slot: bool


def resolve_moe_routing(model) -> MoERouting:
    """``model``'s routing off its config and router classes; ``MoERouting(False, 1, False)`` when it
    declares at most one expert."""
    num_experts, top_k = detect_moe_experts_topk(model)
    if num_experts <= 1:
        return MoERouting(False, 1, False)
    return MoERouting(True, top_k or 1, has_discard_expert_slot(model))
