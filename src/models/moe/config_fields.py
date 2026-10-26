"""MoE config-field readers: each concept's spellings across the roster, read over a composite config's levels.

One registry per concept, so the "is this MoE?" gate, the load metrics, the EP layers and the pipeline split's cost
model cannot classify one checkpoint differently.
"""

from __future__ import annotations

from typing import NamedTuple

from src.models.config_access import get_config_field
from src.models.moe.router_logits import has_discard_expert_slot

# First non-zero wins, so the unambiguous fields precede bare ``top_k``, which is also the generation sampling
# parameter.
ROUTER_TOPK_FIELDS: tuple[str, ...] = (
    "num_experts_per_tok",
    "top_k_experts",
    "moe_router_topk",
    "num_active_experts",
    "top_k",
)
_ROUTER_EXPERT_COUNT_FIELDS = (
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
    "num_moe_experts",
    "moe_num_experts",
)
# The per-expert FFN width expert-TP shards. The MoE-specific spellings come first: a hybrid dense+MoE config
# carries both, and only the MoE one is sharded.
EXPERT_FFN_WIDTH_FIELDS: tuple[str, ...] = (
    "moe_intermediate_size",
    "expert_intermediate_size",
    "moe_ffn_hidden_size",
    "intermediate_size",  # GptOss: the expert FFN is the only FFN, so this is the expert width
)


def _first_router_field(cfg, fields: tuple[str, ...], default=None):
    """First set value among ``fields``, searched over every config level."""
    for field in fields:
        if value := get_config_field(cfg, field):
            return value
    return default


def resolve_router_topk(cfg) -> int:
    """Experts routed per token, ``0`` when no spelling is set."""
    return int(_first_router_field(cfg, ROUTER_TOPK_FIELDS, 0))


def resolve_expert_count(cfg) -> int:
    """Routed experts per MoE layer, ``0`` when no spelling is set (a dense model)."""
    return int(_first_router_field(cfg, _ROUTER_EXPERT_COUNT_FIELDS, 0))


def require_expert_count(cfg) -> int:
    """:func:`resolve_expert_count` for a caller that cannot proceed without one."""
    if num_experts := resolve_expert_count(cfg):
        return num_experts
    raise ValueError(
        f"Could not determine the routed-expert count from {type(cfg).__name__}: none of "
        f"{_ROUTER_EXPERT_COUNT_FIELDS} is set on it or on its text config."
    )


def resolve_expert_ffn_width(cfg) -> int | None:
    """The per-expert intermediate dimension expert-TP splits, ``None`` when no spelling is set."""
    value = _first_router_field(cfg, EXPERT_FFN_WIDTH_FIELDS)
    return int(value) if value else None


def require_router_aux_loss_coef(cfg) -> float:
    """The ``router_aux_loss_coef`` a model that returned an ``aux_loss`` weighs it by, read where its head reads
    it (the text config of a composite). Raises when undeclared: a stand-in weight would train a different
    objective than the model's own forward."""
    coef = get_config_field(cfg, "router_aux_loss_coef")
    if coef is None:
        raise ValueError(
            f"The model returned an `aux_loss` but its config ({type(cfg).__name__}) declares no "
            f"`router_aux_loss_coef`, so the trainer cannot weigh the router balancing term. Set "
            f"router_aux_loss_coef on the model config (model_init_kwargs), or disable the aux loss "
            f"(moe_balancing) for this family."
        )
    return float(coef)


def has_routed_experts(cfg) -> bool:
    """Whether the config describes a MoE model, at either level of a composite config."""
    return resolve_expert_count(cfg) > 0


class MoERouting(NamedTuple):
    """How a model routes: between more than one expert, ``top_k`` picks per token, and whether the
    trailing slot is a discard every load statistic leaves out."""

    routes_between_experts: bool
    top_k: int
    exclude_last_slot: bool


def resolve_moe_routing(model) -> MoERouting:
    """``model``'s routing off its config and router classes; ``MoERouting(False, 1, False)`` when it
    declares at most one expert."""
    config = getattr(model, "config", None)
    if resolve_expert_count(config) <= 1:
        return MoERouting(False, 1, False)
    return MoERouting(True, resolve_router_topk(config) or 1, has_discard_expert_slot(model))
