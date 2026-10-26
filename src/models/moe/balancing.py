"""The ``moe_balancing`` modes and the order a split verdict converges in."""

from typing import get_args

from src.models.moe_balancing import BalancingMode

# Resolved-mode precedence for agreeing one mode across ranks, most informed first: a stage that found a
# bias-capable router has evidence a stage holding no MoE layer does not.
BALANCING_MODE_PRECEDENCE: tuple[BalancingMode, ...] = ("bias_update", "bias_update_transient", "aux_loss", "none")
assert set(BALANCING_MODE_PRECEDENCE) == set(get_args(BalancingMode)) - {"auto"}, "a balancing mode has no precedence"
