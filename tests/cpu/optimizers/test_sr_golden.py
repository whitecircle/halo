#!/usr/bin/env python
"""Bit-exact golden of the stochastic-rounding seeds and the eager AdamWBF16 step;
``tests/gpu/optimizers/test_adamw_bf16_fused.py`` holds the fused launch to it."""

import pytest
import torch
from torch import nn

from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.optimizers.stochastic_rounding import sr_seed_pair, stochastic_round_to_bf16

# bf16 bit patterns (int16 view) after three eager steps: (param, exp_avg, exp_avg_sq).
GOLDEN_STATE = {
    "weight": (
        [-16528, -16554, -16590, -16622, -16678, -16747, -16863, -16964]
        + [15691, 15927, 16036, 16103, 16150, 16167, 16203, 16236],
        [0, 15064, -18068, 14736, -18040, -17887, -17729, 15124]
        + [-17987, 14882, -17890, 14657, -17645, 15107, 14606, 14846],
        [0, 13351, 13342, 13030, 13381, 13360, 13256, 13410]
        + [13386, 13361, 13411, 13399, 13408, 13365, 13387, 13411],
    ),
    "bias": (
        [-16634, -16711, -16826, -17043, 15796, 15964, 16039, 16116],
        [14734, -18237, 13954, -18254, 14233, -18169, 14694, 14235],
        [12621, 12487, 12497, 12604, 12447, 12475, 12558, 12405],
    ),
}
GOLDEN_SR = [-16320, -16375, -16475, -16678, 16092, 16293, 16393, 16448]
GOLDEN_SEEDS = {
    (0xB165EED, 1, 0): (471263526, 590835204),
    (0xB165EED, 7, 3): (1048164521, 459064998),
    (0xC0FFEE, 2, 2**63 + 5): (466973916, 1048173815),
}


def _bits(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().contiguous().view(torch.int16).flatten().tolist()


@pytest.mark.parametrize(("key", "step", "index"), list(GOLDEN_SEEDS))
def test_sr_seed_pair_matches_golden(key, step, index):
    assert sr_seed_pair(key, step, index) == GOLDEN_SEEDS[(key, step, index)]


def test_stochastic_round_to_bf16_matches_golden():
    x = torch.linspace(-3.0, 3.0, 8) + 1e-3
    assert _bits(stochastic_round_to_bf16(x, seed=12345)) == GOLDEN_SR


def test_eager_adamw_bf16_steps_match_golden():
    weight = nn.Parameter(((torch.arange(16, dtype=torch.float32) - 7.5) / 8).reshape(2, 8).to(torch.bfloat16))
    bias = nn.Parameter(torch.linspace(-0.5, 0.5, 8).to(torch.bfloat16))
    optimizer = AdamWBF16(
        [("layer.weight", weight), ("layer.bias", bias)], lr=1e-2, weight_decay=0.1, use_triton=False
    )
    for step in range(1, 4):
        weight.grad = (torch.sin(torch.arange(16, dtype=torch.float32) * step) * 1e-2).reshape(2, 8).to(torch.bfloat16)
        bias.grad = (torch.cos(torch.arange(8, dtype=torch.float32) * step) * 1e-3).to(torch.bfloat16)
        optimizer.step()

    for name, param in (("weight", weight), ("bias", bias)):
        state = optimizer.state[param]
        assert (_bits(param), _bits(state["exp_avg"]), _bits(state["exp_avg_sq"])) == GOLDEN_STATE[name], name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
