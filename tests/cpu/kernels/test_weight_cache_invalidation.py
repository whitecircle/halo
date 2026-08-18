"""The low-precision weight cache must invalidate after a real optimizer step.

``cached_fake_quant`` keys its cache on ``weight._version``. An optimizer that writes through
``p.data`` gets an alias carrying its own version counter, so the parameter's ``_version`` never
advances and the cache serves the step-0 quantized weight for the entire run — the forward freezes
while the master weights keep moving, silently invalidating every fp8/fp4 QAT result.

The existing weight-cache test bumps ``_version`` by hand, which cannot catch that; these tests drive
a real optimizer step instead.
"""

import pytest
import torch
from torch import nn

from src.kernels.lowp.quantization import cached_fake_quant
from src.optimizers.adamw_bf16 import AdamWBF16

BLOCK_AXIS = 1


def _step(optimizer, param, scale: float = 1.0):
    param.grad = torch.full_like(param, 0.1) * scale
    optimizer.step()


@pytest.mark.parametrize("optimizer_name", ["adamw_bf16", "torch_adamw"])
def test_optimizer_step_advances_the_parameter_version(optimizer_name):
    """Every optimizer the toolkit ships must bump ``_version`` on the parameter it updates."""
    param = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))
    if optimizer_name == "adamw_bf16":
        optimizer = AdamWBF16([param], lr=1e-2, weight_decay=0.0, use_triton=False)
    else:
        optimizer = torch.optim.AdamW([param], lr=1e-2, weight_decay=0.0, foreach=False)

    before = param._version
    _step(optimizer, param)
    after = param._version

    assert after > before, (
        f"{optimizer_name} left the parameter's _version at {after} after a step that provably "
        "changed its values. Writes must go through p.detach() (which shares the version counter), "
        "not p.data (which does not) — otherwise the low-precision weight cache never invalidates."
    )


def test_cached_fake_quant_tracks_the_weight_across_steps():
    """The cached quantized weight must follow the master weight, not freeze at step 0."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))
    optimizer = AdamWBF16([param], lr=1e-1, weight_decay=0.0, use_triton=False)

    first = cached_fake_quant(param, "mxfp8", BLOCK_AXIS).clone()
    master_start = param.detach().clone()

    deltas = []
    for _ in range(4):
        _step(optimizer, param)
        deltas.append((cached_fake_quant(param, "mxfp8", BLOCK_AXIS) - first).abs().max().item())

    master_moved = (param.detach() - master_start).abs().max().item()
    assert master_moved > 0, "the optimizer did not move the master weight; test premise broken"
    assert deltas[-1] > 0, (
        f"the cached mxfp8 weight is byte-identical to step 0 after 4 optimizer steps that moved the "
        f"master weight by {master_moved:.3e}. The cache is stale, so the low-precision forward is "
        "frozen at the initial weights while training continues."
    )
    assert deltas == sorted(deltas), (
        f"cached-weight drift {deltas} is not monotonically increasing; the cache is invalidating only intermittently."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
