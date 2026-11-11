#!/usr/bin/env python
"""FLA's q/k L2 norm under ``HALO_FLA_L2NORM_FIXED_NB``: the stock kernels compile and tune again for each new row-count
bucket; with the constant-NB launch one compiled kernel and one tuned config serve every row count, the values stay those
of the stock launch to fp32 rounding, and a gated-delta-rule call is unchanged."""

import os
from unittest import mock

import torch
from fla.modules import l2norm as fla_l2norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from halo.kernels import fla_launch
from halo.kernels.fla_launch import apply_fla_launch_options, triton_autotuner
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err

KNOB = "HALO_FLA_L2NORM_FIXED_NB"
WIDTH = 128
# FLA's bucket is ``cdiv(rows, 65536)``: rows on both sides of its edges.
ROWS = (1, 65536, 65537, 131073)
BUCKETS = (65536, 131072, 196608)
# Buckets no earlier launch of the stock kernels used.
FRESH_BUCKETS = (262144, 327680)
TOLERANCE = {torch.float32: 1e-6, torch.bfloat16: 4e-3}
GDN_TOLERANCE = 5e-3


def _tuners():
    return [triton_autotuner(getattr(fla_l2norm, name)) for name in ("l2norm_fwd_kernel", "l2norm_bwd_kernel")]


class _GatedDeltaRule(torch.nn.Module):
    """The one attribute and device the warm-up reads off a gated-delta-rule layer."""

    def __init__(self):
        super().__init__()
        self.head_k_dim = WIDTH
        self.weight = torch.nn.Parameter(torch.zeros(1, device="cuda"))


def _binaries() -> int:
    device = torch.cuda.current_device()
    return sum(len(tuner.fn.device_caches[device][0]) for tuner in _tuners())


def _run(rows: int, dtype: torch.dtype):
    x = torch.randn(
        rows, WIDTH, device="cuda", dtype=dtype, generator=torch.Generator(device="cuda").manual_seed(rows)
    )
    y, rstd = fla_l2norm.l2norm_fwd(x)
    return y, fla_l2norm.l2norm_bwd(y, rstd, x)


def _gated_delta_rule(dtype: torch.dtype):
    generator = torch.Generator(device="cuda").manual_seed(5)
    q, k, v = (torch.randn(1, 777, 2, WIDTH, device="cuda", dtype=dtype, generator=generator) for _ in range(3))
    g = -torch.rand(1, 777, 2, device="cuda", dtype=torch.float32, generator=generator)
    beta = torch.rand(1, 777, 2, device="cuda", dtype=dtype, generator=generator)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    out, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    out.float().pow(2).sum().backward()
    return out.detach(), q.grad, k.grad


def run_stock() -> dict:
    results = {(rows, dtype): _run(rows, dtype) for dtype in TOLERANCE for rows in ROWS}
    results["gdn"] = _gated_delta_rule(torch.bfloat16)
    return results


def check_stock_compiles_per_bucket() -> None:
    """Non-vacuity: without the install each new bucket costs a binary and a tuned config, which the fix removes."""
    before, tuned = _binaries(), sum(len(tuner.cache) for tuner in _tuners())
    for rows in FRESH_BUCKETS:
        _run(rows, torch.bfloat16)
    assert _binaries() > before and sum(len(tuner.cache) for tuner in _tuners()) > tuned


def check_one_binary_and_one_pick_for_every_bucket() -> None:
    """From empty caches, so the warm-up must itself compile the kernels the launches below then reuse."""
    for tuner in _tuners():
        tuner.fn.device_caches.clear()
        tuner.cache.clear()
    assert _binaries() == 0
    fla_launch.warm_fla_l2norm(torch.nn.ModuleList([_GatedDeltaRule()]))
    binaries, picks = _binaries(), [len(tuner.cache) for tuner in _tuners()]
    assert binaries > 0 and all(picks), "the warm-up compiled or tuned nothing"
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (*ROWS, *BUCKETS):
            _run(rows, dtype)
    assert _binaries() == binaries, f"{_binaries() - binaries} binaries compiled after the warm-up"
    assert [len(tuner.cache) for tuner in _tuners()] == picks, "a row count tuned again"
    assert all("NB" not in tuner.keys for tuner in _tuners())


def check_values_match_the_stock_launch(stock: dict) -> None:
    for rows in ROWS:
        for dtype, tolerance in TOLERANCE.items():
            y, dx = _run(rows, dtype)
            ref_y, ref_dx = stock[(rows, dtype)]
            for got, want in ((y, ref_y), (dx, ref_dx)):
                assert fro_rel_err(got, want) <= tolerance, (rows, dtype, fro_rel_err(got, want))
    for got, want in zip(_gated_delta_rule(torch.bfloat16), stock["gdn"], strict=True):
        assert fro_rel_err(got, want) <= GDN_TOLERANCE


def check_a_wrong_launch_fails_the_comparison(stock: dict) -> None:
    """A launcher that perturbs ``eps`` in the real launch path must break the comparison the other checks rely on."""

    def wrong_eps(self, *args, **kwargs):
        return self.fn.run(*args, **{**kwargs, "eps": 100.0})

    with mock.patch.object(fla_launch._ConstantNB, "run", wrong_eps):
        y, dx = _run(65537, torch.float32)
    ref_y, ref_dx = stock[(65537, torch.float32)]
    assert fro_rel_err(y, ref_y) > 10 * TOLERANCE[torch.float32]
    assert fro_rel_err(dx, ref_dx) > 10 * TOLERANCE[torch.float32]


@gpu_test_main(exact_world_size=1, prefix="fla_l2norm_constant_nb", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    os.environ.pop(KNOB, None)
    apply_fla_launch_options()
    stock = run_stock()
    record_check(checks, "stock_compiles_per_bucket", check_stock_compiles_per_bucket)
    os.environ[KNOB] = "1"
    apply_fla_launch_options()
    record_check(checks, "one_binary_and_one_pick", check_one_binary_and_one_pick_for_every_bucket)
    record_check(checks, "values_match_stock", lambda: check_values_match_the_stock_launch(stock))
    record_check(checks, "negative_control", lambda: check_a_wrong_launch_fails_the_comparison(stock))
    return {"checks": checks}


if __name__ == "__main__":
    run()
