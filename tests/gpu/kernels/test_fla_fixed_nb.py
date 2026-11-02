#!/usr/bin/env python
"""``HALO_FLA_FIXED_NB``: FLA's l2norm and gated-norm kernels skip the per-``NB`` recompiles and stay bit-identical.

Both kernels take ``NB = cdiv(rows, 65536)`` as an unread autotune-key constexpr, rows being tokens times heads. One
process runs the q/k l2norm and the gated output norm forward and backward at Qwen3.5/3.6-35B-A3B's linear-attention
width over token counts that span four ``NB`` values, first with the flag (applied through ``finalize_loaded_model``)
and then with the stock launch. The stock arm reuses the flag arm's autotune picks, which are keyed on the real ``NB``
in both, so the two arms launch the same configs and every output and gradient must match bit for bit.

With the flag each kernel's compiles stay within twice its autotune configs whatever the token counts; the stock arm,
the control, must compile again for every new ``NB`` (otherwise the token counts never left one ``NB`` and the first
check would pass vacuously).

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_fla_fixed_nb.py
"""

import os
from collections import Counter

import torch
from fla.modules import FusedRMSNormGated
from fla.modules.l2norm import l2norm
from triton import knobs

from halo.kernels.fla_launch import FIXED_NB_FLAG, FIXED_NB_KERNELS, triton_autotuner
from halo.model_loading.finalization import finalize_loaded_model
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

SEED = 1234
# Qwen3.5/3.6-35B-A3B's GatedDeltaNet after the q/k repeat: 32 heads of width 128.
HEADS, HEAD_DIM = 32, 128
# Rows = tokens x 32 heads, so these launch with NB = 1, 3, 5 and 7.
TOKEN_COUNTS = (2048, 6144, 10240, 14336)


def _kernels():
    """``(qualname, Autotuner)`` for every table kernel."""
    for module, kernel_names in FIXED_NB_KERNELS.items():
        for kernel_name in kernel_names:
            yield f"{module.__name__}.{kernel_name}", triton_autotuner(getattr(module, kernel_name))


def _inputs(device) -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    rows = []
    for tokens in TOKEN_COUNTS:
        shape = (1, tokens, HEADS, HEAD_DIM)
        rows.append({name: torch.randn(*shape, generator=gen).to(torch.bfloat16).to(device) for name in "qgxo"})
    return rows


def _run(row: dict[str, torch.Tensor], norm: FusedRMSNormGated) -> dict[str, torch.Tensor]:
    """Both kernels' outputs and gradients for one row."""
    q, x, g = (row[name].detach().clone().requires_grad_() for name in "qxg")
    norm.weight.grad = None
    q_normed = l2norm(q)
    gated = norm(x, g)
    (q_normed.float() * row["o"]).sum().backward()
    (gated.float() * row["o"]).sum().backward()
    torch.cuda.synchronize()
    return {
        "l2norm": q_normed,
        "dq": q.grad,
        "gated": gated,
        "dx": x.grad,
        "dg": g.grad,
        "dw": norm.weight.grad.clone(),
    }


def _arm(rows, norm) -> tuple[list[dict[str, torch.Tensor]], Counter]:
    """Every row's results, and the compiles each table kernel ran meanwhile."""
    names = {qualname.rsplit(".", 1)[1] for qualname, _ in _kernels()}
    compiles: Counter = Counter()

    def count(*, fn, **_):
        if fn.name.rsplit(".", 1)[-1] in names:
            compiles[fn.name.rsplit(".", 1)[-1]] += 1

    knobs.runtime.jit_post_compile_hook = count
    try:
        results = [_run(row, norm) for row in rows]
    finally:
        knobs.runtime.jit_post_compile_hook = None
    return results, compiles


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="test_fla_fixed_nb")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    # An empty cache, as on a node's first run: compiles are real. Triton reads it at first launch.
    os.environ["TRITON_CACHE_DIR"] = os.path.join(ctx.cache_dir, "triton")
    torch.cuda.set_device(ctx.device)
    rows = _inputs(ctx.device)
    norm = FusedRMSNormGated(HEAD_DIM, eps=1e-6, device=ctx.device, dtype=torch.bfloat16)
    stock_fns = {qualname: tuner.fn for qualname, tuner in _kernels()}

    os.environ[FIXED_NB_FLAG] = "1"
    finalize_loaded_model(torch.nn.Module())
    fixed, fixed_compiles = _arm(rows, norm)
    for qualname, tuner in _kernels():
        tuner.fn = stock_fns[qualname]
    del os.environ[FIXED_NB_FLAG]
    stock, stock_compiles = _arm(rows, norm)
    config_counts = {qualname.rsplit(".", 1)[1]: len(tuner.configs) for qualname, tuner in _kernels()}
    log(f"  compiles with {FIXED_NB_FLAG}: {dict(fixed_compiles)} (configs {config_counts})")
    log(f"  compiles of the stock launch after it: {dict(stock_compiles)}")

    def bit_identical():
        for tokens, a, b in zip(TOKEN_COUNTS, fixed, stock, strict=True):
            for name in a:
                assert torch.equal(a[name], b[name]), f"{name} at {tokens} tokens differs between the arms"

    def skips_per_nb_recompiles():
        # Twice, not once: an integer argument (the backward's BS) still specializes on divisibility by 16, and
        # BS = cdiv(rows, SM count) can cross it between token counts. Per-NB recompiles would be configs x 4.
        for name, configs in config_counts.items():
            assert 0 < fixed_compiles[name] <= 2 * configs, (
                f"{name}: {fixed_compiles[name]} compiles for {configs} configs"
            )

    def stock_recompiles_per_nb():
        for name in config_counts:
            assert stock_compiles[name] >= len(TOKEN_COUNTS) - 1, (
                f"{name}: the stock launch compiled {stock_compiles[name]} new kernels over {len(TOKEN_COUNTS)} NB "
                f"values; the token counts no longer span several NB"
            )

    record_check(checks, "outputs_and_grads_bit_identical", bit_identical)
    record_check(checks, "fixed_nb_skips_per_nb_recompiles", skips_per_nb_recompiles)
    record_check(checks, "stock_control_recompiles_per_nb", stock_recompiles_per_nb)
    return {
        "checks": checks,
        "metrics": {"fixed_compiles": sum(fixed_compiles.values()), "stock_compiles": sum(stock_compiles.values())},
    }


if __name__ == "__main__":
    run()
