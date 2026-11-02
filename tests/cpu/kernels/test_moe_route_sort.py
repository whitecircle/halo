#!/usr/bin/env python
"""The counting sort behind the EP grouped-GEMM routing (``halo/kernels/moe_route_sort.py``) against the torch
construction it replaces at ``ep_size > 1``: every output and the gate-weight gradient bit for bit, over the roster's EP
shapes (top_k 1 to 10, 1 to 128 local experts, an idle expert, rows with no local slot, a receive with none at all, a
zero-token rank), at the launch's tiling and at one that spreads each receive over many blocks and scan chunks; and
every launch's Triton specialization over a sweep of row, block and slot counts against the layer warm-up's, so the
binaries the warm-up compiles serve every dispatch.

The kernels run under Triton's interpreter in a fresh process (``TRITON_INTERPRET`` is read when a kernel is defined).
The CUDA leg is ``tests/gpu/parallelism/ep/test_ep_sort_sync_free.py``, and
``tests/gpu/parallelism/ep/test_ep_receive_counts.py`` runs it on DeepEP's own counts. The compile check builds the
three kernels for sm_90, sm_100 and sm_103 at every case's shape, without a GPU.

Run: python tests/cpu/kernels/test_moe_route_sort.py
"""

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from halo.kernels import moe_route_sort
from halo.kernels.moe_route_sort import sort_routing_by_expert
from tests.common.repo import probe_findings
from tests.common.route_sort_reference import CASES

_MARKER = "ROUTE_SORT_MISMATCHES:"
# The launch's tiling, and one of 64-slot blocks and one-block scan chunks: every receive then spans several blocks
# and the scan carries its running offsets across chunks.
_TILINGS = {"launch": "", "many_blocks": "moe_route_sort._BLOCK_SLOTS = 64\nmoe_route_sort._SCAN_TILE = 1\n"}
_PROBE = """
import os
os.environ["TRITON_INTERPRET"] = "1"
from halo.kernels import moe_route_sort
{setup}from tests.common.route_sort_reference import {check}
print({marker!r} + "|".join({check}({args})))
"""
_TARGETS = tuple(GPUTarget("cuda", arch, 32) for arch in (90, 100, 103))
# Every reference case's (top_k, local experts), the shapes the GPU leg runs on Blackwell; sm_90 has only this check.
_SHAPES = sorted({(case.top_k, case.local_experts) for case in CASES})


@pytest.mark.parametrize("tiling", _TILINGS)
def test_the_counting_sort_matches_the_torch_construction(tiling):
    probe = _PROBE.format(setup=_TILINGS[tiling], check="counting_sort_mismatches", marker=_MARKER, args='"cpu"')
    assert probe_findings(probe, _MARKER) == []


def test_the_comparison_and_the_count_check_hold_their_controls():
    probe = _PROBE.format(setup="", check="control_findings", marker=_MARKER, args="")
    assert probe_findings(probe, _MARKER) == []


def test_every_dispatch_size_launches_the_binaries_the_warm_up_compiled():
    """A launch whose specialization the layer warm-up did not compile builds its binary between a dispatch and its
    combine, stalling every peer in DeepEP's barrier. Over the roster's EP layouts, expert TP included, every row,
    block and slot count launches the warm-up's key. The 64-slot tiling keeps the sixteen-block receives cheap to
    interpret and moves only the tile constants, which every launch of a layout shares."""
    probe = _PROBE.format(setup=_TILINGS["many_blocks"], check="launch_key_findings", marker=_MARKER, args="")
    assert probe_findings(probe, _MARKER) == []


def test_the_launch_key_check_fails_its_negative_control():
    """The placement kernel rebuilt with every integer argument specialized, at gpt-oss's layout under expert TP: its
    row and slot counts class apart from the warm-up's (1, a multiple of 16, neither), and the check names them."""
    rebuild = "import triton\nmoe_route_sort._place_kernel = triton.jit(moe_route_sort._place_kernel.fn)\n"
    probe = _PROBE.format(
        setup=_TILINGS["many_blocks"] + rebuild, check="launch_key_findings", marker=_MARKER, args="((4, 8),)"
    )
    findings = probe_findings(probe, _MARKER)
    assert findings and all(finding.startswith("_place_kernel") for finding in findings), findings
    assert any("n_rows" in finding for finding in findings) and any("n_valid" in finding for finding in findings)


def test_mis_sized_counts_raise():
    with pytest.raises(ValueError, match="2 receive counts for 4 local experts"):
        sort_routing_by_expert(torch.zeros(2, 2, dtype=torch.long), torch.rand(2, 2), 4, (3, 1))


@pytest.mark.parametrize("target", _TARGETS, ids=lambda target: f"sm_{target.arch}")
@pytest.mark.parametrize(("top_k", "local_experts"), _SHAPES)
def test_the_kernels_compile_for_every_target(target, top_k, local_experts):
    k_pad, e_pad = triton.next_power_of_2(top_k), triton.next_power_of_2(local_experts)
    block = {"TOP_K": top_k, "K_PAD": k_pad, "ROWS": moe_route_sort._BLOCK_SLOTS // k_pad, "E_PAD": e_pad}
    kernels = (
        (moe_route_sort._histogram_kernel, block),
        (moe_route_sort._scan_kernel, {"E_PAD": e_pad, "CHUNK": max(1, moe_route_sort._SCAN_TILE // e_pad)}),
        (moe_route_sort._place_kernel, {**block, "LANE_BITS": moe_route_sort._LANE_BITS}),
    )
    pointer_types = {"experts_ptr": "*i64", "weights_ptr": "*fp32", "sorted_token_ptr": "*i64"}
    pointer_types |= {"sorted_weight_ptr": "*fp32", "sorted_expert_ptr": "*i64", "inv_map_ptr": "*i64"}
    for kernel, constexprs in kernels:
        signature = {
            name: "constexpr"
            if name in constexprs
            else pointer_types.get(name, "*i32" if name.endswith("_ptr") else "i32")
            for name in kernel.arg_names
        }
        asm = triton.compile(ASTSource(fn=kernel, signature=signature, constexprs=constexprs), target=target).asm
        assert f'ttg.target = "cuda:{target.arch}"' in asm["ttgir"], (
            f"{kernel.__name__} not compiled for sm_{target.arch}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
