#!/usr/bin/env python
"""The counting sort behind EP grouped-GEMM routing (``halo/kernels/moe_route_sort.py``) against the torch
construction it replaces, bit for bit, under Triton's interpreter in a fresh process."""

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
# 64-slot blocks and one-block scan chunks: every receive spans several blocks and the scan carries offsets across
# chunks.
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
    """An uncompiled specialization builds between a dispatch and its combine, stalling every peer in DeepEP's
    barrier. The 64-slot tiling keeps interpreted receives cheap."""
    probe = _PROBE.format(setup=_TILINGS["many_blocks"], check="launch_key_findings", marker=_MARKER, args="")
    assert probe_findings(probe, _MARKER) == []


def test_the_launch_key_check_fails_its_negative_control():
    """Every integer specialized at gpt-oss's layout under expert TP: row and slot counts class apart from the
    warm-up's."""
    rebuild = "import triton\nmoe_route_sort._place_kernel = triton.jit(moe_route_sort._place_kernel.fn)\n"
    probe = _PROBE.format(
        setup=_TILINGS["many_blocks"] + rebuild, check="launch_key_findings", marker=_MARKER, args="((4, 8),)"
    )
    findings = probe_findings(probe, _MARKER)
    assert findings and all(finding.startswith("_place_kernel") for finding in findings), findings
    assert any("n_rows" in finding for finding in findings) and any("n_valid" in finding for finding in findings)


def test_every_receive_size_backward_launches_only_kernels_the_warm_up_loaded():
    """A kernel's first launch between dispatch and combine stalls every peer."""
    probe = _PROBE.format(setup=_TILINGS["many_blocks"], check="backward_kernel_findings", marker=_MARKER, args="")
    assert probe_findings(probe, _MARKER) == []


def test_the_backward_kernel_check_fails_its_negative_control():
    """``index_select`` takes a separate kernel for an index of up to 16 rows."""
    rebind = "import torch\ntorch.Tensor.gather = lambda self, dim, index: self.index_select(dim, index)\n"
    probe = _PROBE.format(
        setup=_TILINGS["many_blocks"] + rebind, check="backward_kernel_findings", marker=_MARKER, args=""
    )
    findings = probe_findings(probe, _MARKER)
    assert findings and all("index_select" in finding for finding in findings), findings


class _ReportsCuda(torch.Tensor):
    """Takes the sort's CUDA branch without a GPU."""

    @property
    def is_cuda(self) -> bool:
        return True


@pytest.mark.parametrize("counts", ((2, 0), None), ids=("receive_counts", "ep_size_1"))
def test_on_cuda_every_sort_is_the_counting_sort(monkeypatch, counts):
    """The torch construction's argsort picks kernels by slot count."""
    calls = []
    for name in ("_counting_sort", "_argsort_routing"):
        monkeypatch.setattr(moe_route_sort, name, lambda *args, _name=name: calls.append((_name, args[3])))
    experts, weights = torch.zeros(1, 2, dtype=torch.long), torch.rand(1, 2)
    sort_routing_by_expert(experts.as_subclass(_ReportsCuda), weights, 2, counts)
    sort_routing_by_expert(experts, weights, 2, counts)
    assert calls == [("_counting_sort", counts), ("_argsort_routing", counts)]


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
