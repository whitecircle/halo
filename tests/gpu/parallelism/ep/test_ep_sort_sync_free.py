#!/usr/bin/env python
"""The grouped-GEMM token sort must not synchronize the host.

A host sync per MoE layer per forward drains the launch queue and stops the CPU running ahead of the
GPU; at 60 MoE layers that is 60 stalls per forward. The offending ops are the ones whose OUTPUT SHAPE
is data-dependent (``bincount``, ``unique_consecutive(return_counts=True)``, ``nonzero``) — CUDA can
only size their output by reading the device back.

``torch.cuda.set_sync_debug_mode("error")`` turns any such read into an exception, so this asserts the
property directly rather than trusting a claim no check enforces.

Scope: at ``ep_size == 1`` the whole sort must be sync-free. At ``ep_size > 1`` one sync remains, in
the ``nonzero`` that compacts DeepEP's ``-1`` padding — asserted here too, so that when it is removed
this test is what records it.

Run with 1 GPU:
    torchrun --nproc_per_node=1 \
        tests/gpu/parallelism/ep/test_ep_sort_sync_free.py
"""

import contextlib
import os
from types import SimpleNamespace

import torch

from src.distributed.expert_parallel.base_layer import EPMoELayerBase

EXPERTS_PER_RANK = 8
TOP_K = 4
NUM_TOKENS = 512
HIDDEN = 256


def make_layer(ep_size):
    """Minimal stand-in carrying only what the sort reads off ``self``."""
    return SimpleNamespace(
        ep_size=ep_size,
        experts_per_rank=EXPERTS_PER_RANK,
        _build_inv_map=EPMoELayerBase._build_inv_map,  # staticmethod — no instance to bind
    )


def make_inputs(ep_size, device):
    torch.manual_seed(0)
    tokens = torch.randn(NUM_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    experts = torch.randint(0, EXPERTS_PER_RANK, (NUM_TOKENS, TOP_K), device=device)
    if ep_size > 1:
        # DeepEP marks slots routed to another rank with -1.
        experts = torch.where(torch.rand_like(experts, dtype=torch.float) < 0.5, experts, -1)
    weights = torch.rand(NUM_TOKENS, TOP_K, device=device, dtype=torch.float32)
    return tokens, experts, weights


@contextlib.contextmanager
def sync_is_an_error():
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


def run_sort(ep_size, device):
    stub = make_layer(ep_size)
    tokens, experts, weights = make_inputs(ep_size, device)
    torch.cuda.synchronize()
    with sync_is_an_error():
        out = EPMoELayerBase._sort_tokens_for_grouped_mm(stub, tokens, experts, weights)
    torch.cuda.synchronize()
    return out


def main():
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)

    # ep_size == 1: no DeepEP padding to compact, so nothing in the sort may read the device back.
    sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, _ = run_sort(1, device)
    print(f"[ep1] sort ran sync-free: {sorted_tokens.shape[0]} rows, offs={offs.tolist()}")

    # The counts must still be right — a sync-free histogram that miscounts is worse than a sync.
    reference = torch.zeros(EXPERTS_PER_RANK, device=device, dtype=torch.long)
    unique, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    reference[unique.long()] = counts.long()
    expected_offs = torch.cumsum(reference, 0).to(torch.int32)
    assert torch.equal(offs, expected_offs), f"offsets disagree: {offs.tolist()} vs {expected_offs.tolist()}"
    assert int(offs[-1]) == NUM_TOKENS * TOP_K, (
        f"every (token, slot) pair must land in some expert at ep1: {int(offs[-1])} != {NUM_TOKENS * TOP_K}"
    )
    assert sorted_token_idx.numel() == NUM_TOKENS * TOP_K
    assert sorted_weights.numel() == NUM_TOKENS * TOP_K

    # ep_size > 1: exactly one sync remains, the nonzero that compacts the -1 padding. When that is
    # removed, this assertion is what fails and tells you to update the claim.
    try:
        run_sort(4, device)
    except RuntimeError as exc:
        assert "synchroniz" in str(exc).lower(), f"unexpected failure at ep>1: {exc}"
        print("[ep4] still synchronizes, as documented (nonzero padding compaction)")
    else:
        raise AssertionError(
            "the ep_size>1 sort no longer synchronizes — the padding-compaction sync was removed. "
            "That is the intended direction: update this test and the sync note in "
            "_sort_tokens_for_grouped_mm."
        )

    print("\nPASS  the grouped-GEMM sort is sync-free at ep1 and its offsets match the reference")


if __name__ == "__main__":
    main()
