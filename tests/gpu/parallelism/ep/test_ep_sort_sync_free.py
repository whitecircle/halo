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
from types import SimpleNamespace

import torch

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from tests.common.harness import gpu_test_main
from tests.common.utils import log

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


@gpu_test_main(exact_world_size=1, prefix="ep_sort_sync_free", partial_state=False)
def run(ctx):
    checks: dict[str, bool] = {}
    device = ctx.device

    # ep_size == 1: no DeepEP padding to compact, so nothing in the sort may read the device back
    # (a sync raises out of run_sort and fails the run).
    sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, _ = run_sort(1, device)
    log(f"[ep1] sort ran sync-free: {sorted_tokens.shape[0]} rows, offs={offs.tolist()}")

    # The counts must still be right — a sync-free histogram that miscounts is worse than a sync.
    reference = torch.zeros(EXPERTS_PER_RANK, device=device, dtype=torch.long)
    unique, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    reference[unique.long()] = counts.long()
    expected_offs = torch.cumsum(reference, 0).to(torch.int32)
    checks["ep1_offsets_match_reference"] = torch.equal(offs, expected_offs)
    # Every (token, slot) pair must land in some expert at ep1.
    checks["ep1_every_slot_lands_in_an_expert"] = int(offs[-1]) == NUM_TOKENS * TOP_K
    checks["ep1_sorted_index_covers_every_slot"] = sorted_token_idx.numel() == NUM_TOKENS * TOP_K
    checks["ep1_sorted_weights_cover_every_slot"] = sorted_weights.numel() == NUM_TOKENS * TOP_K

    # ep_size > 1: exactly one sync remains, the nonzero that compacts the -1 padding. When that is
    # removed, this check is what fails and tells you to update the claim.
    try:
        run_sort(4, device)
    except RuntimeError as exc:
        checks["ep4_padding_compaction_still_syncs"] = "synchroniz" in str(exc).lower()
        log(f"[ep4] {exc}")
    else:
        checks["ep4_padding_compaction_still_syncs"] = False
        log(
            "the ep_size>1 sort no longer synchronizes — the padding-compaction sync was removed. "
            "That is the intended direction: update this test and the sync note in "
            "_sort_tokens_for_grouped_mm."
        )
    return {"checks": checks}


if __name__ == "__main__":
    run()
