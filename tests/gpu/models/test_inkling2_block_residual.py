#!/usr/bin/env python
"""Inkling 2's block attention residual at Inkling-2-Small's width (4608), over the eight finished blocks of the final
residual, 8,192 tokens, one and three queries: the toolkit's matmul forward matches upstream's SDPA residual and its
forward and backward peak below half of it, whose memory-efficient backward pads each token's queries to a tile."""

import time
from types import SimpleNamespace

import torch
from transformers.models.inkling2.modeling_inkling2 import Inkling2BlockAttentionResidual

from halo.models.families.inkling2 import UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD, _block_attention_residual_forward
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

DEV = "cuda"
HIDDEN, BLOCKS, TOKENS = 4608, 8, 8192


def _scored(forward, module, block_states, partial, probe):
    """``(outputs, input gradients, seconds, peak GiB above the inputs)`` of one forward and backward."""
    block_states.grad = partial.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    outputs = torch.stack(forward(module, block_states, partial))
    (outputs * probe).sum().backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return outputs.detach(), (block_states.grad.float(), partial.grad.clone()), seconds, peak


def check_matches_upstream_in_less_memory(num_queries: int) -> None:
    module = Inkling2BlockAttentionResidual(SimpleNamespace(hidden_size=HIDDEN), num_queries=num_queries).to(DEV)
    generator = torch.Generator(device=DEV).manual_seed(num_queries)
    torch.nn.init.normal_(module.query, std=1.0, generator=generator)
    # The decoder hands the finished blocks in the run dtype and the partial block in fp32.
    block_states = torch.randn(
        1, TOKENS, BLOCKS, HIDDEN, device=DEV, dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    partial = torch.randn(1, TOKENS, HIDDEN, device=DEV, generator=generator).requires_grad_()
    probe = torch.randn(num_queries, 1, TOKENS, HIDDEN, device=DEV, generator=generator)
    runs = {}
    for name, forward in (
        ("upstream", UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD),
        ("toolkit", _block_attention_residual_forward),
    ):
        _scored(forward, module, block_states, partial, probe)
        runs[name] = _scored(forward, module, block_states, partial, probe)
    upstream, toolkit = runs["upstream"], runs["toolkit"]
    log(
        f"  {num_queries} queries, {TOKENS:,} tokens fwd+bwd: upstream {upstream[2] * 1e3:.1f} ms / {upstream[3]:.2f} "
        f"GiB peak, toolkit {toolkit[2] * 1e3:.1f} ms / {toolkit[3]:.2f} GiB peak"
    )
    torch.testing.assert_close(toolkit[0], upstream[0], rtol=1e-4, atol=1e-4)
    # The block states' gradient is bf16: the two fp32 sums round to neighbouring values.
    torch.testing.assert_close(toolkit[1][0], upstream[1][0], rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(toolkit[1][1], upstream[1][1], rtol=1e-4, atol=1e-4)
    assert toolkit[3] < upstream[3] / 2, (
        f"the toolkit residual peaks at {toolkit[3]:.2f} GiB, upstream {upstream[3]:.2f}"
    )


@gpu_test_main(exact_world_size=1, prefix="inkling2_block_residual", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "one_query", lambda: check_matches_upstream_in_less_memory(1))
    record_check(checks, "three_queries", lambda: check_matches_upstream_in_less_memory(3))
    return {"checks": checks}


if __name__ == "__main__":
    run()
