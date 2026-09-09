#!/usr/bin/env python
"""The weight sync must stage a weight on the sync device as a copy, never through host memory.

The rest of the buffering contract is CPU-only and lives in
``tests/cpu/grpo/test_weight_sync_param_buffer.py``; this case needs a real GPU. A snapshot that
transits pinned host memory is copied out and back over PCIe before the NCCL broadcast, and those two
copies cap the push well below the fabric; one that aliases the source is rewritten by the PEFT
unmerge or the optimizer step before it goes out.

What it pins (the fixture is the CPU suite's own ``_bare_client``, imported so both tiers exercise
the same client construction):

  * a CUDA weight is staged on its own device before ``init_communicator``, as a copy carrying the
    source values and dtype;
  * the copy is issued on the caller's stream and completed by the flush's event wait alone: the
    copies run on a side stream here, as a worker thread's would, and the reader's stream has no
    ordering against them other than that wait;
  * a host-resident source lands on the client's sync device once one is set.

Run with 1 GPU:
    torchrun --nproc_per_node=1 tests/gpu/kernels/test_weight_sync_param_buffer_cuda.py
"""

import torch

from tests.common.harness import gpu_test_main
from tests.common.utils import log
from tests.cpu.grpo.test_weight_sync_param_buffer import _bare_client

# 32 MiB fp32: a copy long enough that a reader with no ordering against it would see it in flight.
_STAGED_ELEMENTS = 8 * 1024 * 1024


def run(ctx) -> dict:
    client = _bare_client()
    weights = torch.randn(_STAGED_ELEMENTS, device=ctx.device)
    side = torch.cuda.Stream(device=ctx.device)
    side.wait_stream(torch.cuda.current_stream(ctx.device))
    with torch.cuda.stream(side):
        client.update_named_param("w", weights)
    client._complete_snapshots()
    _, stored = client._param_buffer[0]

    # A second client: with a small HALO_WEIGHT_SYNC_CHUNK_MB the second param would flush the first.
    client = _bare_client()
    client._sync_device = ctx.device
    client.update_named_param("v", weights.cpu())
    client._complete_snapshots()
    _, from_host = client._param_buffer[0]

    checks = {
        "cuda_source_staged_on_its_device": stored.device == weights.device,
        "snapshot_is_a_copy": stored.data_ptr() != weights.data_ptr(),
        "snapshot_keeps_the_source_dtype": stored.dtype == weights.dtype,
        "snapshot_carries_the_source_values_after_the_flush_wait": torch.equal(stored, weights),
        "host_source_lands_on_the_sync_device": from_host.device == ctx.device,
        "host_source_carries_the_values": torch.equal(from_host, weights),
    }
    for name, ok in checks.items():
        if not ok:
            log(f"{name}: FAIL (stored on {stored.device}, source on {weights.device})")

    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="weight_sync_param_buffer_cuda", partial_state=False)(run)

if __name__ == "__main__":
    main()
