#!/usr/bin/env python
"""The weight sync stages a weight on the sync device as a copy, never through host memory.

Transiting pinned host memory copies out and back over PCIe before the NCCL broadcast, capping the push well below the
fabric; aliasing the source lets the next optimizer step rewrite it before it goes out.
"""

import torch

from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.cpu.weight_sync.test_param_buffer import _bare_client

# 32 MiB fp32: long enough that a reader with no ordering against it would see it in flight.
_STAGED_ELEMENTS = 8 * 1024 * 1024


def run(ctx) -> dict:
    client = _bare_client()
    weights = torch.randn(_STAGED_ELEMENTS, device=ctx.device)
    side = torch.cuda.Stream(device=ctx.device)
    side.wait_stream(torch.cuda.current_stream(ctx.device))
    with torch.cuda.stream(side):
        client.update_named_param("w", weights)
    client.complete_snapshots()
    _, stored = client._param_buffer[0]

    # With a small HALO_WEIGHT_SYNC_CHUNK_MB the second param would flush the first.
    client = _bare_client()
    client._sync_device = ctx.device
    client.update_named_param("v", weights.cpu())
    client.complete_snapshots()
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
