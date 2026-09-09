#!/usr/bin/env python
"""The weight sync must stage a CUDA source weight on the sync device as a copy, never through host memory.

The rest of the buffering contract is CPU-only and lives in
``tests/cpu/grpo/test_weight_sync_param_buffer.py``; this case needs a real GPU. A snapshot that
transits pinned host memory is copied out and back over PCIe before the NCCL broadcast, and those two
copies cap the push at a third of the fabric's rate; one that aliases the source is rewritten by the
PEFT unmerge or the optimizer step before it goes out.

What it pins (the fixture is the CPU suite's own ``_bare_client``, imported so both tiers exercise
the same client construction):

  * a CUDA weight is staged on its device before ``init_communicator`` (the source's own device), and
    on the client's sync device once one is set;
  * the snapshot is a copy, not an alias;
  * the snapshot carries the source values;
  * the staged copies are covered by the event the flush waits on, recorded on the copy's stream.

Run with 1 GPU:
    torchrun --nproc_per_node=1 tests/gpu/kernels/test_weight_sync_param_buffer_cuda.py
"""

import torch

from tests.common.harness import gpu_test_main
from tests.common.utils import log
from tests.cpu.grpo.test_weight_sync_param_buffer import _bare_client


def run(ctx) -> dict:
    client = _bare_client()
    weights = torch.randn(8, 8, device=ctx.device)

    client.update_named_param("w", weights)
    torch.cuda.current_stream().synchronize()  # the flush path syncs before reading; mirror it
    _, stored = client._param_buffer[0]

    client._sync_device = ctx.device
    client.update_named_param("v", weights)
    torch.cuda.current_stream().synchronize()
    _, on_sync_device = client._param_buffer[1]

    event = client._staged_event
    checks = {
        "cuda_source_staged_on_its_device": stored.device == weights.device,
        "snapshot_lands_on_the_sync_device": on_sync_device.device == ctx.device,
        "snapshot_is_a_copy": stored.data_ptr() != weights.data_ptr(),
        "snapshot_carries_the_source_values": torch.equal(stored, weights),
        "staged_copies_are_covered_by_the_flush_event": event is not None and event.query(),
    }
    for name, ok in checks.items():
        if not ok:
            log(f"{name}: FAIL (stored on {stored.device}, source on {weights.device})")

    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="weight_sync_param_buffer_cuda", partial_state=False)(run)

if __name__ == "__main__":
    main()
