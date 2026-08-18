#!/usr/bin/env python
"""The weight sync must offload a CUDA source weight to PINNED host memory, not buffer it on-device.

The rest of the buffering contract is CPU-only and lives in
``tests/cpu/grpo/test_weight_sync_param_buffer.py``; this one case needs a real GPU, and inside the
CPU tier it was a ``skipif(not cuda)`` that the CUDA-hidden CPU suite never ran and no GPU manifest
entry covered — so the one property that can only fail on a GPU went unasserted everywhere.

What it pins (the fixture is the CPU suite's own ``_bare_client``, imported so both tiers exercise
the same client construction):

  * a CUDA weight is buffered on the HOST — buffering on-device holds a full model copy on the
    forwarding rank until the flush;
  * the host buffer is PINNED — the per-pack re-upload to the engine is async and needs page-locked
    memory;
  * the snapshot carries the source values.

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
    checks = {
        "cuda_source_buffered_on_host": stored.device.type == "cpu",
        # A non-CPU buffer is not pinned memory at all, so this only means what it says on the host.
        "host_buffer_is_pinned": stored.device.type == "cpu" and stored.is_pinned(),
        "snapshot_carries_the_source_values": torch.equal(stored.cpu(), weights.cpu()),
    }
    if not checks["cuda_source_buffered_on_host"]:
        log(f"CUDA weight buffered on {stored.device} — full model copy on GPU")
    elif not checks["host_buffer_is_pinned"]:
        log("CPU buffer must be pinned for the per-pack async re-upload")

    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="weight_sync_param_buffer_cuda", partial_state=False)(run)

if __name__ == "__main__":
    main()
