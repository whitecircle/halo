#!/usr/bin/env python
"""
End-to-end step-throughput A/B of the two DeepEP transport backends: legacy Buffer (V1) vs
ElasticBuffer (V2). The backends differ only in the MoE all-to-all (a minority of step FLOPs), so
this confirms picking ``legacy`` for the no-token-ceiling property does not cost training throughput.

Loads the SAME MoE model at EP=2 once per backend and times forward+backward (median over N iters,
CUDA-synced) on a fixed input, reporting median ms/step, tokens/s, and the legacy/elastic ratio.

Run with 2 GPUs (single node — legacy is intranode-only):
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/bench_ep_buffer_backends.py
"""

import statistics
import sys

import torch
from accelerate import PartialState

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
SEQ_LEN = 4096
WARMUP = 3
ITERS = 10


def time_backend(backend, local_rank):
    device = f"cuda:{local_rank}"
    cfg = ParallelismConfig(ep_size=EP_SIZE, use_grouped_gemm=has_grouped_mm(), ep_buffer_backend=backend)
    model = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=cfg,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )[0]
    model.train()
    input_ids = torch.randint(0, 1000, (1, SEQ_LEN), device=device, dtype=torch.long)
    labels = input_ids.clone()

    def step(m):
        m.zero_grad(set_to_none=True)
        out = m(input_ids=input_ids, labels=labels)
        out.loss.backward()

    for _ in range(WARMUP):
        step(model)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step(model)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms = statistics.median(times)
    log(f"  [{backend}] fwd+bwd median={ms:.1f} ms  ({SEQ_LEN / (ms / 1e3):.0f} tok/s/GPU)")

    del model
    cleanup_memory()
    return ms


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()
    ensure_model_downloaded(MODEL_NAME, rank)

    elastic_ms = time_backend("elastic", local_rank)
    barrier()
    legacy_ms = time_backend("legacy", local_rank)
    barrier()

    if rank == 0:
        log(f"\n{'=' * 60}\n  EP backend step throughput (EP={EP_SIZE}, seq={SEQ_LEN})\n{'=' * 60}")
        log(f"  elastic (V2): {elastic_ms:.1f} ms/step")
        log(f"  legacy  (V1): {legacy_ms:.1f} ms/step")
        log(f"  legacy/elastic: {legacy_ms / elastic_ms:.2f}x")
    teardown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
