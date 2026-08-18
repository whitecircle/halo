#!/usr/bin/env python
"""
ep_size==1 vLLM weight-sync correctness: the EP expert gather must run even though
``is_ep_mode`` is False.

At ep_size==1 the MoE is still EP-wrapped (``needs_ep_wrappers`` is True for the grouped-GEMM
path) but ``is_ep_mode`` is False (``ep_group_size == 1``). ``gather_and_send_weights`` must not gate
the EP expert reshape on ``is_ep_mode``: gating it there drops the ep1 experts onto the DENSE path,
which forwards them to vLLM in the EP-internal layout — transposed ``gate_up_proj [E,H,2M]`` or, on the
GptOss grouped-GEMM path, the de-interleaved ``gate_proj_gmm`` / ``up_proj_gmm`` attributes. vLLM's
fused loader rejects those names/shapes, stops consuming the NCCL stream, and the producer hangs in
``packed_broadcast_producer`` — the ep1 online/environmental-GRPO weight-sync hang.

This test drives the real ``gather_and_send_weights`` through a recording sender and asserts the
expert weights are forwarded under the vLLM-expected checkpoint names/shapes:

  1. ``...experts.gate_up_proj`` and ``...experts.down_proj`` appear once per MoE layer.
  2. No EP-internal attribute name leaks (``gate_proj_gmm`` / ``up_proj_gmm`` / a bare
     ``...experts.gate_proj`` raw weight) — the exact thing that hangs vLLM.
  3. The gathered expert tensors are 3D ``[num_local_experts, ...]`` (full grouped weight).
  4. Dense params (``lm_head`` / embeddings) are still forwarded — the dense path still runs.

(``fsdp_shard_ep1_experts=True`` DTensor experts materialize through the same gather —
``test_ep1_knob_weight_sync.py`` pins that path under ZeRO-2 and ZeRO-3.)

A gather that skips the ep1 experts fails (1)/(2): the names come out as the internal ``_gmm``
attributes.

Run with 1 or 2 GPUs (ep1 needs no EP collective, so 1 suffices; 2 exercises the DP path):
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep1_weight_sync_names.py

Requirements:
    - 1x (or 2x) GPU with >=80GB memory
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import torch
from torch.distributed.tensor import DTensor

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B

# EP-internal attribute names that must NEVER reach vLLM (the gather must reshape them away).
FORBIDDEN_NAME_SUBSTRINGS = ("gate_proj_gmm", "up_proj_gmm")


class RecordingSender:
    """Capture every (name, shape, is_dtensor) forwarded to vLLM — no NCCL, no server."""

    def __init__(self):
        self.params: list[tuple[str, tuple, bool]] = []

    def update_named_param(self, name: str, data: torch.Tensor) -> None:
        self.params.append((name, tuple(data.shape), isinstance(data, DTensor)))

    def reset_prefix_cache(self) -> None:
        pass

    @property
    def names(self) -> list[str]:
        return [n for n, _, _ in self.params]


def run(ctx):
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    parallelism_config = ParallelismConfig(ep_size=1, use_grouped_gemm=has_grouped_mm())
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )

    num_experts = model.config.num_local_experts
    n_ep_layers = sum(1 for _, m in model.named_modules() if hasattr(m, "ep_config"))
    log(f"  EP layers (ep_size=1): {n_ep_layers}, num_local_experts={num_experts}")
    assert n_ep_layers > 0, "model was not EP-wrapped at ep_size=1 — grouped-GEMM wrappers missing"

    # All ranks run the gather (collective at ep>1; a pure copy here). Only rank 0 records.
    recorder = RecordingSender() if ctx.rank == 0 else None
    gather_and_send_weights(model, recorder)
    ctx.barrier()

    checks = {}
    if ctx.rank == 0:
        names = recorder.names
        gate_up = [(n, s, d) for n, s, d in recorder.params if n.endswith("experts.gate_up_proj")]
        down = [(n, s, d) for n, s, d in recorder.params if n.endswith("experts.down_proj")]
        forbidden = [n for n in names if any(sub in n for sub in FORBIDDEN_NAME_SUBSTRINGS)]
        # A bare per-expert-less raw ``...experts.gate_proj``/``up_proj`` (NOT the unfused
        # per-expert ``experts.N.gate_proj.weight``) would also be an internal leak.
        raw_split = [n for n in names if n.endswith(("experts.gate_proj", "experts.up_proj"))]

        log(f"  forwarded params: {len(names)}  gate_up_proj: {len(gate_up)}  down_proj: {len(down)}")
        if forbidden:
            log(f"  LEAKED internal names: {forbidden[:4]}")

        checks["gate_up_present"] = len(gate_up) == n_ep_layers
        checks["down_present"] = len(down) == n_ep_layers
        checks["no_internal_gmm_leak"] = len(forbidden) == 0
        checks["no_raw_split_leak"] = len(raw_split) == 0
        checks["gate_up_3d_grouped"] = all(len(s) == 3 and s[0] == num_experts for _, s, _ in gate_up)
        checks["down_3d_grouped"] = all(len(s) == 3 and s[0] == num_experts for _, s, _ in down)
        checks["expert_weights_materialized"] = all(not d for _, _, d in gate_up + down)
        checks["dense_params_forwarded"] = any("lm_head" in n or "embed" in n for n in names)

    # (fsdp_shard_ep1_experts=True is the FSDP-sharded-expert path — exercised by
    # test_ep1_knob_weight_sync.py, which needs the real FSDP2 wrapping this name test omits.)
    del model
    cleanup_memory()

    # Only rank 0 holds the recorder, so its verdict is what every rank reports.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=1, prefix="ep1_weight_sync_names")(run)

if __name__ == "__main__":
    main()
