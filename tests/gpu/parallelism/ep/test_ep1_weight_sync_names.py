#!/usr/bin/env python
"""ep_size==1 vLLM weight sync: the EP expert gather runs even though ``has_ep_group`` is False.

Gating it on ``has_ep_group`` sends ep1 experts down the dense path in the EP-internal layout. vLLM's fused loader
rejects those names, stops consuming the NCCL stream, and the producer hangs in ``packed_broadcast_producer``.
"""

import torch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import supports_grouped_mm
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log
from tests.common.weight_sync import RecordingSender

MODEL_NAME = GPT_OSS_20B

# Must never reach vLLM.
FORBIDDEN_NAME_SUBSTRINGS = ("gate_proj_gmm", "up_proj_gmm")


def run(ctx):
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    parallelism_config = ParallelismConfig(ep_size=1, use_grouped_gemm=supports_grouped_mm())
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )

    num_experts = model.config.num_local_experts
    n_ep_layers = len(ep_layers(model))
    log(f"  EP layers (ep_size=1): {n_ep_layers}, num_local_experts={num_experts}")
    assert n_ep_layers > 0, "model was not EP-wrapped at ep_size=1 — grouped-GEMM EP layers missing"

    # COLLECTIVE at ep>1; a pure copy here. Only rank 0 records.
    recorder = RecordingSender() if ctx.rank == 0 else None
    gather_and_send_weights(model, recorder)
    ctx.barrier()

    checks = {}
    if ctx.rank == 0:
        names = recorder.names
        gate_up = [param for param in recorder.params if param.name.endswith("experts.gate_up_proj")]
        down = [param for param in recorder.params if param.name.endswith("experts.down_proj")]
        forbidden = [n for n in names if any(sub in n for sub in FORBIDDEN_NAME_SUBSTRINGS)]
        # A raw ``...experts.gate_proj``/``up_proj``, unlike per-expert ``experts.N.gate_proj.weight``, is an internal
        # leak too.
        raw_split = [n for n in names if n.endswith(("experts.gate_proj", "experts.up_proj"))]

        log(f"  forwarded params: {len(names)}  gate_up_proj: {len(gate_up)}  down_proj: {len(down)}")
        if forbidden:
            log(f"  LEAKED internal names: {forbidden[:4]}")

        checks["gate_up_present"] = len(gate_up) == n_ep_layers
        checks["down_present"] = len(down) == n_ep_layers
        checks["no_internal_gmm_leak"] = len(forbidden) == 0
        checks["no_raw_split_leak"] = len(raw_split) == 0
        checks["gate_up_3d_grouped"] = all(len(p.shape) == 3 and p.shape[0] == num_experts for p in gate_up)
        checks["down_3d_grouped"] = all(len(p.shape) == 3 and p.shape[0] == num_experts for p in down)
        checks["expert_weights_materialized"] = all(not p.is_dtensor for p in gate_up + down)
        checks["dense_params_forwarded"] = any("lm_head" in n or "embed" in n for n in names)

    # test_ep1_knob_weight_sync.py covers fsdp_shard_ep1_experts=True, which needs the FSDP2 wrapping this test omits.
    del model
    cleanup_memory()

    # Only rank 0 holds the recorder, so its verdict is what every rank reports.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=1, prefix="ep1_weight_sync_names")(run)

if __name__ == "__main__":
    main()
