#!/usr/bin/env python
"""The vLLM weight sync must run under expert TP for a plain full fine-tune, and reassemble the shards.

``_send_ep_expert_weights`` gathers with ``merge_lora=True`` unconditionally — it cannot know which
layers hold native grouped adapters, and the fold is a no-op on an adapter-less weight. A
``_reject_merge_lora_under_expert_tp`` keyed on that flag ALONE kills every vLLM-syncing GRPO run at
``expert_tp_size > 1``, PEFT or not. The other registered weight-sync tests all run at
``expert_tp_size == 1``, where neither the guard nor the ETP gather is reached.

What this pins at ``ep_size=1`` / ``expert_tp_size=2`` (pure ETP), on a tiny gpt-oss MoE:

  * **the sync completes** with no adapters anywhere;
  * **the ETP shards are reassembled**, bitwise, against an un-EP-patched copy of the same seeded
    weights. Under expert TP each rank holds only ``intermediate / expert_tp_size`` of every expert,
    so a sync that shipped this rank's slice under the full-tensor name would keep the served policy
    permanently stale in exactly half its expert FFN — right shapes are not enough, hence the value
    comparison. Anti-vacuity: the reference's experts must be pairwise distinct.

The adapter-present direction (which must still raise) needs no GPU and is pinned on the CPU tier in
``tests/cpu/grpo/test_weight_sync_expert_tp_guard.py``.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_etp_weight_sync.py
"""

import sys

import torch
from transformers import GptOssConfig, GptOssForCausalLM

from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights
from tests.common.harness import gpu_test_main, log
from tests.common.models import TINY_GPTOSS_CONFIG

ETP_SIZE = 2
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
N_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
EXPERT_KEYS = ("experts.gate_up_proj", "experts.down_proj")


class RecordingSender:
    """Capture every (name, tensor) forwarded to vLLM — no NCCL, no server."""

    def __init__(self):
        self.params: list[tuple[str, torch.Tensor]] = []

    def update_named_param(self, name: str, data: torch.Tensor) -> None:
        self.params.append((name, data.detach().clone()))

    def reset_prefix_cache(self) -> None:
        pass

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.params]


def build_model() -> GptOssForCausalLM:
    torch.manual_seed(1234)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)
    return GptOssForCausalLM(config).to(torch.bfloat16)


@gpu_test_main(exact_world_size=2, prefix="etp_weight_sync")
def run(ctx):
    checks, metrics = {}, {}
    config = ParallelismConfig(ep_size=1, expert_tp_size=ETP_SIZE)
    model = build_model().to(ctx.device)
    patch_moe_model_for_ep(model, config.create_ep_config())
    create_ep_buffers(model)

    etp_layers = [m for m in model.modules() if getattr(m, "expert_tp_size", 1) > 1]
    checks["layers_really_expert_tp_sharded"] = len(etp_layers) == N_LAYERS
    # The whole point: this run holds no adapter, so the gather has nothing to fold.
    checks["no_expert_lora_present"] = bool(etp_layers) and not any(m.has_expert_lora for m in etp_layers)

    # The un-patched ground truth: same seeded init, never sharded.
    reference = {
        name: param.detach().float().cpu()
        for name, param in build_model().named_parameters()
        if name.endswith(EXPERT_KEYS)
    }
    checks["reference_has_all_layers"] = len(reference) == N_LAYERS * len(EXPERT_KEYS)
    checks["reference_experts_pairwise_distinct"] = all(
        not torch.equal(tensor[i], tensor[j])
        for tensor in reference.values()
        for i in range(NUM_EXPERTS)
        for j in range(i + 1, NUM_EXPERTS)
    )

    recorder = RecordingSender()
    try:
        gather_and_send_weights(model, recorder)
        checks["sync_completed_under_expert_tp"] = True
    except NotImplementedError as exc:
        log(f"weight sync rejected a fold-free run under expert TP: {exc}")
        checks["sync_completed_under_expert_tp"] = False
        return {"checks": checks, "metrics": metrics}

    gathered = {name: data.float().cpu() for name, data in recorder.params if name.endswith(EXPERT_KEYS)}
    checks["gathered_all_expert_tensors"] = len(gathered) == N_LAYERS * len(EXPERT_KEYS)
    checks["dense_params_forwarded"] = any("lm_head" in name or "embed" in name for name in recorder.names)

    mismatched = [
        name for name, tensor in gathered.items() if name not in reference or not torch.equal(tensor, reference[name])
    ]
    if mismatched:
        log(f"ETP gather mismatched the unsharded reference for {len(mismatched)}: {mismatched[:4]}")
        for name in mismatched[:2]:
            log(f"  {name}: got {tuple(gathered[name].shape)}, want {tuple(reference[name].shape)}")
    checks["etp_gather_matches_unsharded_reference"] = bool(gathered) and not mismatched

    metrics["forwarded_params"] = len(recorder.params)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
