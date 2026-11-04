#!/usr/bin/env python
"""The vLLM weight sync runs under expert TP for a plain full fine-tune and reassembles the shards.

``_send_ep_expert_weights`` always gathers with ``merge_lora=True``, so a refusal keyed on that flag kills every
vLLM-syncing GRPO run at ``expert_tp_size > 1``. Shipping this rank's slice under the full name keeps half the FFN
stale with right shapes.
"""

import sys

import torch
from transformers import GptOssConfig, GptOssForCausalLM

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.rank_log import log
from tests.common.weight_sync import RecordingSender

ETP_SIZE = 2
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
N_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]
EXPERT_KEYS = ("experts.gate_up_proj", "experts.down_proj")


def build_model() -> GptOssForCausalLM:
    torch.manual_seed(1234)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)
    return GptOssForCausalLM(config).to(torch.bfloat16)


@gpu_test_main(exact_world_size=2, prefix="etp_weight_sync")
def run(ctx):
    checks, metrics = {}, {}
    config = ParallelismConfig(ep_size=1, expert_tp_size=ETP_SIZE)
    model = build_model().to(ctx.device)
    apply_ep_layers(model, config.ep_config())

    etp_layers = [m for m in model.modules() if getattr(m, "expert_tp_size", 1) > 1]
    checks["layers_really_expert_tp_sharded"] = len(etp_layers) == N_LAYERS
    # This run holds no adapter, so the gather has nothing to fold.
    checks["no_expert_lora_present"] = bool(etp_layers) and not any(m.has_expert_lora for m in etp_layers)

    # Same seeded init, never sharded.
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

    recorder = RecordingSender(keep_values=True)
    try:
        gather_and_send_weights(model, recorder)
        checks["sync_completed_under_expert_tp"] = True
    except NotImplementedError as exc:
        log(f"weight sync rejected a fold-free run under expert TP: {exc}")
        checks["sync_completed_under_expert_tp"] = False
        return {"checks": checks, "metrics": metrics}

    gathered = {p.name: p.value.float().cpu() for p in recorder.params if p.name.endswith(EXPERT_KEYS)}
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
