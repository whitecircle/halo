#!/usr/bin/env python
"""vLLM weight-sync expert gather at ``ep_size > 1``: global count, values and order, on a tiny gpt-oss MoE.

Shards concatenated in the wrong order keep shapes and names right, vLLM loads them, and the served model routes every
token to the wrong expert. Init weights match on both ranks, so only a trained model tells "in rank order" from "rank 0
twice".
"""

import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM, default_data_collator
from trl import SFTConfig

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG
from tests.common.rank_log import log
from tests.common.weight_sync import RecordingSender

EP_SIZE = 2
SEQ_LEN = 64
NUM_EXPERTS = TINY_GPTOSS_CONFIG["num_local_experts"]
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
N_LAYERS = TINY_GPTOSS_CONFIG["num_hidden_layers"]

# The grouped expert weights vLLM's fused loader expects, per MoE layer.
EXPERT_KEYS = ("experts.gate_up_proj", "experts.down_proj")
# EP-internal attribute names that must never reach vLLM (they hung it at ep1).
FORBIDDEN_NAME_SUBSTRINGS = ("gate_proj_gmm", "up_proj_gmm")


def build_model() -> GptOssForCausalLM:
    torch.manual_seed(1234)
    config = GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)
    return GptOssForCausalLM(config).to(torch.bfloat16)


def build_dataset() -> Dataset:
    generator = torch.Generator().manual_seed(7)
    rows = torch.randint(4, TINY_GPTOSS_CONFIG["vocab_size"], (16, SEQ_LEN), generator=generator)
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": rows.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _local_fused_gate_up(module) -> torch.Tensor | None:
    """Re-interleave the grouped-mm pair. On SM90+ the EP layer never sets ``gate_up_proj``, so probing it alone
    silently narrows the order contract to ``down_proj``."""
    gate = getattr(module, "gate_proj_gmm", None)
    up = getattr(module, "up_proj_gmm", None)
    if gate is None or up is None:
        return None
    gate, up = _materialize(gate.data), _materialize(up.data)
    fused = torch.empty(*gate.shape[:-1], gate.shape[-1] * 2, dtype=gate.dtype, device=gate.device)
    fused[..., ::2] = gate
    fused[..., 1::2] = up
    return fused


def local_expert_weights(model) -> dict[str, torch.Tensor]:
    shards = {}
    for name, module in model.named_modules():
        if not hasattr(module, "ep_config"):
            continue
        for key in EXPERT_KEYS:
            attribute = key.split(".")[-1]
            raw = getattr(module, attribute, None)
            data = _materialize(raw.data) if raw is not None else None
            if data is None and attribute == "gate_up_proj":
                data = _local_fused_gate_up(module)
            if data is None:
                continue
            shards[f"{name}.{key}"] = data.detach().float().cpu()
    return shards


def gather_on_all_ranks(model) -> RecordingSender:
    """COLLECTIVE, recording so the per-rank views can be compared."""
    recorder = RecordingSender(keep_values=True)
    gather_and_send_weights(model, recorder)
    return recorder


def flat_signature(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensors[name].flatten() for name in sorted(tensors)])


@gpu_test_main(exact_world_size=2, prefix="ep2_weight_sync_values")
def run(ctx):
    checks, metrics = {}, {}
    config = ParallelismConfig(ep_size=EP_SIZE)
    model = build_model().to(ctx.device)
    apply_ep_layers(model, config.ep_config())

    # The same seeded init, never EP-patched: the reference layout and expert order.
    dense_reference = {
        name: param.detach().float().cpu()
        for name, param in build_model().named_parameters()
        if name.endswith(EXPERT_KEYS)
    }
    checks["dense_reference_has_all_layers"] = len(dense_reference) == N_LAYERS * len(EXPERT_KEYS)
    # Anti-vacuity: an ordering bug shows only if the experts differ from one another.
    distinct = all(
        not torch.equal(tensor[i], tensor[j])
        for tensor in dense_reference.values()
        for i in range(NUM_EXPERTS)
        for j in range(i + 1, NUM_EXPERTS)
    )
    checks["reference_experts_pairwise_distinct"] = distinct

    recorder = gather_on_all_ranks(model)
    gathered = {p.name: p.value.float().cpu() for p in recorder.params if p.name.endswith(EXPERT_KEYS)}
    checks["gathered_all_expert_tensors"] = len(gathered) == N_LAYERS * len(EXPERT_KEYS)
    checks["gathered_experts_are_global_count"] = bool(gathered) and all(
        tensor.shape[0] == NUM_EXPERTS for tensor in gathered.values()
    )
    checks["no_internal_gmm_leak"] = not any(
        sub in name for name in recorder.names for sub in FORBIDDEN_NAME_SUBSTRINGS
    )
    checks["dense_params_forwarded"] = any("lm_head" in name or "embed" in name for name in recorder.names)

    mismatched = [
        name
        for name, tensor in gathered.items()
        if name not in dense_reference or not torch.equal(tensor, dense_reference[name])
    ]
    if mismatched:
        log(f"expert gather mismatched the dense reference for {len(mismatched)} tensors: {mismatched[:4]}")
    checks["init_gather_matches_dense_reference"] = not mismatched

    # Only trained weights can tell "concatenated in rank order" from "rank 0's shard twice".
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=ctx.output_dir,
            max_steps=1,
            per_device_train_batch_size=4,
            learning_rate=5e-2,  # Large enough that every expert moves out of bf16 rounding.
            lr_scheduler_type="constant",
            max_length=SEQ_LEN,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            seed=42,
            bf16=True,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            # TRL's SFT column pruning would drop attention_mask from the pre-tokenized rows.
            remove_unused_columns=False,
        ),
        train_dataset=build_dataset(),
        processing_class=tokenizer,
        data_collator=default_data_collator,
        parallelism_config=config,
        moe_balancing="none",
    )
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train()

    trained_recorder = gather_on_all_ranks(trainer.model)
    trained_gathered = {p.name: p.value.float().cpu() for p in trained_recorder.params if p.name.endswith(EXPERT_KEYS)}
    checks["trained_gather_differs_from_init"] = any(
        not torch.equal(tensor, dense_reference[name]) for name, tensor in trained_gathered.items()
    )
    # Only meaningful after fully_shard, since nothing can be a DTensor before the trainer exists.
    checks["nothing_forwarded_as_dtensor"] = not any(p.is_dtensor for p in trained_recorder.params)

    # This rank's own shard must sit at [rank * local, (rank + 1) * local).
    shards = local_expert_weights(trainer.model)
    start = ctx.rank * LOCAL_EXPERTS
    misplaced = [
        name
        for name, shard in shards.items()
        if name not in trained_gathered
        or not torch.equal(trained_gathered[name][start : start + LOCAL_EXPERTS], shard)
    ]
    if misplaced:
        log(f"[rank {ctx.rank}] local shard not at global offset {start} for: {misplaced[:4]}")
    placed = torch.tensor(0.0 if misplaced else 1.0, device=ctx.device)
    dist.all_reduce(placed, op=dist.ReduceOp.MIN)
    checks["local_shard_lands_at_rank_offset"] = bool(placed)
    checks["local_shards_are_half_the_experts"] = bool(shards) and all(
        shard.shape[0] == LOCAL_EXPERTS for shard in shards.values()
    )
    # Under grouped-mm the fused gate_up_proj exists only as the pair; probing it alone reduces the checks to
    # down_proj.
    checks["local_shards_cover_all_expert_keys"] = len(shards) == N_LAYERS * len(EXPERT_KEYS)

    # COLLECTIVE: every rank must end up with the same global tensor.
    signature = flat_signature(trained_gathered).to(ctx.device)
    peers = [torch.zeros_like(signature) for _ in range(ctx.world_size)]
    dist.all_gather(peers, signature)
    checks["gathered_experts_identical_across_ranks"] = all(torch.equal(signature, peer) for peer in peers)

    metrics["forwarded_params"] = len(trained_recorder.names)
    metrics["gathered_expert_tensors"] = len(trained_gathered)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
