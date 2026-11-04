#!/usr/bin/env python
"""ep_size==1 with the default fsdp_shard_ep1_experts: the EP gather must materialize the FSDP-sharded experts.

``gather_expert_state_dict`` reshapes expert data, which a DTensor can't mix with a plain buffer (``aten.copy_``).
Unmaterialized, the gather raises or ships a partial shard, corrupting a checkpoint or hanging the vLLM NCCL broadcast.
Runs under both ZeRO2 and ZeRO3 resharding, and needs >1 rank to shard across.
"""

import re

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag, env_str
from halo.hardware import supports_grouped_mm
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log
from tests.common.weight_sync import RecordingSender

# Override to validate another MoE family.
MODEL_NAME = env_str("HALO_TEST_EP1_KNOB_MODEL", GPT_OSS_20B)
ATTN = env_str("HALO_TEST_EP1_KNOB_ATTN", "flash_attention_2")
SEQ_LEN = 128
SEED = 42


# A per-expert hub key names one expert (``experts.3.gate_proj.weight``); a fused one holds them all.
_PER_EXPERT_KEY = re.compile(r"(?:^|\.)experts\.(\d+)\.")


def holds_every_expert(shapes: dict[str, tuple[int, ...]], num_experts: int) -> bool:
    """A plain local FSDP shard passes every dtype/finiteness check; only the expert count tells it apart."""
    groups: dict[tuple[str, str], set[int]] = {}
    for key, shape in shapes.items():
        match = _PER_EXPERT_KEY.search(key)
        if match:
            groups.setdefault((key[: match.start(1)], key[match.end(1) :]), set()).add(int(match.group(1)))
        elif shape[0] != num_experts:
            return False
    return bool(shapes) and all(indices == set(range(num_experts)) for indices in groups.values())


def gather_is_full(reshard, tokenizer, local_rank, output_dir):
    device = f"cuda:{local_rank}"
    mode = "ZeRO3" if reshard else "ZeRO2"
    log(f"\n===== reshard_after_forward={reshard} ({mode}) fsdp_shard_ep1_experts=ON =====")

    # HALO_TEST_EP1_KNOB_LAZY=0 routes the load through from_pretrained + EP patching, for a checkpoint whose expert
    # layout the lazy loader declines.
    pc = ParallelismConfig(
        ep_size=1,
        use_grouped_gemm=supports_grouped_mm(),
        fsdp_reshard_after_forward=reshard,
        ep_lazy_loading=env_flag("HALO_TEST_EP1_KNOB_LAZY", True),
    )
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN,
        use_liger_kernel=True,
    )
    cfg = SFTConfig(
        output_dir=output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=SEQ_LEN,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    ds = create_sft_dataset(num_samples=8, tokenizer=tokenizer, seed=SEED)
    trainer = DistributedSFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tokenizer, parallelism_config=pc
    )
    wrapped = trainer.model

    # The experts settle into their sharded resting state, the state the weight sync sees.
    ids = torch.randint(0, 1000, (1, SEQ_LEN), device=device)
    torch.distributed.broadcast(ids, src=0)
    wrapped.train()
    wrapped.zero_grad(set_to_none=True)
    wrapped(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()).loss.backward()

    layer = next(m for m in ep_layers(wrapped) if hasattr(m, "gather_expert_state_dict"))
    raw = getattr(layer, "gate_proj_gmm" if hasattr(layer, "gate_proj_gmm") else "gate_up_proj").data
    raw_sharded = isinstance(raw, DTensor) and raw.to_local().numel() < raw.numel()

    gathered = layer.gather_expert_state_dict(device="cpu")
    # Key format is family-specific (fused for GptOss/GLM4/Zaya, per-expert for Qwen3/Bailing); holds_every_expert
    # reads both.
    vals = list(gathered.values())
    shapes = {key: tuple(v.shape) for key, v in gathered.items()}
    full = (
        len(vals) >= 2
        and all(not isinstance(v, DTensor) and torch.isfinite(v).all() for v in vals)
        and holds_every_expert(shapes, layer.num_experts)
    )
    log(f"  raw expert sharded-at-rest={raw_sharded}  gather full/plain={full}  tensors={len(vals)} shapes={shapes}")

    # No DTensor may reach vLLM, and both expert and dense params must be present.
    recorder = RecordingSender()
    gather_and_send_weights(wrapped, recorder)
    names = recorder.names
    any_dtensor = any(param.is_dtensor for param in recorder.params)
    has_experts = any("expert" in n.lower() for n in names)
    has_dense = any("lm_head" in n or "embed" in n for n in names)
    sent_experts_full = holds_every_expert(
        {param.name: param.shape for param in recorder.params if ".experts." in param.name}, layer.num_experts
    )
    log(
        f"  gather_and_send_weights: {len(names)} params, any_dtensor={any_dtensor}, experts={has_experts}, "
        f"dense={has_dense}, every expert sent={sent_experts_full}"
    )

    trainer.cleanup_ep()
    del trainer, model, wrapped
    cleanup_memory()
    return {
        "raw_sharded": raw_sharded,
        "gather_full": full,
        "send_all_plain": not any_dtensor and has_experts and has_dense,
        "send_experts_full": sent_experts_full,
    }


def run(ctx) -> dict:
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    z2 = gather_is_full(False, tokenizer, ctx.local_rank, ctx.output_dir)
    barrier()
    z3 = gather_is_full(True, tokenizer, ctx.local_rank, ctx.output_dir)
    barrier()

    return {
        "checks": {
            "zero2_raw_sharded": z2["raw_sharded"],
            "zero2_gather_full": z2["gather_full"],
            "zero2_send_all_plain": z2["send_all_plain"],
            "zero2_send_experts_full": z2["send_experts_full"],
            "zero3_raw_sharded": z3["raw_sharded"],
            "zero3_gather_full": z3["gather_full"],
            "zero3_send_all_plain": z3["send_all_plain"],
            "zero3_send_experts_full": z3["send_experts_full"],
        }
    }


main = gpu_test_main(min_world_size=2, prefix="ep1_knob_weight_sync")(run)

if __name__ == "__main__":
    main()
