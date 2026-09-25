#!/usr/bin/env python
"""
ep_size==1 with the DEFAULT fsdp_shard_ep1_experts=True: the EP gather (checkpoint save AND GRPO
vLLM weight sync) must materialize the FSDP-sharded experts.

At ep1 the replicated experts are FSDP-sharded DTensors. ``gather_expert_state_dict`` reshapes
``self.<attr>.data`` (transpose / GptOss interleave / Qwen3·Bailing·Zaya unfuse) — operations a
plain-buffer reshape cannot mix with a DTensor (``aten.copy_`` mixed Tensor/DTensor), and which would
otherwise ship a silently-partial shard (corrupting a checkpoint, or hanging the vLLM NCCL
broadcast). ``materialize_dtensor`` ``full_tensor``-s each expert weight first.

Builds the real FSDP2-wrapped ``DistributedSFTTrainer`` at ep1 (default knob), runs one
forward+backward (experts settle into their sharded resting state), and asserts — under BOTH
resharding modes (ZeRO2 ``reshard_after_forward=False`` and ZeRO3 ``=True``):
  1. The raw expert weight is genuinely FSDP-sharded (a DTensor whose local shard < global).
  2. ``gather_expert_state_dict`` returns plain (non-DTensor), finite tensors covering every expert
     (fused keys: leading dim == num_experts; per-expert keys: every index), not one rank's shard.
  3. The full ``gather_and_send_weights`` (the RL entrypoint) forwards every param plain (no DTensor
     reaches vLLM), with expert + dense params present and every expert in the expert tensors.

A gather blind to the sharded experts raises (mixed Tensor/DTensor) or returns a partial shard.

Run with 2 GPUs (the sharded path needs >1 rank to shard across):
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep1_knob_weight_sync.py

Other MoE families (validates the per-family gather path) via env:
    HALO_TEST_EP1_KNOB_MODEL=$HALO_DATA_ROOT/models/GLM-4.7-Flash-patched \
        HALO_TEST_EP1_KNOB_ATTN=flash_attention_2  # fused-GLU
    HALO_TEST_EP1_KNOB_MODEL=$HALO_DATA_ROOT/models/ZAYA1-8B-patched HALO_TEST_EP1_KNOB_ATTN=sdpa  # Zaya

Requirements:
    - 2x GPU with >=80GB memory; DeepEP installed
    - Default model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import re

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_flag, env_str
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log
from tests.common.weight_sync import RecordingSender

# Default gpt-oss; override HALO_TEST_EP1_KNOB_MODEL / HALO_TEST_EP1_KNOB_ATTN to validate another MoE family
# (e.g. HALO_TEST_EP1_KNOB_MODEL=$HALO_DATA_ROOT/models/GLM-4.7-Flash-patched for the fused-GLU base gather path).
MODEL_NAME = env_str("HALO_TEST_EP1_KNOB_MODEL", GPT_OSS_20B)
ATTN = env_str("HALO_TEST_EP1_KNOB_ATTN", "flash_attention_2")
SEQ_LEN = 128
SEED = 42


# A per-expert hub key names one expert (``experts.3.gate_proj.weight``); a fused one holds them all.
_PER_EXPERT_KEY = re.compile(r"(?:^|\.)experts\.(\d+)\.")


def holds_every_expert(shapes: dict[str, tuple[int, ...]], num_experts: int) -> bool:
    """Whether expert tensors keyed by name cover all ``num_experts`` experts, not one rank's shard.

    A plain local FSDP shard passes every dtype/finiteness check; only the expert count tells it apart.
    Fused keys carry the experts on the leading dim; per-expert keys must name every index for each
    layer and projection, so one tensor's missing expert cannot hide behind another's.
    """
    groups: dict[tuple[str, str], set[int]] = {}
    for key, shape in shapes.items():
        match = _PER_EXPERT_KEY.search(key)
        if match:
            groups.setdefault((key[: match.start(1)], key[match.end(1) :]), set()).add(int(match.group(1)))
        elif shape[0] != num_experts:
            return False
    return bool(shapes) and all(indices == set(range(num_experts)) for indices in groups.values())


def gather_is_full(reshard, tokenizer, local_rank, output_dir):
    """Build the FSDP2-wrapped trainer at ep1 + knob-on, run one backward, and gather the first EP
    layer's experts. Returns a dict of the structural facts the asserts check."""
    device = f"cuda:{local_rank}"
    mode = "ZeRO3" if reshard else "ZeRO2"
    log(f"\n===== reshard_after_forward={reshard} ({mode}) fsdp_shard_ep1_experts=ON =====")

    # fsdp_shard_ep1_experts is left at its default (True) — this validates the DEFAULT ep1 behavior.
    # HALO_TEST_EP1_KNOB_LAZY=0 routes the load through from_pretrained + EP patching instead of the lazy
    # safetensors loader, for a checkpoint whose expert layout the lazy loader declines.
    pc = ParallelismConfig(
        ep_size=1,
        use_grouped_gemm=has_grouped_mm(),
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
        fsdp="",
    )
    ds = create_sft_dataset(num_samples=8, tokenizer=tokenizer, seed=SEED)
    trainer = DistributedSFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tokenizer, parallelism_config=pc
    )
    wrapped = trainer.model

    # forward+backward → the experts settle into their sharded resting state (the state the weight
    # sync sees; the optimizer step updates values, not shard layout).
    ids = torch.randint(0, 1000, (1, SEQ_LEN), device=device)
    torch.distributed.broadcast(ids, src=0)
    wrapped.train()
    wrapped.zero_grad(set_to_none=True)
    wrapped(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone()).loss.backward()

    layer = next(
        m for _, m in wrapped.named_modules() if hasattr(m, "ep_config") and hasattr(m, "gather_expert_state_dict")
    )
    raw = getattr(layer, "gate_proj_gmm" if hasattr(layer, "gate_proj_gmm") else "gate_up_proj").data
    raw_sharded = isinstance(raw, DTensor) and raw.to_local().numel() < raw.numel()

    gathered = layer.gather_expert_state_dict(device="cpu")
    # full = every gathered expert tensor materialized to a plain (non-DTensor) finite tensor that holds
    # every expert. Key format is family-specific (fused ``experts.gate_up_proj`` for GptOss/GLM4/Zaya/…,
    # per-expert ``experts.{i}.gate_proj.weight`` for Qwen3/Bailing); holds_every_expert reads both.
    vals = list(gathered.values())
    shapes = {key: tuple(v.shape) for key, v in gathered.items()}
    full = (
        len(vals) >= 2
        and all(not isinstance(v, DTensor) and torch.isfinite(v).all() for v in vals)
        and holds_every_expert(shapes, layer.num_experts)
    )
    log(f"  raw expert sharded-at-rest={raw_sharded}  gather full/plain={full}  tensors={len(vals)} shapes={shapes}")

    # The full RL weight-sync entrypoint: EP expert gather + dense-param full_tensor + expert
    # exclusion from the dense send. Every forwarded tensor must be plain (no DTensor reaches vLLM),
    # and both expert and dense (lm_head/embed) params must be present.
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
