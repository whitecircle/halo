#!/usr/bin/env python
"""
Test: PEFT ``modules_to_save`` on the MoE ROUTER under Expert Parallelism (EP=2, GptOss-20B).

The router/gate is absorbed into the EP module (FSDP-ignored), so its grad is DP-averaged by a hook
the EP layer registers on the ORIGINAL router param at construction. ``modules_to_save`` trains a COPY
that hook never saw; ``EPMoELayerBase.reattach_router_grad_sync`` (driven by the trainer mixin) must
re-attach the DP-average hook to the copy or its grad silently drifts across DP ranks. Checks, each of
which FAILS when a piece of the wiring breaks:

  1. Setup: the trainer builds without the loud safety net firing
     (``_validate_ep_peft_trainable_params_synced``), the router copy is trainable, and it is the
     ``modules_to_save`` copy (``.modules_to_save.`` in the name), not the frozen original.
  2. POSITIVE — DP sync: a backward with RANK-DISTINCT inputs leaves the router copy's grad IDENTICAL
     on every rank (the re-attached hook all-reduced it). This is the whole point: without the reattach
     the copy would carry each rank's local, unaveraged grad.
  3. NEGATIVE control: remove the EP grad-sync hooks, redo the same rank-distinct backward, and confirm
     the router copy's grad now DIFFERS across ranks — proving the sync in check 2 came from the hook,
     not from some incidental averaging.
  4. Expert adapters coexist: native grouped expert-LoRA is still present (has_ep_lora), so the router
     path does not disturb expert-LoRA.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_router_modules_to_save.py

Requirements:
    - 2x GPUs (>=80GB), DeepEP installed. Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded).
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig

from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.loading.peft_setup import setup_peft_model, split_expert_lora_targets
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import gpu_mem_gb, log

MODEL_NAME = env_str("HALO_TEST_MODEL", GPT_OSS_20B)
ATTN_IMPL = env_str("HALO_TEST_ATTN", "flex_attention")
EP_SIZE = env_int("HALO_TEST_EP", 2)
MAX_SEQ_LENGTH = 512
LORA_R = 8
LORA_ALPHA = 16
SEED = 42

_PEFT_ARGS = SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)


def _router_copy_params(model):
    """Trainable modules_to_save router copies: ``(name, param)`` with ``.modules_to_save.`` in the name."""
    return [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and "modules_to_save" in n and (".router." in n or ".gate." in n)
    ]


def _grads_identical_across_ranks(param, world_size) -> bool:
    """True iff ``param.grad`` is bit-identical on every rank (all_gather + compare to rank 0)."""
    grad = param.grad
    assert grad is not None, "grad is None — backward did not populate it"
    gathered = [torch.zeros_like(grad) for _ in range(world_size)]
    dist.all_gather(gathered, grad.contiguous())
    return all(torch.equal(gathered[0], g) for g in gathered)


def _rank_distinct_backward(model, tokenizer, rank, local_rank):
    """Forward+backward on text unique to this rank, so an un-synced grad differs across ranks."""
    for _, p in model.named_parameters():
        if p.grad is not None:
            p.grad = None
    text = f"Rank {rank} unique sequence: " + " ".join(f"tok{rank}_{i}" for i in range(40))
    enc = tokenizer([text], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    input_ids = enc["input_ids"].to(local_rank)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()


def run(ctx) -> dict:
    log(f"\n{'=' * 70}")
    log("  EP router modules_to_save DP-sync test")
    log(f"{'=' * 70}")
    log(f"  World size: {ctx.world_size}, ep={EP_SIZE}, model: {MODEL_NAME}")

    checks = {}

    log("\n[1/5] Ensuring model downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log("\n[2/5] Loading tokenizer + model (EP=2, attention LoRA + expert LoRA + modules_to_save=[router])...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(16, tokenizer, seed=SEED)

    # Production path: q_proj attention LoRA + expert LoRA (gate/up/down peeled to native grouped-LoRA)
    # + router full-trained via modules_to_save. get_peft_model wraps the router in a ModulesToSaveWrapper.
    model_config = ModelConfig(
        model_name_or_path=MODEL_NAME,
        use_peft=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "gate_up_proj", "down_proj"],
        lora_modules_to_save=["router"],
        lora_task_type="CAUSAL_LM",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    parallelism_config.expert_lora = split_expert_lora_targets(model_config)
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        use_liger_kernel=True,
    )
    peft_config = setup_peft_model(_PEFT_ARGS, model, model_config, "CAUSAL_LM")
    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

    log("\n[3/5] Building trainer (runs reattach + loud safety-net validation)...")
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    # If _validate_ep_peft_trainable_params_synced were to fire wrongly, or reattach were missing so a
    # trainable router copy went unsynced, construction raises here — the test fails loudly.
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
    )
    barrier()
    checks["trainer_setup_ok"] = True

    tmodel = trainer.model
    router_copies = _router_copy_params(tmodel)
    checks["router_copy_trainable"] = len(router_copies) > 0
    checks["expert_lora_present"] = has_ep_lora(tmodel)
    log(f"  router modules_to_save copies: {len(router_copies)}, expert-LoRA present: {checks['expert_lora_present']}")
    if not router_copies:
        raise RuntimeError("no trainable modules_to_save router copy found — modules_to_save did not wrap it")

    # --- Check 2 (POSITIVE): rank-distinct backward → router copy grad identical across ranks ---
    log("\n[4/5] POSITIVE: rank-distinct backward, router copy grad must be DP-synced...")
    _rank_distinct_backward(tmodel, tokenizer, ctx.rank, ctx.local_rank)
    pos = all(_grads_identical_across_ranks(p, ctx.world_size) for _, p in _router_copy_params(tmodel))
    checks["router_grad_synced"] = pos
    log(f"  router copy grad identical across ranks: {pos}")

    # --- Check 3 (NEGATIVE control): remove EP hooks → same backward → grads must now DIFFER ---
    log("\n[5/5] NEGATIVE control: drop EP hooks, grads must differ across ranks...")
    for layer in ep_layers(tmodel):
        for hook in layer._grad_sync_hooks:
            hook.remove()
        layer._grad_sync_hooks.clear()
    _rank_distinct_backward(tmodel, tokenizer, ctx.rank, ctx.local_rank)
    neg_diff = any(not _grads_identical_across_ranks(p, ctx.world_size) for _, p in _router_copy_params(tmodel))
    checks["router_grad_drifts_without_hook"] = neg_diff
    log(f"  router copy grad DIFFERS across ranks without the hook: {neg_diff}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_ep_router_mts")(run)

if __name__ == "__main__":
    main()
