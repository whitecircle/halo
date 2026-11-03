#!/usr/bin/env python
"""AdamWBF16 end to end with EP=8 on GPT-OSS-20B: full training, LoRA + EP, and SR under EP."""

import argparse
import math
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.ep import load_ep_model
from halo.model_loading.parallel.model_source import resolve_model_source
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.optimizers.adamw_bf16 import AdamWBF16
from halo.runtime.ranks import get_global_world_size, get_local_world_size
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log

MODEL_PATH = GPT_OSS_20B
EP_SIZE = 8
NUM_TRAIN_SAMPLES = 16
MAX_SEQ_LENGTH = 64
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 2


def _load_ep_policy() -> nn.Module:
    ep_config = EPConfig(
        ep_size=EP_SIZE,
        stage_world_size=get_global_world_size(),
        nvlink_domain_size=get_local_world_size(),
        experts_fsdp_managed=False,
    )
    pc = SimpleNamespace(
        ep_config=lambda: ep_config, ep_lazy_loading=True, fp32_non_ep_params=False, max_concurrent_loading=None
    )
    from_pretrained_kwargs = {
        "config": AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        # Auto-detect: FA4 on Blackwell handles gpt-oss sinks; FA2 forces the Hopper-only S-aux path.
        "attn_implementation": None,
    }
    source = resolve_model_source(MODEL_PATH, None, tag="policy")
    return load_ep_model(source, pc, AutoModelForCausalLM, from_pretrained_kwargs, preserve_checkpoint_precision=False)


def test_full_training(output_dir: str) -> bool:
    log(f"\n{'=' * 70}")
    log(f"TEST 1: Full Training with bf16_optimizer (EP={EP_SIZE})")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer)

    log("Loading model with EP...")
    model = _load_ep_policy()

    total_params = sum(p.numel() for p in model.parameters())
    bf16_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    fp32_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    log(f"Total params: {total_params / 1e6:.1f}M  (bf16={bf16_params / 1e6:.1f}M, fp32={fp32_params / 1e6:.1f}M)")

    log("Enabling gradient checkpointing...")
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})

    config = SFTConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=1e-5,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,  # Already enabled above.
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    # bf16_optimizer is auto-enabled from config.bf16=True.
    log("Creating trainer (bf16_optimizer auto-enabled)...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log("Starting training...")
    dist.barrier()
    train_result = trainer.train()
    # Free this model's DeepEP buffers before the next test loads its own.
    trainer.cleanup_ep()

    issues = []

    if not isinstance(trainer.optimizer, AdamWBF16):
        issues.append(f"Optimizer is {type(trainer.optimizer).__name__}, expected AdamWBF16")

    # Counted, so "no param matched the guard" cannot silently assert nothing.
    checked_states = 0
    for p in model.parameters():
        if p in trainer.optimizer.state and p.requires_grad:
            state = trainer.optimizer.state[p]
            if "exp_avg" in state:
                checked_states += 1
                if p.dtype == torch.bfloat16 and state["exp_avg"].dtype != torch.bfloat16:
                    issues.append(f"exp_avg dtype is {state['exp_avg'].dtype}, expected bf16")
                    break
                if p.dtype == torch.bfloat16 and state["exp_avg_sq"].dtype != torch.bfloat16:
                    issues.append(f"exp_avg_sq dtype is {state['exp_avg_sq'].dtype}, expected bf16")
                    break
    if checked_states == 0:
        issues.append("no model parameter carries optimizer state — the state-dtype check asserted nothing")

    # This config enables no fp32 knob, so every param must be bf16.
    bf16_numel = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
    fp32_named = [n for n, p in model.named_parameters() if p.dtype == torch.float32]
    if bf16_numel == 0:
        issues.append("no bf16 parameters after training — model dtype regressed")
    if fp32_named:
        issues.append(f"{len(fp32_named)} unexpected fp32 param(s) with no fp32 knob enabled, e.g. {fp32_named[:3]}")

    loss = train_result.training_loss
    log(f"Training loss: {loss:.4f}")
    if not math.isfinite(loss):
        issues.append(f"Training loss is not finite: {loss}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    log(f"Peak GPU memory: {mem:.1f} GB")

    if issues:
        for issue in issues:
            log(f"  FAIL: {issue}")
        log("\u2717 Full Training Test FAILED")
        return False

    log("  Optimizer: AdamWBF16 \u2713")
    log("  States dtype: bf16 \u2713")
    log(f"  Training loss: {loss:.4f} \u2713")
    log(f"  Peak memory: {mem:.1f} GB \u2713")
    log("\u2713 Full Training Test PASSED")
    return True


def test_lora_training(output_dir: str) -> bool:
    log(f"\n{'=' * 70}")
    log(f"TEST 2: LoRA Training with bf16_optimizer (EP={EP_SIZE})")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer)

    log("Loading model with EP...")
    model = _load_ep_policy()

    log("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    log("Enabling gradient checkpointing...")
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})

    config = SFTConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=1e-4,  # Higher LR for LoRA.
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,  # Already enabled above.
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    log("Creating trainer (bf16_optimizer auto-enabled) + LoRA...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log("Starting training...")
    dist.barrier()
    train_result = trainer.train()
    # Free this model's DeepEP buffers before the next test loads its own.
    trainer.cleanup_ep()

    issues = []

    if not isinstance(trainer.optimizer, AdamWBF16):
        issues.append(f"Optimizer is {type(trainer.optimizer).__name__}, expected AdamWBF16")

    # Counted, so the state guard cannot silently skip every param.
    lora_param_count = 0
    checked_lora_states = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            lora_param_count += 1
            if p in trainer.optimizer.state:
                state = trainer.optimizer.state[p]
                if "exp_avg" in state:
                    checked_lora_states += 1
                    if state["exp_avg"].dtype != torch.bfloat16:
                        issues.append(f"LoRA param {name}: exp_avg is {state['exp_avg'].dtype}")
                        break

    if lora_param_count == 0:
        issues.append("No trainable LoRA parameters found")
    if checked_lora_states == 0:
        issues.append("no trainable param carries optimizer state — the LoRA state check asserted nothing")

    loss = train_result.training_loss
    log(f"Training loss: {loss:.4f}")
    if not math.isfinite(loss):
        issues.append(f"Training loss is not finite: {loss}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    log(f"Peak GPU memory: {mem:.1f} GB")

    if issues:
        for issue in issues:
            log(f"  FAIL: {issue}")
        log("\u2717 LoRA Training Test FAILED")
        return False

    log("  Optimizer: AdamWBF16 \u2713")
    log(f"  LoRA trainable params: {lora_param_count} \u2713")
    log("  States dtype: bf16 \u2713")
    log(f"  Training loss: {loss:.4f} \u2713")
    log(f"  Peak memory: {mem:.1f} GB \u2713")
    log("\u2713 LoRA Training Test PASSED")
    return True


def test_sr_under_ep() -> bool:
    """SR on a synthetic gradient at the real EP=8 shard shape and group. exp_avg_sq tracks an fp32 EMA (rel-err < 5%)
    while the nearest control misses by > 15%. All 8 ranks replicate non-expert params, so SR must round alike or
    replicas drift."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"\n{'=' * 70}")
    log(f"TEST 3: SR correctness under EP (de-bias + replica seeds, world={world_size})")
    log(f"{'=' * 70}")

    device = torch.device("cuda", torch.cuda.current_device())
    beta1, beta2 = 0.9, 0.999
    grad_val = 1e-4
    n_steps = 500
    # One expert's local FFN shard slice (gate/up column block).
    size = 8192

    g32 = torch.full((size,), grad_val, dtype=torch.float32, device=device)
    easq_fp32 = torch.zeros(size, dtype=torch.float32, device=device)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())

    easq_nearest = torch.zeros(size, dtype=torch.bfloat16, device=device)
    for _ in range(n_steps):
        acc = easq_nearest.float()
        acc.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        easq_nearest = acc.to(torch.bfloat16)
    nearest_mean = float(easq_nearest.float().mean())

    p = nn.Parameter(torch.zeros(size, dtype=torch.bfloat16, device=device))
    opt = AdamWBF16([("p", p)], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)
    for _ in range(n_steps):
        p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
        opt.step()
    sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

    log(f"[rank {rank}] expert-shard easq: fp32={fp32_mean:.3e} nearest={nearest_mean:.3e} SR={sr_mean:.3e}")

    issues = []
    nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
    if not nearest_bias > 0.15:
        issues.append(f"premise broken: nearest bias only {nearest_bias:.1%}")
    sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
    if not sr_rel_err < 0.05:
        issues.append(f"SR easq on expert shard should track fp32, rel-err {sr_rel_err:.1%}")
    if not abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean):
        issues.append("SR easq on expert shard must be closer to fp32 than to nearest reference")

    # Identical fixed input and state on every rank; the SR-rounded weight must match bit for bit.
    torch.manual_seed(0)
    pr = nn.Parameter(torch.full((size,), 0.5, dtype=torch.bfloat16, device=device))
    opt_r = AdamWBF16([("p", pr)], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)
    # A signed gradient whose Adam step lands below the bf16 ULP of 0.5 (~0.004), so SR carries the update.
    for _ in range(8):
        pr.grad = torch.full((size,), 1e-3, dtype=torch.bfloat16, device=device)
        opt_r.step()
    w_local = pr.data.clone()

    # Negative control: an identical nearest-rounded run must differ, or the bit-identity is vacuous.
    w_ref32 = torch.full((size,), 0.5, dtype=torch.float32, device=device)
    ea = torch.zeros(size, dtype=torch.float32, device=device)
    ev = torch.zeros(size, dtype=torch.float32, device=device)
    for s in range(1, 9):
        grad = torch.full((size,), 1e-3, dtype=torch.float32, device=device)
        ea.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        ev.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bc1 = 1.0 - beta1**s
        bc2_sqrt = math.sqrt(1.0 - beta2**s)
        denom = ev.sqrt().div(bc2_sqrt).add(1e-8)
        w_ref32.addcdiv_(ea, denom, value=-(1e-3 / bc1))
    w_nearest = w_ref32.to(torch.bfloat16)
    if torch.equal(w_local, w_nearest):
        issues.append("SR weight identical to nearest reference — SR did not fire; bit-identity vacuous")

    gathered = [torch.empty_like(w_local) for _ in range(world_size)]
    dist.all_gather(gathered, w_local)
    for other_rank, g in enumerate(gathered):
        if not torch.equal(g, gathered[0]):
            issues.append(
                f"SR-rounded weight differs between rank 0 and rank {other_rank} — "
                "per-rank SR seeds diverged (replicas would drift)"
            )
            break

    # So the test fails if any rank's checks failed.
    local_ok = torch.tensor([1 if not issues else 0], device=device)
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    all_ok = bool(local_ok.item())

    if issues:
        for issue in issues:
            log(f"  FAIL [rank {rank}]: {issue}")
    if not all_ok:
        log("✗ SR-under-EP Test FAILED")
        return False

    log("  Expert-shard exp_avg_sq unbiased vs fp32 (nearest control biased) ✓")
    log("  SR-rounded weight bit-identical across the EP=8 replicate group ✓")
    log("✓ SR-under-EP Test PASSED")
    return True


@gpu_test_main(exact_world_size=EP_SIZE, prefix="bf16opt_ep")
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "full", "lora", "sr"], help="Which test to run"
    )
    args = parser.parse_args()

    log(f"GPU: {torch.cuda.get_device_name(ctx.device)}")
    log(f"World size: {ctx.world_size}")
    log(f"Model: {MODEL_PATH}")

    checks: dict[str, bool] = {}
    if args.mode in ("all", "full"):
        checks["full"] = test_full_training(os.path.join(ctx.output_dir, "full"))
        cleanup_memory()
    if args.mode in ("all", "lora"):
        checks["lora"] = test_lora_training(os.path.join(ctx.output_dir, "lora"))
        cleanup_memory()
    if args.mode in ("all", "sr"):
        checks["sr"] = test_sr_under_ep()
    return {"checks": checks}


if __name__ == "__main__":
    run()
