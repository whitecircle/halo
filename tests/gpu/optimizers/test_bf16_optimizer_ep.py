#!/usr/bin/env python
"""
Test: AdamWBF16 optimizer with EP=8 on GPT-OSS-20B MoE model.

Verifies that bf16_optimizer works end-to-end with Expert Parallelism:
1. Full training (no LoRA): bf16_optimizer=True, loss decreases, states in bf16
2. LoRA + EP training: bf16_optimizer=True with attention-only LoRA

Usage (8 GPUs required):
    torchrun --nproc_per_node=8 \
        tests/gpu/optimizers/test_bf16_optimizer_ep.py

    # LoRA only:
    torchrun --nproc_per_node=8 \
        tests/gpu/optimizers/test_bf16_optimizer_ep.py --mode lora

    # Full training only:
    torchrun --nproc_per_node=8 \
        tests/gpu/optimizers/test_bf16_optimizer_ep.py --mode full
"""

import argparse
import math
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.loading import load_ep_model
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing
from src.distributed.parallelism_config import ParallelismConfig
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, init_distributed, teardown_distributed
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = GPT_OSS_20B
EP_SIZE = 8
NUM_TRAIN_SAMPLES = 16
MAX_SEQ_LENGTH = 64
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 2


# ── Test 1: Full Training with bf16_optimizer ─────────────────────────────────


def test_full_training():
    """Test bf16_optimizer with EP=8, no LoRA (full fine-tuning)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"\n{'=' * 70}")
    log(f"TEST 1: Full Training with bf16_optimizer (EP={EP_SIZE})")
    log(f"{'=' * 70}")

    if world_size != EP_SIZE:
        log(f"ERROR: Requires {EP_SIZE} GPUs, got {world_size}")
        return False

    output_dir = tempfile.mkdtemp(prefix="bf16opt_ep_full_")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer)

        # Load model with EP
        log("Loading model with EP...")
        ep_config = EPConfig(ep_size=EP_SIZE)
        model = load_ep_model(
            MODEL_PATH,
            ep_config,
            AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=None,  # auto-detect (FA4 on Blackwell handles gpt-oss sinks; FA2 forces the Hopper-only S-aux path)
        )

        total_params = sum(p.numel() for p in model.parameters())
        bf16_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
        fp32_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
        log(f"Total params: {total_params / 1e6:.1f}M  (bf16={bf16_params / 1e6:.1f}M, fp32={fp32_params / 1e6:.1f}M)")

        # Enable gradient checkpointing with EP-safe dispatch caching
        log("Enabling gradient checkpointing...")
        enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})

        # Create config
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=1e-5,
            max_length=MAX_SEQ_LENGTH,
            bf16=True,
            gradient_checkpointing=False,  # Already enabled above
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataset_num_proc=1,
            dataloader_num_workers=0,
        )

        # Create trainer - bf16_optimizer auto-enabled from config.bf16=True
        log("Creating trainer (bf16_optimizer auto-enabled)...")
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        # Verify optimizer type
        log("Starting training...")
        dist.barrier()
        train_result = trainer.train()

        # ── Verification ──────────────────────────────────────────────
        issues = []

        # 1. Verify optimizer is AdamWBF16
        if not isinstance(trainer.optimizer, AdamWBF16):
            issues.append(f"Optimizer is {type(trainer.optimizer).__name__}, expected AdamWBF16")

        # 2. Verify optimizer states are bf16 — counted, so "no param matched the guard" cannot
        # silently assert nothing.
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

        # 3. Verify model params stayed bf16. This config enables no fp32 knob (fp32_router /
        # fp32_experts / fp32_non_ep_params — the documented exceptions), so every param must be bf16.
        bf16_numel = sum(p.numel() for p in model.parameters() if p.dtype == torch.bfloat16)
        fp32_named = [n for n, p in model.named_parameters() if p.dtype == torch.float32]
        if bf16_numel == 0:
            issues.append("no bf16 parameters after training — model dtype regressed")
        if fp32_named:
            issues.append(
                f"{len(fp32_named)} unexpected fp32 param(s) with no fp32 knob enabled, e.g. {fp32_named[:3]}"
            )

        # 4. Verify training loss is valid
        loss = train_result.training_loss
        log(f"Training loss: {loss:.4f}")
        if loss != loss:  # NaN check
            issues.append("Training loss is NaN")
        if loss > 100:
            issues.append(f"Training loss too high: {loss}")

        # 5. Log memory stats
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

    except Exception as e:
        log(f"\u2717 TEST FAILED: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        if "trainer" in locals():
            trainer.cleanup_ep()
        dist.barrier()
        cleanup_dirs(output_dir)
        cleanup_memory()


# ── Test 2: LoRA + EP with bf16_optimizer ─────────────────────────────────────


def test_lora_training():
    """Test bf16_optimizer with EP=8 + LoRA (attention-only)."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"\n{'=' * 70}")
    log(f"TEST 2: LoRA Training with bf16_optimizer (EP={EP_SIZE})")
    log(f"{'=' * 70}")

    if world_size != EP_SIZE:
        log(f"ERROR: Requires {EP_SIZE} GPUs, got {world_size}")
        return False

    output_dir = tempfile.mkdtemp(prefix="bf16opt_ep_lora_")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer)

        # Load model with EP
        log("Loading model with EP...")
        ep_config = EPConfig(ep_size=EP_SIZE)
        model = load_ep_model(
            MODEL_PATH,
            ep_config,
            AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=None,  # auto-detect (FA4 on Blackwell handles gpt-oss sinks; FA2 forces the Hopper-only S-aux path)
        )

        # Apply LoRA (attention-only, not MoE experts)
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

        # Ensure uniform dtype for FSDP
        for name, param in model.named_parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        # Enable gradient checkpointing with EP-safe dispatch caching
        log("Enabling gradient checkpointing...")
        enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})

        # Create config
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=1e-4,  # Higher LR for LoRA
            max_length=MAX_SEQ_LENGTH,
            bf16=True,
            gradient_checkpointing=False,  # Already enabled above
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataset_num_proc=1,
            dataloader_num_workers=0,
        )

        # Create trainer - bf16_optimizer auto-enabled from config.bf16=True
        log("Creating trainer (bf16_optimizer auto-enabled) + LoRA...")
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        # Train
        log("Starting training...")
        dist.barrier()
        train_result = trainer.train()

        # ── Verification ──────────────────────────────────────────────
        issues = []

        # 1. Verify optimizer is AdamWBF16
        if not isinstance(trainer.optimizer, AdamWBF16):
            issues.append(f"Optimizer is {type(trainer.optimizer).__name__}, expected AdamWBF16")

        # 2. Verify LoRA params are trainable and have optimizer states — counted, so the state
        # guard cannot silently skip every param.
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

        # 3. Verify training loss is valid
        loss = train_result.training_loss
        log(f"Training loss: {loss:.4f}")
        if loss != loss:
            issues.append("Training loss is NaN")
        if loss > 100:
            issues.append(f"Training loss too high: {loss}")

        # 4. Log memory stats
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

    except Exception as e:
        log(f"\u2717 TEST FAILED: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        if "trainer" in locals():
            trainer.cleanup_ep()
        dist.barrier()
        cleanup_dirs(output_dir)
        cleanup_memory()


# ── Test 3: SR correctness under EP (de-bias on a sharded expert + replica seeds) ──


def test_sr_under_ep():
    """SR de-biasing and per-rank-identical seeds, validated on an EP topology.

    Two load-bearing SR claims that the dtype/loss<100 checks above do not touch:

      A. UNBIASEDNESS on a sharded expert param: each rank owns a DIFFERENT expert's local
         FFN shard under EP=8. We drive a fixed tiny gradient on a bf16 param shaped like one
         such local expert shard through AdamWBF16 and assert its exp_avg_sq tracks an fp32
         single-rank reference EMA (rel-err < 5%), while the nearest-rounded control overshoots
         by a large single-signed margin (premise guard > 15%). This is exactly
         test_sr_removes_second_moment_bias, now exercised on each rank's expert shard.

      B. REPLICA SEED IDENTITY: EP is orthogonal to DP, so under EP=8 on 8 GPUs all 8 ranks
         form one data-parallel replicate group for replicated (non-expert) params. The SR RNG
         (_SR_RNG) is seeded identically on every rank precisely so SR rounds identically across
         replicas — otherwise replicas would silently drift apart. We run an identical
         AdamWBF16 step on identical inputs on every rank, all-gather the SR-rounded weight, and
         assert it is BIT-IDENTICAL across the whole replicate group. A negative control inside
         the run (weight differs from a nearest-rounded reference) proves SR actually fired.

    This is a standalone numeric check (not routed through the trainer's sharded expert tensors,
    which are not addressable as plain bf16 params for a fixed-point statistical test). The
    expert-shard shape and the EP=8 replicate group are real; the gradient is synthetic.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"\n{'=' * 70}")
    log(f"TEST 3: SR correctness under EP (de-bias + replica seeds, world={world_size})")
    log(f"{'=' * 70}")

    if world_size != EP_SIZE:
        log(f"ERROR: Requires {EP_SIZE} GPUs, got {world_size}")
        return False

    try:
        import torch.nn as nn

        device = torch.device("cuda", torch.cuda.current_device())
        beta1, beta2 = 0.9, 0.999
        grad_val = 1e-4
        n_steps = 500
        # Shape of one expert's local FFN shard slice (gate/up column block); small but realistic.
        size = 8192

        # ── A. Unbiasedness on this rank's expert shard ───────────────────────────
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
        opt = AdamWBF16([p], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)
        for _ in range(n_steps):
            p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16, device=device)
            opt.step()
        sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

        log(f"[rank {rank}] expert-shard easq: fp32={fp32_mean:.3e} nearest={nearest_mean:.3e} SR={sr_mean:.3e}")

        issues = []
        nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
        if nearest_bias <= 0.15:
            issues.append(f"premise broken: nearest bias only {nearest_bias:.1%}")
        sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
        if sr_rel_err >= 0.05:
            issues.append(f"SR easq on expert shard should track fp32, rel-err {sr_rel_err:.1%}")
        if not abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean):
            issues.append("SR easq on expert shard must be closer to fp32 than to nearest reference")

        # ── B. Replica seed identity across the EP=8 data-parallel group ──────────
        # Identical fixed input + state on every rank; SR-rounded weight must match bit-for-bit.
        torch.manual_seed(0)  # identical grad/init across ranks (no per-rank randomness)
        pr = nn.Parameter(torch.full((size,), 0.5, dtype=torch.bfloat16, device=device))
        opt_r = AdamWBF16([pr], lr=1e-3, betas=(beta1, beta2), weight_decay=0.0)
        # A small SIGNED gradient whose Adam step lands below the bf16 ULP of 0.5 (~0.004),
        # so SR is what carries the update — makes the rounding non-trivial.
        for _ in range(8):
            pr.grad = torch.full((size,), 1e-3, dtype=torch.bfloat16, device=device)
            opt_r.step()
        w_local = pr.data.clone()

        # Negative control: an identical NEAREST-rounded reference run must differ from the SR
        # weight, proving SR actually altered the write (otherwise bit-identity is vacuous).
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

        # All-gather the SR weight across the (EP=8) replicate group and require bit-identity.
        gathered = [torch.empty_like(w_local) for _ in range(world_size)]
        dist.all_gather(gathered, w_local)
        for other_rank, g in enumerate(gathered):
            if not torch.equal(g, gathered[0]):
                issues.append(
                    f"SR-rounded weight differs between rank 0 and rank {other_rank} — "
                    "per-rank _SR_RNG seeds diverged (replicas would drift)"
                )
                break

        # Reduce pass/fail across ranks so the test fails if ANY rank's checks failed.
        local_ok = torch.tensor([1 if not issues else 0], device=device)
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        all_ok = bool(local_ok.item())

        if issues:
            for issue in issues:
                log(f"  FAIL [rank {rank}]: {issue}")
        if not all_ok:
            log("✗ SR-under-EP Test FAILED")
            return False

        log("  Expert-shard exp_avg_sq unbiased vs fp32 (nearest control overshoots) ✓")
        log("  SR-rounded weight bit-identical across the EP=8 replicate group ✓")
        log("✓ SR-under-EP Test PASSED")
        return True

    except Exception as e:
        log(f"✗ TEST FAILED: {e}")
        if rank == 0:
            traceback.print_exc()
        return False
    finally:
        dist.barrier()
        cleanup_memory()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--mode", type=str, default="all", choices=["all", "full", "lora", "sr"], help="Which test to run"
    )
    parser.add_argument("--steps", type=int, default=NUM_TRAIN_STEPS)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()  # accelerate's logging utility requires the state to be initialized first

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"World size: {world_size}")
    log(f"Model: {MODEL_PATH}")

    results = {}

    if args.mode in ("all", "full"):
        results["full"] = test_full_training()

    if args.mode in ("all", "lora"):
        results["lora"] = test_lora_training()

    if args.mode in ("all", "sr"):
        results["sr"] = test_sr_under_ep()

    # Summary
    log(f"\n{'=' * 70}")
    log("SUMMARY")
    log(f"{'=' * 70}")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        log(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        log("\n\u2713 ALL TESTS PASSED")
    else:
        log("\n\u2717 SOME TESTS FAILED")

    teardown_distributed()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
