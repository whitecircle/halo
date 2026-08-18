#!/usr/bin/env python
"""
Test: LoRA and QLoRA with Context Parallelism (CP=2) on a dense model (Qwen3-0.6B).

Validates that PEFT LoRA/QLoRA adapters work correctly with Context Parallelism
(Ulysses attention) on a dense (non-MoE) model.

CP splits sequences across GPUs via Ulysses attention, requiring that LoRA
adapters on attention projections produce correct gradients when operating on
partial sequences per rank.

Tests:
  1. LoRA + CP=2 -> train 5 steps -> save -> verify checkpoint -> reload adapter
  2. QLoRA + CP=2 -> train 5 steps -> save -> verify checkpoint -> reload adapter
  3. Resume: fresh CP+LoRA model -> peft.restore_adapters from test 1's save ->
     every restored adapter tensor equals the saved one (the CP key-remap path)

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_cp_dense.py
"""

import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

from src.distributed.checkpoint.peft import PeftAdapterSaver, restore_adapters
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import adapter_save_checks, snapshot_adapters
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

# Default to Qwen3-0.6B for fast CI; override to a larger dense model (e.g.
# Qwen3-4B) via HALO_TEST_LORA_CP_MODEL=Qwen/Qwen3-4B-Instruct-2507 for scale checks.
MODEL_NAME = env_str("HALO_TEST_LORA_CP_MODEL", QWEN3_0_6B)
CP_SIZE = 2
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_R = 8
LORA_ALPHA = 16


# Helpers


def _verify_lora_updated(before: dict, model) -> dict[str, bool]:
    checks = {}
    after = snapshot_adapters(model, expert_lora=False)
    updated = sum(1 for name in before if name in after and not torch.equal(before[name], after[name]))
    total = len(before)
    # With few training steps, lora_A may not update initially because lora_B
    # starts at zero (so d(loss)/d(lora_A) = lora_B^T @ ... = 0 in step 1).
    # Require at least 25% of params to be updated as a reasonable threshold.
    lora_ok = updated > 0 and updated >= total * 0.25
    checks["lora_updated"] = lora_ok
    log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({updated}/{total} params changed)")
    return checks


def _validate_training(train_result, trainer, max_steps) -> tuple[dict[str, bool], list]:
    training_loss = train_result.training_loss
    step_losses = [
        entry["loss"] for entry in trainer.state.log_history if "loss" in entry and "eval_loss" not in entry
    ]

    checks = {}

    loss_finite = math.isfinite(training_loss)
    checks["loss_finite"] = loss_finite
    log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")

    all_finite = all(math.isfinite(l) for l in step_losses)
    checks["all_steps_finite"] = all_finite
    log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

    steps_ok = train_result.global_step == max_steps
    checks["steps_completed"] = steps_ok
    log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{max_steps})")

    # A band, not `< 100`: no finite LM loss on this model reaches 100, so that bound was strictly
    # weaker than the finiteness check beside it and could not fail. The floor catches a collapsed
    # or masked-away objective, the ceiling an untrained/garbage forward.
    loss_reasonable = 0.05 < training_loss < 20.0
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss in band (0.05, 20): {'PASS' if loss_reasonable else 'FAIL'} ({training_loss:.4f})")

    return checks, step_losses


def _verify_checkpoint_reload(
    save_dir: str,
    tokenizer,
    rank: int,
    local_rank: int,
    quantization_config=None,
) -> dict[str, bool]:
    """Reload saved adapter and verify it produces finite logits. Rank 0 only."""
    if rank != 0:
        return {}

    checks = {}
    try:
        log("  Reloading base model for checkpoint verification...")
        load_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": {"": local_rank},
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)

        log(f"  Loading adapter from {save_dir}...")
        reloaded = PeftModel.from_pretrained(base_model, save_dir)
        checks["adapter_reload"] = True
        log("  Adapter reload: PASS")

        reloaded.eval()
        test_input = tokenizer("What is 2 + 2?", return_tensors="pt").to(f"cuda:{local_rank}")
        with torch.no_grad():
            output = reloaded(**test_input)

        logits_finite = torch.isfinite(output.logits).all().item()
        checks["reload_logits_finite"] = logits_finite
        log(f"  Reload logits finite: {'PASS' if logits_finite else 'FAIL'}")

        del reloaded, base_model
        cleanup_memory()

    except Exception as e:
        log(f"  Checkpoint reload FAILED: {e}")
        traceback.print_exc()
        checks["adapter_reload"] = False

    return checks


# Test runners


def run_lora_cp(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + CP=2 on Qwen3-0.6B dense model."""
    model = None
    trainer = None

    try:
        log(f"  Loading model with CP={CP_SIZE}...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=False,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e6:.1f}M params, GPU: {gpu_mem_gb():.2f} GB")

        # Apply LoRA
        log(f"  Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M ({100 * trainable / total_params:.2f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        # Create trainer
        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_cp_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        # Train
        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        # Validate
        log("\n  --- Validation (LoRA+CP) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        lora_checks = _verify_lora_updated(lora_before, model)
        checks.update(lora_checks)

        # Save checkpoint
        save_dir = os.path.join(base_output_dir, "lora_cp_save")
        log("\n  --- Checkpoint Save (LoRA+CP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)

        # Reload and verify on rank 0
        reload_checks = _verify_checkpoint_reload(save_dir, tokenizer, rank, local_rank)
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        detail = f"loss={train_result.training_loss:.6f}"
        return all_passed, detail

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        del trainer
        del model
        cleanup_memory()
        barrier()


def run_qlora_cp(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run QLoRA + CP=2 on Qwen3-0.6B dense model."""
    model = None
    trainer = None

    try:
        # Quantization config
        model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            use_bnb_nested_quant=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        quantization_config = get_quantization_config(model_config)
        log(f"  Quantization: {quantization_config.quant_method}")

        # Load model with CP + quantization
        log(f"  Loading model with CP={CP_SIZE} + 4-bit quantization...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=False,
            quantization_config=quantization_config,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e6:.1f}M params, GPU: {gpu_mem_gb():.2f} GB")

        # Apply LoRA
        log(f"  Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M ({100 * trainable / total_params:.2f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        # Create trainer
        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "qlora_cp_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        # Train
        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        # Validate
        log("\n  --- Validation (QLoRA+CP) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        lora_checks = _verify_lora_updated(lora_before, model)
        checks.update(lora_checks)

        # Save checkpoint
        save_dir = os.path.join(base_output_dir, "qlora_cp_save")
        log("\n  --- Checkpoint Save (QLoRA+CP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)

        # Reload and verify on rank 0
        reload_checks = _verify_checkpoint_reload(
            save_dir,
            tokenizer,
            rank,
            local_rank,
            quantization_config=quantization_config,
        )
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        detail = f"loss={train_result.training_loss:.6f}"
        return all_passed, detail

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        del trainer
        del model
        cleanup_memory()
        barrier()


def run_lora_cp_resume(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Save -> construct-fresh -> resume for LoRA + CP=2: the CP key-remap path.

    Builds a FRESH CP-wrapped LoRA model (zero-init adapters), then drives the real resume seam —
    ``peft.restore_adapters``, which remaps the CP-normalized saved keys onto the live
    CP-wrapped names — against the adapter checkpoint test 1 wrote. Without the remap every saved key
    reads as unexpected and the loader's "CP-remap regression" guard raises; with it, every restored
    adapter tensor must equal the saved one.
    """
    save_dir = os.path.join(base_output_dir, "lora_cp_save")
    model = None
    trainer = None

    try:
        log(f"  Loading FRESH model with CP={CP_SIZE} for resume...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=False,
        )
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_cp_resume"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        checks = {}
        before = snapshot_adapters(trainer.model, expert_lora=False)

        log("  Restoring adapters via peft.restore_adapters (collective, all ranks)...")
        restore_adapters(save_dir, trainer.model, is_cp_mode=True)

        after = snapshot_adapters(trainer.model, expert_lora=False)
        saved = load_file(os.path.join(save_dir, "adapter_model.safetensors"))

        # Live -> saved (portable) key spelling, derived independently of the remap under test.
        live_to_saved = {
            live: PeftAdapterSaver._normalize_cp_adapter_key(live.replace(".default.", ".")) for live in after
        }
        matched = {live: key for live, key in live_to_saved.items() if key in saved}
        coverage_ok = len(matched) == len(after) == len(saved) > 0
        checks["saved_keys_cover_live"] = coverage_ok
        log(
            f"  Saved keys cover live adapters: {'PASS' if coverage_ok else 'FAIL'} "
            f"({len(matched)} matched / {len(after)} live / {len(saved)} saved)"
        )

        equal = sum(1 for live, key in matched.items() if torch.equal(after[live].to(saved[key].dtype), saved[key]))
        restored_ok = coverage_ok and equal == len(matched)
        checks["restored_equal_saved"] = restored_ok
        log(f"  Restored tensors equal saved: {'PASS' if restored_ok else 'FAIL'} ({equal}/{len(matched)})")

        # Anti-vacuity: the restore actually overwrote the fresh zero-init adapters.
        changed = sum(1 for name in before if not torch.equal(before[name], after[name]))
        checks["restore_changed_fresh_weights"] = changed > 0
        log(f"  Restore changed fresh adapters: {'PASS' if changed > 0 else 'FAIL'} ({changed}/{len(before)})")

        all_passed = all(checks.values())
        return all_passed, f"{equal}/{len(saved)} adapter tensors restored"

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        del trainer
        del model
        cleanup_memory()
        barrier()


# Main


def main() -> int:
    """Run LoRA/QLoRA + CP tests on Qwen3-0.6B dense model. Returns 0 on success, 1 on failure."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    state = PartialState()
    rank = state.process_index
    local_rank = state.local_process_index
    world_size = state.num_processes

    base_output_dir, cache_dir = setup_cache_dirs("test_lora_cp_dense", rank)
    # setup_cache_dirs mkdtemps a DIFFERENT random dir per rank; saves and the resume leg's
    # cross-rank file consensus need one shared path — use rank 0's everywhere.
    dirs = [base_output_dir]
    dist.broadcast_object_list(dirs, src=0)
    base_output_dir = dirs[0]

    log(f"\n{'#' * 70}")
    log("  LoRA/QLoRA + CP Test on Dense Model")
    log(f"  World size: {world_size}, CP size: {CP_SIZE}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    if world_size != CP_SIZE:
        log(f"\nERROR: This test requires exactly {CP_SIZE} GPUs, got {world_size}")
        if dist.is_initialized():
            teardown_distributed()
        return 1

    ensure_model_downloaded(MODEL_NAME, rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    parallelism_config = ParallelismConfig(cp_size=CP_SIZE)
    results: dict[str, tuple[bool, str]] = {}

    # ── Test 1: LoRA + CP=2 ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 1: LoRA + CP=2 (Qwen3-0.6B Dense)")
    log(f"{'=' * 70}")

    success, detail = run_lora_cp(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["lora_cp"] = (success, detail)

    # ── Test 2: QLoRA + CP=2 ────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 2: QLoRA (4-bit) + CP=2 (Qwen3-0.6B Dense)")
    log(f"{'=' * 70}")

    success, detail = run_qlora_cp(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["qlora_cp"] = (success, detail)

    # ── Test 3: LoRA + CP=2 adapter resume ──────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 3: LoRA + CP=2 adapter resume (save -> fresh model -> restore)")
    log(f"{'=' * 70}")

    success, detail = run_lora_cp_resume(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        base_output_dir,
    )
    results["lora_cp_resume"] = (success, detail)

    # ── Summary ──────────────────────────────────────────────────────────
    log(f"\n{'#' * 70}")
    log("  FINAL RESULTS")
    log(f"{'#' * 70}")
    for name, (passed, detail) in results.items():
        status = "PASSED" if passed else "FAILED"
        log(f"  {name:20s} {status} -- {detail}")
    log(f"{'#' * 70}")

    all_passed = all(p for p, _ in results.values())
    log(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    cleanup_dirs(base_output_dir, cache_dir)
    if dist.is_initialized():
        teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
