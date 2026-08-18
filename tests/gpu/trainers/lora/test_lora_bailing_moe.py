#!/usr/bin/env python
"""
Test: LoRA adapters with EP and ETP on Bailing MoE (Ring-mini-linear-2.0, ~16B).

Validates that PEFT LoRA adapters work correctly with Expert Parallelism and
Expert Tensor Parallelism on Bailing/Ling MoE model, including checkpoint
save and reload verification.

Tests:
  1. LoRA + EP=2 -> train 5 steps -> save -> verify checkpoint -> reload adapter
  2. LoRA + ETP (expert_tp_size=2) -> train 5 steps -> save -> verify checkpoint -> reload

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_bailing_moe.py [--mode ep|etp|all]

Requirements:
    - 2x GPUs with >=40GB memory each
    - DeepEP installed
    - flash-linear-attention installed (for linear attention layers)
    - Model: inclusionAI/Ring-mini-linear-2.0 (auto-downloaded)
"""

import argparse
import glob
import math
import os
import traceback

import torch

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

apply_remote_code_compat_shims()

from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import BAILING_MOE_RING_MINI
from tests.common.peft_helpers import adapter_save_checks, snapshot_adapters
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = BAILING_MOE_RING_MINI
MAX_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 1e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42

# Bailing fuses QKV; attention only, since expert layers are distributed across ranks
LORA_TARGET_MODULES = ["query_key_value", "dense", "g_proj"]
LORA_R = 8
LORA_ALPHA = 16


def _verify_checkpoint_reload(
    save_dir: str,
    tokenizer,
    rank: int,
    local_rank: int,
    trained_lora: dict[str, torch.Tensor],
) -> dict[str, bool]:
    if rank != 0:
        return {}
    checks = {}
    try:
        log("  Reloading base model for checkpoint verification...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": local_rank},
        )
        # A bare load leaves this family's non-persistent buffers (Lightning-Attention ``slope``,
        # rotary ``inv_freq``) on transformers-5's uninitialized memory, so the reload control runs the
        # seam every toolkit load path runs — without it it measures the allocator, not the checkpoint.
        finalize_loaded_model(base_model)
        log(f"  Loading adapter from {save_dir}...")
        reloaded = PeftModel.from_pretrained(base_model, save_dir)

        # "from_pretrained did not raise" is vacuous: lora_B is zero-init, so an all-zero adapter
        # loads cleanly and gives finite logits — the trained values must be compared to what returned.
        restored = {
            n.replace(".default.", "."): p.detach().cpu() for n, p in reloaded.named_parameters() if "lora_" in n
        }
        matched = [k for k in trained_lora if k.replace(".default.", ".") in restored]
        checks["adapter_reload"] = bool(matched) and all(
            torch.allclose(trained_lora[k].float(), restored[k.replace(".default.", ".")].float(), atol=1e-6)
            for k in matched
        )
        log(f"  Adapter reload: {'PASS' if checks['adapter_reload'] else 'FAIL'} ({len(matched)} tensors compared)")

        reloaded.eval()
        test_input = tokenizer("What is 2 + 2?", return_tensors="pt").to(f"cuda:{local_rank}")
        with torch.no_grad():
            output = reloaded(**test_input)
        logits_finite = torch.isfinite(output.logits).all().item()
        checks["reload_logits_finite"] = logits_finite
        log(f"  Reload logits finite: {'PASS' if logits_finite else 'FAIL'}")

        # Finiteness catches the reused-page reading of an unrepaired buffer but not the zeroed one:
        # zeros are finite, and the model they build has a dead RoPE and no decay. Swept off the
        # model's own non-persistent set, so it follows the load rather than this family's two names.
        unset = [
            name
            for name, tensor in base_model.named_non_persistent_buffers()
            if tensor.is_floating_point()
            and tensor.numel()
            and not (torch.isfinite(tensor).all().item() and tensor.abs().amax().item() > 0)
        ]
        checks["reload_buffers_initialized"] = not unset
        log(f"  Reload buffers initialized: {'PASS' if not unset else f'FAIL {unset[:5]}'}")
        if not logits_finite:
            # separates a bad EP gather (non-finite saved tensor) from a bad reloaded forward
            for shard in sorted(glob.glob(os.path.join(save_dir, "*.safetensors"))):
                bad = {k: v for k, v in load_file(shard).items() if not torch.isfinite(v).all()}
                log(f"  [diag] {os.path.basename(shard)}: {len(bad)} non-finite of {len(load_file(shard))} tensors")
                for key, val in list(bad.items())[:10]:
                    log(f"  [diag]   {key} shape={tuple(val.shape)} nan={val.isnan().sum().item()}")
            log(f"  [diag] logits nan={output.logits.isnan().sum().item()} inf={output.logits.isinf().sum().item()}")

        del reloaded, base_model
        cleanup_memory()
    except Exception as e:
        log(f"  Checkpoint reload FAILED: {e}")
        traceback.print_exc()
        checks["adapter_reload"] = False
    return checks


def run_lora_ep(
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + EP=2 on Bailing MoE."""
    model = None
    trainer = None

    try:
        parallelism_config = ParallelismConfig(ep_size=2)
        log(f"  Config: {parallelism_config.mode_string}")

        log("  Loading model with EP=2...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e9:.2f}B params, GPU: {gpu_mem_gb():.2f} GB")

        log(f"  Applying LoRA (r={LORA_R}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total_params:.3f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_ep_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
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
        log(f"  EP mode: {trainer.is_ep_mode}")

        log(f"  Training for {MAX_STEPS} steps...")
        barrier()
        train_result = trainer.train()

        checks = {}
        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        updated = sum(1 for n in lora_before if n in lora_after and not torch.equal(lora_before[n], lora_after[n]))
        # ALL must move: `updated > 0` also passes a partially-wired adapter
        lora_ok = updated == len(lora_before) and updated > 0
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({updated}/{len(lora_before)})")

        save_dir = os.path.join(base_output_dir, "lora_ep_save")
        log("\n  --- Checkpoint Save (LoRA+EP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = _verify_checkpoint_reload(save_dir, tokenizer, rank, local_rank, lora_after)
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        return all_passed, f"loss={training_loss:.6f}"

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        barrier()


def run_lora_etp(
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + ETP (expert_tp_size=2) on Bailing MoE."""
    model = None
    trainer = None

    try:
        parallelism_config = ParallelismConfig(ep_size=1, expert_tp_size=2)
        log(f"  Config: {parallelism_config.mode_string}")

        log("  Loading model with ETP...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e9:.2f}B params, GPU: {gpu_mem_gb():.2f} GB")

        log(f"  Applying LoRA (r={LORA_R}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total_params:.3f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_etp_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
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

        log(f"  Training for {MAX_STEPS} steps...")
        barrier()
        train_result = trainer.train()

        checks = {}
        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        updated = sum(1 for n in lora_before if n in lora_after and not torch.equal(lora_before[n], lora_after[n]))
        # ALL must move: `updated > 0` also passes a partially-wired adapter
        lora_ok = updated == len(lora_before) and updated > 0
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({updated}/{len(lora_before)})")

        save_dir = os.path.join(base_output_dir, "lora_etp_save")
        log("\n  --- Checkpoint Save (LoRA+ETP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = _verify_checkpoint_reload(save_dir, tokenizer, rank, local_rank, lora_after)
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        return all_passed, f"loss={training_loss:.6f}"

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        barrier()


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ep", "etp", "all"], default="all")
    args, _ = parser.parse_known_args()

    log(f"\n{'#' * 70}")
    log("  LoRA + Bailing MoE (EP / ETP) Test")
    log(f"  World size: {ctx.world_size}, Mode: {args.mode}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)

    results: dict[str, tuple[bool, str]] = {}

    if args.mode in ("ep", "all"):
        log(f"\n{'=' * 70}")
        log("  TEST 1: LoRA + EP=2 (Bailing MoE)")
        log(f"{'=' * 70}")
        success, detail = run_lora_ep(
            tokenizer,
            train_dataset,
            eval_dataset,
            ctx.rank,
            ctx.local_rank,
            ctx.output_dir,
        )
        results["lora_ep"] = (success, detail)

    if args.mode in ("etp", "all"):
        log(f"\n{'=' * 70}")
        log("  TEST 2: LoRA + ETP (expert_tp_size=2) (Bailing MoE)")
        log(f"{'=' * 70}")
        success, detail = run_lora_etp(
            tokenizer,
            train_dataset,
            eval_dataset,
            ctx.rank,
            ctx.local_rank,
            ctx.output_dir,
        )
        results["lora_etp"] = (success, detail)

    for name, (passed, detail) in results.items():
        log(f"  {name:20s} {'PASSED' if passed else 'FAILED'} -- {detail}")

    return {"checks": {name: passed for name, (passed, _) in results.items()}}


main = gpu_test_main(exact_world_size=2, prefix="test_lora_bailing_moe")(run)

if __name__ == "__main__":
    main()
