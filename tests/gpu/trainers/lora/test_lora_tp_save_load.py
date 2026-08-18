#!/usr/bin/env python
"""
Test: LoRA adapter save/load across parallelism modes (+ TP rejection).

Validates that PEFT LoRA adapters save/reload correctly on the SUPPORTED modes, and that the
UNSUPPORTED LoRA+TP combination is rejected at trainer construction. Three sub-tests:

  A. Qwen3-0.6B (dense) with TP=2  — LoRA+TP must be REJECTED (adapters are not integrated into
                                      the TP DTensor graph → rank-inconsistent adapter). Asserts
                                      DistributedSFTTrainer raises ValueError; no training.
  C. GptOss-20B (MoE) with EP=2    — EP-only (attention LoRA replicated): train, save, reload.
  D. Qwen3-0.6B (dense) with FSDP2 — FSDP2-wrapped LoRA params: train, save, reload.

(Labels skip B: an EP+TP sub-test would hit the identical LoRA+TP guard sub-test A covers, at the
cost of a 20B model load.)

Supported sub-tests (C, D):
  1. Load model, apply LoRA (r=8, alpha=16) to q_proj + v_proj
  2. Train 5 steps with save_strategy="steps" at step 5
  3. Verify checkpoint files exist + adapter weights match gathered live weights + key names are
     standard PEFT format (no parallelism artifacts)

Usage:
    # Individual modes (2 GPUs)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_tp_save_load.py --mode qwen3_tp

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_tp_save_load.py --mode gptoss_ep

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_tp_save_load.py --mode qwen3_fsdp

    # All (default)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_tp_save_load.py --mode all
"""

import argparse
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

# Dense model for the Qwen3 sub-tests (A: TP=2, D: FSDP2). Defaults to
# Qwen3-0.6B for fast CI; override to a larger model (e.g. Qwen3-4B) via
# HALO_TEST_LORA_SAVE_LOAD_MODEL=Qwen/Qwen3-4B-Instruct-2507 for scale checks.
QWEN3_MODEL = env_str("HALO_TEST_LORA_SAVE_LOAD_MODEL", QWEN3_0_6B)

MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


# Helpers


def create_lora_config() -> LoraConfig:
    """Create a standard LoRA configuration for testing."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def snapshot_lora_weights(model) -> dict:
    """Capture a snapshot of all LoRA adapter weights (gathering DTensors)."""
    snapshot = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            data = param.data
            if hasattr(data, "full_tensor"):
                data = data.full_tensor()
            snapshot[name] = data.clone().cpu()
    return snapshot


def verify_lora_updated(before: dict, after: dict) -> tuple:
    """Check that LoRA weights changed during training."""
    if not before or not after:
        return False, "No LoRA parameters found"

    updated = sum(1 for name in before if name in after and not torch.equal(before[name], after[name]))
    total = len(before)
    unchanged = total - updated
    return updated > 0 and unchanged == 0, f"{updated}/{total} LoRA params updated"


def verify_saved_adapters(checkpoint_dir: str, live_weights: dict, rank: int) -> tuple:
    """Verify saved adapter files exist and weights match live model.

    Returns (passed, details_str).
    """
    checks = {}
    details = []

    # Check 1: adapter_config.json exists
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    checks["adapter_config"] = os.path.isfile(config_path)
    details.append(f"  adapter_config.json: {'FOUND' if checks['adapter_config'] else 'MISSING'}")

    # Check 2: adapter_model.safetensors exists
    safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    has_safetensors = os.path.isfile(safetensors_path)
    has_bin = os.path.isfile(bin_path)
    checks["adapter_weights"] = has_safetensors or has_bin
    details.append(f"  adapter_model: {'safetensors' if has_safetensors else 'bin' if has_bin else 'MISSING'}")

    if not checks["adapter_weights"]:
        return False, "\n".join(details)

    # Check 3: Load saved weights and compare to live gathered weights
    if has_safetensors:
        saved_dict = safetensors_load_file(safetensors_path, device="cpu")
    else:
        saved_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    details.append(f"  Saved keys: {len(saved_dict)}")

    # Check 4: Key names should be clean PEFT format (no DTensor/EP artifacts)
    bad_keys = [k for k in saved_dict if "DTensor" in k or "ep_" in k or "full_tensor" in k]
    checks["clean_keys"] = len(bad_keys) == 0
    if bad_keys:
        details.append(f"  BAD keys (parallelism artifacts): {bad_keys[:5]}")
    else:
        details.append("  Key format: clean PEFT format")

    # Check 5: All keys should have standard LoRA structure
    # Expected pattern: model.layers.N.self_attn.{q_proj,v_proj}.lora_{A,B}.weight
    lora_keys = [k for k in saved_dict if "lora_" in k]
    checks["has_lora_keys"] = len(lora_keys) > 0
    details.append(f"  LoRA keys: {len(lora_keys)}")
    if lora_keys:
        details.append(f"  Sample key: {lora_keys[0]}")

    # Check 6: Compare saved weights to live gathered weights
    # Build mapping from PEFT save keys to live parameter names.
    # Live keys: base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
    # Saved keys: model.layers.N.self_attn.q_proj.lora_A.weight
    match_count = 0
    mismatch_count = 0
    missing_count = 0

    # Build a normalized lookup from live weights
    # Live keys: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    # Saved keys: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # The only difference is '.default.' (PEFT adapter name) in live keys
    def normalize_key(key: str) -> str:
        """Strip '.default.' adapter name from PEFT keys for comparison."""
        return key.replace(".default.", ".")

    live_by_normalized = {normalize_key(k): (k, v) for k, v in live_weights.items()}

    for saved_key, saved_tensor in saved_dict.items():
        normalized_saved = normalize_key(saved_key)
        if normalized_saved in live_by_normalized:
            _, live_tensor = live_by_normalized[normalized_saved]
            if torch.allclose(saved_tensor.float(), live_tensor.float(), atol=1e-6):
                match_count += 1
            else:
                mismatch_count += 1
                max_diff = (saved_tensor.float() - live_tensor.float()).abs().max().item()
                details.append(f"  MISMATCH: {saved_key} (max_diff={max_diff:.6e})")
        else:
            missing_count += 1
            if missing_count <= 3:
                details.append(f"  NOT MATCHED: {saved_key}")

    checks["weights_match"] = match_count > 0 and mismatch_count == 0
    details.append(
        f"  Weight comparison: {match_count} matched, {mismatch_count} mismatched, {missing_count} unmatched"
    )

    all_passed = all(checks.values())
    return all_passed, "\n".join(details)


# Sub-test A: Qwen3-0.6B with TP=2


def run_qwen3_tp_rejected(rank: int, local_rank: int, world_size: int) -> bool:
    """LoRA + TP=2 must be REJECTED at trainer construction.

    TP shards the attention base layers as DTensors, but PEFT adds lora_A/lora_B as plain tensors
    outside the TP graph: the replicated matrix diverges across ranks (per-rank init, never
    broadcast) and the sharded one is corrupted by the TP replicated-grad sync. The adapter would
    be rank-inconsistent and would not reload onto a non-TP model. The trainer must fail fast with
    a clear ValueError rather than train a silently-wrong adapter.

    This test FAILS if the guard is removed — the trainer would then construct successfully.
    """
    log(f"\n{'=' * 70}")
    log(f"  SUB-TEST A: {QWEN3_MODEL} — LoRA + TP=2 must be REJECTED")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("lora_tp_reject_qwen3", rank)
    model = None

    try:
        log("[A.1] Loading tokenizer + datasets...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

        log("[A.2] Loading model with TP=2 + LoRA...")
        ensure_model_downloaded(QWEN3_MODEL, rank)
        parallelism_config = ParallelismConfig(tp_size=2)
        model, _ = load_distributed_model(
            model_name_or_path=QWEN3_MODEL,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        model = get_peft_model(model, create_lora_config())

        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            bf16=True,
            use_liger_kernel=False,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            save_strategy="no",
            dataloader_num_workers=0,
            fsdp="",
        )

        # The guard fires in _setup_distributed_modes (after super().__init__), deterministically
        # on every rank — so all ranks raise together; no collective desync.
        log("[A.3] Asserting DistributedSFTTrainer(LoRA, TP=2) raises ValueError...")
        raised, err = False, ""
        try:
            DistributedSFTTrainer(
                model=model,
                args=config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                parallelism_config=parallelism_config,
            )
        except ValueError as e:
            raised, err = True, str(e)

        checks = {
            "construction_rejected": raised,
            "error_explains_tp": ("Tensor Parallelism" in err) or ("tp_size" in err),
        }
        log(f"  LoRA+TP rejected at construction: {'PASS' if checks['construction_rejected'] else 'FAIL'}")
        log(f"  Error explains TP incompatibility: {'PASS' if checks['error_explains_tp'] else 'FAIL'}")
        if raised:
            log(f"  Rejection message: {err.splitlines()[0]}")

        all_passed = all(checks.values())
        log(f"\n  Sub-test A (Qwen3 LoRA+TP rejection): {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    except Exception as e:
        log(f"\n  Sub-test A FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        del model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)


# Sub-test C: GptOss-20B with EP=2 (no TP)


def run_gptoss_ep_save_load(rank: int, local_rank: int, world_size: int) -> bool:
    """Test LoRA save/load with EP=2 on GptOss-20B (MoE model, no TP)."""
    log(f"\n{'=' * 70}")
    log("  SUB-TEST C: GptOss-20B — LoRA + EP=2 Save/Load")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("lora_ep_save_gptoss", rank)
    model = None
    trainer = None

    try:
        log("[C.1] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(GPT_OSS_20B, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("[C.2] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED + 30)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 31)

        log("[C.3] Loading model with EP=2...")
        ensure_model_downloaded(GPT_OSS_20B, rank)
        parallelism_config = ParallelismConfig(ep_size=2)
        log(f"  Config: {parallelism_config.mode_string}")

        model, _ = load_distributed_model(
            model_name_or_path=GPT_OSS_20B,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        log(f"  Model loaded: {model.config.model_type}")
        log(f"  GPU memory: {gpu_mem_gb():.2f} GB")

        log("[C.4] Applying LoRA adapters...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        lora_before = snapshot_lora_weights(model)
        log(f"  LoRA params tracked: {len(lora_before)}")

        log("[C.5] Configuring trainer (save at step 5)...")
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=MAX_STEPS,
            save_total_limit=1,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log(f"  EP mode: {trainer.is_ep_mode}, TP mode: {trainer.is_tp_mode}")

        log(f"[C.6] Training ({MAX_STEPS} steps)...")
        train_result = trainer.train()

        log("[C.7] Validating training...")
        training_loss = train_result.training_loss
        step_losses = [
            entry["loss"] for entry in trainer.state.log_history if "loss" in entry and "eval_loss" not in entry
        ]
        log(f"  Training loss: {training_loss:.6f}")
        log(f"  Per-step losses: {[f'{sl:.4f}' for sl in step_losses]}")

        checks = {}

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'}")

        all_finite = all(math.isfinite(sl) for sl in step_losses)
        checks["all_steps_finite"] = all_finite
        log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

        lora_after = snapshot_lora_weights(model)
        lora_updated, lora_details = verify_lora_updated(lora_before, lora_after)
        checks["lora_updated"] = lora_updated
        log(f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} ({lora_details})")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        checks["ep_mode"] = trainer.is_ep_mode
        checks["no_tp_mode"] = not trainer.is_tp_mode
        log(f"  EP mode active: {'PASS' if checks['ep_mode'] else 'FAIL'}")
        log(f"  TP mode inactive: {'PASS' if checks['no_tp_mode'] else 'FAIL'}")

        log("[C.8] Validating saved adapter checkpoint...")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{MAX_STEPS}")
        dist.barrier()

        if rank == 0:
            save_ok, save_details = verify_saved_adapters(checkpoint_dir, lora_after, rank)
            checks["save_load"] = save_ok
            log(save_details)
            log(f"  Save/load roundtrip: {'PASS' if save_ok else 'FAIL'}")
        else:
            checks["save_load"] = True

        # MIN, not broadcast: rank 0 owns save_load, but every other check is per-rank, and a
        # broadcast overwrites each peer's verdict with rank 0's — a failure seen only on rank 1
        # (adapter desync, a NaN on one shard) would be reported as a pass by both ranks.
        result_t = torch.tensor([1 if all(checks.values()) else 0], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.all_reduce(result_t, op=dist.ReduceOp.MIN)

        all_passed = result_t.item() == 1
        log(f"\n  Sub-test C (GptOss LoRA+EP save/load): {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    except Exception as e:
        log(f"\n  Sub-test C FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)


# Sub-test D: Qwen3-0.6B with FSDP2 (standard data parallelism)


def run_qwen3_fsdp_save_load(rank: int, local_rank: int, world_size: int) -> bool:
    """Test LoRA save/load with FSDP2 on Qwen3-0.6B (dense model, no EP/TP/CP)."""
    log(f"\n{'=' * 70}")
    log(f"  SUB-TEST D: {QWEN3_MODEL} — LoRA + FSDP2 Save/Load")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("lora_fsdp_save_qwen3", rank)
    model = None
    trainer = None

    try:
        log("[D.1] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("[D.2] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED + 40)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 41)

        # FSDP2 mode: load via load_distributed_model with default (no parallelism) config
        # The mixin wraps with FSDP2 automatically for torchrun
        log("[D.3] Loading model for FSDP2...")
        ensure_model_downloaded(QWEN3_MODEL, rank)
        parallelism_config = ParallelismConfig()

        model, _ = load_distributed_model(
            model_name_or_path=QWEN3_MODEL,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        log(f"  Model loaded: {model.config.model_type}")
        log(f"  GPU memory: {gpu_mem_gb():.2f} GB")

        log("[D.4] Applying LoRA adapters...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        lora_before = snapshot_lora_weights(model)
        log(f"  LoRA params tracked: {len(lora_before)}")

        log("[D.5] Configuring trainer (save at step 5)...")
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=MAX_STEPS,
            save_total_limit=1,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log(f"  EP mode: {trainer.is_ep_mode}, TP mode: {trainer.is_tp_mode}")
        log(f"  FSDP wrapped: {trainer._fsdp_wrapped}")

        log(f"[D.6] Training ({MAX_STEPS} steps)...")
        train_result = trainer.train()

        log("[D.7] Validating training...")
        training_loss = train_result.training_loss
        log(f"  Training loss: {training_loss:.6f}")

        checks = {}

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'}")

        lora_after = snapshot_lora_weights(model)
        lora_updated, lora_details = verify_lora_updated(lora_before, lora_after)
        checks["lora_updated"] = lora_updated
        log(f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} ({lora_details})")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        checks["no_ep_mode"] = not trainer.is_ep_mode
        checks["no_tp_mode"] = not trainer.is_tp_mode
        log(f"  EP mode inactive: {'PASS' if checks['no_ep_mode'] else 'FAIL'}")
        log(f"  TP mode inactive: {'PASS' if checks['no_tp_mode'] else 'FAIL'}")

        log("[D.8] Validating saved adapter checkpoint...")
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{MAX_STEPS}")
        dist.barrier()

        if rank == 0:
            save_ok, save_details = verify_saved_adapters(checkpoint_dir, lora_after, rank)
            checks["save_load"] = save_ok
            log(save_details)
            log(f"  Save/load roundtrip: {'PASS' if save_ok else 'FAIL'}")
        else:
            checks["save_load"] = True

        # MIN, not broadcast: rank 0 owns save_load, but every other check is per-rank, and a
        # broadcast overwrites each peer's verdict with rank 0's — a failure seen only on rank 1
        # (adapter desync, a NaN on one shard) would be reported as a pass by both ranks.
        result_t = torch.tensor([1 if all(checks.values()) else 0], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.all_reduce(result_t, op=dist.ReduceOp.MIN)

        all_passed = result_t.item() == 1
        log(f"\n  Sub-test D (Qwen3 LoRA+FSDP2 save/load): {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    except Exception as e:
        log(f"\n  Sub-test D FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        del trainer
        del model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)


# Main

ALL_MODES = ["qwen3_tp", "gptoss_ep", "qwen3_fsdp"]


def main() -> int:
    """Run LoRA save/load tests across parallelism modes. Returns 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="all",
        choices=ALL_MODES + ["all"],
    )
    args, _ = parser.parse_known_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()

    modes = ALL_MODES if args.mode == "all" else [args.mode]

    log(f"\n{'#' * 70}")
    log("  LoRA Save/Load Test (Parallelism Modes)")
    log(f"  World size: {world_size}, GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Modes: {modes}")
    log(f"{'#' * 70}")

    if world_size < 2:
        log(f"\nERROR: This test requires at least 2 GPUs, got {world_size}")
        teardown_distributed()
        return 1

    results = {}

    for mode in modes:
        if mode == "qwen3_tp":
            ensure_model_downloaded(QWEN3_MODEL, rank)
            results["qwen3_tp"] = run_qwen3_tp_rejected(rank, local_rank, world_size)
        elif mode == "gptoss_ep":
            ensure_model_downloaded(GPT_OSS_20B, rank)
            results["gptoss_ep"] = run_gptoss_ep_save_load(rank, local_rank, world_size)
        elif mode == "qwen3_fsdp":
            ensure_model_downloaded(QWEN3_MODEL, rank)
            results["qwen3_fsdp"] = run_qwen3_fsdp_save_load(rank, local_rank, world_size)

        dist.barrier()
        cleanup_memory()

    # Summary
    log(f"\n{'#' * 70}")
    log("  RESULTS SUMMARY")
    log(f"{'#' * 70}")
    for name, passed in results.items():
        log(f"  {name}: {'PASSED' if passed else 'FAILED'}")

    all_passed = all(results.values())
    log(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    log(f"{'#' * 70}\n")

    if dist.is_initialized():
        dist.barrier()
    teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
