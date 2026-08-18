#!/usr/bin/env python
"""
Comprehensive Checkpoint Save/Load Roundtrip Test for GptOss-20B (MoE).

Validates that DistributedSFTTrainer can train, save, and reload GptOss-20B
checkpoints correctly under EP and EP+TP parallelism configurations. For each
mode the test:

1. Loads model with appropriate parallelism config (EP or EP+TP)
2. Trains for a few steps (verifies loss is finite and decreasing)
3. Saves checkpoint via trainer.save_model()
4. Tears down trainer/model (frees GPU, avoids collective hazards)
5. Rank 0 reloads checkpoint via from_pretrained (CPU) and verifies:
   a. Checkpoint files exist (config.json + weights)
   b. Parameter count and names match the model architecture
   c. No NaN/Inf in any parameter
   d. Weights differ from untrained pretrained (training happened)
   e. Forward pass produces valid logits (no NaN/Inf, correct shape)
   f. Save-load roundtrip is lossless (save again, compare state dicts)

Modes tested (one per invocation via --mode flag):
  ep2      : ParallelismConfig(ep_size=2)          -- Expert Parallelism
  ep2_tp2  : ParallelismConfig(ep_size=2, tp_size=2) -- EP + Tensor Parallelism

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts, top_k=4)

Run examples (2 GPUs, one mode at a time to manage memory):
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_checkpoint_roundtrip_gptoss_20b.py --mode ep2

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_checkpoint_roundtrip_gptoss_20b.py --mode ep2_tp2
"""

import argparse
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
MAX_STEPS = 3
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 16
NUM_EVAL_SAMPLES = 4
SEED = 42
WEIGHT_ATOL = 1e-6
# GptOss-20B is ~20.7B params; the gathered save must carry all of them, not one EP rank's half.
EXPECTED_FULL_PARAM_RANGE = (18e9, 23e9)


def train_and_save(
    mode_name: str,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    save_dir: str,
    base_output_dir: str,
    local_rank: int,
) -> tuple[bool, float, list[float]]:
    """Train model and save checkpoint. All ranks must call this."""
    model = None
    trainer = None

    try:
        log(f"  [1/3] Loading model ({mode_name})...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )

        mem_gb = torch.cuda.memory_allocated() / 1e9
        log(f"  Model loaded, GPU mem: {mem_gb:.2f} GB")

        log(f"  [2/3] Training for {MAX_STEPS} steps...")
        config = SFTConfig(
            output_dir=os.path.join(base_output_dir, f"trainer_{mode_name}"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # Already applied in load_distributed_model
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
            fsdp="",  # Mixin handles FSDP wrapping
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer.is_ep_mode, f"Trainer should be in EP mode for {mode_name}"
        if parallelism_config.is_tp_mode:
            assert trainer.is_tp_mode, f"Trainer should be in TP mode for {mode_name}"
        log("  Parallelism modes confirmed")

        train_result = trainer.train()
        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"  Training loss: {training_loss:.6f}")
        log(f"  Step losses: {[f'{l:.4f}' for l in step_losses]}")

        log(f"  [3/3] Saving checkpoint to {save_dir}...")
        trainer.save_model(save_dir)
        dist.barrier()
        log("  Checkpoint saved")

        return True, training_loss, step_losses

    except Exception as e:
        log(f"  Train/save FAILED: {e}")
        traceback.print_exc()
        return False, float("nan"), []

    finally:
        del trainer
        del model
        cleanup_memory()
        dist.barrier()


def verify_checkpoint_on_rank0(
    save_dir: str,
    tokenizer,
    local_rank: int,
) -> tuple[bool, dict[str, bool], str]:
    """Comprehensive checkpoint verification. Only called on rank 0."""
    checks: dict[str, bool] = {}
    details: list[str] = []

    details.append("  [Check 1] Checkpoint files:")
    files = os.listdir(save_dir) if os.path.exists(save_dir) else []
    has_config = os.path.exists(os.path.join(save_dir, "config.json"))
    has_safetensors = os.path.exists(os.path.join(save_dir, "model.safetensors")) or any(
        f.startswith("model") and f.endswith(".safetensors") for f in files
    )
    has_pytorch = os.path.exists(os.path.join(save_dir, "pytorch_model.bin")) or any(
        f.startswith("pytorch_model") and f.endswith(".bin") for f in files
    )
    has_model = has_safetensors or has_pytorch

    checks["files_exist"] = has_config and has_model
    weight_fmt = "safetensors" if has_safetensors else ("pytorch" if has_pytorch else "MISSING")
    details.append(f"    config.json: {'YES' if has_config else 'NO'}")
    details.append(f"    Model weights: {weight_fmt}")
    details.append(f"    All files: {sorted(files)}")

    if not checks["files_exist"]:
        return False, checks, "\n".join(details)

    details.append("  [Check 2] Loading checkpoint with from_pretrained...")
    try:
        reloaded_model = AutoModelForCausalLM.from_pretrained(
            save_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        checks["loads_ok"] = True
        details.append("    Loaded successfully")
    except Exception as e:
        checks["loads_ok"] = False
        details.append(f"    FAILED to load: {e}")
        return False, checks, "\n".join(details)

    reloaded_sd = reloaded_model.state_dict()

    details.append("  [Check 3] Parameter inventory:")
    num_params = sum(p.numel() for p in reloaded_model.parameters())
    num_keys = len(reloaded_sd)
    details.append(f"    Total parameters: {num_params:,}")
    details.append(f"    State dict keys: {num_keys}")

    # The EP non-grouped-mm gather can drop 2D expert biases silently, so every layer carrying an
    # `experts.gate_up_proj` must also carry both biases.
    moe_layer_indices = sorted(
        {
            int(k.split(".")[2])
            for k in reloaded_sd
            if k.startswith("model.layers.") and k.endswith(".mlp.experts.gate_up_proj")
        }
    )
    missing_biases: list[str] = []
    for L in moe_layer_indices:
        for bias_attr in ("gate_up_proj_bias", "down_proj_bias"):
            key = f"model.layers.{L}.mlp.experts.{bias_attr}"
            if key not in reloaded_sd:
                missing_biases.append(key)
    checks["expert_biases_present"] = not missing_biases
    if missing_biases:
        details.append(
            f"    EXPERT BIASES MISSING ({len(missing_biases)}/{len(moe_layer_indices) * 2}): "
            f"{missing_biases[:3]}{'...' if len(missing_biases) > 3 else ''}"
        )
    else:
        details.append(f"    Expert biases complete: 2 × {len(moe_layer_indices)} MoE layers")

    min_params, max_params = EXPECTED_FULL_PARAM_RANGE
    checks["param_count_reasonable"] = min_params < num_params < max_params
    details.append(
        f"    In expected range ({min_params / 1e9:.0f}B-{max_params / 1e9:.0f}B): {checks['param_count_reasonable']}"
    )

    details.append("  [Check 4] Weight integrity (NaN/Inf scan):")
    nan_params = []
    inf_params = []
    for name, param in reloaded_sd.items():
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)

    checks["no_nan_weights"] = len(nan_params) == 0
    checks["no_inf_weights"] = len(inf_params) == 0
    if nan_params:
        details.append(f"    NaN in {len(nan_params)} params: {nan_params[:5]}")
    if inf_params:
        details.append(f"    Inf in {len(inf_params)} params: {inf_params[:5]}")
    if not nan_params and not inf_params:
        details.append(f"    All {num_keys} parameters clean (no NaN/Inf)")

    details.append("  [Check 5] Training effect (compare vs untrained pretrained):")
    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        pretrained_sd = pretrained_model.state_dict()

        changed_params = 0
        total_compared = 0
        max_change = 0.0
        max_change_param = ""

        for key in reloaded_sd:
            if key in pretrained_sd and reloaded_sd[key].shape == pretrained_sd[key].shape:
                total_compared += 1
                diff = (reloaded_sd[key].float() - pretrained_sd[key].float()).abs().max().item()
                if diff > 1e-8:
                    changed_params += 1
                if diff > max_change:
                    max_change = diff
                    max_change_param = key

        checks["weights_changed"] = changed_params > 0 and total_compared > 0
        details.append(f"    Changed params: {changed_params}/{total_compared}")
        details.append(f"    Max weight change: {max_change:.2e} ({max_change_param})")
        del pretrained_model, pretrained_sd
    except Exception as e:
        # Not "assume OK": without the reference this has verified nothing, and a pass would
        # green-light a checkpoint that never received a gradient.
        checks["weights_changed"] = False
        details.append(f"    FAILED (could not load pretrained for comparison): {e}")

    details.append("  [Check 6] Forward pass validation:")
    try:
        device = f"cuda:{local_rank}"
        reloaded_model = reloaded_model.to(device)
        reloaded_model.eval()

        prompt = "What is 2 + 2?"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = reloaded_model(**inputs)

        logits = outputs.logits
        batch, seq_len, vocab = logits.shape

        shape_ok = batch == 1 and seq_len == inputs["input_ids"].shape[1]
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        logit_min = logits.min().item()
        logit_max = logits.max().item()
        range_ok = abs(logit_min) < 1000 and abs(logit_max) < 1000

        checks["forward_shape"] = shape_ok
        checks["forward_no_nan"] = not has_nan
        checks["forward_no_inf"] = not has_inf
        checks["forward_range"] = range_ok

        details.append(f"    Logits shape: {logits.shape} (valid={shape_ok})")
        details.append(f"    NaN={has_nan}, Inf={has_inf}")
        details.append(f"    Logit range: [{logit_min:.2f}, {logit_max:.2f}] (valid={range_ok})")

        reloaded_model = reloaded_model.cpu()
    except Exception as e:
        checks["forward_shape"] = False
        checks["forward_no_nan"] = False
        checks["forward_no_inf"] = False
        checks["forward_range"] = False
        details.append(f"    FAILED: {e}")
        traceback.print_exc()

    details.append("  [Check 7] Save-load roundtrip (save again, reload, compare):")
    try:
        roundtrip_dir = save_dir + "_roundtrip"
        os.makedirs(roundtrip_dir, exist_ok=True)
        reloaded_model.save_pretrained(roundtrip_dir, safe_serialization=True)

        roundtrip_model = AutoModelForCausalLM.from_pretrained(
            roundtrip_dir,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        roundtrip_sd = roundtrip_model.state_dict()

        mismatches = 0
        max_diff = 0.0
        max_diff_param = ""
        for key in reloaded_sd:
            if key not in roundtrip_sd:
                mismatches += 1
                continue
            if reloaded_sd[key].shape != roundtrip_sd[key].shape:
                mismatches += 1
                continue
            diff = (reloaded_sd[key].float() - roundtrip_sd[key].float()).abs().max().item()
            if diff > max_diff:
                max_diff = diff
                max_diff_param = key
            if diff > WEIGHT_ATOL:
                mismatches += 1

        checks["roundtrip_lossless"] = mismatches == 0
        details.append(f"    Mismatches: {mismatches}/{len(reloaded_sd)}")
        details.append(f"    Max diff: {max_diff:.2e} ({max_diff_param})")

        del roundtrip_model, roundtrip_sd
        import shutil

        shutil.rmtree(roundtrip_dir, ignore_errors=True)

    except Exception as e:
        checks["roundtrip_lossless"] = False
        details.append(f"    FAILED: {e}")
        traceback.print_exc()

    del reloaded_model, reloaded_sd
    cleanup_memory()

    all_passed = all(checks.values())
    return all_passed, checks, "\n".join(details)


def run_mode_test(
    mode_name: str,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    base_output_dir: str,
    rank: int,
    local_rank: int,
) -> tuple[bool, str]:
    """Run full train -> save -> verify cycle for one parallelism mode."""
    save_dir = os.path.join(base_output_dir, f"{mode_name}_checkpoint")
    all_checks: dict[str, bool] = {}

    log(f"\n  -- Phase 1: Train + Save ({mode_name}) --")
    train_ok, training_loss, step_losses = train_and_save(
        mode_name=mode_name,
        parallelism_config=parallelism_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        save_dir=save_dir,
        base_output_dir=base_output_dir,
        local_rank=local_rank,
    )

    all_checks["train_ok"] = train_ok
    all_checks["loss_finite"] = math.isfinite(training_loss)
    if not train_ok:
        return False, _format_checks(all_checks)

    # The group must go down on every rank BEFORE rank 0's minutes-long two-checkpoint reload:
    # teardown_distributed opens with an unconditional collective, so a peer arriving first would
    # sit there until the NCCL watchdog fired. No extra barrier — teardown's own is the rendezvous,
    # and a second would desync against the ``train_ok`` early return above.
    teardown_distributed()

    log(f"\n  -- Phase 2: Verify Checkpoint ({mode_name}) --")
    if rank == 0:
        verify_ok, verify_checks, verify_detail = verify_checkpoint_on_rank0(
            save_dir=save_dir,
            tokenizer=tokenizer,
            local_rank=local_rank,
        )
        log(verify_detail)
        all_checks.update(verify_checks)
    else:
        verify_ok = True

    # No broadcast of the verify result either — the group is already down. torchrun aggregates
    # exit codes, so a failed rank-0 verify still fails the run.
    overall = train_ok and verify_ok
    return overall, _format_checks(all_checks)


def _format_checks(checks: dict[str, bool]) -> str:
    parts = [f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()]
    return ", ".join(parts)


MODE_LABELS = {
    "ep2": "EP=2 (Expert Parallelism)",
    "ep2_tp2": "EP=2+TP=2 (Expert + Tensor Parallelism)",
    # ep2/ep2_tp2 both take the grouped-mm path on SM90+, which handles expert biases; only an
    # explicitly off-grouped run reaches the fallback gather that can drop them.
    "ep2_no_gmm": "EP=2 (no grouped_mm — exercises bias-gather fallback)",
    # Pure expert-TP: experts are reconstructed across the ETP group before the HF-layout save.
    "etp": "ETP (ep_size=1, expert_tp_size=2 — expert-TP gather path)",
}


def _build_parallelism_config(mode: str) -> ParallelismConfig:
    """Build ParallelismConfig AFTER dist is initialized."""
    if mode == "ep2":
        return ParallelismConfig(ep_size=2)
    elif mode == "ep2_tp2":
        return ParallelismConfig(ep_size=2, tp_size=2)
    elif mode == "ep2_no_gmm":
        return ParallelismConfig(ep_size=2, use_grouped_gemm=False)
    elif mode == "etp":
        return ParallelismConfig(ep_size=1, expert_tp_size=2)
    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    """Run checkpoint save/load roundtrip test. Returns 0 on success, 1 on failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ep2",
        choices=list(MODE_LABELS.keys()),
        help="Parallelism mode to test (default: ep2)",
    )
    args, _ = parser.parse_known_args()

    rank, world_size, local_rank = init_distributed()

    from accelerate import PartialState

    PartialState()

    base_output_dir, cache_dir = setup_cache_dirs("test_ckpt_rt_oss20b", rank)

    mode_label = MODE_LABELS[args.mode]
    parallelism_config = _build_parallelism_config(args.mode)

    log(f"\n{'#' * 70}")
    log(f"  Checkpoint Roundtrip Test: {mode_label}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  World size: {world_size}, GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Steps: {MAX_STEPS}, Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    if world_size < 2:
        log(f"\nERROR: Need at least 2 GPUs, got {world_size}")
        teardown_distributed()
        return 1

    success = False

    try:
        log("\n[Setup] Loading tokenizer...")
        ensure_model_downloaded(MODEL_NAME, rank)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"  Tokenizer: vocab_size={tokenizer.vocab_size}")

        log("[Setup] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
        log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        log(f"\n{'=' * 60}")
        log(f"  {mode_label}")
        log(f"{'=' * 60}")

        success, detail = run_mode_test(
            mode_name=args.mode,
            parallelism_config=parallelism_config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            base_output_dir=base_output_dir,
            rank=rank,
            local_rank=local_rank,
        )

        log(f"\n{'#' * 70}")
        if success:
            log(f"  PASSED: {mode_label}")
        else:
            log(f"  FAILED: {mode_label}")
        log(f"  Details: {detail}")
        log(f"{'#' * 70}\n")

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()

    finally:
        cleanup_memory()
        cleanup_dirs(base_output_dir, cache_dir)
        teardown_distributed()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
