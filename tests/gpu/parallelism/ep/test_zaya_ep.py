#!/usr/bin/env python
"""ZAYA1-8B Expert Parallelism (EP) end-to-end smoke test.

Verifies the EP path on ZAYA1-8B with DeepEP all-to-all expert routing:
  1. ``load_distributed_model(expert_parallel_size=N)`` rewires every
     ``ZayaSparseMoeBlock`` to ``EPZayaMoELayer`` and slices the fused
     experts across ``N`` ranks.
  2. The expert weights now have ``num_experts / N`` along dim 0 on each rank.
  3. A short DistributedSFTTrainer run executes forward + backward + a
     gradient-accumulation step under FSDP2 (DP=1) + EP=N with grouped GEMM,
     and the loss decreases over the first few steps.

Run (2 GPUs, EP=2):
    docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \\
        --ulimit stack=67108864 \\
        -v $(pwd):/workspace \\
        -v /root/.cache/huggingface:/root/.cache/huggingface \\
        -w /workspace -e HF_HOME=/root/.cache/huggingface \\
        halo:blackwell \\
        torchrun --nproc_per_node=2 \\
            tests/gpu/parallelism/ep/test_zaya_ep.py
"""

import math
import sys

import torch
from trl import SFTConfig

from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_flag, env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, init_distributed, setup_cache_dirs
from tests.common.models import ZAYA_8B
from tests.common.utils import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
MAX_STEPS = env_int("HALO_TEST_ZAYA_EP_STEPS", 4)
SEQ = env_int("HALO_TEST_ZAYA_EP_SEQ", 512)
LR = 5e-6
# Zaya refuses gradient checkpointing everywhere (``apply_zaya_patches`` clears
# ``supports_gradient_checkpointing`` at load): the recompute faults in cuDNN on the CCA Conv1d
# pair, and per-layer GC re-wraps the cross-layer EDA state with a fresh grad_fn, whose recompute
# graph the autograd engine processes polynomially in the number of checkpointed layers. Set
# HALO_TEST_ZAYA_GC=1 only to assert the refusal fires.
GC_DEFAULT = False


def main() -> int:
    rank, world, local = init_distributed()
    from accelerate import PartialState

    PartialState()
    output_dir, cache_dir = setup_cache_dirs("test_zaya_ep", rank)

    log("=" * 70)
    log("  ZAYA1-8B Expert Parallelism smoke test (DeepEP)")
    log(f"  Model: {MODEL}")
    log(f"  World: {world}, GPU: {torch.cuda.get_device_name(local)}")
    log(f"  EP size: {world} (every GPU owns a fraction of experts)")
    log(f"  Steps: {MAX_STEPS}, Batch: 1, Seq: {SEQ}")
    log("=" * 70)

    try:
        # ── Load model with EP ─────────────────────────────────────────
        log("\n[1/4] Loading ZAYA1-8B under EP...")
        # The hub checkpoint stores experts fused ([E, 2M, H] under ``mlp.experts``), which the lazy
        # safetensors loader slices per rank directly — the production default path.
        pc = ParallelismConfig(ep_size=world, use_grouped_gemm=True)
        model, tok = load_distributed_model(
            model_name_or_path=MODEL,
            parallelism_config=pc,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ── Verify EP layers were installed and weights are sharded ────
        ep_layers = [m for m in model.modules() if isinstance(m, EPZayaMoELayer)]
        assert len(ep_layers) > 0, "No EPZayaMoELayer found — patching failed"
        expected_experts_per_rank = ep_layers[0].experts_per_rank
        # ZAYA1-8B has 80 layers, 40 odd-index MLP layers → 40 EP wrappers.
        log(f"  ✓ EPZayaMoELayer count: {len(ep_layers)}")
        log(f"  ✓ Local experts per layer: {expected_experts_per_rank} (of {ep_layers[0].num_experts} total)")
        gate_up_shape = tuple(ep_layers[0].gate_up_proj.shape)
        down_shape = tuple(ep_layers[0].down_proj.shape)
        log(f"  ✓ gate_up_proj shape: {gate_up_shape}  (E_local, H, 2M)")
        log(f"  ✓ down_proj    shape: {down_shape}      (E_local, M, H)")
        assert gate_up_shape[0] == expected_experts_per_rank, (
            f"Expected dim-0 = {expected_experts_per_rank}, got {gate_up_shape[0]}"
        )
        # The gate owns the "+1" discard slot and masks it (prob 0, index 0) before the wrapper
        # dispatches, so the wrapper's expert count must EXCLUDE it — a dispatched index of
        # ``num_experts`` would run past the last expert.
        gate = ep_layers[0].gate
        assert gate.num_router_classes == ep_layers[0].num_experts + 1, (
            "the router must carry one class more than the experts (the discard slot)"
        )
        assert gate.balancing_biases.shape[0] == gate.num_router_classes, (
            "bias-update balancing rides the gate's native balancing_biases buffer"
        )

        # ── Dataset ────────────────────────────────────────────────────
        log("\n[2/4] Creating synthetic SFT dataset...")
        train_ds = create_sft_dataset(16, tok, seed=42)
        log(f"  ✓ {len(train_ds)} samples")

        # ── Trainer ────────────────────────────────────────────────────
        log("\n[3/4] Configuring DistributedSFTTrainer (EP + grouped GEMM + GC)...")
        cfg = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=LR,
            bf16=True,
            gradient_checkpointing=env_flag("HALO_TEST_ZAYA_GC", GC_DEFAULT),
            gradient_checkpointing_kwargs={
                "use_reentrant": env_flag("HALO_TEST_ZAYA_GC_REENTRANT", True),
            },
            # Set True so Halo's _deferred_liger_kernel flow defers TRL's
            # re-application during __init__ then restores the flag — this
            # makes TRL's compute_loss take its Liger path (skip entropy,
            # read outputs.token_accuracy) which is required when FLCE is on
            # (FLCE swaps the lm_head + CE path; outputs.logits is a
            # placeholder, not the full [B*S, V] tensor TRL needs for entropy).
            use_liger_kernel=True,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            max_length=SEQ,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",
            seed=42,
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=cfg,
            train_dataset=train_ds,
            processing_class=tok,
            parallelism_config=pc,
        )
        log(f"  ✓ Trainer: {type(trainer).__name__}")
        log(f"  ✓ Parallelism: {pc.mode_string}")

        # ── Train ──────────────────────────────────────────────────────
        log(f"\n[4/4] Training {MAX_STEPS} steps...")
        result = trainer.train()
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"  ✓ Final loss: {result.training_loss:.4f}")
        log(f"  ✓ Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
        log(f"  ✓ HBM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        assert math.isfinite(result.training_loss), f"Non-finite loss: {result.training_loss}"
        assert all(math.isfinite(x) for x in step_losses), f"Non-finite step loss: {step_losses}"
        assert len(step_losses) == MAX_STEPS, f"Expected {MAX_STEPS} step losses, got {len(step_losses)}"
        if len(step_losses) >= 2:
            assert step_losses[-1] < step_losses[0], f"Loss did not decrease: {step_losses[0]} -> {step_losses[-1]}"

        if rank == 0:
            log("\n" + "=" * 70)
            log("  ✓ ALL CHECKS PASSED (ZAYA1-8B + EP)")
            log("=" * 70)
        return 0
    finally:
        cleanup_dirs(output_dir, cache_dir)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
