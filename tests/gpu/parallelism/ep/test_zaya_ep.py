#!/usr/bin/env python
"""ZAYA1-8B Expert Parallelism (EP) end-to-end smoke test.

Verifies the EP path on ZAYA1-8B with DeepEP all-to-all expert routing:
  1. ``load_distributed_model(expert_parallel_size=N)`` rewires every
     ``ZayaSparseMoeBlock`` to ``EPZayaMoELayer`` and slices the fused
     experts across ``N`` ranks.
  2. The expert weights now have ``num_experts / N`` along dim 0 on each rank.
  3. A short DistributedSFTTrainer run executes forward + backward + a
     gradient-accumulation step under FSDP2 (DP=1) + EP=N with grouped GEMM,
     logging a finite loss every step. Each step sees a different sample, so the
     loss trend is not asserted.

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

import torch
from trl import SFTConfig

from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_flag, env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.utils import log, step_losses

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


@gpu_test_main(min_world_size=2, prefix="test_zaya_ep")
def run(ctx):
    checks: dict[str, bool] = {}
    world = ctx.world_size

    log("=" * 70)
    log("  ZAYA1-8B Expert Parallelism smoke test (DeepEP)")
    log(f"  Model: {MODEL}")
    log(f"  World: {world}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  EP size: {world} (every GPU owns a fraction of experts)")
    log(f"  Steps: {MAX_STEPS}, Batch: 1, Seq: {SEQ}")
    log("=" * 70)

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
    assert ep_layers, "no EP layers were installed on the Zaya model"
    first = ep_layers[0]
    # ZAYA1-8B has 80 layers, 40 odd-index MLP layers → 40 EP wrappers.
    log(f"  EPZayaMoELayer count: {len(ep_layers)}")
    log(f"  Local experts per layer: {first.experts_per_rank} (of {first.num_experts} total)")
    gate_up_shape = tuple(first.gate_up_proj.shape)
    log(f"  gate_up_proj shape: {gate_up_shape}  (E_local, H, 2M)")
    log(f"  down_proj    shape: {tuple(first.down_proj.shape)}      (E_local, M, H)")
    checks["experts_sliced_ep_way"] = gate_up_shape[0] == first.experts_per_rank == first.num_experts // world
    # The gate owns the "+1" discard slot and masks it (prob 0, index 0) before the wrapper
    # dispatches, so the wrapper's expert count must EXCLUDE it — a dispatched index of
    # ``num_experts`` would run past the last expert.
    gate = first.gate
    checks["router_carries_the_discard_slot"] = gate.num_router_classes == first.num_experts + 1
    # Bias-update balancing rides the gate's native balancing_biases buffer.
    checks["balancing_biases_cover_every_router_class"] = gate.balancing_biases.shape[0] == gate.num_router_classes

    # ── Dataset ────────────────────────────────────────────────────
    log("\n[2/4] Creating synthetic SFT dataset...")
    train_ds = create_sft_dataset(16, tok, seed=42)
    log(f"  {len(train_ds)} samples")

    # ── Trainer ────────────────────────────────────────────────────
    log("\n[3/4] Configuring DistributedSFTTrainer (EP + grouped GEMM + GC)...")
    cfg = SFTConfig(
        output_dir=ctx.output_dir,
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
    ctx.on_teardown(trainer.cleanup_ep)
    log(f"  Trainer: {type(trainer).__name__}")
    log(f"  Parallelism: {pc.mode_string}")

    # ── Train ──────────────────────────────────────────────────────
    log(f"\n[4/4] Training {MAX_STEPS} steps...")
    result = trainer.train()
    losses = step_losses(trainer)
    log(f"  Final loss: {result.training_loss:.4f}")
    log(f"  Per-step losses: {[f'{loss:.4f}' for loss in losses]}")
    log(f"  HBM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    checks["training_loss_finite"] = math.isfinite(result.training_loss)
    checks["step_losses_finite"] = all(math.isfinite(loss) for loss in losses)
    checks["logged_every_step"] = len(losses) == MAX_STEPS
    checks["loss_decreased"] = len(losses) >= 2 and losses[-1] < losses[0]
    return {"checks": checks, "metrics": ctx.metrics(trainer)}


if __name__ == "__main__":
    run()
