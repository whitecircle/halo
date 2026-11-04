#!/usr/bin/env python
"""ZAYA1-8B under EP on DeepEP: experts sliced across ranks and a finite loss each step. Samples differ, so no loss
trend is asserted."""

import torch
from trl import SFTConfig

from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_int, env_str
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses, training_run_checks

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
MAX_STEPS = env_int("HALO_TEST_ZAYA_EP_STEPS", 4)
SEQ = env_int("HALO_TEST_ZAYA_EP_SEQ", 512)
LR = 5e-6


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

    log("\n[1/4] Loading ZAYA1-8B under EP...")
    # The hub stores experts fused ([E, 2M, H] under ``mlp.experts``), which the lazy loader slices per rank directly.
    pc = ParallelismConfig(ep_size=world, use_grouped_gemm=True)
    model, tok = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ep_layers = [m for m in model.modules() if isinstance(m, EPZayaMoELayer)]
    assert ep_layers, "no EP layers were installed on the Zaya model"
    first = ep_layers[0]
    # 80 layers, 40 odd-index MLP layers: 40 EP layers.
    log(f"  EPZayaMoELayer count: {len(ep_layers)}")
    log(f"  Local experts per layer: {first.experts_per_rank} (of {first.num_experts} total)")
    gate_up_shape = tuple(first.gate_up_proj.shape)
    log(f"  gate_up_proj shape: {gate_up_shape}  (E_local, H, 2M)")
    log(f"  down_proj    shape: {tuple(first.down_proj.shape)}      (E_local, M, H)")
    checks["experts_sliced_ep_way"] = gate_up_shape[0] == first.experts_per_rank == first.num_experts // world
    # The gate owns the "+1" discard slot and masks it before dispatch, so the wrapper's expert count must exclude it:
    # a dispatched index of ``num_experts`` would run past the last expert.
    gate = first.gate
    checks["router_carries_the_discard_slot"] = gate.num_router_classes == first.num_experts + 1
    checks["balancing_biases_cover_every_router_class"] = gate.balancing_biases.shape[0] == gate.num_router_classes

    log("\n[2/4] Creating synthetic SFT dataset...")
    train_ds = create_sft_dataset(16, tok, seed=42)
    log(f"  {len(train_ds)} samples")

    log("\n[3/4] Configuring DistributedSFTTrainer (EP + grouped GEMM + GC)...")
    cfg = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        bf16=True,
        # Zaya refuses gradient checkpointing: the recompute faults in cuDNN on the CCA Conv1d pair, and per-layer GC
        # re-wraps the EDA state with a fresh grad_fn, whose recompute graph autograd processes polynomially in the
        # layer count.
        gradient_checkpointing=False,
        # The mixin defers TRL's re-application and restores the flag, so TRL's compute_loss takes its Liger path:
        # under FLCE outputs.logits is a placeholder, not the [B*S, V] tensor TRL needs for entropy.
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=SEQ,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
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

    log(f"\n[4/4] Training {MAX_STEPS} steps...")
    result = trainer.train()
    log(f"  HBM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    checks |= training_run_checks(result, trainer, MAX_STEPS)
    checks["logged_every_step"] = len(step_losses(trainer)) == MAX_STEPS
    return {"checks": checks, "metrics": ctx.metrics(trainer)}


if __name__ == "__main__":
    run()
