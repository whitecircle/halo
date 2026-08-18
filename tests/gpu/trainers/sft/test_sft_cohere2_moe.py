#!/usr/bin/env python
"""Cohere2 MoE EP=2 SFT smoke + gathered-save round-trip on a tiny random-init model.

Rank 0 materializes a tiny Cohere2 MoE checkpoint (real tokenizer vocab) on disk, then all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — the family declares
     ``_supports_lazy_loading = False``, so this exercises the ``from_pretrained`` route (including
     the per-expert→fused checkpoint conversion) plus the generic inv_freq recompute.
  2. Run a short DistributedSFTTrainer run (forward + backward + optimizer under FSDP2 + EP, tied
     embeddings, parallel-residual blocks, interleaved sliding/NoPE attention).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss (a dropped/transposed expert axis or a
     missed shared-expert average shifts it by >>1), and ``logit_scale`` must survive the save.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_cohere2_moe.py
"""

import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.cohere2_moe import Cohere2MoeConfig
from trl import SFTConfig

from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_COHERE2_MOE_CONFIG
from tests.common.utils import cleanup_memory, log

SEED = 42
NUM_TRAIN_STEPS = 3
MAX_SEQ_LENGTH = 256
LOSS_TOL = 5e-2  # EP-vs-plain reload forward noise (bf16 + grouped-GEMM vs loop); dropped experts shift >>1


def _materialize_checkpoint(base_dir: str, tokenizer) -> None:
    """Rank 0: build + save the tiny model with the real tokenizer's vocab."""
    torch.manual_seed(SEED)
    config = Cohere2MoeConfig(
        **{**TINY_COHERE2_MOE_CONFIG, "vocab_size": len(tokenizer), "attn_implementation": "eager"}
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)


def _fixed_batch(device, vocab_size: int):
    torch.manual_seed(SEED + 7)
    ids = torch.randint(0, vocab_size, (2, 64), device=device)
    return ids, ids.clone()


def _forward_loss(model, ids, labels) -> float:
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids, labels=labels).loss.item()


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)  # tokenizer only
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # setup_cache_dirs is per-rank; these must be rank-shared, keyed by port to avoid collisions.
    tag = os.environ.get("MASTER_PORT", "0")
    temp_root = os.path.dirname(ctx.output_dir.rstrip("/"))
    base_dir = os.path.join(temp_root, f"cohere2_moe_tiny_base_{tag}")
    save_dir = os.path.join(temp_root, f"cohere2_moe_tiny_trained_{tag}")
    if ctx.rank == 0:
        for d in (base_dir, save_dir):
            shutil.rmtree(d, ignore_errors=True)
        ctx.on_teardown(lambda: [shutil.rmtree(d, ignore_errors=True) for d in (base_dir, save_dir)])
        _materialize_checkpoint(base_dir, tokenizer)
    barrier()

    pc = ParallelismConfig(ep_size=ctx.world_size)
    model, _ = load_distributed_model(
        model_name_or_path=base_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        # On: the family's Liger spec patches Cohere2MoeMLP, which under EP is the SHARED expert the
        # wrapper adopts unchanged — so this run is the one that exercises the fused GLU surviving EP.
        use_liger_kernel=True,
    )

    ep_layers = [m for m in model.modules() if isinstance(m, EPCohere2MoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == TINY_COHERE2_MOE_CONFIG["num_hidden_layers"]
    checks["average_combination_scaled"] = all(ep._output_scale == 0.5 for ep in ep_layers)
    vocab_size = model.config.vocab_size

    train_dataset = create_sft_dataset(16, tokenizer, seed=SEED)
    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=1,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",
        ddp_find_unused_parameters=True,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    result = trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"train losses: {losses}")
    metrics["final_train_loss"] = result.training_loss
    checks["trained_all_steps"] = result.global_step == NUM_TRAIN_STEPS
    checks["train_losses_finite"] = all(torch.isfinite(torch.tensor(losses)).tolist()) and len(losses) > 0

    ids, labels = _fixed_batch(device, vocab_size)
    ep_loss = _forward_loss(model, ids, labels)
    metrics["ep_loss_post_train"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(torch.tensor(ep_loss)))

    barrier()
    trainer.save_model(save_dir)
    barrier()

    reloaded = (
        AutoModelForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
        .to(device=device)
        .eval()
    )
    checks["logit_scale_survives_save"] = reloaded.config.logit_scale == TINY_COHERE2_MOE_CONFIG["logit_scale"]
    rl_loss = _forward_loss(reloaded, ids, labels)
    metrics["reload_loss"] = rl_loss
    delta = abs(rl_loss - ep_loss)
    metrics["reload_loss_delta"] = delta
    log(f"EP loss {ep_loss:.6f} vs reloaded plain-HF loss {rl_loss:.6f} (|Δ|={delta:.2e})")
    checks["reload_loss_matches"] = delta < LOSS_TOL

    del reloaded
    cleanup_memory()
    ctx.on_teardown(lambda: trainer.cleanup_ep() if hasattr(trainer, "cleanup_ep") else None)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="sft_cohere2_moe")(run)

if __name__ == "__main__":
    main()
