#!/usr/bin/env python
"""DeepSeek-V4 EP=2 SFT smoke + gathered-save round-trip on a tiny random-init model.

Rank 0 materializes a tiny DeepSeek-V4 checkpoint (real tokenizer vocab, randomized tid2eid) on
disk, then all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — exercising the eager-force guard (a
     flash_attention request must be overridden to eager), the lazy fused-expert loader, and the
     per-rope-type inv_freq recompute.
  2. Run a short DistributedSFTTrainer run (forward + backward + optimizer under FSDP2 + EP).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss (a dropped/transposed expert axis shifts
     it by >>1), and the hash tid2eid table must survive the round-trip.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_deepseek_v4_moe.py
"""

import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4 import DeepseekV4Config
from trl import SFTConfig

from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_DSV4_CONFIG
from tests.common.utils import cleanup_memory, log

SEED = 42
NUM_TRAIN_STEPS = 3
MAX_SEQ_LENGTH = 256
LOSS_TOL = 5e-2  # EP-vs-plain reload forward noise (bf16 + grouped-GEMM vs loop); dropped experts shift >>1


def _materialize_checkpoint(base_dir: str, tokenizer) -> None:
    """Rank 0: build + save the tiny model with the real tokenizer's vocab."""
    torch.manual_seed(SEED)
    config = DeepseekV4Config(**{**TINY_DSV4_CONFIG, "vocab_size": len(tokenizer), "attn_implementation": "eager"})
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    # DISTINCT experts per token id (DeepEP dispatch + the wrapper's init guard require it).
    gen = torch.Generator().manual_seed(SEED)
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], config.n_routed_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)


def _fixed_batch(tokenizer, device, vocab_size: int):
    torch.manual_seed(SEED + 7)
    ids = torch.randint(0, vocab_size, (2, 64), device=device)
    labels = ids.clone()
    return ids, labels


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
    base_dir = os.path.join(temp_root, f"dsv4_tiny_base_{tag}")
    save_dir = os.path.join(temp_root, f"dsv4_tiny_trained_{tag}")
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
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # guard must override to eager
        use_liger_kernel=True,  # FLCE-only applier
    )
    attn = getattr(model.config, "_attn_implementation", None)
    checks["eager_forced"] = attn == "eager"
    log(f"attn_implementation resolved to: {attn}")

    ep_layers = [m for m in model.modules() if isinstance(m, EPDeepseekV4MoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == len(TINY_DSV4_CONFIG["layer_types"])
    checks["hash_layer_present"] = any(ep.is_hash for ep in ep_layers)
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
        # FLCE-only forward returns no logits: with the flag off, TRL's metric path slices None.
        use_liger_kernel=True,
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

    ids, labels = _fixed_batch(tokenizer, device, vocab_size)
    ep_loss = _forward_loss(model, ids, labels)
    metrics["ep_loss_post_train"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(torch.tensor(ep_loss)))

    barrier()
    trainer.save_model(save_dir)
    barrier()

    # .to(bf16) flattens `_keep_in_fp32_modules_strict` norms, whose fp32 output otherwise crashes
    # the stock eager forward; those modules upcast internally, and the EP loader does the same.
    reloaded = (
        AutoModelForCausalLM.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    # tid2eid must survive train + gathered save (frozen buffer, not trained).
    hash_ep = next(ep for ep in ep_layers if ep.is_hash)
    checks["tid2eid_roundtrip"] = torch.equal(
        reloaded.model.layers[0].mlp.gate.tid2eid.cpu(), hash_ep.gate.tid2eid.cpu()
    )
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


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="sft_dsv4_moe")(run)

if __name__ == "__main__":
    main()
