#!/usr/bin/env python
"""A native MXFP4 gpt-oss checkpoint (the ``openai/gpt-oss-*`` layout) through the whole EP=2 life of a run: lazy load
with the experts decoded to bf16, train with a checkpoint, resume from it, gathered save, sharded save merged by
``merge_ep_shards``, then a stock ``from_pretrained``.

The weights are bf16 from the load on, so no save may carry the release's ``quantization_config``: the resume would
refuse its own checkpoint, and a stock load would build an MXFP4 quantizer for a bf16 file. Each save's ``config.json``
is checked, the merge must equal the gathered save bit for bit, and the stock reload's loss must match the EP model's.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_native_mxfp4_roundtrip.py
"""

import json
import math
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GptOssForCausalLM
from trl import SFTConfig

from halo.checkpoint.parallel.ep import save_ep_sharded_checkpoint
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import fixed_batch_loss, safetensors_state_dict, save_context
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_output_dir, world_all
from tests.common.ep_reference import random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.native_mxfp4 import write_native_mxfp4
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import step_losses

SEED = 42
SAVE_AT_STEP = 2
TOTAL_STEPS = 4
MAX_SEQ_LENGTH = 256


def _sft_config(output_dir: str, max_steps: int, save_at: int | None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        seed=SEED,
    )


def _trainer(model_path: str, pc, tokenizer, dataset, config, *, preserve_checkpoint_precision: bool = False):
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_liger_kernel=False,
        trust_remote_code=True,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
    )
    return DistributedSFTTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer, parallelism_config=pc
    )


def _claims_mxfp4(directory: str) -> bool:
    with open(os.path.join(directory, "config.json")) as f:
        return "quantization_config" in json.load(f)


def _rank0_merge_and_reload(sharded_dir: str, merged_dir: str, gathered_dir: str, ids, labels, ep_loss: float) -> dict:
    """Rank 0 alone; the verdict travels as data, since a rank-local raise would leave the peer in a collective."""
    merge_ep_shards(sharded_dir, merged_dir, verbose=False)
    merged, gathered = safetensors_state_dict(merged_dir), safetensors_state_dict(gathered_dir)
    differing = sorted(k for k in set(merged) & set(gathered) if not torch.equal(merged[k], gathered[k]))
    reloaded = GptOssForCausalLM.from_pretrained(merged_dir, dtype=torch.bfloat16, attn_implementation="eager")
    reloaded = reloaded.to(ids.device).eval()
    reload_loss = fixed_batch_loss(reloaded, ids, labels)
    log(f"EP loss {ep_loss:.6f} vs stock reload of the merge {reload_loss:.6f} (|Δ|={abs(reload_loss - ep_loss):.2e})")
    if differing:
        log(f"merged and gathered differ at {differing[:4]}")
    return {
        "checks": {
            "merged_config_does_not_claim_mxfp4": not _claims_mxfp4(merged_dir),
            "merged_keys_equal_gathered": set(merged) == set(gathered),
            "merged_tensors_equal_gathered": not differing,
            "stock_reload_builds_no_quantizer": getattr(reloaded, "hf_quantizer", None) is None,
            "stock_reload_loss_matches_ep": abs(reload_loss - ep_loss) < TOL.parallel_vs_baseline_loss_abs,
        },
        "reload_loss": reload_loss,
    }


@gpu_test_main(exact_world_size=2, prefix="native_mxfp4_roundtrip")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    root = shared_output_dir(ctx)
    source = os.path.join(root, "gpt_oss_native_mxfp4")
    train_out = os.path.join(root, "train_out")
    ckpt_dir = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    gathered_dir, sharded_dir, merged_dir = (os.path.join(root, kind) for kind in ("gathered", "sharded", "merged"))

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if ctx.rank == 0:
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES["gpt_oss"], source, tokenizer, SEED)
        write_native_mxfp4(source)
    ctx.barrier()
    checks["source_claims_mxfp4"] = _claims_mxfp4(source)
    checks["lazy_gate_admits_source"] = is_lazy_loadable(source)

    pc = ParallelismConfig(ep_size=ctx.world_size)
    dataset = create_sft_dataset(32, tokenizer, seed=SEED)

    log(f"--- train {SAVE_AT_STEP} steps from the MXFP4 source, checkpoint at step {SAVE_AT_STEP} ---")
    trainer = _trainer(source, pc, tokenizer, dataset, _sft_config(train_out, SAVE_AT_STEP, SAVE_AT_STEP))
    ctx.on_teardown(trainer.cleanup_ep)
    checks["loaded_config_does_not_claim_mxfp4"] = getattr(trainer.model.config, "quantization_config", None) is None
    trainer.train()
    first = step_losses(trainer)
    checks["first_phase_losses_finite"] = len(first) == SAVE_AT_STEP and all(math.isfinite(x) for x in first)
    del trainer
    cleanup_memory()
    ctx.barrier()
    checks["checkpoint_config_does_not_claim_mxfp4"] = not _claims_mxfp4(ckpt_dir)

    log(f"--- resume from checkpoint-{SAVE_AT_STEP} to step {TOTAL_STEPS} ---")
    trainer = _trainer(
        ckpt_dir,
        pc,
        tokenizer,
        dataset,
        _sft_config(train_out, TOTAL_STEPS, None),
        preserve_checkpoint_precision=True,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train(resume_from_checkpoint=ckpt_dir)
    resumed = step_losses(trainer)
    checks["resumed_to_the_last_step"] = trainer.state.global_step == TOTAL_STEPS
    checks["resumed_losses_finite"] = bool(resumed) and all(math.isfinite(x) for x in resumed)

    vocab_size = trainer.model.config.get_text_config().vocab_size  # The tokenizer's, as the build sizes it.
    ids, labels = random_token_batch(vocab_size, batch=2, seq=64, device=ctx.device, seed=SEED)
    ep_loss = fixed_batch_loss(trainer.model, ids, labels)
    metrics["ep_loss"] = ep_loss
    checks["ep_loss_finite"] = math.isfinite(ep_loss)

    log("--- gathered save, sharded save, merge, stock reload ---")
    trainer.save_model(gathered_dir)
    save_ep_sharded_checkpoint(save_context(trainer.model), sharded_dir)
    ctx.barrier()
    checks["gathered_config_does_not_claim_mxfp4"] = not _claims_mxfp4(gathered_dir)
    verdict = [
        _rank0_merge_and_reload(sharded_dir, merged_dir, gathered_dir, ids, labels, ep_loss) if ctx.rank == 0 else None
    ]
    dist.broadcast_object_list(verdict, src=0)
    checks |= verdict[0]["checks"]
    metrics["reload_loss"] = verdict[0]["reload_loss"]
    checks = {name: world_all(ok, ctx.device) for name, ok in checks.items()}
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
