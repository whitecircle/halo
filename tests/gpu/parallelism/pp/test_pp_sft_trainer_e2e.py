#!/usr/bin/env python
"""End-to-end PP through the real DistributedSFTTrainer: the stage split, FSDP wrap, schedule, clip and sharded save.

Without the signature-column pin HF silently drops input_ids/labels from the dataset; double accumulation shows as a
stalled LR.
"""

import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3ForCausalLM, default_data_collator
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model, full_batch_eval_loss
from tests.common.pp_trainers import (
    CheckpointCallsInTrain,
    build_pp_args,
    build_pp_trainer,
    build_sft_rows,
    losses_identical_across_ranks,
)
from tests.common.rank_log import log
from tests.common.tolerances import TOL

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 12


@gpu_test_main(exact_world_size=4, prefix="pp_sft_e2e")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    args = build_pp_args(
        SFTConfig,
        ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=4,
        max_length=SEQ_LEN,
        # Trained dtype == save dtype, so the save is lossless and the reload must reproduce the eval loss to float
        # precision.
        bf16=True,
        gradient_checkpointing=True,  # TRL's default; exercises the stage's non-reentrant GC delegate.
        # 8 eval rows over dp=2 at eval_bs==train_bs: one full batch per replica, evals at steps 6 and 12.
        eval_strategy="steps",
        eval_steps=6,
        per_device_eval_batch_size=4,
    )

    eval_dataset = build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"], n_rows=8, seed=23)
    trainer, steps, _ = build_pp_trainer(
        DistributedSFTTrainer,
        model=build_dense_model(),
        args=args,
        train_dataset=build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"]),
        eval_dataset=eval_dataset,
        processing_class=AutoTokenizer.from_pretrained(QWEN3_0_6B),
        data_collator=default_data_collator,
        parallelism_config=config,
    )

    checks, metrics = {}, {}
    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == 2
    )

    checkpointed = CheckpointCallsInTrain(trainer)
    trainer.train()
    checks["train_checkpointed_the_stage_layers"] = checkpointed.calls > 0
    metrics["checkpointed_layer_calls"] = checkpointed.calls

    # Every rank of a chain must consume the same batch: stage 0 embeds the rows whose labels the last stage scores.
    # Accelerate's default global-rank sharding breaks this.
    chain_ok = True
    for step in steps:
        step_sum = step["input_ids"].to(torch.float64).sum()
        peers = [torch.zeros_like(step_sum) for _ in range(config.pp_size)]
        dist.all_gather(peers, step_sum, group=trainer._pp_chain_group)
        chain_ok &= all(torch.equal(peers[0], p) for p in peers)
    checks["chain_ranks_consume_identical_batches"] = chain_ok and len(steps) == N_STEPS

    # Stage-summed loss-token count / stage_world_size (= G/dp). An unwired one fails here though the loss still
    # decreases.
    stage_world = trainer.parallelism_config.stage_world_size
    normalizer_ok = len(steps) == N_STEPS
    for step in steps:
        expected = (step["labels"][:, 1:] != -100).sum().float().reshape(1)
        dist.all_reduce(expected, group=trainer._pp_stage_group)
        normalizer_ok &= abs(step["num_items"] - float(expected) / stage_world) < TOL.recomputed_normalizer_abs
    checks["normalizer_is_stage_global_count_over_dp"] = normalizer_ok
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"[rank {ctx.rank}] losses={[f'{v:.4f}' for v in losses]}")

    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and all(torch.isfinite(torch.tensor(losses)))
    # A tiny model memorizing 64 rows: a flat loss means a schedule built without its loss_fn.
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # A divergence means chains saw different data or the broadcast source is wrong.
    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and all(torch.isfinite(torch.tensor(grad_norms)))

    # The step-12 eval sees exactly the weights the save captures, so a reload verifies it below.
    eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["eval_ran_twice_and_finite"] = len(eval_losses) == 2 and all(torch.isfinite(torch.tensor(eval_losses)))
    checks["eval_loss_identical_across_ranks"] = losses_identical_across_ranks(eval_losses or [float("nan")])
    metrics["final_eval_loss"] = eval_losses[-1] if eval_losses else float("nan")

    # The harness gives each rank its own output dir, but the PP save is multi-writer single-directory: use rank 0's.
    save_dir = shared_output_dir(ctx)
    # TRL copied the full-size tokenizer's special-token ids into the tiny config; restore ids in the tiny vocab, or
    # from_pretrained's padding_idx assert reflects the size mismatch.
    trainer.model.config.pad_token_id, trainer.model.config.eos_token_id = 0, 1
    trainer.save_model(save_dir)
    dist.barrier()
    checks_reload = True
    if ctx.rank == 0:
        # So a reload failure cannot strand ranks 1..3 at the broadcast below.
        try:
            reloaded = Qwen3ForCausalLM.from_pretrained(save_dir)
            # from_pretrained doesn't require it: a served export without generation_config.json samples with class
            # defaults.
            for artifact in ("config.json", "generation_config.json"):
                if not os.path.isfile(os.path.join(save_dir, artifact)):
                    log(f"[reload] export missing {artifact}")
                    checks_reload = False
            reference_keys = set(build_dense_model().state_dict())
            reloaded_keys = set(reloaded.state_dict())
            if reloaded_keys != reference_keys:
                log(f"reload key mismatch: missing={reference_keys - reloaded_keys}")
                checks_reload = False
            # A bf16 run's export is uniformly bf16; an fp32 tensor means the writer bypassed save_dtype_caster.
            off_dtype = [
                k for k, v in reloaded.state_dict().items() if v.is_floating_point() and v.dtype != torch.bfloat16
            ]
            if off_dtype:
                log(f"[reload] tensors off the save dtype: {off_dtype[:5]}")
                checks_reload = False
            # A reload equal to init means the save captured the wrong tensors. In fp32: init is default precision,
            # export bf16.
            init_sd = build_dense_model().state_dict()
            moved = sum(
                not torch.allclose(reloaded.state_dict()[k].float(), init_sd[k].float()) for k in reference_keys
            )
            log(f"[reload] params changed from init: {moved}/{len(reference_keys)}")
            if moved < len(reference_keys) * 0.9:
                checks_reload = False
            # HF's eval_loss is the average of the two DP shards' mean CE here. Catches a wrong normalizer, stale
            # weights or a bad save.
            reference_eval = full_batch_eval_loss(reloaded, eval_dataset, 4)
            log(f"[reload] reference eval={reference_eval:.6f} logged={eval_losses[-1]:.6f}")
            if abs(reference_eval - eval_losses[-1]) / abs(reference_eval) > 1e-4:
                checks_reload = False
        except Exception as exc:
            log(f"[reload] FAILED: {exc!r}")
            checks_reload = False
    flag = torch.tensor([1.0 if checks_reload else 0.0], device=torch.cuda.current_device())
    dist.broadcast(flag, src=0)
    checks["save_reloads_into_unsplit_model"] = bool(flag.item())

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
