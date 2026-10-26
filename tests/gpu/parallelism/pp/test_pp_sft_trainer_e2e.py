#!/usr/bin/env python
"""End-to-end pipeline parallelism through the real DistributedSFTTrainer.

The runtime-level PP tests drive the engine directly; this one goes through the whole trainer
stack — ``_maybe_prepare_pipeline_model`` splitting the model before TRL sees it, the stage-scoped
FSDP wrap, the schedule-driven ``training_step`` with the loss broadcast, the PP-aware gradient
clip, and the sharded save with global names. It fails when any of those seams breaks:

  * the signature-column pin (without it HF silently drops input_ids/labels from the dataset)
  * gradient_accumulation_steps -> microbatch conversion (double accumulation shows as a stalled LR)
  * the loss broadcast (ranks would log different losses)
  * the per-stage safetensors save + merged index (reload into the unsplit class must round-trip,
    at the artifact contract's save dtype — the writer routes every tensor through save_dtype_caster)

Topology: 4 GPUs as 2 simulated nodes of 2 (NVLink domain = 2), pipeline crossing the node boundary.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_sft_trainer_e2e.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM, default_data_collator
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.tolerances import TOL
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 12


def build_dataset(vocab_size: int, n_rows: int = 16, seed: int = 7) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, vocab_size, (n_rows, SEQ_LEN), generator=g)
    # Ragged prompt masks (-100 prefixes of varying length): DP replicas then hold unequal loss-token
    # counts, which is what exercises the num_items_in_batch -> G/dp normalizer threading — the
    # failure mode is silently mis-scaled gradients, visible here as a stalled loss.
    labels = rows.clone()
    for i in range(labels.size(0)):
        labels[i, : 4 + (i % 11)] = -100
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


def _mean_ce(model, batch: dict) -> float:
    """Per-batch mean CE exactly as the eval step computes it (sum CE / non-ignored count)."""
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits, shift_labels = logits[:, :-1, :], batch["labels"][:, 1:]
    total = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(total / (shift_labels != -100).sum().clamp(min=1))


@gpu_test_main(exact_world_size=4, prefix="pp_sft_e2e")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    args = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # becomes the pipeline's microbatch count
        learning_rate=5e-3,
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        # Production precision. Also load-bearing for the reload check: trained dtype == save dtype
        # (save_dtype_caster exports bf16), so the save is lossless and the reloaded model must
        # reproduce the logged eval loss to float precision.
        bf16=True,
        gradient_checkpointing=True,  # TRL's default; exercises the stage's non-reentrant GC delegate
        # Loss-only PP evaluation through the schedule's forward-only path: 8 eval rows over dp=2
        # at eval_bs==train_bs = one full batch per replica, evals at steps 6 and 12.
        eval_strategy="steps",
        eval_steps=6,
        per_device_eval_batch_size=4,
    )

    eval_dataset = build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=8, seed=23)
    trainer = DistributedSFTTrainer(
        model=build_dense_model(),
        args=args,
        train_dataset=build_dataset(TINY_QWEN3_CONFIG["vocab_size"]),
        eval_dataset=eval_dataset,
        processing_class=AutoTokenizer.from_pretrained(QWEN3_0_6B),
        data_collator=default_data_collator,
        parallelism_config=config,
    )

    checks, metrics = {}, {}
    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == 2
    )

    # Spy on the runtime step to prove the loss normalizer is actually wired: it must equal the
    # stage-summed loss-token count / stage_world_size (= G/dp) every step. An unwired normalizer
    # (None, or HF's never-computed num_items_in_batch) fails here even though the loss still
    # decreases.
    captured: list[tuple[float, float]] = []
    batch_sums: list[torch.Tensor] = []
    real_step = trainer._pp_runtime.step

    def spying_step(input_ids, labels, **kwargs):
        local = float((labels[:, 1:] != -100).sum())
        normalizer = kwargs.get("num_items_in_batch")
        captured.append((float(normalizer) if normalizer is not None else float("nan"), local))
        batch_sums.append(input_ids.to(torch.float64).sum())
        return real_step(input_ids, labels, **kwargs)

    trainer._pp_runtime.step = spying_step

    trainer.train()

    # Chain consistency: every rank of a pipeline chain must consume the same batch (stage 0 embeds
    # the rows whose labels the last stage scores). Accelerate's default global-rank sharding breaks
    # exactly this — this check fails without the PP-aware dataloader path.
    chain_ok = True
    for step_sum in batch_sums:
        peers = [torch.zeros_like(step_sum) for _ in range(config.pp_size)]
        dist.all_gather(peers, step_sum, group=trainer._pp_chain_group)
        chain_ok &= all(torch.equal(peers[0], p) for p in peers)
    checks["chain_ranks_consume_identical_batches"] = chain_ok and len(batch_sums) == N_STEPS

    stage_world = trainer.parallelism_config.stage_world_size
    normalizer_ok = len(captured) == N_STEPS
    for norm_value, local_count in captured:
        expected = torch.tensor([local_count], device=torch.cuda.current_device())
        dist.all_reduce(expected, group=trainer._pp_stage_group)
        normalizer_ok &= abs(norm_value - float(expected) / stage_world) < TOL.recomputed_normalizer_abs
    checks["normalizer_is_stage_global_count_over_dp"] = normalizer_ok
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"[rank {ctx.rank}] losses={[f'{v:.4f}' for v in losses]}")

    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and all(torch.isfinite(torch.tensor(losses)))
    # Tiny model memorizing 64 fixed rows: the loss must actually move, or the schedule never
    # produced gradients (a schedule built without its loss_fn).
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # The broadcast makes every rank log the same loss; a divergence means chains saw different data
    # or the broadcast source is wrong.
    local = torch.tensor(losses, device=torch.cuda.current_device())
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and all(torch.isfinite(torch.tensor(grad_norms)))

    # Loss-only PP evaluation: two evals ran through the schedule's forward-only path, every rank
    # logs the same value (broadcast + world gather), and the value is verified against a reload
    # below (the step-12 eval sees exactly the weights the save captures).
    eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["eval_ran_twice_and_finite"] = len(eval_losses) == 2 and all(torch.isfinite(torch.tensor(eval_losses)))
    local_eval = torch.tensor(eval_losses or [float("nan")], device=torch.cuda.current_device())
    gathered_eval = [torch.zeros_like(local_eval) for _ in range(ctx.world_size)]
    dist.all_gather(gathered_eval, local_eval)
    checks["eval_loss_identical_across_ranks"] = all(torch.allclose(local_eval, g) for g in gathered_eval)
    metrics["final_eval_loss"] = eval_losses[-1] if eval_losses else float("nan")

    # Save: per-stage shards under global names, merged index, reload into the unsplit class.
    # The harness hands every rank its own isolated output dir; the PP save is a multi-writer
    # single-directory format, so all ranks must agree on rank 0's directory.
    dirs: list = [None] * ctx.world_size
    dist.all_gather_object(dirs, ctx.output_dir)
    save_dir = dirs[0]
    # TRL copied the (full-size) tokenizer's special-token ids into the tiny config; restore ids
    # that exist in the tiny vocab so from_pretrained's padding_idx assert reflects PP mechanics,
    # not the test's tokenizer/model size mismatch.
    trainer.model.config.pad_token_id, trainer.model.config.eos_token_id = 0, 1
    trainer.save_model(save_dir)
    dist.barrier()
    checks_reload = True
    if ctx.rank == 0:
        # try/except so a reload failure cannot strand ranks 1..3 at the broadcast below.
        try:
            reloaded = Qwen3ForCausalLM.from_pretrained(save_dir)
            # from_pretrained does not require the file, so its absence is asserted, not inferred:
            # a served export without generation_config.json samples with the class defaults.
            for artifact in ("config.json", "generation_config.json"):
                if not os.path.isfile(os.path.join(save_dir, artifact)):
                    log(f"[reload] export missing {artifact}")
                    checks_reload = False
            reference_keys = set(build_dense_model().state_dict())
            reloaded_keys = set(reloaded.state_dict())
            if reloaded_keys != reference_keys:
                log(f"reload key mismatch: missing={reference_keys - reloaded_keys}")
                checks_reload = False
            # Artifact contract: the PP writer routes every tensor through save_dtype_caster, so a
            # bf16 run's export is uniformly bf16. An fp32 tensor here means the caster was bypassed.
            off_dtype = [
                k for k, v in reloaded.state_dict().items() if v.is_floating_point() and v.dtype != torch.bfloat16
            ]
            if off_dtype:
                log(f"[reload] tensors off the save dtype: {off_dtype[:5]}")
                checks_reload = False
            # Training must have moved the weights — a reload equal to init means the save captured
            # the wrong tensors (or training silently did nothing). Compared in fp32: init is built
            # in default precision, the export is bf16.
            init_sd = build_dense_model().state_dict()
            moved = sum(
                not torch.allclose(reloaded.state_dict()[k].float(), init_sd[k].float()) for k in reference_keys
            )
            log(f"[reload] params changed from init: {moved}/{len(reference_keys)}")
            if moved < len(reference_keys) * 0.9:
                checks_reload = False
            # The step-12 eval_loss must reproduce on the reloaded full model: HF's eval_loss is
            # the mean of the gathered per-batch means, = the average of the two DP shards' mean
            # CE here. Catches a wrong eval normalizer, a stale-weights eval, or a bad save alike.
            eval_batches = [eval_dataset[i : i + 4] for i in (0, 4)]
            shard_means = [
                _mean_ce(reloaded, {k: torch.tensor(v) for k, v in batch.items()}) for batch in eval_batches
            ]
            reference_eval = sum(shard_means) / len(shard_means)
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
