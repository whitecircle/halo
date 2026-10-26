#!/usr/bin/env python
"""End-to-end PP+EP through the real DistributedSFTTrainer (pp2 × ep2, gpt-oss tiny MoE).

Exercises the full trainer stack on a MoE model: EP patching before the ctor, the split with the
threaded ``moe_balancing`` knob, the EP-aware stage FSDP wrap, the schedule-driven step with the
deferred EP grad sweep, bias-update balancing with a stage-scoped counter reduce, and the EP-aware
PP save (family gather under global names). It fails when any of those seams breaks:

  * losses not finite / not identical across ranks / not decreasing (dead grads or desync)
  * balancing biases zero (counters never recorded / reduce misrouted) or unequal across a
    stage's ranks (reduce_group not injected)
  * reload key-set drift (the EP gather naming — gpt-oss re-interleave, 2D biases — vs
    from_pretrained's expectations)
  * router_balancing_biases.pt missing global (unsplit-model) names for any of the 8 layers
  * the aux-loss + live-coefficient config not raising at construction (the severed-path gate)

Topology: 4 GPUs as 2 simulated NVLink domains of 2. Run:
    torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ep_sft_trainer_e2e.py
"""

import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GptOssForCausalLM, default_data_collator
from trl import SFTConfig

from halo.callbacks.balancing import BalancingBiasCallback
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG
from tests.common.pp_equivalence import build_gptoss_model
from tests.common.pp_trainers import (
    build_bias_balanced_sft_trainer,
    build_pp_args,
    build_sft_rows,
    expect_construction_error,
    losses_identical_across_ranks,
    probe_step,
    reload_reproduces_probe_loss,
)
from tests.common.utils import log

PP_SIZE = 2
EP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 12


def build_args(output_dir: str) -> SFTConfig:
    return build_pp_args(
        SFTConfig,
        output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=4,
        max_length=SEQ_LEN,
        bf16=True,
        gradient_checkpointing=True,  # TRL's default; exercises the stage's non-reentrant GC + EP dispatch replay
    )


@gpu_test_main(exact_world_size=4, prefix="pp_ep_sft_e2e")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    args = build_args(ctx.output_dir)

    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, ep_size=EP_SIZE
    )
    rows = build_sft_rows(TINY_GPTOSS_CONFIG["vocab_size"])
    trainer = build_bias_balanced_sft_trainer(build_gptoss_model(), args, config, tokenizer, rows)
    ctx.on_teardown(trainer.cleanup_ep)

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == 2
    )
    balancing_cbs = [c for c in trainer.callback_handler.callbacks if isinstance(c, BalancingBiasCallback)]
    checks["bias_balancing_callback_built"] = bool(balancing_cbs)
    checks["balancing_reduce_group_is_stage_scoped"] = balancing_cbs[0].reduce_group is trainer._pp_stage_group

    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"[rank {ctx.rank}] losses={[f'{v:.4f}' for v in losses]}")
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all())
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    # ── balancing biases: nonzero after training, identical across the stage's ranks ────────────
    stage = trainer._top_level_model()
    biases = {
        name: module.balancing_biases.detach().clone()
        for name, module in stage.named_modules()
        if getattr(module, "balancing_biases", None) is not None
    }
    checks["stage_has_4_balancing_routers"] = len(biases) == 4
    flat = torch.cat([b.flatten() for b in biases.values()])
    checks["balancing_biases_nonzero"] = bool((flat.abs() > 0).any())
    stage_gathered = [torch.zeros_like(flat) for _ in range(dist.get_world_size(trainer._pp_stage_group))]
    dist.all_gather(stage_gathered, flat, group=trainer._pp_stage_group)
    checks["balancing_biases_identical_across_stage"] = all(torch.equal(flat, g) for g in stage_gathered)

    # ── loss probe on the final weights: one more pipeline step (no optimizer update follows), so
    # the reloaded full model must reproduce this rank's per-replica loss on the same batch. This
    # is the value check on the EP gather — a save that permutes experts or swaps layers under each
    # other's global names keeps the key set intact but cannot reproduce the loss.
    probe_batch, probe_loss = probe_step(trainer, rows, 4)
    metrics["probe_loss"] = float(probe_loss)

    # ── save: EP gather under global names, reload into the unsplit class ───────────────────────
    save_dir = shared_output_dir(ctx)
    trainer.model.config.pad_token_id, trainer.model.config.eos_token_id = 0, 1
    trainer.save_model(save_dir)
    dist.barrier()
    reload_ok = True
    biases_file_ok = True
    values_ok = True
    if ctx.rank == 0:
        try:
            reloaded = GptOssForCausalLM.from_pretrained(save_dir)
            reference_keys = set(build_gptoss_model().state_dict())
            reloaded_keys = set(reloaded.state_dict())
            if reloaded_keys != reference_keys:
                log(
                    f"reload key mismatch: missing={sorted(reference_keys - reloaded_keys)[:8]} "
                    f"extra={sorted(reloaded_keys - reference_keys)[:8]}"
                )
                reload_ok = False
            else:
                init_sd = build_gptoss_model().state_dict()
                moved = sum(
                    not torch.allclose(reloaded.state_dict()[k].to(torch.bfloat16), init_sd[k]) for k in reference_keys
                )
                log(f"[reload] params changed from init: {moved}/{len(reference_keys)}")
                if moved < len(reference_keys) * 0.9:
                    reload_ok = False
        except Exception as exc:
            log(f"[reload] FAILED: {exc!r}")
            reload_ok = False
        try:
            # Value check: the reloaded model on this rank's probe batch must reproduce the probe
            # loss the pipeline computed on the final weights (grouped-mm EP vs dense HF experts).
            values_ok = reload_reproduces_probe_loss(
                reloaded.to(ctx.device).to(torch.bfloat16),
                probe_batch,
                rows,
                config.data_parallel_size,
                probe_loss,
                ctx.device,
            )
        except Exception as exc:
            log(f"[reload-value] FAILED: {exc!r}")
            values_ok = False
        try:
            saved_biases = torch.load(f"{save_dir}/router_balancing_biases.pt", weights_only=False)
            expected = {f"model.layers.{i}.mlp" for i in range(8)}
            if set(saved_biases) != expected:
                log(f"balancing-bias keys: {sorted(saved_biases)} (expected {sorted(expected)})")
                biases_file_ok = False
        except Exception as exc:
            log(f"[biases file] FAILED: {exc!r}")
            biases_file_ok = False
    # Bias values: the saved file must hold this run's actual biases — compare the stage's own
    # (globally renamed) biases against the file on every stage's first rank.
    bias_values_ok = True
    if trainer.parallelism_config.stage_local_rank == 0:
        saved_biases = torch.load(f"{save_dir}/router_balancing_biases.pt", weights_only=False)
        for local_name, bias in biases.items():
            global_name = stage.global_parameter_name(local_name)
            if global_name not in saved_biases or not torch.allclose(
                saved_biases[global_name].to(bias.device, bias.dtype), bias
            ):
                log(f"[bias-values] mismatch for {global_name}")
                bias_values_ok = False
    flags = torch.tensor(
        [1.0 if reload_ok else 0.0, 1.0 if biases_file_ok else 0.0, 1.0 if values_ok else 0.0],
        device=ctx.device,
    )
    dist.broadcast(flags, src=0)
    bias_values_ok = world_all(bias_values_ok)
    checks["save_reloads_into_unsplit_model"] = bool(flags[0].item())
    checks["balancing_biases_saved_globally_for_all_8_layers"] = bool(flags[1].item())
    checks["reloaded_model_reproduces_probe_loss"] = bool(flags[2].item())
    checks["saved_bias_values_match_in_memory"] = bias_values_ok

    # ── negative: aux-loss balancing with a live coefficient must be rejected at construction ───
    # (a PP stage severs the *ForCausalLM aux-loss add — silently unbalanced experts otherwise).
    expect_construction_error(
        DistributedSFTTrainer,
        "aux_loss_pp_rejected_at_construction",
        checks,
        "aux",
        model=build_gptoss_model(router_aux_loss_coef=0.02),
        args=build_args(ctx.output_dir),
        train_dataset=rows,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        parallelism_config=config,
        moe_balancing="aux_loss",
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
