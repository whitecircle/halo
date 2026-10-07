#!/usr/bin/env python
"""EP8 full fine-tuning: run-start KL scores survive train, resume, evaluation and HF export.

Uses local tiny Qwen3-MoE weights and a full-logits EP8 oracle, without context parallelism.
Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_ep_reference.py
Use --ep-loading lazy or --ep-loading eager to select checkpoint loading.
"""

import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import (
    build_offline_grpo_trainer,
    doubled_head_kl_verdict,
    ep_loading_parser,
    full_logits_kl,
    offline_grpo_config,
    offline_grpo_dataset,
    pure_kl_batch,
    reference_oracle_rows,
    resumed_export_verdict,
    save_offline_moe_base,
    swept_reference_error,
)
from tests.common.tolerances import TOL
from tests.common.utils import (
    cleanup_memory,
    finish_phase,
    log,
    optimizer_state_matches,
    resumed_loss_deltas,
    step_losses,
)

SEED = 1337
BETA = 0.2
STEPS = 2
SAVE_STEP = 1


class _SavedState(RestorePointSnapshot):
    def extra(self):
        return {
            "masters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.trainer.model.named_parameters()
                if not isinstance(parameter, DTensor) and parameter.dtype == torch.float32
            }
        }


def _build(ctx, source, output, train, evaluation=None, *, ep_loading, checkpoint=None, beta=BETA):
    parallelism = ParallelismConfig(ep_size=8, ep_fp32_router=True, ep_lazy_loading=ep_loading == "lazy")
    args = offline_grpo_config(
        output,
        steps=STEPS,
        save_steps=SAVE_STEP,
        seed=SEED,
        kl_beta=beta,
        evaluate=evaluation is not None,
        save=evaluation is not None,
    )
    return build_offline_grpo_trainer(ctx, source, parallelism, args, train, evaluation, checkpoint=checkpoint)


def run(ctx):
    ep_loading = ep_loading_parser().parse_args().ep_loading
    pin_deterministic_ep_dispatch()
    shared = shared_output_dir(ctx)
    base, output = os.path.join(shared, "base"), os.path.join(shared, "train")
    if ctx.rank == 0:
        save_offline_moe_base(base, SEED)
    dist.barrier()
    train, evaluation = offline_grpo_dataset(16), offline_grpo_dataset(8, 3)
    expected = reference_oracle_rows(
        _build(ctx, base, os.path.join(shared, "oracle"), train, ep_loading=ep_loading, beta=0.0)
    )
    trainer = _build(ctx, base, output, train, evaluation, ep_loading=ep_loading)
    checks = {
        "noncp_full_ft_without_live_reference": not trainer.parallelism_config.is_cp_mode and trainer.ref_model is None
    }
    references = trainer._reference_storage_by_split
    initial_train = references["training"].values.clone()
    initial_eval = references["evaluation"].values.clone()
    checks["swept_reference_matches_full_logits"] = (
        swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol
    )
    batch = pure_kl_batch(trainer)
    checks["independent_kl_oracle_is_nonzero"], checks["kl_matches_independent_full_logits"] = doubled_head_kl_verdict(
        trainer, batch, lambda model: full_logits_kl(model, batch, BETA), "EP8 non-CP pure KL"
    )
    saved = _SavedState("save", trainer, capture_optimizer=True)
    trainer.add_callback(saved)
    result = trainer.train()
    losses = step_losses(trainer)
    checks["train_and_evaluate_with_kl"] = (
        result.global_step == STEPS
        and math.isfinite(result.training_loss)
        and math.isfinite(trainer.evaluate()["eval_loss"])
    )
    checks["production_bf16_optimizer"] = isinstance(
        getattr(trainer.optimizer, "optimizer", trainer.optimizer), AdamWBF16
    )
    continuous = os.path.join(shared, "continuous")
    trainer.save_model(continuous)
    checkpoint = os.path.join(output, f"checkpoint-{SAVE_STEP}")
    checks["reference_sidecar_saved"] = os.path.isfile(os.path.join(checkpoint, REFERENCE_LOGPS_FILE))
    captured = saved.captured or {}
    finish_phase(trainer)
    saved.trainer = None
    del trainer, saved, references
    cleanup_memory()

    with patch.object(OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("reswept reference")):
        resumed = _build(ctx, checkpoint, output, train, evaluation, ep_loading=ep_loading, checkpoint=checkpoint)
    reshard_fsdp2_modules(resumed.model)
    masters = captured.get("masters", {})
    current = dict(resumed.model.named_parameters())
    checks["fp32_router_restore_has_coverage"] = bool(masters)
    checks["fp32_masters_restored_exactly_before_forward"] = bool(masters) and all(
        current[name].dtype == torch.float32 and torch.equal(before, current[name].detach().cpu())
        for name, before in masters.items()
    )
    checks["train_reference_restored_bit_exact"] = torch.equal(
        initial_train, resumed._reference_storage_by_split["training"].values
    )
    checks["eval_reference_restored_bit_exact"] = torch.equal(
        initial_eval, resumed._reference_storage_by_split["evaluation"].values
    )
    restored = RestorePointSnapshot("train_begin", resumed, capture_optimizer=True)
    resumed.add_callback(restored)
    resumed.train(resume_from_checkpoint=checkpoint)
    resumed_state = restored.captured or {}
    optimizer_exact, reason = optimizer_state_matches(
        captured.get("optimizer") or {"state": {}}, resumed_state.get("optimizer") or {"state": {}}
    )
    checks["optimizer_restored_bit_exact"] = optimizer_exact
    if not optimizer_exact:
        log(f"EP8 non-CP optimizer restore: {reason}")
    checks["scheduler_and_step_restored"] = (
        resumed_state.get("global_step") == SAVE_STEP and resumed_state.get("sched_last_epoch") == SAVE_STEP
    )
    deltas = resumed_loss_deltas(losses, step_losses(resumed), save_step=SAVE_STEP, total_steps=STEPS)
    checks["resumed_loss_matches_continuous"] = bool(deltas) and max(deltas) < TOL.replayed_resume_loss_abs
    checks["resumed_evaluation_finite"] = math.isfinite(resumed.evaluate()["eval_loss"])
    export = os.path.join(shared, "export")
    resumed.save_model(export)
    dist.barrier()
    if ctx.rank == 0:
        (
            checks["resumed_export_bit_exact"],
            checks["resumed_step_updates_weights"],
            checks["hf_export_loads_and_scores"],
        ) = resumed_export_verdict(continuous, export, checkpoint, ctx.device)
    metrics = ctx.metrics(resumed)
    finish_phase(resumed)
    restored.trainer = None
    del resumed
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_ep_reference")(run)

if __name__ == "__main__":
    main()
