#!/usr/bin/env python
"""TP2×DP2 and ETP2×DP2 run-start KL: ordered scores, exact resume and HF export.

Local tiny Qwen3 checkpoints exercise distinct DP shards and collective-model siblings.
Run: torchrun --nproc_per_node=4 tests/gpu/trainers/grpo/test_offline_grpo_sibling_reference.py --mode tp
Use --mode etp for Qwen3-MoE with pure expert tensor parallelism.
"""

import argparse
import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.parallelism_config import ParallelismConfig
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import (
    OFFLINE_VOCAB,
    build_offline_grpo_trainer,
    doubled_head_kl_verdict,
    full_logits_kl,
    make_offline_tokenizer,
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

SEED, BETA, STEPS, SAVE_STEP = 1337, 0.2, 2, 1


def sibling_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tp", "etp"), required=True)
    return parser


def _save_base(path, mode):
    if mode == "etp":
        save_offline_moe_base(path, SEED)
        return
    torch.manual_seed(SEED)
    config = Qwen3Config(
        vocab_size=len(OFFLINE_VOCAB),
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    make_offline_tokenizer().save_pretrained(path)


def _build(ctx, mode, source, output, train, evaluation=None, *, checkpoint=None, beta=BETA):
    parallelism = ParallelismConfig(tp_size=2) if mode == "tp" else ParallelismConfig(expert_tp_size=2)
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
    mode = sibling_parser().parse_args().mode
    shared = shared_output_dir(ctx)
    base, output = os.path.join(shared, "base"), os.path.join(shared, "train")
    if ctx.rank == 0:
        _save_base(base, mode)
    dist.barrier()
    train, evaluation = offline_grpo_dataset(16), offline_grpo_dataset(8, 3)
    expected = reference_oracle_rows(_build(ctx, mode, base, os.path.join(shared, "oracle"), train, beta=0.0))
    trainer = _build(ctx, mode, base, output, train, evaluation)
    rank_map = trainer._data_parallel_rank_by_global_rank()
    checks = {
        "two_dp_shards_with_two_siblings_each": sorted(rank_map) == [0, 0, 1, 1],
        "noncp_full_ft_without_live_reference": not trainer.parallelism_config.is_cp_mode
        and trainer.ref_model is None,
        "distinct_rows_make_order_check_nonvacuous": any(not torch.equal(expected[0], row) for row in expected[2::2]),
    }
    checks["swept_reference_matches_full_logits_in_dataset_order"] = (
        swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol
    )
    initial_train = trainer._reference_storage_by_split["training"].values.clone()
    initial_eval = trainer._reference_storage_by_split["evaluation"].values.clone()
    batch = pure_kl_batch(trainer)
    checks["independent_kl_oracle_is_nonzero"], checks["kl_matches_independent_full_logits"] = doubled_head_kl_verdict(
        trainer, batch, lambda model: full_logits_kl(model, batch, BETA), f"{mode.upper()}2×DP2 KL"
    )
    saved = RestorePointSnapshot("save", trainer, capture_optimizer=True)
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
    del trainer, saved
    cleanup_memory()

    with patch.object(OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("reswept reference")):
        resumed = _build(ctx, mode, checkpoint, output, train, evaluation, checkpoint=checkpoint)
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
    checks["optimizer_restore_has_state_coverage"] = bool(captured.get("optimizer", {}).get("state"))
    checks["optimizer_restored_bit_exact"] = optimizer_exact
    if not optimizer_exact:
        log(f"{mode} optimizer restore: {reason}")
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


main = gpu_test_main(exact_world_size=4, prefix="offline_grpo_sibling_reference")(run)

if __name__ == "__main__":
    main()
