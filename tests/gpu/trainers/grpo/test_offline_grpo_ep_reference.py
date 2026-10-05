#!/usr/bin/env python
"""EP8 full fine-tuning: run-start KL scores survive train, resume, evaluation and HF export.

Uses local tiny Qwen3-MoE weights and a full-logits EP8 oracle, without context parallelism.
Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_ep_reference.py
Set HALO_TEST_OFFLINE_GRPO_EP_LAZY=0 to exercise eager checkpoint loading.
"""

import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_flag
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset, save_offline_moe_base
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


def _build(source, output, train, evaluation=None, *, checkpoint=None, beta=BETA):
    parallelism = ParallelismConfig(
        ep_size=8, ep_fp32_router=True, ep_lazy_loading=env_flag("HALO_TEST_OFFLINE_GRPO_EP_LAZY", True)
    )
    model, _ = load_distributed_model(
        model_name_or_path=source,
        parallelism_config=parallelism,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        use_liger_kernel=False,
        preserve_checkpoint_precision=checkpoint is not None,
    )
    args = OfflineGRPOConfig(
        output_dir=output,
        max_steps=STEPS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-3,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        kl_beta=beta,
        use_chunked_grpo_logprobs=True,
        loss_type="grpo",
        policy_gradient_formulation="reinforce",
        min_log_prob=None,
        max_grad_norm=0.0,
        logging_steps=1,
        eval_strategy="steps" if evaluation is not None else "no",
        eval_steps=1,
        save_strategy="steps" if evaluation is not None else "no",
        save_steps=SAVE_STEP,
        save_total_limit=STEPS,
        report_to="none",
        max_prompt_length=32,
        max_completion_length=32,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        fsdp="",
    )
    return OfflineGRPOTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=make_offline_tokenizer(),
        parallelism_config=parallelism,
        resume_checkpoint=checkpoint,
        moe_balancing="none",
    )


def _full_logps(model, batch):
    ids = torch.cat([batch["prompt_input_ids"], batch["completion_input_ids"]], dim=1)
    attention = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    width = batch["completion_input_ids"].size(1)
    return logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -width:]


def _reference_oracle(source, output, train):
    oracle = _build(source, output, train, beta=0.0)
    batch = oracle._prepare_inputs(oracle.data_collator(list(oracle.train_dataset)))
    oracle.model.eval()
    with torch.no_grad():
        logps = _full_logps(oracle.model, batch)
    rows = [
        row[: int(mask.sum())].detach().cpu().clone()
        for row, mask in zip(logps, batch["completion_attention_mask"], strict=True)
    ]
    finish_phase(oracle)
    del oracle
    cleanup_memory()
    return rows


def _kl_oracle(trainer, checks):
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[index] for index in range(2)]))
    batch["advantage"].zero_()
    reshard_fsdp2_modules(trainer.model)
    original = trainer.model.get_output_embeddings().weight.detach().clone()
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.mul_(2)
            policy = _full_logps(trainer.model, batch)
            delta = batch[REF_PER_TOKEN_LOGPS_COLUMN] - policy
            mask = batch["completion_attention_mask"]
            weights = batch["group_size"].float().reciprocal()
            expected = BETA * (((delta.exp() - delta - 1) * mask).sum(1) / mask.sum(1) * weights).sum() / weights.sum()
            actual = trainer.compute_loss(trainer.model, batch)
    finally:
        reshard_fsdp2_modules(trainer.model)
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.copy_(original)
        trainer.model.train(was_training)
    error = abs((actual - expected).item())
    checks["independent_kl_oracle_is_nonzero"] = bool(expected > 0)
    checks["kl_matches_independent_full_logits"] = error < TOL.exact_objective_rel * expected.item()
    log(f"EP8 non-CP pure KL: expected={expected.item():.5g}, error={error:.5g}")


def run(ctx):
    pin_deterministic_ep_dispatch()
    paths = [ctx.output_dir]
    dist.broadcast_object_list(paths, src=0)
    shared = paths[0]
    base, output = os.path.join(shared, "base"), os.path.join(shared, "train")
    if ctx.rank == 0:
        save_offline_moe_base(base, SEED)
    dist.barrier()
    train, evaluation = offline_grpo_dataset(16), offline_grpo_dataset(8, 3)
    expected = _reference_oracle(base, os.path.join(shared, "oracle"), train)
    trainer = _build(base, output, train, evaluation)
    checks = {
        "noncp_full_ft_without_live_reference": not trainer.parallelism_config.is_cp_mode and trainer.ref_model is None
    }
    references = trainer._reference_storage_by_split
    initial_train = references["training"].values.clone()
    initial_eval = references["evaluation"].values.clone()
    errors = [
        (torch.as_tensor(actual) - reference).abs().max().item()
        for actual, reference in zip(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected, strict=True)
    ]
    checks["swept_reference_matches_full_logits"] = bool(errors) and max(errors) < TOL.logprob_atol
    _kl_oracle(trainer, checks)
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
        resumed = _build(checkpoint, output, train, evaluation, checkpoint=checkpoint)
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
        before, after = (
            load_file(os.path.join(continuous, "model.safetensors")),
            load_file(os.path.join(export, "model.safetensors")),
        )
        previous = load_file(os.path.join(checkpoint, "model.safetensors"))
        checks["resumed_export_bit_exact"] = before.keys() == after.keys() and all(
            torch.equal(before[name], after[name]) for name in before
        )
        checks["resumed_step_updates_weights"] = previous.keys() == after.keys() and any(
            not torch.equal(previous[name], after[name]) for name in previous
        )
        served = AutoModelForCausalLM.from_pretrained(export, dtype=torch.bfloat16, attn_implementation="eager")
        checks["hf_export_loads_and_scores"] = bool(served(torch.tensor([[3, 8, 4, 5, 1]])).logits.isfinite().all())
        del served
    metrics = ctx.metrics(resumed)
    finish_phase(resumed)
    restored.trainer = None
    del resumed
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_ep_reference")(run)

if __name__ == "__main__":
    main()
