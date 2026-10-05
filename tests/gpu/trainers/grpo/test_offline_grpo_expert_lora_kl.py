#!/usr/bin/env python
"""Native EP8 expert LoRA: KL stays anchored to the configured base through resume/export.

Uses the training script's reference loader and a local tiny Qwen3-MoE, with no Hub downloads.
Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_expert_lora_kl.py
"""

import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM

from scripts.training.offline_grpo import _load_kl_reference
from src.args.distributed_args import DistributedArguments
from src.args.offline_grpo_args import OfflineGRPOScriptArguments
from src.checkpoint.format import RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.environment import resolve_resume_weights_source
from src.training.script_runner import ScriptRuntime
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import offline_grpo_dataset, save_offline_moe_base
from tests.common.peft_helpers import (
    assert_only_adapters_trainable,
    is_expert_lora_active,
    load_peft_model_from_config,
    peft_model_config,
    snapshot_adapters,
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

SEED = 913
STEPS = 3
SAVE_STEP = 1
BETA = 0.2
LR = 0.01


def _config(output):
    return OfflineGRPOConfig(
        output_dir=output,
        max_steps=STEPS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=LR,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        kl_beta=BETA,
        use_chunked_grpo_logprobs=True,
        loss_type="grpo",
        policy_gradient_formulation="reinforce",
        min_log_prob=None,
        max_grad_norm=0.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=SAVE_STEP,
        save_total_limit=STEPS,
        report_to="none",
        max_prompt_length=16,
        max_completion_length=16,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        fsdp="",
    )


def _build(ctx, base, output, train, evaluation, *, checkpoint=None):
    parallelism = ParallelismConfig(ep_size=8, merge_expert_lora_on_save=True)
    configured = peft_model_config("expert_lora", parallelism, model_name=base)
    model, tokenizer, peft_config = load_peft_model_from_config(
        configured, parallelism, attn_implementation="flash_attention_2", use_liger_kernel=False
    )
    args = _config(output)
    # A trained runtime source must not replace model_name_or_path as the reference's base.
    runtime = ScriptRuntime(parallelism, "ep8", ctx.local_rank, checkpoint, checkpoint or base)
    reference = _load_kl_reference(
        OfflineGRPOScriptArguments(),
        runtime,
        args,
        configured,
        DistributedArguments(expert_parallel_size=8),
        policy=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        attn_default="flash_attention_2",
    )
    return OfflineGRPOTrainer(
        model=model,
        ref_model=reference,
        args=args,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=parallelism,
        resume_checkpoint=checkpoint,
        moe_balancing="none",
    )


def _token_logps(model, ids):
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        return logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _reference_oracle(ctx, reference, base, checkpoint, checks):
    ids = torch.tensor([[3, 8, 4, 5, 8, 8, 1], [3, 9, 4, 6, 10, 7, 1]], device=ctx.device)
    # Separate full models share the loader's requested kernel, not its choice of weights source.
    original = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    original.to(ctx.device).eval()
    expected = _token_logps(original, ids)
    actual = _token_logps(reference, ids)
    checks["live_reference_matches_configured_base_exactly"] = torch.equal(expected, actual)
    del original
    trained = (
        AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        .to(ctx.device)
        .eval()
    )
    wrong = _token_logps(trained, ids)
    effect = (wrong - expected).abs().max().item()
    checks["trained_checkpoint_is_a_distinct_kl_anchor"] = effect > TOL.control_min_loss_shift(TOL.weight_atol)
    checks["reference_is_not_trained_checkpoint"] = not torch.equal(actual, wrong)
    log(
        f"native expert-LoRA reference: base max error={(actual - expected).abs().max().item():.5g}, trained-anchor shift={effect:.5g}"
    )
    del trained
    cleanup_memory()


def _kl_oracle(ctx, trainer, export, checks):
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[index] for index in range(2)]))
    batch["advantage"].zero_()
    ids = torch.cat([batch["prompt_input_ids"], batch["completion_input_ids"]], dim=1)
    mask = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
    width = batch["completion_input_ids"].size(1)
    oracle = (
        AutoModelForCausalLM.from_pretrained(export, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        .to(ctx.device)
        .eval()
    )
    with torch.no_grad():
        oracle.get_output_embeddings().weight.mul_(2)
        logits = oracle(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1, :].float()
        policy = logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -width:]
        logits = trainer.ref_model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1, :].float()
        reference = logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -width:]
        delta = torch.minimum(reference, policy + KL_LOGRATIO_CLAMP) - policy
        valid = batch["completion_attention_mask"]
        weights = batch["group_size"].float().reciprocal()
        expected = BETA * (((delta.exp() - delta - 1) * valid).sum(1) / valid.sum(1) * weights).sum() / weights.sum()
    del oracle
    reshard_fsdp2_modules(trainer.model)
    original = trainer.model.get_output_embeddings().weight.detach().clone()
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.mul_(2)
            actual = trainer.compute_loss(trainer.model, batch)
    finally:
        reshard_fsdp2_modules(trainer.model)
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.copy_(original)
        trainer.model.train(was_training)
    error = abs((actual - expected).item())
    checks["live_reference_kl_oracle_is_nonzero"] = bool(expected > 0)
    checks["live_reference_kl_matches_independent_logits"] = error < TOL.exact_objective_rel * expected.item()
    log(f"native expert-LoRA pure KL: expected={expected.item():.5g}, error={error:.5g}")


def run(ctx):
    pin_deterministic_ep_dispatch()
    paths = [ctx.output_dir]
    dist.broadcast_object_list(paths, src=0)
    shared = paths[0]
    base, output = os.path.join(shared, "base"), os.path.join(shared, "train")
    if ctx.rank == 0:
        save_offline_moe_base(base, SEED)
    dist.barrier()
    train, evaluation = offline_grpo_dataset(24), offline_grpo_dataset(8, 4)
    trainer = _build(ctx, base, output, train, evaluation)
    only_adapters, reason = assert_only_adapters_trainable(trainer.model)
    checks = {
        "native_expert_only_no_peft_wrapper": is_expert_lora_active(trainer.model)
        and not isinstance(trainer.model, PeftModel),
        "only_native_adapters_train": only_adapters,
        "native_lora_keeps_live_frozen_reference": trainer.ref_model is not None and not trainer._precompute_reference,
        "live_reference_parameters_are_frozen": trainer.ref_model is not None
        and all(not parameter.requires_grad for parameter in trainer.ref_model.parameters()),
    }
    log(reason)
    initial = snapshot_adapters(trainer.model, expert_lora=True)
    at_save = RestorePointSnapshot("save", trainer, capture_optimizer=True, expert_lora=True)
    trainer.add_callback(at_save)
    result = trainer.train()
    continuous_losses = step_losses(trainer)
    continuous = snapshot_adapters(trainer.model, expert_lora=True)
    checks["train_with_kl_finite"] = trainer.state.global_step == STEPS and math.isfinite(result.training_loss)
    checks["evaluation_with_kl_finite"] = math.isfinite(trainer.evaluate()["eval_loss"])
    checks["adapters_have_nonzero_update"] = bool(initial) and any(
        not torch.equal(initial[name], continuous[name]) for name in initial
    )
    checkpoint = os.path.join(output, f"checkpoint-{SAVE_STEP}")
    checks["merged_checkpoint_keeps_unmerged_resume_adapter"] = os.path.isfile(
        os.path.join(checkpoint, RESUME_ADAPTER_MARKER_FILE)
    ) and os.path.isdir(os.path.join(checkpoint, RESUME_ADAPTER_DIR))
    finish_phase(trainer)
    at_save.trainer = None
    del trainer
    cleanup_memory()

    source = resolve_resume_weights_source(
        checkpoint,
        SimpleNamespace(model_name_or_path=base),
        ParallelismConfig(ep_size=8, merge_expert_lora_on_save=True),
    )
    checks["policy_rebuild_uses_original_base"] = source == base
    with patch.object(
        OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("swept live LoRA reference")
    ):
        resumed = _build(ctx, source, output, train, evaluation, checkpoint=checkpoint)
    _reference_oracle(ctx, resumed.ref_model, base, checkpoint, checks)
    restored = RestorePointSnapshot("train_begin", resumed, capture_optimizer=True, expert_lora=True)
    resumed.add_callback(restored)
    resumed.train(resume_from_checkpoint=checkpoint)
    final = snapshot_adapters(resumed.model, expert_lora=True)
    before, after = (at_save.captured or {}).get("adapters", {}), (restored.captured or {}).get("adapters", {})
    checks["saved_adapters_restored_bit_exactly"] = (
        bool(before)
        and before.keys() == after.keys()
        and all(torch.equal(before[name], after[name]) for name in before)
    )
    saved_optimizer = (at_save.captured or {}).get("optimizer") or {"state": {}}
    restored_optimizer = (restored.captured or {}).get("optimizer") or {"state": {}}
    optimizer_exact, reason = optimizer_state_matches(saved_optimizer, restored_optimizer)
    checks["saved_optimizer_state_restored_bit_exactly"] = optimizer_exact
    if not optimizer_exact:
        log(f"Native expert-LoRA optimizer restore: {reason}")
    checks["resumed_final_adapters_bit_exact"] = continuous.keys() == final.keys() and all(
        torch.equal(continuous[name], final[name]) for name in continuous
    )
    deltas = resumed_loss_deltas(continuous_losses, step_losses(resumed), save_step=SAVE_STEP, total_steps=STEPS)
    checks["resumed_losses_match_continuous"] = deltas is not None and max(deltas) < TOL.replayed_resume_loss_abs
    checks["resumed_evaluation_finite"] = math.isfinite(resumed.evaluate()["eval_loss"])
    export = os.path.join(shared, "export")
    resumed.save_model(export)
    dist.barrier()
    _kl_oracle(ctx, resumed, export, checks)
    if ctx.rank == 0:
        served = AutoModelForCausalLM.from_pretrained(export, dtype=torch.bfloat16, attn_implementation="eager")
        checks["merged_hf_export_loads_and_scores"] = bool(
            served(torch.tensor([[3, 8, 4, 5, 1]])).logits.isfinite().all()
        )
        del served
    metrics = ctx.metrics(resumed)
    finish_phase(resumed)
    del resumed
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_expert_lora_kl")(run)

if __name__ == "__main__":
    main()
