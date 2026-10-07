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
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.environment import resolve_resume_weights_source
from src.training.script_runner import ScriptRuntime
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import (
    completion_logps,
    doubled_head_kl_verdict,
    doubled_output_head,
    export_scores_finite,
    offline_grpo_config,
    offline_grpo_dataset,
    offline_grpo_trainer,
    pure_kl_batch,
    pure_kl_objective,
    save_offline_moe_base,
    token_logps,
)
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


def _fa2_model(path, device):
    """A separate full model on the loader's requested kernel, independent of the trainer's weights source."""
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    return model.to(device).eval()


def _build(ctx, base, output, train, evaluation, *, checkpoint=None):
    parallelism = ParallelismConfig(ep_size=8, merge_expert_lora_on_save=True)
    configured = peft_model_config("expert_lora", parallelism, model_name=base)
    model, tokenizer, peft_config = load_peft_model_from_config(
        configured, parallelism, attn_implementation="flash_attention_2", use_liger_kernel=False
    )
    args = offline_grpo_config(
        output,
        steps=STEPS,
        save_steps=SAVE_STEP,
        seed=SEED,
        kl_beta=BETA,
        learning_rate=LR,
        sequence_length=16,
        lr_scheduler_type="constant",
    )
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
    return offline_grpo_trainer(
        ctx,
        model,
        parallelism,
        args,
        train,
        evaluation,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        ref_model=reference,
        peft_config=peft_config,
    )


def _reference_oracle(ctx, reference, base, checkpoint, checks):
    ids = torch.tensor([[3, 8, 4, 5, 8, 8, 1], [3, 9, 4, 6, 10, 7, 1]], device=ctx.device)
    original = _fa2_model(base, ctx.device)
    with torch.no_grad():
        expected = token_logps(original, ids)
        actual = token_logps(reference, ids)
    checks["live_reference_matches_configured_base_exactly"] = torch.equal(expected, actual)
    del original
    trained = _fa2_model(checkpoint, ctx.device)
    with torch.no_grad():
        wrong = token_logps(trained, ids)
    effect = (wrong - expected).abs().max().item()
    checks["trained_checkpoint_is_a_distinct_kl_anchor"] = effect > TOL.control_min_loss_shift(TOL.weight_atol)
    checks["reference_is_not_trained_checkpoint"] = not torch.equal(actual, wrong)
    log(
        f"native expert-LoRA reference: base max error={(actual - expected).abs().max().item():.5g}, trained-anchor shift={effect:.5g}"
    )
    del trained
    cleanup_memory()


def _kl_oracle(ctx, trainer, export, checks):
    batch = pure_kl_batch(trainer)
    oracle = _fa2_model(export, ctx.device)
    with doubled_output_head(oracle), torch.no_grad():
        policy = completion_logps(oracle, batch)
        reference = completion_logps(trainer.ref_model, batch)
    expected = pure_kl_objective(policy, reference, batch["completion_attention_mask"], batch["group_size"], BETA)
    del oracle
    checks["live_reference_kl_oracle_is_nonzero"], checks["live_reference_kl_matches_independent_logits"] = (
        doubled_head_kl_verdict(trainer, batch, lambda _model: expected, "native expert-LoRA pure KL")
    )


def run(ctx):
    pin_deterministic_ep_dispatch()
    shared = shared_output_dir(ctx)
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
        checks["merged_hf_export_loads_and_scores"] = export_scores_finite(export, ctx.device)
    metrics = ctx.metrics(resumed)
    finish_phase(resumed)
    del resumed
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_expert_lora_kl")(run)

if __name__ == "__main__":
    main()
