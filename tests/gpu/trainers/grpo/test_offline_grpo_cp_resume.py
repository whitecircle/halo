#!/usr/bin/env python
"""Offline GRPO: train, evaluate, restore the original KL, resume, and export.

Qwen3 runs pure CP; GPT-OSS and Cohere2 MoE also exercise EP2+CP2 with fp32 expert masters.
Every family uses a random tiny checkpoint and an offline tokenizer, with no Hub downloads.
Run with: torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_offline_grpo_cp_resume.py --cp-size {1,2}
"""

import argparse
import functools
import math
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    TrainerCallback,
)

from src.checkpoint.format import REFERENCE_LOGPS_FILE, load_full_state_dict
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.extension import deep_ep
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.attention import model_has_sinks
from src.models.patches.gpt_oss_sinks import (
    SinksPolicy,
    apply_sinks_policy,
    is_sink_key,
    neutralized_sink_value,
    stamped_sinks_policy,
)
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.grpo.reference_logps import OfflineGRPOReferenceLogpsMixin
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import loading_problems
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

SEED = 721
STEPS = 3
CHECKPOINT_STEP = 2
LEARNING_RATE = 1e-3
MOE_FAMILIES = ("gpt_oss", "cohere2_moe")


class _ReferenceProbe(OfflineGRPOReferenceLogpsMixin):
    """Exercise the production sidecar restore without constructing another Trainer."""


class _ResumeState(TrainerCallback):
    def __init__(self):
        self.optimizer_has_moments = False
        self.optimizer_moment_sums = None
        self.optimizer_steps = None
        self.optimizer_lrs = None
        self.scheduler_epoch = None
        self.is_bf16_optimizer = False

    def on_train_begin(self, args, state, control, **kwargs):
        optimizer = kwargs["optimizer"]
        self.is_bf16_optimizer = isinstance(getattr(optimizer, "optimizer", optimizer), AdamWBF16)
        self.optimizer_has_moments = any(
            bool(torch.any(moment.to_local() if hasattr(moment, "to_local") else moment).item())
            for item in optimizer.state.values()
            if (moment := item.get("exp_avg_sq")) is not None
        )
        self.optimizer_moment_sums = _optimizer_moment_sums(optimizer)
        self.optimizer_steps = _optimizer_steps(optimizer)
        self.optimizer_lrs = [group["lr"] for group in optimizer.param_groups]
        self.scheduler_epoch = kwargs["lr_scheduler"].last_epoch
        return control


class _StepTwoState(TrainerCallback):
    def __init__(self):
        self.optimizer_moment_sums = None
        self.optimizer_steps = None
        self.optimizer_lrs = None
        self.expert_masters = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == CHECKPOINT_STEP:
            self.optimizer_moment_sums = _optimizer_moment_sums(kwargs["optimizer"])
            self.optimizer_steps = _optimizer_steps(kwargs["optimizer"])
            self.optimizer_lrs = [group["lr"] for group in kwargs["optimizer"].param_groups]
            self.expert_masters = _expert_masters(model)
        return control


def _optimizer_moment_sums(optimizer):
    totals = {}
    for key in ("exp_avg", "exp_avg_sq"):
        tensors = [
            value.to_local() if hasattr(value, "to_local") else value
            for item in optimizer.state.values()
            if (value := item.get(key)) is not None
        ]
        totals[key] = sum(float(tensor.float().square().sum().item()) for tensor in tensors)
    return totals


def _optimizer_steps(optimizer):
    def step_number(value):
        local = value.to_local() if hasattr(value, "to_local") else value
        return float(local.item() if hasattr(local, "item") else local)

    return sorted({step_number(step) for item in optimizer.state.values() if (step := item.get("step")) is not None})


def _expert_masters(model):
    """Local FSDP-ignored expert masters, by wrapper ownership rather than checkpoint spelling."""
    expert_ids = {
        id(parameter)
        for layer in model.modules()
        if isinstance(layer, EPMoELayerBase)
        for _, parameter in layer.expert_named_params()
    }
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if id(parameter) in expert_ids
    }


def _save_tiny_model(path, family="qwen3"):
    fast = make_offline_tokenizer()
    torch.manual_seed(SEED)
    model = (
        tiny_family_model(TINY_MOE_FAMILIES[family], fast, overrides={"num_hidden_layers": 2})
        if family in MOE_FAMILIES
        else Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(fast),
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=128,
                tie_word_embeddings=False,
                pad_token_id=fast.pad_token_id,
                eos_token_id=fast.eos_token_id,
            )
        )
    )
    model.to(torch.bfloat16).save_pretrained(path)
    fast.save_pretrained(path)


def _config(output_dir, save):
    return OfflineGRPOConfig(
        output_dir=output_dir,
        max_steps=STEPS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        kl_beta=0.05,
        use_chunked_grpo_logprobs=True,
        loss_type="grpo",
        policy_gradient_formulation="reinforce",
        min_log_prob=None,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=CHECKPOINT_STEP,
        save_total_limit=2,
        report_to="none",
        max_prompt_length=16,
        max_completion_length=16,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        fsdp="",
    )


def _parallelism_config(cp_size, ep_size, fp32_masters):
    return ParallelismConfig(cp_size=cp_size, ep_size=ep_size, ep_fp32_experts=fp32_masters)


def _build_trainer(
    source,
    output_dir,
    tokenizer,
    train,
    evaluation,
    *,
    cp_size,
    ep_size=1,
    fp32_masters=False,
    checkpoint=None,
    save=False,
):
    parallelism = _parallelism_config(cp_size, ep_size, fp32_masters)
    model, _ = load_distributed_model(
        model_name_or_path=source,
        parallelism_config=parallelism,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="flash_attention_2",
        use_liger_kernel=False,
        reset_sinks=True,
        preserve_checkpoint_precision=checkpoint is not None,
    )
    trainer = OfflineGRPOTrainer(
        model=model,
        args=_config(output_dir, save),
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        parallelism_config=parallelism,
        resume_checkpoint=checkpoint,
        moe_balancing="none",
    )
    assert trainer.parallelism_config.is_cp_mode == (cp_size > 1)
    if cp_size > 1:
        assert trainer.cp_config.cp_size == cp_size
    return trainer


def _eager_oracle(source, device="cpu"):
    """Same sink-free policy as Ulysses, with independent full-row eager attention."""
    model, info = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    assert not loading_problems(info), f"incomplete oracle/export checkpoint: {loading_problems(info)}"
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED, attn_implementation="eager")
    return model.to(device).eval()


def _sink_checks(model, checkpoint=None):
    if not model_has_sinks(model.config):
        return {}
    checks = {"gptoss_sink_policy_neutralized": stamped_sinks_policy(model) is SinksPolicy.NEUTRALIZED}
    if checkpoint is not None:
        state = load_full_state_dict(checkpoint)
        sinks = {name: value for name, value in state.items() if is_sink_key(name)}
        checks["gptoss_exported_sinks_neutralized"] = len(sinks) == model.config.num_hidden_layers and all(
            torch.all(value == neutralized_sink_value(value.dtype)).item() for value in sinks.values()
        )
    return checks


def _reference_rows(dataset):
    return [torch.as_tensor(row, dtype=torch.float32) for row in dataset[REF_PER_TOKEN_LOGPS_COLUMN]]


def _unsharded_reference_error(source, dataset, swept_rows, device):
    """Compare the initial sweep with a separate full-row causal-LM score."""
    model = _eager_oracle(source, device)
    errors = []
    with torch.inference_mode():
        for row, swept in zip(dataset, swept_rows, strict=True):
            prompt = row["prompt_input_ids"]
            completion = row["completion_input_ids"]
            tokens = torch.tensor([prompt + completion], dtype=torch.long, device=device)
            logits = model(tokens).logits[:, len(prompt) - 1 : len(prompt) - 1 + len(completion)].float()
            targets = torch.tensor(completion, dtype=torch.long, device=device).view(1, -1, 1)
            unsharded = logits.log_softmax(dim=-1).gather(-1, targets).view(-1).cpu()
            errors.append(float((unsharded - swept).abs().max()))
    del model
    return max(errors)


def _record_training_rows(trainer):
    """Capture the logical rows used by each train step, excluding eval sweeps."""
    rows = []
    original = trainer._compute_loss_inner

    def observed(model, inputs):
        if model.training:
            ids = (
                inputs["input_ids"]
                if trainer.parallelism_config.is_cp_mode
                else torch.cat([inputs["prompt_input_ids"], inputs["completion_input_ids"]], dim=1)
            )
            rows.append(ids.detach().cpu().clone())
        return original(model, inputs)

    trainer._compute_loss_inner = observed
    return rows


def _kl_oracle(ctx, trainer, export):
    batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[index] for index in range(2)]))
    batch["advantage"] = torch.zeros_like(batch["advantage"])
    if trainer.parallelism_config.is_cp_mode:
        ids, attention = batch["input_ids"], batch["attention_mask"]
        reference = batch[REF_PER_TOKEN_LOGPS_COLUMN][:, 1:]
        valid = batch["labels"][:, 1:] != LABEL_IGNORE_INDEX
    else:
        ids = torch.cat([batch["prompt_input_ids"], batch["completion_input_ids"]], dim=1)
        attention = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
        reference = batch[REF_PER_TOKEN_LOGPS_COLUMN]
        valid = batch["completion_attention_mask"].bool()
    expected = torch.zeros((), device=ctx.device)
    if ctx.rank == 0:
        oracle = _eager_oracle(export, ctx.device)
        with torch.no_grad():
            oracle.get_output_embeddings().weight.mul_(2)
            logits = oracle(input_ids=ids, attention_mask=attention).logits[:, :-1]
            logps = logits.float().log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            if not trainer.parallelism_config.is_cp_mode:
                logps = logps[:, -reference.size(1) :]
            delta = torch.minimum(reference, logps + KL_LOGRATIO_CLAMP) - logps
            weights = batch["group_size"].float().reciprocal()
            expected = (
                trainer.beta
                * (((delta.exp() - delta - 1) * valid).sum(1) / valid.sum(1).clamp(min=1) * weights).sum()
                / weights.sum()
            )
        del oracle
        cleanup_memory()
    dist.broadcast(expected, src=0)
    reshard_fsdp2_modules(trainer.model)
    head = trainer.model.get_output_embeddings().weight
    original = head.detach().clone()
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            head.mul_(2)
            actual = trainer.compute_loss(trainer.model, batch)
    finally:
        reshard_fsdp2_modules(trainer.model)
        with torch.no_grad():
            trainer.model.get_output_embeddings().weight.copy_(original)
        trainer.model.train(was_training)
    error = abs((actual - expected).item())
    log(f"CP{trainer.parallelism_config.cp_size} nonzero KL oracle: expected={expected.item():.5g}, error={error:.5g}")
    return bool(expected > 0), error < TOL.exact_objective_rel * expected.item()


def _probe(checkpoint, dataset, settings):
    probe = _ReferenceProbe()
    probe.beta = 0.05
    probe._init_reference_logps(resume_checkpoint=checkpoint)
    return probe._restore_reference_logps_or_none(dataset, "training", settings=settings)


def _negative_sidecar_checks(ctx, checkpoint, tokenized_dataset, settings):
    indices = list(range(len(tokenized_dataset)))
    indices[0], indices[2] = indices[2], indices[0]
    reordered = tokenized_dataset.select(indices)
    assert [len(row) for row in reordered["completion_input_ids"]] == [
        len(row) for row in tokenized_dataset["completion_input_ids"]
    ]
    assert reordered["prompt_input_ids"] != tokenized_dataset["prompt_input_ids"]
    try:
        _probe(checkpoint, reordered, settings)
    except ValueError as exc:
        reordered_rejected = "differ from the saved run" in str(exc)
    else:
        reordered_rejected = False

    sidecar = os.path.join(checkpoint, REFERENCE_LOGPS_FILE)
    held = sidecar + ".held"
    if ctx.rank == 0:
        os.replace(sidecar, held)
    dist.barrier()
    try:
        try:
            _probe(checkpoint, tokenized_dataset, settings)
        except RuntimeError as exc:
            missing_rejected = REFERENCE_LOGPS_FILE in str(exc) and "TRAINED" in str(exc)
        else:
            missing_rejected = False
    finally:
        dist.barrier()
        if ctx.rank == 0:
            os.replace(held, sidecar)
        dist.barrier()
    return reordered_rejected, missing_rejected


def _export_comparison(checkpoint, continuous_export, resumed_export):
    checkpoint_weights = load_full_state_dict(checkpoint)
    continuous = load_full_state_dict(continuous_export)
    resumed = load_full_state_dict(resumed_export)
    same_keys = checkpoint_weights.keys() == continuous.keys() == resumed.keys()
    if not same_keys:
        return False, False, False
    # Export rounding of fp32 masters alone must not count as a third optimizer update.
    step_change = torch.linalg.vector_norm(
        torch.cat(
            [
                (continuous[name].float() - checkpoint_weights[name].to(continuous[name].dtype).float()).flatten()
                for name in continuous
            ]
        )
    )
    resume_error = torch.linalg.vector_norm(
        torch.cat([(continuous[name].float() - resumed[name].float()).flatten() for name in continuous])
    )
    model = _eager_oracle(resumed_export)
    with torch.no_grad():
        logits = model(torch.tensor([[3, 4, 5]], dtype=torch.long)).logits
    log(f"Step-3 export: checkpoint-to-step delta={step_change:.5g}, resume error={resume_error:.5g}")
    return bool(step_change > 0), bool(resume_error == 0), bool(torch.isfinite(logits).all())


def cp_resume_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", choices=(1, 2), type=int, default=2)
    parser.add_argument("--family", choices=("qwen3", *MOE_FAMILIES), default="qwen3")
    parser.add_argument("--ep-size", choices=(1, 2), type=int, default=1)
    parser.add_argument("--fp32-masters", action="store_true")
    return parser


def run(ctx) -> dict:
    parser = cp_resume_parser()
    args = parser.parse_args()
    cp_size, ep_size, fp32_masters = args.cp_size, args.ep_size, args.fp32_masters
    if fp32_masters and (ep_size != 2 or args.family not in MOE_FAMILIES):
        parser.error("--fp32-masters requires a MoE family with --ep-size 2")
    if ep_size > 1:
        if args.family not in MOE_FAMILIES:
            parser.error("--ep-size 2 requires a MoE family")
        # Exact resume compares the same routed-token order, not atomic receive-slot races.
        buffer_cls = deep_ep().ElasticBuffer
        buffer_cls.__init__ = functools.partialmethod(buffer_cls.__init__, deterministic=True)
    log(f"offline GRPO {args.family}: EP{ep_size}/CP{cp_size}, fp32 expert masters={fp32_masters}")
    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    shared = dirs[0]
    base = os.path.join(shared, f"tiny_{args.family}")
    output = os.path.join(shared, "train")
    continuous_export = os.path.join(shared, "continuous_export")
    resumed_export = os.path.join(shared, "resumed_export")
    if ctx.rank == 0:
        _save_tiny_model(base, args.family)
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(base)
    train, evaluation = offline_grpo_dataset(8), offline_grpo_dataset(4, 6)

    layout = {"cp_size": cp_size, "ep_size": ep_size, "fp32_masters": fp32_masters}
    trainer = _build_trainer(base, output, tokenizer, train, evaluation, **layout, save=True)
    step_two = _StepTwoState()
    trainer.add_callback(step_two)
    checks = {"run_start_reference_swept": REF_PER_TOKEN_LOGPS_COLUMN in trainer.train_dataset.column_names}
    checks.update({f"fresh_{name}": value for name, value in _sink_checks(trainer.model).items()})
    initial_train_rows = _reference_rows(trainer.train_dataset)
    initial_eval_rows = _reference_rows(trainer.eval_dataset)
    if ctx.rank == 0:
        completion_lengths = {len(row["completion_input_ids"]) for row in trainer.train_dataset}
        first_batch = [trainer.train_dataset[index] for index in range(2)]
        oracle_error = _unsharded_reference_error(base, trainer.train_dataset, initial_train_rows, ctx.device)
        checks["variable_length_reference_rows"] = len(completion_lengths) >= 2
        if cp_size > 1:
            chunk_width = trainer.data_collator(first_batch)["input_ids"].size(1) // cp_size
            checks["reference_crosses_cp_boundary"] = any(
                len(row["prompt_input_ids"])
                <= chunk_width
                < len(row["prompt_input_ids"]) + len(row["completion_input_ids"])
                for row in first_batch
            )
        checks["reference_matches_unsharded"] = oracle_error < TOL.logprob_atol
        log(f"initial CP{cp_size} reference vs unsharded max logp error={oracle_error:.5g}")
    settings = trainer._reference_settings()
    tokenized = trainer.train_dataset.remove_columns(REF_PER_TOKEN_LOGPS_COLUMN)
    uninterrupted_rows = _record_training_rows(trainer)
    result = trainer.train()
    checks["fresh_train_three_steps"] = trainer.state.global_step == STEPS and math.isfinite(result.training_loss)
    checks["fresh_eval_finite"] = any(
        entry.get("step") == STEPS and math.isfinite(entry["eval_loss"])
        for entry in trainer.state.log_history
        if "eval_loss" in entry
    )
    checkpoint = os.path.join(output, f"checkpoint-{CHECKPOINT_STEP}")
    sidecar_path = os.path.join(checkpoint, REFERENCE_LOGPS_FILE)
    dist.barrier()
    checks["checkpoint_and_reference_sidecar"] = os.path.isfile(sidecar_path)
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    checks["train_and_eval_reference_saved"] = set(payload) == {"training", "evaluation"} and all(
        payload[split]["values"].numel() > 0 for split in ("training", "evaluation")
    )
    checks.update({f"checkpoint_{name}": value for name, value in _sink_checks(trainer.model, checkpoint).items()})
    trainer.save_model(continuous_export)
    dist.barrier()
    checks.update(
        {f"continuous_{name}": value for name, value in _sink_checks(trainer.model, continuous_export).items()}
    )
    checks["kl_oracle_nonzero"], checks["kl_matches_unsharded_after_update"] = _kl_oracle(
        ctx, trainer, continuous_export
    )
    checks["row_reorder_rejected"], checks["missing_reference_rejected"] = _negative_sidecar_checks(
        ctx, checkpoint, tokenized, settings
    )
    trainer.cleanup_ep()
    del trainer
    cleanup_memory()
    dist.barrier()

    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base), _parallelism_config(cp_size, ep_size, fp32_masters)
    )
    checks["path_b_uses_trained_checkpoint"] = source == checkpoint
    with patch.object(
        OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("reswept KL reference")
    ):
        resumed = _build_trainer(source, output, tokenizer, train, evaluation, **layout, checkpoint=checkpoint)
    if fp32_masters:
        expected_masters = step_two.expert_masters
        restored_masters = _expert_masters(resumed.model)
        checks["fp32_expert_masters_off_bf16_grid"] = (
            bool(expected_masters)
            and all(value.dtype == torch.float32 for value in expected_masters.values())
            and any(not torch.equal(value, value.bfloat16().float()) for value in expected_masters.values())
        )
        checks["fp32_expert_masters_exact_before_resumed_forward"] = (
            expected_masters.keys() == restored_masters.keys()
            and bool(expected_masters)
            and all(
                restored_masters[name].dtype == torch.float32
                and torch.equal(expected_masters[name], restored_masters[name])
                for name in expected_masters
            )
        )
    start_export = os.path.join(shared, "resumed_start_export")
    resumed.save_model(start_export)
    if ctx.rank == 0:
        saved = load_full_state_dict(checkpoint)
        restored = load_full_state_dict(start_export)
        # Serving exports round masters to the run dtype; their exact live values are checked above.
        checks["checkpoint_weights_loaded_exactly"] = saved.keys() == restored.keys() and all(
            torch.equal(saved[key].to(restored[key].dtype), restored[key]) for key in saved
        )
    checks["train_reference_restored"] = all(
        torch.equal(before, after)
        for before, after in zip(initial_train_rows, _reference_rows(resumed.train_dataset), strict=True)
    )
    checks["eval_reference_restored"] = all(
        torch.equal(before, after)
        for before, after in zip(initial_eval_rows, _reference_rows(resumed.eval_dataset), strict=True)
    )
    capture = _ResumeState()
    resumed.add_callback(capture)
    resumed_rows = _record_training_rows(resumed)
    resumed_result = resumed.train(resume_from_checkpoint=checkpoint)
    checks["optimizer_moments_restored_exactly"] = (
        step_two.optimizer_moment_sums is not None
        and capture.optimizer_moment_sums is not None
        and all(
            math.isclose(step_two.optimizer_moment_sums[key], capture.optimizer_moment_sums[key], rel_tol=1e-6)
            for key in ("exp_avg", "exp_avg_sq")
        )
    )
    checks["optimizer_step_counters_restored"] = step_two.optimizer_steps == capture.optimizer_steps
    checks["production_adamw_bf16_resumed"] = capture.is_bf16_optimizer
    checks["optimizer_learning_rates_restored"] = step_two.optimizer_lrs == capture.optimizer_lrs
    if ctx.rank == 0:
        log(
            f"optimizer moments at checkpoint/resume: {step_two.optimizer_moment_sums} / {capture.optimizer_moment_sums}"
        )
        log(f"optimizer step counters at checkpoint/resume: {step_two.optimizer_steps} / {capture.optimizer_steps}")
        log(f"optimizer learning rates at checkpoint/resume: {step_two.optimizer_lrs} / {capture.optimizer_lrs}")
    checks["resume_uses_same_step_three_rows"] = (
        len(uninterrupted_rows) == STEPS
        and len(resumed_rows) == STEPS - CHECKPOINT_STEP
        and torch.equal(uninterrupted_rows[CHECKPOINT_STEP], resumed_rows[0])
    )
    if ctx.rank == 0:
        log(f"step-3 rows match on resume: {checks['resume_uses_same_step_three_rows']}")
    checks["resume_next_optimizer_step"] = (
        resumed.state.global_step == STEPS
        and math.isfinite(resumed_result.training_loss)
        and capture.optimizer_has_moments
        and capture.scheduler_epoch == CHECKPOINT_STEP
    )
    checks["resumed_eval_finite"] = any(
        entry.get("step") == STEPS and math.isfinite(entry["eval_loss"])
        for entry in resumed.state.log_history
        if "eval_loss" in entry
    )
    resumed.save_model(resumed_export)
    dist.barrier()
    if ctx.rank == 0:
        checks.update(
            {f"resumed_{name}": value for name, value in _sink_checks(resumed.model, resumed_export).items()}
        )
        changed, matched, loadable = _export_comparison(checkpoint, continuous_export, resumed_export)
        checks["step_three_changed_weights"] = changed
        checks["resumed_step_matches_continuous"] = matched
        checks["hf_export_loads_and_scores"] = loadable
    checks = ctx.broadcast_checks(checks)
    metrics = ctx.metrics(resumed)
    resumed.cleanup_ep()
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=2, prefix="offline_grpo_cp_resume")(run)

if __name__ == "__main__":
    main()
