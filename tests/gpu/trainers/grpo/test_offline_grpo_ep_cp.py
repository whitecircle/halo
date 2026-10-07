#!/usr/bin/env python
"""Offline GRPO's actual trainer on EP8 with CP2/CP4, including KL and resume/export.

The oracle runs full rows through an EP8/CP1 trainer model and independently reduces its logits.
Run: torchrun --nproc_per_node=8 tests/gpu/trainers/grpo/test_offline_grpo_ep_cp.py
Use --ep-loading lazy or --ep-loading eager to select checkpoint loading.
"""

import functools
import hashlib
import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import TrainerCallback

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN, OfflineGRPOCPDataCollatorWithPadding
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.base_layer import get_flash_attn_func
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.cp_grpo import (
    ep_gradient_parameter_names,
    gradient_agreement,
    optimizer_step_agreement,
    same_layout_optimizer_step,
)
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import (
    OFFLINE_VOCAB,
    build_offline_grpo_trainer,
    ep_loading_parser,
    offline_grpo_config,
    offline_grpo_dataset,
    pure_kl_objective,
    resumed_export_verdict,
    save_offline_moe_base,
    swept_reference_error,
    token_logps,
)
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, finish_phase, log

SEED = 1337
BETA = 0.2
STEPS = 2
SAVE_STEP = 1
LR = 1e-3


def _dataset(groups, offset=0):
    rows = []
    for row in offline_grpo_dataset(groups, offset):
        row["prompt"] = " ".join([row["prompt"]] * 3)
        row["completions"] = [" ".join([text] * count) for text, count in zip(row["completions"], (4, 5), strict=True)]
        rows.append(row)
    return Dataset.from_list(rows)


def _build_trainer(
    ctx, source, output, cp_size, train, evaluation=None, checkpoint=None, *, ep_lazy_loading, kl_beta=BETA
):
    parallelism = ParallelismConfig(
        ep_size=8,
        cp_size=cp_size,
        ep_fp32_router=True,
        ep_lazy_loading=ep_lazy_loading,
    )
    args = offline_grpo_config(
        output,
        steps=STEPS,
        save_steps=SAVE_STEP,
        seed=SEED,
        kl_beta=kl_beta,
        learning_rate=LR,
        evaluate=evaluation is not None,
        save=evaluation is not None,
    )
    return build_offline_grpo_trainer(ctx, source, parallelism, args, train, evaluation, checkpoint=checkpoint)


def _snapshot(model, gradients=False):
    reshard_fsdp2_modules(model)
    result = {}
    for name, parameter in model.named_parameters():
        if gradients and not parameter.requires_grad:
            continue
        tensor = parameter.grad if gradients else parameter
        # An idle expert can have None or an explicit zero derivative depending on shard graph edges.
        if tensor is None:
            tensor = torch.zeros_like(parameter)
        result[name] = (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().clone()
    return result


def _missing_gradients(model):
    reshard_fsdp2_modules(model)
    return sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    )


def _gradient_diagnostics(prefix, actual, expected, **bounds):
    failures = []
    for name in sorted(actual.keys() | expected.keys()):
        if name not in actual or name not in expected:
            failures.append(f"{name}: present actual={name in actual}, expected={name in expected}")
            continue
        if not torch.count_nonzero(actual[name]) and not torch.count_nonzero(expected[name]):
            continue
        _, direction, norm, cosine, ratio = gradient_agreement({name: actual[name]}, {name: expected[name]}, **bounds)
        if not direction or not norm:
            error = (actual[name] - expected[name]).abs().max().item()
            failures.append(
                f"{name}: cosine={cosine:.8g}, norm ratio={ratio:.8g}, "
                f"norm actual/reference={actual[name].norm().item():.8g}/{expected[name].norm().item():.8g}, max error={error:.8g}"
            )
    if failures:
        log(f"{prefix} gradient mismatches: " + "; ".join(failures))


def _optimizer_state(model, optimizer, parameters=None):
    if parameters is None:
        reshard_fsdp2_modules(model)
        parameters = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    result, groups = {}, []
    for group in optimizer.param_groups:
        groups.append({key: group[key] for key in ("lr", "betas", "eps", "weight_decay")})
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            result[names[id(parameter)]] = {
                key: (value.to_local() if isinstance(value, DTensor) else value).detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in state.items()
            }
    return {"groups": groups, "parameters": result}


def _state_diagnostics(prefix, actual, expected):
    if actual["groups"] != expected["groups"]:
        log(f"{prefix} optimizer groups actual={actual['groups']}, expected={expected['groups']}")
    failures = []
    for name in sorted(actual["parameters"].keys() | expected["parameters"].keys()):
        current, reference = actual["parameters"].get(name, {}), expected["parameters"].get(name, {})
        for key in sorted(current.keys() | reference.keys()):
            a, b = current.get(key), reference.get(key)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                    error = (
                        (a.float() - b.float()).abs().max().item() if a.shape == b.shape and a.numel() else math.inf
                    )
                    failures.append(f"{name}/{key}: shape={tuple(a.shape)}/{tuple(b.shape)}, max error={error:.8g}")
            elif a != b:
                failures.append(f"{name}/{key}: actual={a}, expected={b}")
    if failures:
        log(f"{prefix} optimizer state mismatches ({len(failures)}): " + "; ".join(failures[:12]))


class _LifecycleProbe(TrainerCallback):
    def __init__(self):
        self.start = None
        self.steps = {}

    def _capture(self, model, optimizer):
        rng = torch.cuda.get_rng_state().cpu().numpy().tobytes()
        state = _optimizer_state(model, optimizer)
        state["rng"] = hashlib.sha256(rng).hexdigest()[:16]
        state["masters"] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if not isinstance(parameter, DTensor) and parameter.dtype == torch.float32
        }
        return state

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.start = self._capture(model, optimizer)
        log(
            f"train begin step={state.global_step}, optimizer groups={self.start['groups']}, CUDA RNG={self.start['rng']}"
        )

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.steps[state.global_step] = self._capture(model, optimizer)
        log(
            f"step={state.global_step}, optimizer groups={self.steps[state.global_step]['groups']}, CUDA RNG={self.steps[state.global_step]['rng']}"
        )


def _perturb_policy(model):
    reshard_fsdp2_modules(model)
    with torch.no_grad():
        model.get_output_embeddings().weight.mul_(2.0)


def _pure_kl_loss(logps, reference, batch):
    """The pure-KL oracle over CP-collated full rows, whose completion tokens the shifted labels mark."""
    valid = batch["labels"][:, 1:] != LABEL_IGNORE_INDEX
    return pure_kl_objective(logps, reference, valid, batch["group_size"], BETA)


def run(ctx):
    args = ep_loading_parser().parse_args()
    build_trainer = functools.partial(_build_trainer, ctx, ep_lazy_loading=args.ep_loading == "lazy")
    pin_deterministic_ep_dispatch()
    log("Exact EP resume premise: deterministic DeepEP dispatch")
    shared = shared_output_dir(ctx)
    base = os.path.join(shared, "tiny_qwen3_moe")
    if ctx.rank == 0:
        save_offline_moe_base(base, SEED)
    dist.barrier()
    cp_kernel = get_flash_attn_func()
    log(f"EP8 CP kernel: {cp_kernel.__module__}.{cp_kernel.__name__}; oracle: full-row FA2")
    train, evaluation = _dataset(16), _dataset(8, 3)
    checks = {}

    oracle = build_trainer(base, os.path.join(shared, "oracle"), 1, train, kl_beta=0.0)
    rows = list(oracle.train_dataset)
    collator = OfflineGRPOCPDataCollatorWithPadding(pad_token_id=OFFLINE_VOCAB["<pad>"], cp_size=4)
    all_rows = {name: tensor.to(ctx.device) for name, tensor in collator(rows).items()}
    oracle.model.eval()
    with torch.no_grad():
        reference = token_logps(oracle.model, all_rows["input_ids"], all_rows["attention_mask"]).detach()
    reference_rows = [
        reference[
            index,
            len(row["prompt_input_ids"]) - 1 : len(row["prompt_input_ids"]) + len(row["completion_input_ids"]) - 1,
        ].cpu()
        for index, row in enumerate(rows)
    ]
    batch = {name: tensor[:2] for name, tensor in all_rows.items()}
    # A pure-KL validation batch catches an absent KL gradient, rather than hiding it under reward gradients.
    batch["advantage"] = torch.zeros_like(batch["advantage"])
    _perturb_policy(oracle.model)
    initial_weights = _snapshot(oracle.model)
    oracle.model.train()
    logps = token_logps(oracle.model, batch["input_ids"], batch["attention_mask"])
    expected_loss = _pure_kl_loss(logps, reference[:2], batch)
    expected_loss.backward()
    expected_gradients = _snapshot(oracle.model, gradients=True)
    oracle_missing = _missing_gradients(oracle.model)
    expected_loss_value = expected_loss.detach().clone()
    expected_logps = logps.detach().clone()
    oracle.create_optimizer()
    checks["oracle_uses_adamw_bf16"] = isinstance(oracle.optimizer, AdamWBF16)
    finish_phase(oracle)
    del oracle, all_rows, logps, expected_loss
    cleanup_memory()

    for cp_size in (1, 2, 4):
        prefix = f"ep8_cp{cp_size}"
        output = os.path.join(shared, prefix)
        trainer = build_trainer(base, output, cp_size, train, evaluation)
        swept = trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN]
        reference_error = swept_reference_error(swept, reference_rows)
        checks[f"{prefix}_reference_oracle"] = reference_error < TOL.logprob_atol
        local_batch = {
            name: tensor.to(ctx.device)
            for name, tensor in trainer.data_collator([trainer.train_dataset[index] for index in range(2)]).items()
        }
        local_batch["advantage"] = torch.zeros_like(local_batch["advantage"])
        _perturb_policy(trainer.model)
        current_initial = _snapshot(trainer.model)
        checks[f"{prefix}_initial_weight_oracle"] = current_initial.keys() == initial_weights.keys() and all(
            torch.equal(current_initial[name], initial_weights[name]) for name in initial_weights
        )
        del current_initial
        trainer.model.train()
        with torch.no_grad():
            if cp_size > 1:
                got_logps, got_labels = trainer._cp_chunked_logps(
                    trainer.model, local_batch["input_ids"], local_batch["attention_mask"], local_batch["labels"]
                )
                chunk = local_batch["input_ids"].size(1) // cp_size
                start = trainer.cp_config.cp_rank * chunk
                valid = got_labels != LABEL_IGNORE_INDEX
                expected_tokens = expected_logps[:, start : start + got_logps.size(1)]
            else:
                ids = torch.cat([local_batch["prompt_input_ids"], local_batch["completion_input_ids"]], dim=1)
                mask = torch.cat(
                    [local_batch["prompt_attention_mask"], local_batch["completion_attention_mask"]], dim=1
                )
                width = local_batch["completion_input_ids"].size(1)
                _, got_logps = trainer._get_per_token_logps(trainer.model, ids, mask, width)
                valid = local_batch["completion_attention_mask"].bool()
                start = local_batch["prompt_input_ids"].size(1) - 1
                expected_tokens = expected_logps[:, start : start + width]
        token_error = (got_logps[valid] - expected_tokens[valid]).abs()
        error = torch.tensor(token_error.max().item() if token_error.numel() else 0.0, device=ctx.device)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        checks[f"{prefix}_policy_token_oracle"] = error.item() < TOL.logprob_atol
        loss = trainer.compute_loss(trainer.model, local_batch)
        loss_error = abs((loss.detach() - expected_loss_value).item())
        checks[f"{prefix}_kl_loss_oracle"] = loss_error < TOL.exact_objective_rel * expected_loss_value.abs().item()
        loss.backward()
        gradients = _snapshot(trainer.model, gradients=True)
        log(f"{prefix} absent gradients CP1={oracle_missing}, CP{cp_size}={_missing_gradients(trainer.model)}")
        ep_names = ep_gradient_parameter_names(
            trainer.model, [layer for layer in trainer.model.modules() if isinstance(layer, EPMoELayerBase)]
        )
        partitions = {
            "dense": (set(expected_gradients) - ep_names, {}),
            "expert_router": (
                ep_names,
                {"cosine_min": TOL.ep_grad_cosine_min, "norm_ratio_band": TOL.ep_grad_norm_ratio_band},
            ),
        }
        names = gradients.keys() == expected_gradients.keys()
        checks[f"{prefix}_gradient_names"] = names
        summaries = []
        for kind, (selected, bounds) in partitions.items():
            actual_part = {name: gradients[name] for name in selected}
            expected_part = {name: expected_gradients[name] for name in selected}
            _gradient_diagnostics(prefix + " " + kind, actual_part, expected_part, **bounds)
            _, directions, norms, minimum_cosine, norm_ratio = gradient_agreement(actual_part, expected_part, **bounds)
            checks[f"{prefix}_{kind}_parameter_coverage"] = bool(selected)
            checks[f"{prefix}_{kind}_every_parameter_gradient_direction"] = directions
            checks[f"{prefix}_{kind}_every_parameter_gradient_norm"] = norms
            summaries.append(f"{kind}: minimum cosine={minimum_cosine:.5g}, norm ratio={norm_ratio:.5g}")
        trainer.create_optimizer()
        checks[f"{prefix}_uses_adamw_bf16"] = isinstance(trainer.optimizer, AdamWBF16)
        expected_weights, step_optimizer, step_parameters = same_layout_optimizer_step(
            trainer.model, trainer.optimizer, expected_gradients
        )
        trainer.optimizer.step()
        actual_weights = _snapshot(trainer.model)
        for kind, (selected, bounds) in partitions.items():
            checks[f"{prefix}_{kind}_optimizer_step_oracle"] = optimizer_step_agreement(
                {name: actual_weights[name] for name in selected},
                {name: expected_weights[name] for name in selected},
                {name: initial_weights[name] for name in selected},
                **bounds,
            )
        _state_diagnostics(
            prefix + " oracle optimizer step",
            _optimizer_state(trainer.model, trainer.optimizer),
            _optimizer_state(None, step_optimizer, step_parameters),
        )
        for kind, (selected, bounds) in partitions.items():
            _gradient_diagnostics(
                prefix + " " + kind + " optimizer update",
                {name: actual_weights[name] - initial_weights[name] for name in selected},
                {name: expected_weights[name] - initial_weights[name] for name in selected},
                **bounds,
            )
        log(
            f"{prefix} optimizer step max error={max((actual_weights[name] - expected_weights[name]).abs().max().item() for name in initial_weights):.8g}"
        )
        del step_optimizer, step_parameters, expected_weights, actual_weights
        trainer.model.zero_grad(set_to_none=True)
        log(
            f"{prefix}: reference error={reference_error:.5g}, token error={error.item():.5g}, "
            f"KL loss error={loss_error:.5g}; " + "; ".join(summaries)
        )
        continuous_probe = _LifecycleProbe()
        trainer.add_callback(continuous_probe)
        result = trainer.train()
        checks[f"{prefix}_train"] = trainer.state.global_step == STEPS and math.isfinite(result.training_loss)
        checks[f"{prefix}_evaluate"] = math.isfinite(trainer.evaluate()["eval_loss"])
        continuous = os.path.join(shared, prefix + "_continuous")
        trainer.save_model(continuous)
        checkpoint = os.path.join(output, f"checkpoint-{SAVE_STEP}")
        checks[f"{prefix}_reference_checkpointed"] = os.path.isfile(os.path.join(checkpoint, REFERENCE_LOGPS_FILE))
        finish_phase(trainer)
        del trainer
        with patch.object(
            OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("reswept reference")
        ):
            resumed = build_trainer(checkpoint, output, cp_size, train, evaluation, checkpoint)
        expected_masters = continuous_probe.steps[SAVE_STEP]["masters"]
        reshard_fsdp2_modules(resumed.model)
        restored_masters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in resumed.model.named_parameters()
            if name in expected_masters
        }
        checks[f"{prefix}_fp32_master_restore_has_coverage"] = bool(expected_masters)
        checks[f"{prefix}_fp32_masters_exact_before_resumed_forward"] = (
            expected_masters.keys() == restored_masters.keys()
            and all(
                restored_masters[name].dtype == torch.float32
                and torch.equal(expected_masters[name], restored_masters[name])
                for name in expected_masters
            )
        )
        for name, master in expected_masters.items():
            restored = restored_masters[name]
            log(
                f"{prefix} fp32 master BEFORE resumed forward {name}: "
                f"dtype={restored.dtype}, max error={(master - restored).abs().max().item():.8g}"
            )
        checks[f"{prefix}_reference_restored"] = all(
            torch.equal(torch.as_tensor(before), torch.as_tensor(after))
            for before, after in zip(swept, resumed.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], strict=True)
        )
        resumed_probe = _LifecycleProbe()
        resumed.add_callback(resumed_probe)
        resumed.train(resume_from_checkpoint=checkpoint)
        _state_diagnostics(prefix + " checkpoint restore", resumed_probe.start, continuous_probe.steps[SAVE_STEP])
        _state_diagnostics(prefix + " final resume", resumed_probe.steps[STEPS], continuous_probe.steps[STEPS])
        for name, master in continuous_probe.steps[SAVE_STEP]["masters"].items():
            restored = resumed_probe.start["masters"][name]
            if not torch.equal(master, restored):
                log(f"{prefix} restored fp32 master {name}: max error={(master - restored).abs().max().item():.8g}")
        export = os.path.join(shared, prefix + "_resumed")
        resumed.save_model(export)
        dist.barrier()
        if ctx.rank == 0:
            (
                checks[f"{prefix}_resume_exact"],
                checks[f"{prefix}_resumed_step_changes_weights"],
                checks[f"{prefix}_hf_export_scores"],
            ) = resumed_export_verdict(continuous, export, checkpoint, ctx.device)
        metrics = ctx.metrics(resumed)
        finish_phase(resumed)
        del resumed
    return {"checks": ctx.broadcast_checks(checks), "metrics": metrics}


main = gpu_test_main(exact_world_size=8, prefix="offline_grpo_ep_cp")(run)

if __name__ == "__main__":
    main()
