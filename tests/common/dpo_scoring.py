"""Full-model DPO head equivalence on identical dense-FSDP2 or EP2 layouts and frozen reference scores."""

from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

from halo.distributed.fsdp import fsdp2_param_groups, reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.module_tree import hidden_state_backbone
from halo.models.unwrapping import base_transformers_model
from halo.runtime.dtensor import to_local
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all
from tests.common.ep_reference import captured_picks, routing_pinned
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.preference_precompute import column
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    shared_tiny_family_checkpoint,
    word_level_tokenizer,
)
from tests.common.tolerances import TOL

FP32_GRAD_REL_L2 = 5e-4
BF16_GRAD_REL_L2 = 5e-3
ZERO_GRAD_ATOL = 1e-12
STEPS = 2
LEARNING_RATE = 0.05


class _FullHeadDPOTrainer(DistributedDPOTrainer):
    def compute_ref_log_probs(self, model, inputs):
        with self.accelerator.autocast():
            return super().compute_ref_log_probs(model, inputs)


def assert_dpo_gradients_match(actual: dict, expected: dict, *, bf16: bool) -> float:
    """Each local gradient shard must match; all ranks together cover every parameter."""
    if not expected or actual.keys() != expected.keys():
        raise AssertionError("DPO gradient snapshots do not cover the same nonempty parameter set")
    worst = 0.0
    for name, want in expected.items():
        got = actual[name]
        if (
            got is None
            or want is None
            or got.shape != want.shape
            or not bool(got.isfinite().all() & want.isfinite().all())
        ):
            raise AssertionError(f"Non-finite or wrong-shaped DPO gradient: {name}")
        error, norm = (got.double() - want.double()).norm(), want.double().norm()
        if not norm:
            if error > ZERO_GRAD_ATOL:
                raise AssertionError(f"Expected-zero DPO gradient changed: {name}")
            continue
        relative = float(error / norm)
        worst = max(worst, relative)
        if relative > (BF16_GRAD_REL_L2 if bf16 else FP32_GRAD_REL_L2):
            raise AssertionError(f"DPO gradient relative L2 {relative:.6g}: {name}")
        if not bf16:
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4, msg=name)
    return worst


def assert_dpo_loss_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.double(), expected.double(), atol=2e-6, rtol=2e-5)


def assert_sgd_update(before, gradient, after, learning_rate):
    expected = before.clone().add_(gradient, alpha=-learning_rate)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)


def assert_sgd_deltas_match(first, second, learning_rate):
    """Compare raw updates with only the two independently predicted storage-rounding errors allowed."""
    deltas, ideals, rounding = [], [], []
    for before, gradient, after in (first, second):
        expected = before.clone().add_(gradient, alpha=-learning_rate)
        ideal = -learning_rate * gradient.double()
        ideals.append(ideal)
        rounding.append((expected.double() - before.double() - ideal).norm())
        deltas.append(after.double() - before.double())
    budget = (ideals[0] - ideals[1]).norm() + rounding[0] + rounding[1] + ZERO_GRAD_ATOL
    if (deltas[0] - deltas[1]).norm() > budget:
        raise AssertionError("DPO SGD delta differs beyond the gradient and predicted storage-rounding error")


def _parity_loss(logps, batch, *, beta, weight):
    chosen, rejected = logps.sum(1, dtype=torch.float64).chunk(2)
    delta = (chosen - batch["ref_chosen_logps"].double()) - (rejected - batch["ref_rejected_logps"].double())
    return -F.logsigmoid(beta * delta).mean() * weight


def _fp32_head(module, hidden):
    with torch.autocast(hidden.device.type, enabled=False):
        return F.linear(
            hidden.float(),
            module.weight.to(hidden.dtype).float(),
            None if module.bias is None else module.bias.float(),
        )


def _local_snapshot(trainer, *, gradients: bool, host: bool = True) -> dict[str, torch.Tensor]:
    reshard_fsdp2_modules(trainer.model)
    experts = trainer._get_sharded_expert_param_ids()
    result = {}
    for name, parameter in trainer.model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad if gradients else parameter
        if value is None:
            if id(parameter) not in experts:
                raise AssertionError(f"Requested non-expert gradient is missing: {name}")
            value = torch.zeros_like(parameter)
        local = to_local(value).detach()
        result[name] = local.float().cpu().clone() if host else local.clone()
    return result


def _batch(rank: int, device):
    rows = [
        {"prompt_ids": [2, 4 + rank], "chosen_ids": [6, 7, 8, 9, 1], "rejected_ids": [10, 1]},
        {"prompt_ids": [2, 5, 6], "chosen_ids": [], "rejected_ids": [11, 12, 13, 1]},
    ]
    return {key: value.to(device) for key, value in DataCollatorForPreference(pad_token_id=1)(rows).items()}


def _full_scores(trainer, batch):
    logits = trainer.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).logits[:, :-1]
    distribution = logits.float().log_softmax(-1)
    mask = batch["completion_mask"][:, 1:].bool() & batch["attention_mask"][:, 1:].bool()
    logps = distribution.gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1).masked_fill(~mask, 0.0)
    entropy = -(distribution.detach().exp() * distribution.detach()).sum(-1).masked_fill(~mask, 0.0)
    return logps, entropy


def _assert_scores_match(got, want):
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def _build_trainer(ctx, checkpoint: str, *, family: str, dtype: torch.dtype, bounded: bool):
    pc = ParallelismConfig(
        ep_size=2 if family == "qwen3_moe" else 1,
        bf16_optimizer=False,
        fp32_non_ep_params=True,
        ep_fp32_experts=family == "qwen3_moe",
    )
    model, tokenizer = load_distributed_model(
        model_name_or_path=checkpoint,
        parallelism_config=pc,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation="sdpa",
        use_liger_kernel=False,
    )
    if not bounded:
        head = base_transformers_model(model).get_output_embeddings()
        head.forward = MethodType(_fp32_head, head)
    hidden_dtypes = []
    observer = hidden_state_backbone(model).register_forward_hook(
        lambda module, inputs, output: hidden_dtypes.append(output.last_hidden_state.dtype)
    )
    rows = Dataset.from_list(
        [
            {"prompt": "t4 t5", "chosen": "t6 t7 t8", "rejected": "t9 t10"},
            {"prompt": "t11", "chosen": "t12", "rejected": "t13 t14 t15"},
            {"prompt": "t16 t17", "chosen": "t18 t19", "rejected": "t20"},
            {"prompt": "t21", "chosen": "t22 t23 t24", "rejected": "t25 t26"},
        ]
    )
    trainer_class = DistributedDPOTrainer if bounded else _FullHeadDPOTrainer
    trainer = trainer_class(
        model=model,
        args=DPOConfig(
            output_dir=f"{ctx.output_dir}/{'bounded' if bounded else 'full'}",
            max_steps=STEPS,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=False,
            precompute_ref_log_probs=True,
            beta=0.7,
            loss_weights=[1.7],
            learning_rate=LEARNING_RATE,
            optim="sgd",
            max_grad_norm=0.0,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_length=32,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            report_to=[],
            save_strategy="no",
            use_liger_kernel=False,
        ),
        train_dataset=rows,
        eval_dataset=rows,
        processing_class=tokenizer,
        parallelism_config=pc,
        use_chunked_dpo_logprobs=bounded,
    )
    trainer.create_optimizer()
    ctx.on_teardown(trainer.cleanup_ep)
    ctx.on_teardown(observer.remove)
    return trainer, hidden_dtypes


def _compare_backward(bounded, full, batch, *, ep: bool, bf16: bool):
    capture = captured_picks(bounded.model) if ep else nullcontext([])
    with capture as picks, bounded.compute_loss_context_manager():
        got_loss = bounded.compute_loss(bounded.model, batch)
    bounded.accelerator.backward(got_loss)
    bounded._finish_fsdp2_backward(bounded.model)
    got_norm = bounded._get_grad_norm(bounded.model)
    gradients = _local_snapshot(bounded, gradients=True)
    replay = routing_pinned(full.model, [torch.cat(chunks) for chunks in picks]) if picks else nullcontext([])
    with replay as routes, full.compute_loss_context_manager():
        want_loss = full.compute_loss(full.model, batch)
    assert_dpo_loss_matches(got_loss, want_loss)
    if ep:
        assert (
            picks
            and routes
            and all(route.duplicated == 0 and route.regret <= TOL.router_near_tie_logit_abs for route in routes)
        )
    full.accelerator.backward(want_loss)
    full._finish_fsdp2_backward(full.model)
    want_norm = full._get_grad_norm(full.model)
    worst = assert_dpo_gradients_match(gradients, _local_snapshot(full, gradients=True), bf16=bf16)
    if ep:
        expert_ids = bounded._get_sharded_expert_param_ids()
        names = {name for name, parameter in bounded.model.named_parameters() if id(parameter) in expert_ids}
        assert names and any(gradients[name].count_nonzero() for name in names), (
            "no local expert gradient was exercised"
        )
    torch.testing.assert_close(
        torch.as_tensor(got_norm),
        torch.as_tensor(want_norm),
        atol=1e-8,
        rtol=BF16_GRAD_REL_L2 if bf16 else FP32_GRAD_REL_L2,
    )
    return got_loss, worst


def run_dpo_scoring_equivalence(ctx, *, family: str, dtype_name: str) -> dict:
    """COLLECTIVE. Public bounded scorer vs full FP32 head on the same storage-rounded model operands."""
    if family not in ("dense", "qwen3_moe") or dtype_name not in ("float32", "bfloat16"):
        raise ValueError("DPO scoring equivalence accepts dense/Qwen3-MoE and float32/bfloat16")
    if family == "qwen3_moe" and dtype_name != "bfloat16":
        raise ValueError("The EP2 DPO row uses BF16 DeepEP operands")
    pin_deterministic_ep_dispatch()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    bf16 = dtype == torch.bfloat16
    tiny = TINY_DENSE_FAMILY if family == "dense" else TINY_MOE_FAMILIES[family]
    checkpoint = shared_tiny_family_checkpoint(
        ctx, tiny, f"dpo_scoring_{family}_{dtype_name}", word_level_tokenizer([f"t{i}" for i in range(2, 48)]), 102
    )
    bounded, bounded_dtypes = _build_trainer(ctx, checkpoint, family=family, dtype=dtype, bounded=True)
    full, full_dtypes = _build_trainer(ctx, checkpoint, family=family, dtype=dtype, bounded=False)
    checks = parallel_shape_checks(bounded.model, bounded.parallelism_config)
    checks["actual_fsdp2_groups_present"] = bool(fsdp2_param_groups(bounded.model))
    checks["numerical_row_uses_fp32_master_parameters"] = all(
        parameter.dtype == torch.float32 for parameter in bounded.model.parameters()
    )
    metrics = {}
    for name, first, second in (
        ("train", bounded.train_dataset, full.train_dataset),
        ("eval", bounded.eval_dataset, full.eval_dataset),
    ):
        for key in bounded.REFERENCE_COLUMNS:
            assert first.features[key].dtype == second.features[key].dtype == "float32"
            _assert_scores_match(column(first, key), column(second, key))
        checks[f"{name}_reference_sweep_matches_full_fp32_head"] = True
    batch = _batch(ctx.rank, ctx.device)
    with torch.no_grad(), bounded.accelerator.autocast():
        got, got_entropy = bounded._chunked_dpo_logps(bounded.model, batch, compute_entropy=True)
    with torch.no_grad(), full.accelerator.autocast():
        want, want_entropy = _full_scores(full, batch)
    _assert_scores_match(got, want)
    torch.testing.assert_close(got_entropy, want_entropy, atol=1e-4, rtol=2e-4)
    _assert_scores_match(got.sum(1, dtype=torch.float64), want.sum(1, dtype=torch.float64))
    checks["per_token_logps_entropy_and_fp64_parity_sums_match"] = True
    for index, (key, values) in enumerate(
        zip(bounded.REFERENCE_COLUMNS, got.sum(1, dtype=torch.float32).chunk(2), strict=True)
    ):
        offsets = [0.1234567, -0.2345678] if index == 0 else [0.3456789, 0.4567891]
        batch[key] = values + torch.tensor(offsets, device=ctx.device)
    parity = _parity_loss(got, batch, beta=bounded.beta, weight=bounded.loss_weights[0])
    full_parity = _parity_loss(want, batch, beta=full.beta, weight=full.loss_weights[0])
    assert_dpo_loss_matches(parity, full_parity)
    checks["fp64_objective_parity_both_arms"] = True
    initial = _local_snapshot(bounded, gradients=False)
    initial_full = _local_snapshot(full, gradients=False)
    assert initial.keys() == initial_full.keys() and all(
        torch.equal(value, initial_full[name]) for name, value in initial.items()
    )
    worst_gradient = 0.0
    for step in range(STEPS):
        for trainer in (bounded, full):
            trainer.model.train()
            trainer.state.global_step = step
            trainer.optimizer.zero_grad(set_to_none=True)
        got_loss, error = _compare_backward(bounded, full, batch, ep=family == "qwen3_moe", bf16=bf16)
        if step == 0:
            assert_dpo_loss_matches(got_loss, parity)
        worst_gradient = max(worst_gradient, error)
        before = [_local_snapshot(trainer, gradients=False, host=False) for trainer in (bounded, full)]
        gradients = [_local_snapshot(trainer, gradients=True, host=False) for trainer in (bounded, full)]
        for trainer in (bounded, full):
            assert all(
                group.get("momentum", 0) == group.get("weight_decay", 0) == 0 and not group.get("maximize", False)
                for group in trainer.optimizer.param_groups
            )
            trainer.optimizer.step()
        updated = [_local_snapshot(trainer, gradients=False, host=False) for trainer in (bounded, full)]
        for name in before[0]:
            for arm in (0, 1):
                assert_sgd_update(before[arm][name], gradients[arm][name], updated[arm][name], LEARNING_RATE)
            assert_sgd_deltas_match(
                (before[0][name], gradients[0][name], updated[0][name]),
                (before[1][name], gradients[1][name], updated[1][name]),
                LEARNING_RATE,
            )
        actual, expected = (_local_snapshot(trainer, gradients=False) for trainer in (bounded, full))
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            torch.testing.assert_close(
                value, expected[name], atol=2e-5, rtol=BF16_GRAD_REL_L2 if bf16 else 2e-4, msg=name
            )
        checks[f"step{step + 1}_loss_all_gradients_norm_and_update_match"] = True
        metrics[f"step{step + 1}_loss"] = float(got_loss.detach())
    moved = any(not torch.equal(value, initial[name]) for name, value in actual.items())
    reference_moved = any(not torch.equal(value, initial_full[name]) for name, value in expected.items())
    checks["optimizer_really_changed_weights"] = world_all(moved and reference_moved, ctx.device)
    empty_rank = {key: value.clone() for key, value in batch.items()}
    if ctx.rank == 1:
        empty_rank["completion_mask"].zero_()
    assert bool(empty_rank["completion_mask"].any()) == (ctx.rank == 0)
    for trainer in (bounded, full):
        trainer.state.global_step = STEPS
        trainer.optimizer.zero_grad(set_to_none=True)
    _, error = _compare_backward(bounded, full, empty_rank, ep=family == "qwen3_moe", bf16=bf16)
    worst_gradient = max(worst_gradient, error)
    checks["zero_target_rank_backward_matches_reference"] = True
    checks["actual_backbone_compute_dtype_matches_row"] = all(
        values and values[-1] == dtype for values in (bounded_dtypes, full_dtypes)
    )
    metrics["worst_parameter_gradient_relative_l2"] = worst_gradient
    return {"checks": checks, "metrics": metrics}
