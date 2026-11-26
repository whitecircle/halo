"""Full-model DPO head equivalence on identical dense-FSDP2 or EP2 layouts and frozen reference scores."""

import json
import math
from contextlib import ExitStack, contextmanager, nullcontext
from copy import copy
from types import MethodType

import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

from halo.distributed.fsdp import fsdp2_param_groups, reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.introspection.module_tree import hidden_state_backbone
from halo.models.introspection.unwrapping import base_transformers_model
from halo.runtime.consensus import barrier_on_exit
from halo.runtime.dtensor import to_local
from halo.runtime.ranks import get_global_rank
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all
from tests.common.dpo_head_gradients import PinnedHeadGradients
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
BF16_DENSE_GRAD_REL_L2 = 2.2e-2
BF16_MOE_GRAD_REL_L2 = 1.9e-2
# The unrounded FP32-head control measures expected BF16 drift; it is not the acceptance gate.
BF16_UNROUNDED_CONTROL_REL_L2 = {"dense": BF16_DENSE_GRAD_REL_L2, "qwen3_moe": 2.35e-2}
ZERO_GRAD_ATOL = 1e-12
STEPS = 2
LEARNING_RATE = 0.05


class _FullHeadDPOTrainer(DistributedDPOTrainer):
    def compute_ref_log_probs(self, model, inputs):
        with self.accelerator.autocast():
            return super().compute_ref_log_probs(model, inputs)


def resolve_dpo_gradient_tolerance(*, family: str, bf16: bool) -> float:
    """Select the family-specific whole-model gradient bound at the comparison's compute precision."""
    if family not in ("dense", "qwen3_moe"):
        raise ValueError("DPO gradient comparison accepts dense or qwen3_moe")
    if not bf16:
        return FP32_GRAD_REL_L2
    return BF16_DENSE_GRAD_REL_L2 if family == "dense" else BF16_MOE_GRAD_REL_L2


def assert_dpo_gradients_match(
    actual: dict, expected: dict, *, bf16: bool, family: str = "dense", tolerance: float | None = None
) -> float:
    """Each local gradient shard must match; all ranks together cover every parameter."""
    tolerance = resolve_dpo_gradient_tolerance(family=family, bf16=bf16) if tolerance is None else tolerance
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
        if relative > tolerance:
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


def assert_dpo_weight_steps_match(first, second, learning_rate):
    """Bound each weight by prior divergence, actual gradient difference and one predicted storage quantum."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise AssertionError("DPO weight-step comparison requires a finite positive learning rate")
    snapshots = (*first, *second)
    reference = first[0]
    if reference.dtype not in (torch.float32, torch.bfloat16) or any(
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
        or not bool(value.isfinite().all())
        for value in snapshots
    ):
        raise AssertionError(
            "DPO weight-step snapshots require finite matching FP32 or BF16 shapes, dtypes and devices"
        )
    spacing = []
    for before, gradient, _after in (first, second):
        predicted = before.clone().add_(gradient, alpha=-learning_rate)
        if not bool(predicted.isfinite().all()):
            raise AssertionError("DPO weight-step prediction is nonfinite")
        upper = torch.nextafter(predicted, torch.full_like(predicted, math.inf))
        lower = torch.nextafter(predicted, torch.full_like(predicted, -math.inf))
        spacing.append(
            torch.maximum(
                torch.where(upper.isfinite(), upper.double() - predicted.double(), 0.0),
                torch.where(lower.isfinite(), predicted.double() - lower.double(), 0.0),
            )
        )
    before, gradient, after = first
    other_before, other_gradient, other_after = second
    budget = (
        (before.double() - other_before.double()).abs()
        + learning_rate * (gradient.double() - other_gradient.double()).abs()
        + torch.maximum(*spacing)
    )
    if bool(((after.double() - other_after.double()).abs() > budget).any()):
        raise AssertionError("DPO stored weight differs beyond its elementwise gradient and storage-quantum budget")


def assert_dpo_updates_match(before, gradients, updated, *, bf16: bool) -> None:
    """COLLECTIVE. Join local native-SGD, raw-delta and stored-weight checks before the next forward."""
    with barrier_on_exit("DPO independent SGD and stored-weight checks"):
        names = before[0].keys()
        assert names and all(snapshot.keys() == names for snapshot in (*before, *gradients, *updated))
        for name in names:
            for arm in (0, 1):
                assert_sgd_update(before[arm][name], gradients[arm][name], updated[arm][name], LEARNING_RATE)
            first = (before[0][name], gradients[0][name], updated[0][name])
            second = (before[1][name], gradients[1][name], updated[1][name])
            assert_sgd_deltas_match(first, second, LEARNING_RATE)
            if bf16:
                assert_dpo_weight_steps_match(first, second, LEARNING_RATE)
            else:
                torch.testing.assert_close(updated[0][name], updated[1][name], atol=2e-5, rtol=2e-4, msg=name)


def verify_dpo_numerical_storage(trainer, *, family: str) -> None:
    """Require dense FP32 masters or ordinary BF16 EP parameters after the parallel wrap."""
    reshard_fsdp2_modules(trainer.model)
    expected = torch.float32 if family == "dense" else torch.bfloat16
    parameters = list(trainer.model.named_parameters())
    if not parameters:
        raise AssertionError("DPO numerical fixture has no model parameters")
    for name, parameter in parameters:
        if parameter.dtype != expected:
            raise AssertionError(f"DPO {family} numerical fixture requires {expected} storage: {name}")


def _parity_loss(logps, batch, *, beta, weight):
    chosen, rejected = logps.sum(1, dtype=torch.float64).chunk(2)
    delta = (chosen - batch["ref_chosen_logps"].double()) - (rejected - batch["ref_rejected_logps"].double())
    return -F.logsigmoid(beta * delta).mean() * weight


def _fp32_head_logits(module, hidden, *, backward_dtype=None):
    with torch.autocast(hidden.device.type, enabled=False):
        rounded_weight = module.weight.to(hidden.dtype).float()
        weight = rounded_weight
        if backward_dtype is not None:
            # The kernel reads rounded values but returns its accumulated gradient in master precision.
            master = module.weight.float()
            weight = master + (rounded_weight - master).detach()
        logits = F.linear(hidden.float(), weight, None if module.bias is None else module.bias.float())
        if backward_dtype is not None and logits.requires_grad:
            logits.register_hook(lambda gradient: gradient.to(backward_dtype).float())
        return logits


def _fp32_head(module, hidden):
    return _fp32_head_logits(module, hidden)


def _native_backward_fp32_head(module, hidden):
    dtype = hidden.dtype if hidden.is_cuda and hidden.dtype == torch.bfloat16 else None
    return _fp32_head_logits(module, hidden, backward_dtype=dtype)


@contextmanager
def _matched_oracle_head(trainer, *, bf16):
    if not bf16:
        yield
        return
    head = base_transformers_model(trainer.model).get_output_embeddings()
    original = head.forward
    head.forward = MethodType(_native_backward_fp32_head, head)
    try:
        yield
    finally:
        head.forward = original


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


@contextmanager
def same_state_dpo_oracle(oracle, policy):
    """Temporarily copy same-layout policy shards into the oracle; restore its weights/state and discard probe grads."""
    if oracle.model is policy.model:
        raise AssertionError("DPO same-state oracle requires two distinct model trajectories")
    for trainer in (policy, oracle):
        reshard_fsdp2_modules(trainer.model)
    source = dict(policy.model.named_parameters())
    target = dict(oracle.model.named_parameters())
    if not source or source.keys() != target.keys():
        raise AssertionError("DPO same-state oracle requires matching nonempty parameter names")
    for name, parameter in target.items():
        current, supplied = to_local(parameter), to_local(source[name])
        if (
            parameter.shape != source[name].shape
            or current.shape != supplied.shape
            or current.dtype != supplied.dtype
            or current.device != supplied.device
        ):
            raise AssertionError(f"DPO same-state oracle requires matching parameter shards: {name}")
    original = {name: to_local(parameter).detach().clone() for name, parameter in target.items()}
    state = oracle.state
    try:
        with torch.no_grad():
            for name, parameter in target.items():
                to_local(parameter).copy_(to_local(source[name]))
        # Sweep markers include TrainerState identity; the probe must not consume the real step's sync.
        oracle.state = copy(state)
        yield
    finally:
        try:
            reshard_fsdp2_modules(oracle.model)
            with torch.no_grad():
                for name, parameter in oracle.model.named_parameters():
                    to_local(parameter).copy_(original[name])
        finally:
            oracle.state = state
            oracle.optimizer.zero_grad(set_to_none=True)


def _batch(rank: int, device):
    rows = [
        {"prompt_ids": [2, 4 + rank], "chosen_ids": [6, 7, 8, 9, 1], "rejected_ids": [10, 1]},
        {"prompt_ids": [2, 5, 6], "chosen_ids": [], "rejected_ids": [11, 12, 13, 1]},
    ]
    batch = DataCollatorForPreference(pad_token_id=1)(rows)
    _assert_fixture_tokens(batch)
    return {key: value.to(device) for key, value in batch.items()}


def _assert_fixture_tokens(batch):
    """Pin the scored IDs and both masks; parity alone cannot detect an extra EOS in both paths."""
    completion = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
        device=batch["completion_mask"].device,
    )
    attention = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
        device=batch["attention_mask"].device,
    )
    if not torch.equal(batch["completion_mask"].bool(), completion):
        raise AssertionError("DPO fixture completion mask changed")
    if not torch.equal(batch["attention_mask"].bool(), attention):
        raise AssertionError("DPO fixture attention mask changed")
    expected = ([6, 7, 8, 9, 1], [], [10, 1], [11, 12, 13, 1])
    for row, wanted in enumerate(expected):
        actual = batch["input_ids"][row][completion[row] & attention[row]].tolist()
        if actual != wanted:
            raise AssertionError(f"DPO fixture scored IDs changed on row {row}: {actual} != {wanted}")


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
        fp32_non_ep_params=family == "dense",
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


def _compare_backward(bounded, full, batch, *, family: str, bf16: bool, diagnose: bool = False):
    ep = family == "qwen3_moe"
    phase = {"rank": get_global_rank(), "step": bounded.state.global_step}
    with ExitStack() as scope:
        diagnosis = scope.enter_context(PinnedHeadGradients().record(full)) if diagnose else None
        capture = captured_picks(bounded.model) if ep else nullcontext([])
        print("__DPO_SCORING_PHASE__ " + json.dumps(dict(phase, phase="bounded forward/backward")), flush=True)
        with capture as picks, bounded.compute_loss_context_manager():
            got_loss = bounded.compute_loss(bounded.model, batch)
        bounded.accelerator.backward(got_loss)
        bounded._finish_fsdp2_backward(bounded.model)
        got_norm = bounded._get_grad_norm(bounded.model)
        with barrier_on_exit("DPO bounded gradient snapshot"):
            gradients = _local_snapshot(bounded, gradients=True)
        with same_state_dpo_oracle(full, bounded):
            replay = routing_pinned(full.model, [torch.cat(chunks) for chunks in picks]) if picks else nullcontext([])
            print("__DPO_SCORING_PHASE__ " + json.dumps(dict(phase, phase="same-state oracle forward")), flush=True)
            with replay as routes, _matched_oracle_head(full, bf16=bf16), full.compute_loss_context_manager():
                want_loss = full.compute_loss(full.model, batch)
            with barrier_on_exit("DPO same-state loss and routing checks"):
                assert_dpo_loss_matches(got_loss, want_loss)
                if ep:
                    assert (
                        picks
                        and routes
                        and all(
                            route.duplicated == 0 and route.regret <= TOL.router_near_tie_logit_abs for route in routes
                        )
                    )
            print("__DPO_SCORING_PHASE__ " + json.dumps(dict(phase, phase="same-state oracle backward")), flush=True)
            full.accelerator.backward(want_loss)
            full._finish_fsdp2_backward(full.model)
            want_norm = full._get_grad_norm(full.model)
            with barrier_on_exit("DPO same-state gradients, norm and expert-activity checks"):
                expected_gradients = _local_snapshot(full, gradients=True)
                if diagnosis and not bf16:
                    report = diagnosis.report(gradients, expected_gradients)
                    report.update(
                        rank=bounded.accelerator.process_index,
                        bounded_loss=float(got_loss.detach()),
                        full_loss=float(want_loss.detach()),
                    )
                    print(
                        "__DPO_HEAD_GRADIENT_DIAGNOSTIC__ " + json.dumps(report),
                        flush=True,
                    )
                worst = assert_dpo_gradients_match(
                    gradients, expected_gradients, bf16=bf16, family=family, tolerance=FP32_GRAD_REL_L2
                )
                torch.testing.assert_close(
                    torch.as_tensor(got_norm),
                    torch.as_tensor(want_norm),
                    atol=1e-8,
                    rtol=FP32_GRAD_REL_L2,
                )
                if ep:
                    expert_ids = bounded._get_sharded_expert_param_ids()
                    names = {
                        name for name, parameter in bounded.model.named_parameters() if id(parameter) in expert_ids
                    }
                    assert names and any(gradients[name].count_nonzero() for name in names), (
                        "no local expert gradient was exercised"
                    )
        if bf16:
            with same_state_dpo_oracle(full, bounded):
                replay = (
                    routing_pinned(full.model, [torch.cat(chunks) for chunks in picks]) if picks else nullcontext()
                )
                with replay, full.compute_loss_context_manager():
                    unrounded_loss = full.compute_loss(full.model, batch)
                with barrier_on_exit("DPO unrounded BF16 loss control"):
                    assert_dpo_loss_matches(got_loss, unrounded_loss)
                full.accelerator.backward(unrounded_loss)
                full._finish_fsdp2_backward(full.model)
                full._get_grad_norm(full.model)
                with barrier_on_exit("DPO unrounded BF16 gradient control"):
                    unrounded_gradients = _local_snapshot(full, gradients=True)
                    assert_dpo_gradients_match(
                        gradients,
                        unrounded_gradients,
                        bf16=True,
                        family=family,
                        tolerance=BF16_UNROUNDED_CONTROL_REL_L2[family],
                    )
                    if diagnosis:
                        report = diagnosis.report(gradients, unrounded_gradients)
                        report.update(
                            rank=bounded.accelerator.process_index,
                            bounded_loss=float(got_loss.detach()),
                            full_loss=float(unrounded_loss.detach()),
                        )
                        print("__DPO_HEAD_GRADIENT_DIAGNOSTIC__ " + json.dumps(report), flush=True)
    print("__DPO_SCORING_PHASE__ " + json.dumps(dict(phase, phase="independent full forward")), flush=True)
    with full.compute_loss_context_manager():
        independent_loss = full.compute_loss(full.model, batch)
    with barrier_on_exit("DPO independent full-head loss check"):
        if not bool(independent_loss.isfinite().all()):
            raise AssertionError("The independent full-head trajectory produced a nonfinite loss")
    print("__DPO_SCORING_PHASE__ " + json.dumps(dict(phase, phase="independent full backward")), flush=True)
    full.accelerator.backward(independent_loss)
    full._finish_fsdp2_backward(full.model)
    full._get_grad_norm(full.model)
    return got_loss, worst, independent_loss


def run_dpo_scoring_equivalence(ctx, *, family: str, dtype_name: str) -> dict:
    """COLLECTIVE. Match native BF16 backward rounding, then bound unrounded FP32-head drift separately.

    Same-state scoring uses identical weights; the full-head control and bounded trainer
    still take independent optimizer steps. The scored token IDs and masks are pinned first.
    """
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
    for trainer in (bounded, full):
        verify_dpo_numerical_storage(trainer, family=family)
    checks["dense_fp32_masters" if family == "dense" else "ordinary_bf16_ep_storage"] = True
    metrics = {}
    expert_ids = bounded._get_sharded_expert_param_ids()
    expert_names = {name for name, parameter in bounded.model.named_parameters() if id(parameter) in expert_ids}
    if family == "qwen3_moe":
        assert expert_names, "the EP numerical row has no local expert parameters"
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
        got_loss, error, independent_loss = _compare_backward(
            bounded, full, batch, family=family, bf16=bf16, diagnose=bf16 and step == 0
        )
        worst_gradient = max(worst_gradient, error)
        with barrier_on_exit("DPO pre-step snapshots and optimizer checks"):
            if step == 0:
                assert_dpo_loss_matches(got_loss, parity)
            before = [_local_snapshot(trainer, gradients=False, host=False) for trainer in (bounded, full)]
            gradients = [_local_snapshot(trainer, gradients=True, host=False) for trainer in (bounded, full)]
            for trainer in (bounded, full):
                assert all(
                    group.get("momentum", 0) == group.get("weight_decay", 0) == 0 and not group.get("maximize", False)
                    for group in trainer.optimizer.param_groups
                )
        for trainer in (bounded, full):
            trainer.optimizer.step()
        with barrier_on_exit("DPO updated parameter snapshots"):
            updated = [_local_snapshot(trainer, gradients=False, host=False) for trainer in (bounded, full)]
        assert_dpo_updates_match(before, gradients, updated, bf16=bf16)
        with barrier_on_exit("DPO local expert-weight movement checks"):
            if expert_names:
                counts = [
                    sum(not torch.equal(before[arm][name], updated[arm][name]) for name in expert_names)
                    for arm in (0, 1)
                ]
                assert all(counts), "the DPO SGD step did not move any local expert parameter"
                checks[f"step{step + 1}_local_expert_weights_really_moved"] = True
                metrics[f"step{step + 1}_local_expert_tensors_updated"] = min(counts)
            actual, expected = (_local_snapshot(trainer, gradients=False) for trainer in (bounded, full))
        checks[f"step{step + 1}_same_state_loss_all_gradients_and_norm_match"] = True
        checks[f"step{step + 1}_independent_updates_meet_native_sgd_and_weight_bounds"] = True
        metrics[f"step{step + 1}_loss"] = float(got_loss.detach())
        metrics[f"step{step + 1}_independent_full_loss"] = float(independent_loss.detach())
        metrics[f"step{step + 1}_independent_loss_drift"] = float(
            (got_loss.detach() - independent_loss.detach()).abs()
        )
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
    _, error, _ = _compare_backward(bounded, full, empty_rank, family=family, bf16=bf16)
    worst_gradient = max(worst_gradient, error)
    checks["zero_target_rank_same_state_backward_matches_reference"] = True
    checks["actual_backbone_compute_dtype_matches_row"] = all(
        values and values[-1] == dtype for values in (bounded_dtypes, full_dtypes)
    )
    metrics["worst_same_state_parameter_gradient_relative_l2"] = worst_gradient
    return {"checks": checks, "metrics": metrics}
