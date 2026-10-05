#!/usr/bin/env python
"""Offline-GRPO scoring equivalence on Qwen3 for CP1, CP2 and CP4.

Exercises the production scoring and reduction helpers through real Ulysses attention,
then applies Halo's mean gradient sync explicitly.

Run on four GPUs: torchrun --nproc_per_node=4 tests/gpu/parallelism/cp/test_cp_grpo_scoring.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.base_layer import get_flash_attn_func
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.fsdp import reshard_fsdp2_modules, setup_fsdp2_for_dp
from src.distributed.pipeline_parallel.losses import loss_token_counts_per_row
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore
from src.trainers.grpo.objective.offline import offline_loss
from tests.common.cp_grpo import (
    boundary_loss_negative_control,
    full_row_loss,
    gradient_agreement,
    optimizer_step_agreement,
)
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

SEED = 918
SEQ = 64
LR = 0.1
GROUP_SIZES = (2, 3)
CONFIG = TINY_QWEN3_CONFIG | {
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 32,
}


def _model(device):
    torch.manual_seed(SEED)
    config = Qwen3Config(**CONFIG, attn_implementation="flash_attention_2")
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).train()


def _batch(device):
    torch.manual_seed(SEED + 1)
    ids = torch.randint(1, CONFIG["vocab_size"], (2, SEQ), device=device)
    mask = torch.ones_like(ids)
    mask[1, -8:] = 0
    ids[1, -8:] = 0
    labels = ids.clone()
    labels[0, :21] = LABEL_IGNORE_INDEX
    labels[1, :36] = LABEL_IGNORE_INDEX
    labels[1, -8:] = LABEL_IGNORE_INDEX
    advantages = torch.tensor([1.25, -0.75], device=device)
    group_sizes = torch.tensor(GROUP_SIZES, device=device)
    return ids, mask, labels, advantages, group_sizes


def _gradient_snapshot(model):
    return {
        name: (param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad).detach().float().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _step(model):
    torch.optim.SGD(model.parameters(), lr=LR).step()
    return {
        name: (param.full_tensor() if isinstance(param, DTensor) else param).detach().float().clone()
        for name, param in model.named_parameters()
    }


def run(ctx) -> dict:
    if ctx.world_size != 4:
        raise ValueError("this suite compares CP2 and CP4 in one four-GPU launch")
    cp_kernel = get_flash_attn_func()
    log(f"CP attention kernel: {cp_kernel.__module__}.{cp_kernel.__name__}; CP1 oracle: flash_attention_2")
    ids, mask, labels, advantages, group_sizes = _batch(ctx.device)
    for tensor in (ids, mask, labels, advantages, group_sizes):
        dist.broadcast(tensor, src=0)

    baseline = _model(ctx.device)
    initial_weights = {name: param.detach().float().clone() for name, param in baseline.named_parameters()}
    logits = baseline(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    full_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    base_loss = full_row_loss(full_logps, labels[:, 1:], advantages, group_sizes)
    base_loss.backward()
    base_loss_value = base_loss.detach().clone()
    full_logps = full_logps.detach().clone()
    base_gradients = _gradient_snapshot(baseline)
    base_weights = _step(baseline)
    del baseline, logits, base_loss
    cleanup_memory()

    checks = {}
    for cp_size in (2, 4):
        cp_config = CPConfig(cp_size=cp_size, world_size=ctx.world_size, gpus_per_node=ctx.world_size)
        model = UlyssesCPModelWrapper(_model(ctx.device), cp_config)
        scorer = ChunkedLogprobsCore()
        scorer.temperature = 1.0
        scorer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
        local_logps, shifted_labels = scorer._cp_chunked_logps(model, ids, mask, labels)
        chunk = SEQ // cp_size
        lo = cp_config.cp_rank * chunk
        hi = lo + shifted_labels.size(1)
        supervised = shifted_labels != LABEL_IGNORE_INDEX
        expected_labels = labels[:, lo + 1 : hi + 1]
        # The final position of a non-final shard predicts the NEXT shard's first label.
        if cp_config.cp_rank < cp_size - 1:
            expected_labels = labels[:, lo + 1 : lo + chunk + 1]
        labels_match = bool(torch.equal(shifted_labels, expected_labels))
        if cp_config.cp_rank == 0:
            labels_match &= bool(shifted_labels[0, -1] == labels[0, chunk])
        logp_error = (local_logps[supervised] - full_logps[:, lo:hi][supervised]).abs()
        group_logp_error = torch.tensor(logp_error.max().item() if logp_error.numel() else 0.0, device=ctx.device)
        group_targets = torch.tensor(logp_error.numel(), device=ctx.device)
        dist.all_reduce(group_logp_error, op=dist.ReduceOp.MAX, group=cp_config.process_group)
        dist.all_reduce(group_targets, op=dist.ReduceOp.SUM, group=cp_config.process_group)
        logps_match = bool(group_targets.item() > 0 and group_logp_error.item() < TOL.logprob_atol)
        if cp_size == 4:
            checks["cp4_zero_target_shard"] = cp_config.cp_rank != 0 or not bool(supervised.any())

        local_loss = -local_logps * advantages.unsqueeze(1)
        cp_loss = offline_loss(
            local_loss,
            supervised,
            group_sizes,
            loss_type="grpo",
            max_completion_length=SEQ,
            row_token_counts=loss_token_counts_per_row(labels),
            cp_config=cp_config,
        )
        loss_match = bool((cp_loss.detach() - base_loss_value).abs().item() < TOL.parallel_vs_baseline_loss_abs)
        cp_loss.backward()

        cp_gradients = _gradient_snapshot(model)
        for grad in cp_gradients.values():
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= ctx.world_size
        same_grad_keys, directions_match, norm_match, minimum_cosine, norm_ratio = gradient_agreement(
            cp_gradients, base_gradients
        )
        if cp_size == 4:
            correct, broken, correct_error, broken_error = boundary_loss_negative_control(
                scorer, model, ids, mask, advantages, group_sizes, full_logps, cp_config
            )
            checks["cp4_boundary_only_loss"] = correct
            checks["cp4_drop_boundary_negative_control"] = broken
            log(f"CP4 boundary-only loss error={correct_error:.4g}; dropped-boundary loss error={broken_error:.4g}")

        # Write the synchronized gradients back to the model before the optimizer step.
        for name, param in model.named_parameters():
            if name in cp_gradients:
                param.grad.copy_(cp_gradients[name].to(param.grad.dtype))
        cp_weights = _step(model)
        same_weight_keys = cp_weights.keys() == base_weights.keys()
        step_error = (
            max((cp_weights[name] - base_weights[name]).abs().max().item() for name in base_weights)
            if same_weight_keys
            else float("inf")
        )
        step_match = (
            same_weight_keys
            and step_error < TOL.kernel_atol
            and optimizer_step_agreement(cp_weights, base_weights, initial_weights)
        )

        checks[f"cp{cp_size}_boundary_labels"] = labels_match
        checks[f"cp{cp_size}_token_logps"] = logps_match
        checks[f"cp{cp_size}_row_loss"] = loss_match
        checks[f"cp{cp_size}_gradient_names"] = same_grad_keys
        checks[f"cp{cp_size}_gradient_norm"] = norm_match
        checks[f"cp{cp_size}_every_parameter_gradient_direction"] = directions_match
        checks[f"cp{cp_size}_optimizer_step"] = step_match
        log(
            f"CP{cp_size}: max supervised logp error={group_logp_error.item():.4g}; "
            f"loss error={(cp_loss.detach() - base_loss_value).abs().item():.4g}; "
            f"minimum gradient cosine={minimum_cosine:.5g}; gradient norm ratio={norm_ratio:.5g}; "
            f"step max error={step_error:.4g}"
        )
        del model, scorer
        cleanup_memory()

    # Production-shaped entry: FSDP2 wraps both decoder layers and the model root. The scorer's
    # redirection must enter that root, unshard lm_head, and keep it live through the local head
    # sweep; a bare CP model above cannot detect a missing root-forward redirect.
    cp_config = CPConfig(cp_size=2, world_size=ctx.world_size, gpus_per_node=ctx.world_size)
    model = UlyssesCPModelWrapper(_model(ctx.device), cp_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    setup_fsdp2_for_dp(
        model,
        dp_size=ctx.world_size,
        args=SimpleNamespace(bf16=True, fp16=False, fp32_grad_reduce=True),
        reshard_after_forward=False,
    )
    checks["fsdp2_root_wrapped"] = isinstance(model, FSDPModule) and any(
        isinstance(param, DTensor) for param in model.parameters()
    )
    checks["fsdp2_gc_enabled"] = bool(model.model.is_gradient_checkpointing)
    layer_calls = {"count": 0}
    hook = model.model.model.layers[0].register_forward_pre_hook(
        lambda _module, _args: layer_calls.__setitem__("count", layer_calls["count"] + 1)
    )
    scorer = ChunkedLogprobsCore()
    scorer.temperature = 1.0
    scorer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    local_logps, shifted_labels = scorer._cp_chunked_logps(model, ids, mask, labels)
    supervised = shifted_labels != LABEL_IGNORE_INDEX
    cp_loss = offline_loss(
        -local_logps * advantages.unsqueeze(1),
        supervised,
        group_sizes,
        loss_type="grpo",
        max_completion_length=SEQ,
        row_token_counts=loss_token_counts_per_row(labels),
        cp_config=cp_config,
    )
    checks["fsdp2_row_loss"] = bool(
        (cp_loss.detach() - base_loss_value).abs().item() < TOL.parallel_vs_baseline_loss_abs
    )
    cp_loss.backward()
    hook.remove()
    checks["fsdp2_gc_recomputed"] = layer_calls["count"] >= 2
    reshard_fsdp2_modules(model)
    fsdp_gradients = _gradient_snapshot(model)
    same_keys, directions_match, norm_match, minimum_cosine, norm_ratio = gradient_agreement(
        fsdp_gradients, base_gradients
    )
    checks["fsdp2_gradient_names"] = same_keys
    checks["fsdp2_every_parameter_gradient_direction"] = directions_match
    checks["fsdp2_gradient_norm"] = norm_match
    log(f"FSDP2+GC: minimum gradient cosine={minimum_cosine:.5g}; gradient norm ratio={norm_ratio:.5g}")
    fsdp_weights = _step(model)
    checks["fsdp2_optimizer_step"] = optimizer_step_agreement(fsdp_weights, base_weights, initial_weights)
    del model, scorer
    cleanup_memory()

    return {"checks": checks}


main = gpu_test_main(exact_world_size=4, prefix="cp_grpo_scoring", partial_state=False)(run)

if __name__ == "__main__":
    main()
