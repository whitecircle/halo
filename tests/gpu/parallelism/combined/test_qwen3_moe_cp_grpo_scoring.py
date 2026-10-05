#!/usr/bin/env python
"""Qwen3-MoE EP8 offline-GRPO scoring: CP1 vs CP2/CP4 on identical logical rows.

Tests the production scoring seam on real DeepEP and Ulysses, including EP-owned expert
gradients and one optimizer step.

Run on one NVLink-connected eight-GPU domain:
    torchrun --nproc_per_node=8 tests/gpu/parallelism/combined/test_qwen3_moe_cp_grpo_scoring.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.base_layer import get_flash_attn_func
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import loss_token_counts_per_row
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore
from src.trainers.grpo.objective.offline import offline_loss
from tests.common.cp_grpo import (
    boundary_loss_negative_control,
    ep_gradient_parameter_names,
    full_row_loss,
    gradient_agreement,
    optimizer_step_agreement,
)
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

SEED = 221
SEQ = 64
LR = 0.1
CONFIG = TINY_QWEN3_MOE_CONFIG | {
    "vocab_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "router_aux_loss_coef": 0.0,
    "attn_implementation": "flash_attention_2",
}


def _batch(device):
    torch.manual_seed(SEED + 1)
    ids = torch.randint(1, CONFIG["vocab_size"], (2, SEQ), device=device)
    mask = torch.ones_like(ids)
    ids[1, -8:] = 0
    mask[1, -8:] = 0
    labels = ids.clone()
    labels[0, :21] = LABEL_IGNORE_INDEX
    labels[1, :36] = LABEL_IGNORE_INDEX
    labels[1, -8:] = LABEL_IGNORE_INDEX
    advantages = torch.tensor([1.2, -0.8], device=device)
    group_sizes = torch.tensor([2, 3], device=device)
    return ids, mask, labels, advantages, group_sizes


def _build_model(device, cp_size):
    torch.manual_seed(SEED)
    config = Qwen3MoeConfig(**CONFIG)
    config.use_cache = False
    model = Qwen3MoeForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.router_aux_loss_coef = 0.0
    pc = ParallelismConfig(
        world_size=8, gpus_per_node=8, nvlink_domain_size=8, ep_size=8, cp_size=cp_size, ep_fp32_router=True
    )
    patch_moe_model_for_ep(model, pc.create_ep_config())
    assert create_ep_buffers(model) == CONFIG["num_hidden_layers"]
    cp_config = pc.create_cp_config() if cp_size > 1 else None
    if cp_size > 1:
        model = UlyssesCPModelWrapper(model, cp_config)
    model.train()
    ep_layers = [layer for layer in model.modules() if isinstance(layer, EPMoELayerBase)]
    assert len(ep_layers) == CONFIG["num_hidden_layers"]
    return model, cp_config, ep_layers


def _sync_non_ep_grads(model, ep_layers):
    expert_ids = {id(param) for layer in ep_layers for param in layer.parameters()}
    for _name, param in model.named_parameters():
        if id(param) in expert_ids:
            continue  # EP router/expert hooks already applied the correct world-wide sync/scale.
        has_grad = torch.tensor(int(param.grad is not None), device=param.device)
        dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        if has_grad.item() == 0:
            continue
        grad = param.grad.float() if param.grad is not None else torch.zeros_like(param, dtype=torch.float32)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= dist.get_world_size()
        if param.grad is None:
            param.grad = grad.to(param.dtype)
        else:
            param.grad.copy_(grad.to(param.grad.dtype))


def _snapshot(model):
    return {
        name: param.grad.detach().float().clone() for name, param in model.named_parameters() if param.grad is not None
    }


def _step(model):
    torch.optim.SGD(model.parameters(), lr=LR).step()
    return {name: param.detach().float().clone() for name, param in model.named_parameters()}


def _destroy_ep(ep_layers):
    dist.barrier()
    for layer in ep_layers:
        layer.dispatcher.destroy()
    dist.barrier()


def run(ctx) -> dict:
    cp_kernel = get_flash_attn_func()
    log(f"CP attention kernel: {cp_kernel.__module__}.{cp_kernel.__name__}; CP1 oracle: flash_attention_2")
    ids, mask, labels, advantages, group_sizes = _batch(ctx.device)
    for tensor in (ids, mask, labels, advantages, group_sizes):
        dist.broadcast(tensor, src=0)

    baseline, _, base_ep_layers = _build_model(ctx.device, cp_size=1)
    initial_weights = {name: param.detach().float().clone() for name, param in baseline.named_parameters()}
    ep_names = ep_gradient_parameter_names(baseline, base_ep_layers)
    parameter_groups = {"dense": initial_weights.keys() - ep_names, "expert_router": ep_names}
    assert all(parameter_groups.values())
    ep_bounds = {"cosine_min": TOL.ep_grad_cosine_min, "norm_ratio_band": TOL.ep_grad_norm_ratio_band}
    logits = baseline(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    full_logps = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    base_loss = full_row_loss(full_logps, labels[:, 1:], advantages, group_sizes)
    base_loss.backward()
    _sync_non_ep_grads(baseline, base_ep_layers)
    base_gradients = _snapshot(baseline)
    base_loss_value = base_loss.detach().clone()
    full_logps = full_logps.detach().clone()
    base_weights = _step(baseline)
    _destroy_ep(base_ep_layers)
    del baseline, base_ep_layers, logits, base_loss
    cleanup_memory()

    checks = {}
    for cp_size in (2, 4):
        model, cp_config, ep_layers = _build_model(ctx.device, cp_size)
        assert ep_gradient_parameter_names(model, ep_layers) == ep_names
        scorer = ChunkedLogprobsCore()
        scorer.temperature = 1.0
        scorer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
        local_logps, local_labels = scorer._cp_chunked_logps(model, ids, mask, labels)
        chunk = SEQ // cp_size
        lo = cp_config.cp_rank * chunk
        hi = lo + local_logps.size(1)
        labels_match = bool(torch.equal(local_labels, labels[:, lo + 1 : hi + 1]))
        if cp_config.cp_rank == 0:
            labels_match &= bool(local_labels[0, -1] == labels[0, chunk])
        local_valid = local_labels != LABEL_IGNORE_INDEX
        local_error = (local_logps[local_valid] - full_logps[:, lo:hi][local_valid]).abs()
        group_error = torch.tensor(local_error.max().item() if local_error.numel() else 0.0, device=ctx.device)
        group_targets = torch.tensor(local_error.numel(), device=ctx.device)
        dist.all_reduce(group_error, op=dist.ReduceOp.MAX, group=cp_config.process_group)
        dist.all_reduce(group_targets, op=dist.ReduceOp.SUM, group=cp_config.process_group)
        logps_match = bool(group_targets.item() > 0 and group_error.item() < TOL.logprob_atol)
        if cp_size == 4:
            checks["cp4_zero_target_shard"] = cp_config.cp_rank != 0 or not bool(local_valid.any())

        cp_loss = offline_loss(
            -local_logps * advantages.unsqueeze(1),
            local_valid,
            group_sizes,
            loss_type="grpo",
            max_completion_length=SEQ,
            row_token_counts=loss_token_counts_per_row(labels),
            cp_config=cp_config,
        )
        loss_match = bool((cp_loss.detach() - base_loss_value).abs().item() < TOL.parallel_vs_baseline_loss_abs)
        cp_loss.backward()
        _sync_non_ep_grads(model, ep_layers)
        cp_gradients = _snapshot(model)
        same_grad_keys = cp_gradients.keys() == base_gradients.keys()
        for kind, names in parameter_groups.items():
            got = {name: gradient for name, gradient in cp_gradients.items() if name in names}
            want = {name: gradient for name, gradient in base_gradients.items() if name in names}
            keys_match, direction_matches, norm_matches, minimum_cosine, norm_ratio = gradient_agreement(
                got, want, **(ep_bounds if kind == "expert_router" else {})
            )
            same_grad_keys &= keys_match
            checks[f"cp{cp_size}_{kind}_gradient_direction"] = direction_matches
            checks[f"cp{cp_size}_{kind}_gradient_norm"] = norm_matches
            log(
                f"EP8/CP{cp_size} {kind}: minimum gradient cosine={minimum_cosine:.5g}, "
                f"gradient norm ratio={norm_ratio:.5g}"
            )
        if cp_size == 4:
            correct, broken, correct_error, broken_error = boundary_loss_negative_control(
                scorer, model, ids, mask, advantages, group_sizes, full_logps, cp_config
            )
            checks["cp4_boundary_only_loss"] = correct
            checks["cp4_drop_boundary_negative_control"] = broken
            log(
                f"EP8/CP4 boundary-only loss error={correct_error:.4g}; dropped-boundary loss error={broken_error:.4g}"
            )

        cp_weights = _step(model)
        same_weight_keys = cp_weights.keys() == base_weights.keys()
        dense_step_error = (
            max((cp_weights[name] - base_weights[name]).abs().max().item() for name in parameter_groups["dense"])
            if same_weight_keys
            else float("inf")
        )
        checks[f"cp{cp_size}_boundary_labels"] = labels_match
        checks[f"cp{cp_size}_token_logps"] = logps_match
        checks[f"cp{cp_size}_row_loss"] = loss_match
        checks[f"cp{cp_size}_gradient_names"] = same_grad_keys
        step_matches = same_weight_keys and dense_step_error < TOL.kernel_atol
        for kind, names in parameter_groups.items():
            group_step_matches = optimizer_step_agreement(
                {name: weight for name, weight in cp_weights.items() if name in names},
                {name: weight for name, weight in base_weights.items() if name in names},
                {name: weight for name, weight in initial_weights.items() if name in names},
                **(ep_bounds if kind == "expert_router" else {}),
            )
            checks[f"cp{cp_size}_{kind}_optimizer_step"] = group_step_matches
            step_matches &= group_step_matches
        checks[f"cp{cp_size}_optimizer_step"] = step_matches
        log(
            f"EP8/CP{cp_size}: max supervised logp error={group_error.item():.4g}, "
            f"loss error={(cp_loss.detach() - base_loss_value).abs().item():.4g}, "
            f"dense step max error={dense_step_error:.4g}"
        )
        _destroy_ep(ep_layers)
        del model, cp_config, ep_layers, scorer
        cleanup_memory()
    return {"checks": checks}


main = gpu_test_main(exact_world_size=8, prefix="qwen3_moe_cp_grpo_scoring", partial_state=False)(run)

if __name__ == "__main__":
    main()
