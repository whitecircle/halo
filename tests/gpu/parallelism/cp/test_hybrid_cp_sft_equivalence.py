#!/usr/bin/env python
"""Native CUDA hybrid CP1/CP2: GDN blocks, FSDP2 loss, every gradient, clipping and consecutive SGD updates."""

import argparse
import copy
import inspect
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
from transformers import DataCollatorForSeq2Seq, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5
from trl import SFTConfig

from halo.distributed.context_parallel import families, ulysses_attention
from halo.distributed.context_parallel.families import Qwen3_5UlyssesGatedDelta
from halo.distributed.context_parallel.sequence_chunks import cp_local_targets
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.selection import flash_kernel, preferred_flash_impl
from halo.models.unwrapping import base_transformers_model
from halo.runtime.dtensor import gather_full_tensors, materialize_dtensor
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir, world_mean
from tests.common.ep_reference import full_grad
from tests.common.harness import gpu_test_main
from tests.common.hybrid_cp import (
    build_gpu_hybrid_model,
    hybrid_hidden_batch,
    hybrid_tensor_agreement,
    hybrid_token_batch,
)
from tests.common.memory import finish_phase
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.rank_log import log
from tests.common.resume_export import save_base_checkpoint
from tests.common.tolerances import TOL

MAX_NORM = 0.01
LEARNING_RATE = 0.1


def _kernel_target(function):
    for _ in range(5):
        if not inspect.isfunction(function):
            break
        captured = inspect.getclosurevars(function).nonlocals
        if "implementation" in captured:
            return captured["implementation"]
        function = captured.get("dispatched", getattr(function, "__wrapped__", None))
    raise AssertionError("Cannot verify the native hub-kernel dispatch target")


def _check_tensor(checks, metrics, name, actual, expected, *, gradient=False):
    passed, values = hybrid_tensor_agreement(actual, expected, gradient=gradient)
    checks[name] = passed
    metrics.update({f"{name}/{key}": value for key, value in values.items()})


def _block_capture(layer, hidden, mask, target, *, checkpointing):
    layer.zero_grad(set_to_none=True)
    inputs = hidden.detach().clone().requires_grad_(True)
    output = checkpoint(layer, inputs, use_reentrant=True) if checkpointing else layer(inputs)
    (output.float() * target * mask[..., None]).sum().backward()
    return output.detach(), inputs.grad.detach(), {name: full_grad(value) for name, value in layer.named_parameters()}


def _gdn_blocks(ctx, batch, checkpointing):
    native = build_gpu_hybrid_model().model.layers[0].linear_attn.to(device=ctx.device, dtype=torch.bfloat16)
    finalize_loaded_model(native)
    hidden, mask, targets = hybrid_hidden_batch(batch, 130, native.hidden_size, device=ctx.device)
    hidden = hidden.to(torch.bfloat16)
    targets = targets / mask.sum()
    reference = _block_capture(native, hidden, mask, targets, checkpointing=checkpointing)
    wrapped = Qwen3_5UlyssesGatedDelta(copy.deepcopy(native), dist.group.WORLD, 2)
    part = slice(ctx.rank * 65, (ctx.rank + 1) * 65)
    observed = _block_capture(wrapped, hidden[:, part], mask[:, part], targets[:, part], checkpointing=checkpointing)
    checks, metrics = {}, {}
    real = mask[:, part, None].expand_as(observed[0])
    _check_tensor(checks, metrics, "gdn_output", observed[0][real], reference[0][:, part][real])
    _check_tensor(checks, metrics, "gdn_input_gradient", observed[1], reference[1][:, part], gradient=True)
    gradients = {}
    for name, value in observed[2].items():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        gradients[name.removeprefix("original_attention.")] = value
    checks["gdn_every_parameter_covered"] = gradients.keys() == reference[2].keys()
    for name, expected in reference[2].items():
        _check_tensor(checks, metrics, f"gdn_gradient/{name}", gradients.get(name), expected, gradient=True)

    # Native dense CP drops the layer mask; only the real causal prefix is qualified.
    masked = copy.deepcopy(native)
    masked.forward = lambda inputs: modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward(masked, inputs, attention_mask=mask)
    masked_reference = _block_capture(masked, hidden, mask, targets, checkpointing=checkpointing)
    real = mask[..., None].expand_as(reference[0])
    _check_tensor(checks, metrics, "gdn_mask_real_prefix", reference[0][real], masked_reference[0][real])
    _check_tensor(checks, metrics, "gdn_mask_input_gradient", reference[1], masked_reference[1], gradient=True)
    for name, expected in masked_reference[2].items():
        _check_tensor(checks, metrics, f"gdn_mask_gradient/{name}", reference[2][name], expected, gradient=True)

    if batch > 1:
        original_inverse = families.heads_to_sequence

        def mixed_rows(core, group):
            return original_inverse(core, group).roll(1, dims=0)

        with torch.no_grad(), patch.object(families, "heads_to_sequence", mixed_rows):
            changed = wrapped(hidden[:, part])
        real = mask[:, part, None].expand_as(changed)
        agrees, _ = hybrid_tensor_agreement(changed[real], reference[0][:, part][real])
        broken = torch.tensor(int(not agrees), device=ctx.device)
        dist.all_reduce(broken, op=dist.ReduceOp.MAX)
        checks["gdn_row_layout_negative_control"] = bool(broken)
    return checks, metrics


def _full_weights(model):
    reshard_fsdp2_modules(model)
    return gather_full_tensors(((name, value.detach()) for name, value in model.named_parameters()), retain=True)


def _build_trainer(ctx, source, tokenizer, batch, checkpointing, cp_size, attention):
    parallelism = ParallelismConfig(cp_size=cp_size, fp32_non_ep_params=True)
    model, _ = load_distributed_model(
        source, parallelism, dtype=torch.bfloat16, attn_implementation=attention, model_class=Qwen3_5ForCausalLM
    )
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    records = hybrid_token_batch(batch, len(tokenizer))
    dataset = Dataset.from_list(
        [{name: value[row].tolist() for name, value in records.items()} for row in range(batch)]
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=os.path.join(shared_output_dir(ctx), f"cp{cp_size}"),
            max_steps=2,
            per_device_train_batch_size=batch,
            learning_rate=LEARNING_RATE,
            optim="sgd",
            weight_decay=0.0,
            bf16=True,
            use_liger_kernel=False,
            gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            max_grad_norm=MAX_NORM,
            report_to=[],
            save_strategy="no",
            logging_steps=1,
            packing=False,
            padding_free=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataloader_num_workers=0,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=2),
        parallelism_config=parallelism,
        bf16_optimizer=False,
    )
    assert trainer._fsdp_wrapped and any(isinstance(value, DTensor) for value in trainer.model.parameters())
    assert base_transformers_model(trainer.model).config._attn_implementation == attention
    trainer.create_optimizer()
    return trainer


def _step(trainer, batch):
    model = trainer.model
    model.train()
    model.zero_grad(set_to_none=True)
    captured = []

    def retain_input(module, args, output):
        output.retain_grad()
        captured.append(output)

    handle = base_transformers_model(model).get_input_embeddings().register_forward_hook(retain_input)
    initial = _full_weights(model)
    with (
        patch.object(modeling_qwen3_5, "causal_conv1d_fn", wraps=modeling_qwen3_5.causal_conv1d_fn) as convolution,
        patch.object(
            modeling_qwen3_5, "torch_chunk_gated_delta_rule", wraps=modeling_qwen3_5.torch_chunk_gated_delta_rule
        ) as delta,
    ):
        with trainer.compute_loss_context_manager():
            loss, outputs = trainer.compute_loss(model, trainer._prepare_inputs(batch), return_outputs=True)
        trainer.accelerator.backward(loss)
        trainer._finish_fsdp2_backward(model)
        gradients = {name: full_grad(value).cpu() for name, value in model.named_parameters() if value.requires_grad}
    handle.remove()
    assert len(captured) == 1 and captured[0].grad is not None
    input_gradient = captured[0].grad.detach().float().cpu() / trainer.parallelism_config.cp_size
    clip_norm = trainer._clip_grad_norm(model)
    trainer._get_grad_norm(model, grad_norm=clip_norm)
    clipped = {name: full_grad(value).cpu() for name, value in model.named_parameters() if value.requires_grad}
    trainer.optimizer.step()
    final = _full_weights(model)
    trainer.state.global_step += 1
    return {
        "loss": world_mean(float(loss.detach()), trainer.accelerator.device),
        "logits": outputs.logits.detach().float().cpu(),
        "gradients": gradients,
        "clipped": clipped,
        "input_gradient": input_gradient,
        "clip_norm": float(materialize_dtensor(clip_norm)),
        "updates": {name: final[name] - value for name, value in initial.items()},
        "convolution_calls": convolution.call_count,
        "delta_rule_calls": delta.call_count,
    }


def _whole_model(ctx, source, tokenizer, batch, checkpointing, attention):
    batches = [hybrid_token_batch(batch, len(tokenizer), boundary_only=boundary) for boundary in (False, True)]
    reference = _build_trainer(ctx, source, tokenizer, batch, checkpointing, 1, attention)
    reference_steps = [_step(reference, copy.deepcopy(values)) for values in batches]
    finish_phase(reference)
    del reference
    trainer = _build_trainer(ctx, source, tokenizer, batch, checkpointing, 2, attention)
    checks, metrics = {}, {}
    for step, (values, expected) in enumerate(zip(batches, reference_steps, strict=True)):
        actual = _step(trainer, copy.deepcopy(values))
        prefix = f"step{step}"
        checks[f"{prefix}/loss"] = abs(actual["loss"] - expected["loss"]) <= TOL.parallel_vs_baseline_loss_abs
        metrics[f"{prefix}/loss_error"] = abs(actual["loss"] - expected["loss"])
        part = slice(ctx.rank * 65, (ctx.rank + 1) * 65)
        real = values["attention_mask"][:, part, None].bool().expand_as(actual["logits"])
        _check_tensor(checks, metrics, f"{prefix}/logits", actual["logits"][real], expected["logits"][:, part][real])
        _check_tensor(
            checks,
            metrics,
            f"{prefix}/input_gradient",
            actual["input_gradient"],
            expected["input_gradient"][:, part],
            gradient=True,
        )
        for field in ("gradients", "clipped", "updates"):
            checks[f"{prefix}/{field}_every_parameter_covered"] = actual[field].keys() == expected[field].keys()
            for name, want in expected[field].items():
                _check_tensor(
                    checks, metrics, f"{prefix}/{field}/{name}", actual[field].get(name), want, gradient=True
                )
        checks[f"{prefix}/clip_active"] = actual["clip_norm"] > MAX_NORM and expected["clip_norm"] > MAX_NORM
        clipped_norm = sum(value.double().square().sum() for value in actual["clipped"].values()).sqrt()
        checks[f"{prefix}/clip_achieved"] = float(clipped_norm) <= MAX_NORM * (1 + TOL.clip_achieved_norm_rel)
        checks[f"{prefix}/native_gdn_executed"] = actual["convolution_calls"] > 0 and actual["delta_rule_calls"] > 0
        checks[f"{prefix}/optimizer_moved_head"] = bool(actual["updates"]["lm_head.weight"].any())
    # Boundary-only supervision vanishes if the production last-target pairing is removed.
    boundary_batch = batches[-1]
    boundary_targets = cp_local_targets(boundary_batch["labels"], ctx.rank, 2)
    checks["boundary_and_empty_shard_pinned"] = bool(
        (boundary_targets != -100).sum() == (batch if ctx.rank == 0 else 0)
    )
    empty_targets = cp_local_targets(batches[0]["labels"], ctx.rank, 2)
    checks["ordinary_zero_target_shard_pinned"] = (
        bool((empty_targets != -100).sum() == 0) if ctx.rank == 0 else bool((empty_targets != -100).any())
    )

    def drop_boundary(labels, cp_rank, cp_size):
        targets = cp_local_targets(labels, cp_rank, cp_size).clone()
        targets[:, -1] = -100
        return targets

    with patch("halo.distributed.context_parallel.ulysses_model.cp_local_targets", drop_boundary):
        changed = _step(trainer, copy.deepcopy(boundary_batch))
    checks["dropped_boundary_changes_loss"] = (
        abs(changed["loss"] - reference_steps[-1]["loss"]) > TOL.parallel_vs_baseline_loss_abs
    )
    checks["dropped_boundary_removes_gradient"] = all(not value.any() for value in changed["gradients"].values())
    finish_phase(trainer)
    return checks, metrics


def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 3), required=True)
    parser.add_argument("--gc", action="store_true")
    args = parser.parse_args()
    ctx.broadcast_seed(114)
    convolution = _kernel_target(modeling_qwen3_5.causal_conv1d_fn)
    delta = _kernel_target(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    assert convolution.__module__.startswith("causal_conv1d") and delta.__module__.startswith("fla.")
    attention = preferred_flash_impl()
    selected = inspect.unwrap(ulysses_attention._flash_attn_func())
    expected = inspect.unwrap(flash_kernel(attention, varlen=False))
    assert (selected.__module__, selected.__qualname__) == (expected.__module__, expected.__qualname__)
    log(
        f"Native GDN CUDA dispatch: {convolution.__module__}.{convolution.__qualname__}, {delta.__module__}.{delta.__qualname__}; attention={attention}, CP={selected.__module__}.{selected.__qualname__}"
    )
    tokenizer = make_offline_tokenizer()
    base = save_base_checkpoint(
        ctx,
        build_gpu_hybrid_model(
            vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        ).to(torch.bfloat16),
        tokenizer,
        os.path.join(shared_output_dir(ctx), "base"),
    )
    checks, metrics = _gdn_blocks(ctx, args.batch_size, args.gc)
    model_checks, model_metrics = _whole_model(ctx, base, tokenizer, args.batch_size, args.gc, attention)
    return {"checks": checks | model_checks, "metrics": metrics | model_metrics}


main = gpu_test_main(exact_world_size=2, prefix="hybrid_cp_equivalence")(run)

if __name__ == "__main__":
    main()
