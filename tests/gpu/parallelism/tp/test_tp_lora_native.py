#!/usr/bin/env python
"""Native PEFT TP=2 CUDA gradients, clipping, fused AdamWBF16 and generation."""

import os

import torch
import torch.distributed as dist

from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.harness import gpu_test_main
from tests.common.tp_lora_native import (
    INPUT_SEED,
    align_reference,
    assert_native_factors,
    assert_replicas_equal,
    full,
    lora_layers,
    native_model,
    reference_grad_norm,
    tp_clip_trainer,
    write_tiny_checkpoint,
)

FORWARD_RTOL = 3e-2
FORWARD_ATOL = 3e-3
GRAD_RTOL = 5e-2
GRAD_ATOL = 5e-3
STEPS = 3


def _assert_gradients(native, reference):
    reference_layers = lora_layers(reference)
    for name, layer in lora_layers(native).items():
        for factor_name in ("lora_A", "lora_B"):
            actual = getattr(layer, factor_name)["default"].weight.grad
            expected = getattr(reference_layers[name], factor_name)["default"].weight.grad
            assert actual is not None and expected is not None
            torch.testing.assert_close(full(actual).float(), expected.float(), rtol=GRAD_RTOL, atol=GRAD_ATOL)


def _run_mode(ctx, path, checkpointing, max_norm):
    native = native_model(
        path,
        ctx.world_size,
        device=ctx.device,
        dtype=torch.bfloat16,
        checkpointing=checkpointing,
        autocast_adapter_dtype=False,
    )
    reference = native_model(
        path,
        ctx.world_size,
        device=ctx.device,
        dtype=torch.bfloat16,
        checkpointing=checkpointing,
        autocast_adapter_dtype=False,
        parallel=False,
    )
    align_reference(native, reference)
    trainable = [param for param in native.parameters() if param.requires_grad]
    assert all(param.dtype == torch.bfloat16 for param in trainable)
    before = [full(param) for param in trainable]
    trainer = tp_clip_trainer(native)
    optimizer = AdamWBF16(trainable, lr=2e-2, weight_decay=0.0)
    generator = torch.Generator(device=ctx.device).manual_seed(INPUT_SEED)
    values = torch.randn(
        (2, 7, native.config.hidden_size), generator=generator, device=ctx.device, dtype=torch.bfloat16
    )
    native.train()
    reference.train()
    for _ in range(2):
        inputs, ref_inputs = values.clone().requires_grad_(True), values.clone().requires_grad_(True)
        output = native(inputs_embeds=inputs, use_cache=False).logits
        expected = reference(inputs_embeds=ref_inputs, use_cache=False).logits
        torch.testing.assert_close(output, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
        probe = torch.linspace(-0.7, 0.9, output.numel(), device=ctx.device).reshape_as(output)
        (output.float() * probe).sum().backward()
        (expected.float() * probe).sum().backward()
        torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=GRAD_RTOL, atol=GRAD_ATOL)
        _assert_gradients(native, reference)
    norm = trainer.accelerator.clip_grad_norm_(native.parameters(), max_norm)
    expected_norm = reference_grad_norm(reference)
    torch.testing.assert_close(norm, expected_norm, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    if max_norm > 0:
        assert expected_norm > max_norm, "the clipping case must actually scale the gradients"
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
    _assert_gradients(native, reference)
    optimizer.step()
    assert_replicas_equal(native, ctx.world_size)
    for step in range(1, STEPS):
        trainer.state.global_step = step
        optimizer.zero_grad(set_to_none=True)
        native(inputs_embeds=values, use_cache=False).logits.float().square().mean().backward()
        trainer.accelerator.clip_grad_norm_(native.parameters(), max_norm)
        if max_norm > 0:
            clipped = trainer._compute_tp_grad_norm([param for param in trainable if param.grad is not None])
            assert clipped <= max_norm * (1 + torch.finfo(torch.bfloat16).eps)
        optimizer.step()
        assert_replicas_equal(native, ctx.world_size)
    assert all(not torch.equal(full(param), previous) for param, previous in zip(trainable, before, strict=True))
    align_reference(native, reference)
    native.eval()
    reference.eval()
    tokens = torch.tensor([[1, 9, 12, 3]], device=ctx.device)
    mask = torch.ones_like(tokens)
    for use_cache in (False, True):
        with torch.no_grad():
            output = native(input_ids=tokens, attention_mask=mask, use_cache=use_cache).logits
            expected = reference(input_ids=tokens, attention_mask=mask, use_cache=use_cache).logits
            generated = native.generate(
                tokens, attention_mask=mask, max_new_tokens=3, use_cache=use_cache, do_sample=False
            )
            expected_tokens = reference.generate(
                tokens, attention_mask=mask, max_new_tokens=3, use_cache=use_cache, do_sample=False
            )
        assert output.grad_fn is None and torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
        torch.testing.assert_close(generated, expected_tokens, rtol=0, atol=0)
        peers = [torch.empty_like(output) for _ in range(ctx.world_size)]
        dist.all_gather(peers, output)
        assert all(torch.equal(peer, peers[0]) for peer in peers[1:])
        peers = [torch.empty_like(generated) for _ in range(ctx.world_size)]
        dist.all_gather(peers, generated)
        assert all(torch.equal(peer, peers[0]) for peer in peers[1:])
    assert_native_factors(native)


def run(ctx):
    path = os.path.join(ctx.output_dir, "tiny_llama")
    if ctx.rank == 0:
        write_tiny_checkpoint(path)
    ctx.barrier()
    checks = {}
    for checkpointing, max_norm in ((False, 0.0), (True, 0.05)):
        label = "checkpointed_clipped" if checkpointing else "plain_unclipped"
        _run_mode(ctx, path, checkpointing, max_norm)
        checks[label] = True
    return {"checks": checks}


main = gpu_test_main(exact_world_size=2, prefix="tp_lora_native")(run)

if __name__ == "__main__":
    main()
