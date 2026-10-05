#!/usr/bin/env python
"""Native PEFT TP equivalence and initialization on a real tiny Llama.

Initialization is tested independently of the aligned collective oracle, with a named ``tp``
mesh and no adapter reinitialization. An upstream initialization defect must remain a failure.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.gloo import run_gloo_ranks
from tests.common.tp_lora_native import (
    ADAPTER_SEED,
    INPUT_SEED,
    LORA_RANK,
    align_reference,
    assert_replicas_equal,
    full,
    lora_layers,
    native_model,
    reference_grad_norm,
    tp_clip_trainer,
    without_native_collective,
    write_tiny_checkpoint,
)

RTOL = 2e-5
ATOL = 2e-6


@pytest.fixture
def checkpoint_path(tmp_path):
    path = str(tmp_path / "llama")
    write_tiny_checkpoint(path)
    return path


def _run_workers(worker, world_size: int, path: str, *args) -> None:
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}):
        run_gloo_ranks(
            worker,
            world_size,
            world_size,
            path,
            *args,
            pg_timeout=timedelta(seconds=60),
            env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        )


def _assert_factor_grads(native, reference) -> None:
    reference_layers = lora_layers(reference)
    for name, layer in lora_layers(native).items():
        for factor_name in ("lora_A", "lora_B"):
            grad = getattr(layer, factor_name)["default"].weight.grad
            reference_grad = getattr(reference_layers[name], factor_name)["default"].weight.grad
            assert grad is not None and reference_grad is not None
            torch.testing.assert_close(full(grad), reference_grad, rtol=RTOL, atol=ATOL)


def _equivalence_worker(rank, world_size, path, checkpointing, max_norm):
    torch.set_num_threads(1)
    native = native_model(path, world_size, checkpointing=checkpointing)
    reference = native_model(path, world_size, checkpointing=checkpointing, parallel=False)
    align_reference(native, reference)
    native.train()
    reference.train()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    for _ in range(2):
        values = torch.randn((2, 7, native.config.hidden_size), generator=generator)
        inputs, ref_inputs = values.clone().requires_grad_(True), values.clone().requires_grad_(True)
        output = native(inputs_embeds=inputs, use_cache=False).logits
        ref_output = reference(inputs_embeds=ref_inputs, use_cache=False).logits
        torch.testing.assert_close(output, ref_output, rtol=RTOL, atol=ATOL)
        probe = torch.linspace(-0.7, 0.9, output.numel()).reshape_as(output)
        (output * probe).sum().backward()
        (ref_output * probe).sum().backward()
        torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=RTOL, atol=ATOL)
        _assert_factor_grads(native, reference)
    trainer = tp_clip_trainer(native)
    ref_norm = reference_grad_norm(reference)
    norm = trainer.accelerator.clip_grad_norm_(native.parameters(), max_norm)
    torch.testing.assert_close(norm, ref_norm, rtol=RTOL, atol=ATOL)
    if max_norm > 0:
        assert ref_norm > max_norm, "the clipping case must actually scale the gradients"
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
    _assert_factor_grads(native, reference)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("checkpointing,max_norm", [(False, 0.0), (True, 0.05)])
def test_native_forward_backward_accumulation_and_halo_clipping(checkpoint_path, world_size, checkpointing, max_norm):
    _run_workers(_equivalence_worker, world_size, checkpoint_path, checkpointing, max_norm)


def _initialization_worker(rank, world_size, path, initialization):
    torch.set_num_threads(1)
    model = native_model(path, world_size, initialization=initialization)
    assert_replicas_equal(model, world_size)
    failures = []
    for name, layer in lora_layers(model).items():
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        assert a.shape == (LORA_RANK, layer.get_base_layer().weight.shape[1])
        assert b.shape == (layer.get_base_layer().weight.shape[0], LORA_RANK)
        sharded = b if name.endswith("q_proj") else a
        local = sharded.to_local()
        peers = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(peers, local)
        if (name.endswith("o_proj") or initialization in (False, "orthogonal")) and all(
            torch.equal(peer, peers[0]) for peer in peers[1:]
        ):
            failures.append(f"{name}: the initializer repeated one local tile across TP ranks")
        if initialization is True and name.endswith("o_proj"):
            bound = 1 / math.sqrt(layer.get_base_layer().weight.shape[1])
            if full(a).abs().max() > bound + 1e-7:
                failures.append(f"{name}: rowwise A used local rather than global fan-in")
        if initialization in (True, "gaussian"):
            assert torch.count_nonzero(full(b)) == 0
        if initialization == "gaussian":
            a_std = full(a).std().item()
            if not 0.5 / LORA_RANK <= a_std <= 2 / LORA_RANK:
                failures.append(f"{name}: gaussian A std {a_std} does not match 1/r")
        if initialization == "orthogonal":
            # Half the rank's orthonormal rows contribute unit-variance samples, scaled by 1/10.
            expected_std = math.sqrt((LORA_RANK // 2) / LORA_RANK) / 10
            for factor_name, factor in (("A", a), ("B", b)):
                factor_std = full(factor).std().item()
                if not 0.5 * expected_std <= factor_std <= 2 * expected_std:
                    failures.append(f"{name}: orthogonal {factor_name} std {factor_std} has the wrong scale")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("initialization", [True, False, "gaussian", "orthogonal"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_named_tp_initialization_has_global_shapes_fan_in_and_no_tiling(checkpoint_path, initialization, world_size):
    _run_workers(_initialization_worker, world_size, checkpoint_path, initialization)


def _rank_seed_worker(rank, world_size, path, initialization):
    torch.set_num_threads(1)
    model = native_model(path, world_size, initialization=initialization, adapter_seed=ADAPTER_SEED + rank)
    assert_replicas_equal(model, world_size)


@pytest.mark.parametrize("initialization", [True, False, "gaussian", "orthogonal"])
def test_native_initialization_synchronizes_replicas_with_per_rank_seeds(checkpoint_path, initialization):
    _run_workers(_rank_seed_worker, 2, checkpoint_path, initialization)


def _optimizer_worker(rank, world_size, path, autocast_adapter_dtype, max_norm):
    torch.set_num_threads(1)
    model = native_model(path, world_size, dtype=torch.bfloat16, autocast_adapter_dtype=autocast_adapter_dtype)
    trainer = tp_clip_trainer(model)
    trainable = [param for param in model.parameters() if param.requires_grad]
    expected_dtype = torch.float32 if autocast_adapter_dtype else torch.bfloat16
    assert all(param.dtype == expected_dtype for param in trainable)
    before = [full(param) for param in trainable]
    optimizer = AdamWBF16(trainable, lr=2e-2, weight_decay=0.0, use_triton=False)
    tokens = torch.tensor([[1, 9, 12, 3, 7, 4, 8]])
    for step in range(3):
        trainer.state.global_step = step
        optimizer.zero_grad(set_to_none=True)
        model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()
        trainer.accelerator.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        assert_replicas_equal(model, world_size)
    assert all(not torch.equal(full(param), previous) for param, previous in zip(trainable, before, strict=True))


@pytest.mark.parametrize("autocast_adapter_dtype", [True, False])
@pytest.mark.parametrize("max_norm", [0.0, 0.05])
def test_adamw_bf16_keeps_native_replicas_identical(checkpoint_path, autocast_adapter_dtype, max_norm):
    _run_workers(_optimizer_worker, 2, checkpoint_path, autocast_adapter_dtype, max_norm)


def _generation_worker(rank, world_size, path, use_cache):
    torch.set_num_threads(1)
    native = native_model(path, world_size)
    reference = native_model(path, world_size, parallel=False)
    align_reference(native, reference)
    native.eval()
    reference.eval()
    tokens = torch.tensor([[1, 9, 12, 3]])
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        output = native(input_ids=tokens, attention_mask=mask, use_cache=use_cache).logits
        ref_output = reference(input_ids=tokens, attention_mask=mask, use_cache=use_cache).logits
        torch.testing.assert_close(output, ref_output, rtol=RTOL, atol=ATOL)
        kwargs = {"attention_mask": mask, "max_new_tokens": 3, "do_sample": False, "use_cache": use_cache}
        generated = native.generate(tokens, **kwargs)
        expected = reference.generate(tokens, **kwargs)
    assert output.grad_fn is None
    torch.testing.assert_close(generated, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_cache", [False, True])
def test_native_no_grad_evaluation_and_generate_match_plain_model(checkpoint_path, use_cache):
    _run_workers(_generation_worker, 2, checkpoint_path, use_cache)


def _negative_collective_worker(rank, world_size, path, target):
    torch.set_num_threads(1)
    native = native_model(path, world_size, targets=(target,))
    reference = native_model(path, world_size, targets=(target,), parallel=False)
    align_reference(native, reference)
    generator = torch.Generator().manual_seed(INPUT_SEED)
    inputs = torch.randn((2, 7, native.config.hidden_size), generator=generator)
    output = native(inputs_embeds=inputs, use_cache=False).logits
    expected = reference(inputs_embeds=inputs, use_cache=False).logits
    torch.testing.assert_close(output, expected, rtol=RTOL, atol=ATOL)
    layer = next(iter(lora_layers(native).values()))
    ref_layer = next(iter(lora_layers(reference).values()))
    factor = layer.lora_B["default"] if target == "q_proj" else layer.lora_A["default"]
    assert isinstance(factor.weight, DTensor)
    with without_native_collective(factor):
        broken = native(inputs_embeds=inputs, use_cache=False).logits
        if target == "q_proj":
            torch.testing.assert_close(broken, expected, rtol=RTOL, atol=ATOL)
            broken.square().sum().backward()
            expected.square().sum().backward()
            actual_grad = layer.lora_A["default"].weight.grad
            expected_grad = ref_layer.lora_A["default"].weight.grad
            assert actual_grad is not None and expected_grad is not None
            assert not torch.allclose(actual_grad, expected_grad, rtol=RTOL, atol=ATOL), (
                "removing native colwise B's backward SUM must break A's gradient"
            )
        else:
            assert not torch.allclose(broken, expected, rtol=RTOL, atol=ATOL), (
                "removing native rowwise A's forward SUM must break the model's output"
            )


@pytest.mark.parametrize("target", ["q_proj", "o_proj"])
def test_disabling_each_native_collective_independently_breaks_equivalence(checkpoint_path, target):
    _run_workers(_negative_collective_worker, 2, checkpoint_path, target)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
