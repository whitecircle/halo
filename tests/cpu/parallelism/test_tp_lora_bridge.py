#!/usr/bin/env python
"""Multi-process equivalence tests for the HF-native TP LoRA compatibility bridge.

The tiny model carries both plain TP layouts: ``q_proj`` is colwise and ``o_proj`` is rowwise.
The tests compare the bridged model with an unsharded PEFT model after copying the bridge's full
adapter tensors into the reference. This isolates placement and collectives from random init while
separate initialization checks catch local-fan-in and tiled-shard failures.

Run: ``python tests/cpu/parallelism/test_tp_lora_bridge.py``.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from peft import get_peft_model
from torch.distributed.tensor import DTensor, Shard

from src.distributed.tensor_parallel.lora import apply_tp_to_lora
from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.ports import free_port
from tests.common.tp_lora_bridge import (
    BASE_SEED,
    HIDDEN_FEATURES,
    IN_FEATURES,
    INPUT_SEED,
    LORA_RANK,
)
from tests.common.tp_lora_bridge import (
    TinyTPModel as _TinyTPModel,
)
from tests.common.tp_lora_bridge import (
    align_reference_from_bridge as _align_reference_from_bridge,
)
from tests.common.tp_lora_bridge import (
    assert_replicated_factors_equal as _assert_replicated_factors_equal,
)
from tests.common.tp_lora_bridge import (
    full as _full,
)
from tests.common.tp_lora_bridge import (
    lora_config as _lora_config,
)
from tests.common.tp_lora_bridge import (
    lora_layers as _lora_layers,
)
from tests.common.tp_lora_bridge import (
    reference as _reference,
)
from tests.common.tp_lora_bridge import (
    reference_grad_norm as _reference_grad_norm,
)
from tests.common.tp_lora_bridge import (
    tp_clip_trainer as _tp_clip_trainer,
)
from tests.common.tp_lora_bridge import (
    tp_peft as _tp_peft,
)

PG_TIMEOUT_SECONDS = 60


def _init_process_group(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=PG_TIMEOUT_SECONDS),
    )


def _assert_forward_and_grads_match(tp_model: nn.Module, reference: nn.Module) -> None:
    generator = torch.Generator().manual_seed(INPUT_SEED)
    x_value = torch.randn((3, IN_FEATURES), generator=generator, dtype=torch.float32)
    x = x_value.clone().requires_grad_(True)
    ref_x = x_value.clone().requires_grad_(True)

    tp_model.train()
    reference.train()
    tp_output = tp_model(x)
    ref_output = reference(ref_x)
    torch.testing.assert_close(tp_output, ref_output, rtol=1e-5, atol=1e-6)

    probe = torch.linspace(-0.7, 0.9, tp_output.numel()).reshape_as(tp_output)
    (tp_output * probe).sum().backward()
    (ref_output * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)

    ref_layers = _lora_layers(reference)
    for name, tp_layer in _lora_layers(tp_model).items():
        ref_layer = ref_layers[name]
        for factor_name in ("lora_A", "lora_B"):
            tp_grad = getattr(tp_layer, factor_name)["default"].weight.grad
            ref_grad = getattr(ref_layer, factor_name)["default"].weight.grad
            assert tp_grad is not None and ref_grad is not None
            torch.testing.assert_close(_full(tp_grad), ref_grad, rtol=1e-5, atol=1e-6)

    tp_model.eval()
    reference.eval()
    with torch.no_grad():
        tp_eval = tp_model(x_value)
        ref_eval = reference(x_value)
    torch.testing.assert_close(tp_eval, ref_eval, rtol=1e-5, atol=1e-6)
    assert tp_eval.grad_fn is None


def _equivalence_worker(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        for checkpointing in (False, True):
            for max_grad_norm in (0.0, 0.15):
                tp_model = _tp_peft(world_size, checkpointing=checkpointing)
                reference = _reference(checkpointing=checkpointing)
                _align_reference_from_bridge(tp_model, reference)
                _assert_forward_and_grads_match(tp_model, reference)

                trainer = _tp_clip_trainer(tp_model)
                ref_norm = _reference_grad_norm(reference)
                tp_norm = trainer.accelerator.clip_grad_norm_(tp_model.parameters(), max_grad_norm)
                torch.testing.assert_close(tp_norm, ref_norm, rtol=1e-5, atol=1e-6)
                if max_grad_norm > 0:
                    assert ref_norm > max_grad_norm, "the clipping test must exercise scaling"
                    clipped_ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_grad_norm)
                    torch.testing.assert_close(clipped_ref_norm, ref_norm, rtol=1e-5, atol=1e-6)

                ref_layers = _lora_layers(reference)
                for name, tp_layer in _lora_layers(tp_model).items():
                    for factor_name in ("lora_A", "lora_B"):
                        tp_grad = getattr(tp_layer, factor_name)["default"].weight.grad
                        ref_grad = getattr(ref_layers[name], factor_name)["default"].weight.grad
                        assert tp_grad is not None and ref_grad is not None
                        torch.testing.assert_close(_full(tp_grad), ref_grad, rtol=1e-5, atol=1e-6)

                factor_ids = {
                    name: (id(layer.lora_A["default"].weight), id(layer.lora_B["default"].weight))
                    for name, layer in _lora_layers(tp_model).items()
                }
                assert apply_tp_to_lora(tp_model) == 0
                assert factor_ids == {
                    name: (id(layer.lora_A["default"].weight), id(layer.lora_B["default"].weight))
                    for name, layer in _lora_layers(tp_model).items()
                }
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_forward_backward_checkpointing_and_clipping_match_unsharded(world_size):
    mp.start_processes(
        _equivalence_worker,
        args=(world_size, free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def _initialization_worker(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        for initialization in (True, False, "gaussian", "orthogonal"):
            model = _tp_peft(world_size, initialization=initialization, bridge=False)
            rng_before = torch.get_rng_state()
            assert apply_tp_to_lora(model) == 2
            assert torch.equal(torch.get_rng_state(), rng_before), "the bridge changed this rank's RNG state"
            rng_peers = [torch.empty_like(rng_before) for _ in range(world_size)]
            dist.all_gather(rng_peers, torch.get_rng_state())
            assert all(torch.equal(peer, rng_peers[0]) for peer in rng_peers[1:]), "TP ranks' RNG states drifted"
            layers = _lora_layers(model)
            q_a = layers["q_proj"].lora_A["default"].weight
            q_b = layers["q_proj"].lora_B["default"].weight
            o_a = layers["o_proj"].lora_A["default"].weight
            o_b = layers["o_proj"].lora_B["default"].weight

            assert not isinstance(q_a, DTensor) and not isinstance(o_b, DTensor)
            assert isinstance(q_b, DTensor) and q_b.placements == (Shard(0),)
            assert isinstance(o_a, DTensor) and o_a.placements[0].dim % 2 == 1
            assert q_b.shape == (HIDDEN_FEATURES, LORA_RANK)
            assert o_a.shape == (LORA_RANK, HIDDEN_FEATURES)
            _assert_replicated_factors_equal(model, world_size)

            full_q_b, full_o_a = _full(q_b), _full(o_a)
            assert torch.count_nonzero(full_o_a) > 0
            if initialization in (True, "gaussian"):
                assert torch.count_nonzero(full_q_b) == 0
            else:
                assert torch.count_nonzero(full_q_b) > 0

            o_shards = [torch.empty_like(o_a.to_local()) for _ in range(world_size)]
            dist.all_gather(o_shards, o_a.to_local())
            assert any(not torch.equal(shard, o_shards[0]) for shard in o_shards[1:]), (
                "rowwise A was initialized as repeated local tiles instead of one global tensor"
            )
            if initialization is True:
                assert full_o_a.abs().max() <= 1 / math.sqrt(HIDDEN_FEATURES) + 1e-7
            if initialization in (False, "orthogonal"):
                q_shards = [torch.empty_like(q_b.to_local()) for _ in range(world_size)]
                dist.all_gather(q_shards, q_b.to_local())
                assert any(not torch.equal(shard, q_shards[0]) for shard in q_shards[1:]), (
                    "colwise B was initialized as repeated local tiles instead of one global tensor"
                )
            if initialization == "gaussian":
                assert 0.5 / LORA_RANK <= full_o_a.std().item() <= 2 / LORA_RANK
            if initialization == "orthogonal":
                assert 0.02 <= full_o_a.std().item() <= 0.2
                assert 0.02 <= full_q_b.std().item() <= 0.2

        rank_seeded = _tp_peft(world_size, bridge=False, adapter_seed=BASE_SEED + rank)
        for name, layer in _lora_layers(rank_seeded).items():
            factor = layer.lora_A["default"] if name == "q_proj" else layer.lora_B["default"]
            peers = [torch.empty_like(factor.weight) for _ in range(world_size)]
            dist.all_gather(peers, factor.weight)
            assert any(not torch.equal(peer, peers[0]) for peer in peers[1:]), (
                f"{name} replicas unexpectedly share the same pre-bridge initialization"
            )
        assert apply_tp_to_lora(rank_seeded) == 2
        _assert_replicated_factors_equal(rank_seeded, world_size)
    finally:
        dist.destroy_process_group()


def test_global_shape_initialization_and_factor_placements():
    world_size = 4
    mp.start_processes(
        _initialization_worker,
        args=(world_size, free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def _optimizer_worker(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        for autocast_adapter_dtype, checkpointing, max_grad_norm in (
            (True, False, 0.0),
            (True, True, 0.2),
            (False, False, 0.0),
            (False, True, 0.2),
        ):
            model = _tp_peft(
                world_size,
                dtype=torch.bfloat16,
                checkpointing=checkpointing,
                autocast_adapter_dtype=autocast_adapter_dtype,
            )
            expected_dtype = torch.float32 if autocast_adapter_dtype else torch.bfloat16
            assert all(param.dtype == expected_dtype for param in model.parameters() if param.requires_grad)
            trainer = _tp_clip_trainer(model)
            optimizer = AdamWBF16(
                [param for param in model.parameters() if param.requires_grad],
                lr=2e-2,
                weight_decay=0.0,
                use_triton=False,
            )
            before = {
                name: param.detach().to_local().clone() if isinstance(param, DTensor) else param.detach().clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            generator = torch.Generator().manual_seed(INPUT_SEED)
            batches = [torch.randn((4, IN_FEATURES), generator=generator, dtype=torch.bfloat16) for _ in range(3)]
            for step, batch in enumerate(batches):
                trainer.state.global_step = step
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch).float().square().mean()
                loss.backward()
                trainer.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                _assert_replicated_factors_equal(model, world_size)

            changed = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                local = param.detach().to_local() if isinstance(param, DTensor) else param.detach()
                changed.append(not torch.equal(local, before[name]))
            assert all(changed), "an LoRA factor did not move after three AdamWBF16 steps"

            model.eval()
            with torch.no_grad():
                output = model(batches[0])
            assert torch.isfinite(output).all()
    finally:
        dist.destroy_process_group()


def test_adamw_bf16_keeps_replicas_equal_with_clipping_on_and_off():
    world_size = 2
    mp.start_processes(
        _optimizer_worker,
        args=(world_size, free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def _align_unbridged_control(tp_model: nn.Module, reference: nn.Module, target_name: str, style: str) -> None:
    tp_layer = _lora_layers(tp_model)[target_name]
    ref_layer = _lora_layers(reference)[target_name]
    a = tp_layer.lora_A["default"].weight.detach()
    b = tp_layer.lora_B["default"].weight.detach()
    with torch.no_grad():
        if style == "colwise":
            b_shards = [torch.empty_like(b) for _ in range(dist.get_world_size())]
            dist.all_gather(b_shards, b)
            ref_layer.lora_B["default"].weight.copy_(torch.cat(b_shards, dim=0))
            dist.broadcast(a, src=0)
            ref_layer.lora_A["default"].weight.copy_(a)
        else:
            a_shards = [torch.empty_like(a) for _ in range(dist.get_world_size())]
            dist.all_gather(a_shards, a)
            ref_layer.lora_A["default"].weight.copy_(torch.cat(a_shards, dim=1))
            dist.broadcast(b, src=0)
            ref_layer.lora_B["default"].weight.copy_(b)


def _negative_collective_worker(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        generator = torch.Generator().manual_seed(INPUT_SEED)
        x_value = torch.randn((3, IN_FEATURES), generator=generator)

        # Without colwise B's TP transform, forward still composes through the base o_proj all-reduce,
        # but A sees only this rank's output slice and its gradient is incomplete.
        tp_q = _tp_peft(world_size, targets=("q_proj",), bridge=False)
        ref_q = _reference(targets=("q_proj",))
        _align_unbridged_control(tp_q, ref_q, "q_proj", "colwise")
        q_out, q_ref = tp_q(x_value), ref_q(x_value)
        torch.testing.assert_close(q_out, q_ref, rtol=1e-5, atol=1e-6)
        q_out.square().mean().backward()
        q_ref.square().mean().backward()
        q_grad = _lora_layers(tp_q)["q_proj"].lora_A["default"].weight.grad
        q_ref_grad = _lora_layers(ref_q)["q_proj"].lora_A["default"].weight.grad
        assert q_grad is not None and q_ref_grad is not None
        assert not torch.allclose(q_grad, q_ref_grad, rtol=1e-4, atol=1e-6), (
            "removing colwise B's backward all-reduce did not break gradient equivalence"
        )

        # Without rowwise A's TP transform, B receives only this rank's partial rank-r intermediate.
        tp_o = _tp_peft(world_size, targets=("o_proj",), bridge=False)
        ref_o = _reference(targets=("o_proj",))
        _align_unbridged_control(tp_o, ref_o, "o_proj", "rowwise")
        o_out, o_ref = tp_o(x_value), ref_o(x_value)
        assert not torch.allclose(o_out, o_ref, rtol=1e-4, atol=1e-6), (
            "removing rowwise A's forward all-reduce did not break forward equivalence"
        )
    finally:
        dist.destroy_process_group()


def test_removing_either_required_collective_breaks_equivalence():
    world_size = 2
    mp.start_processes(
        _negative_collective_worker,
        args=(world_size, free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def _target_rejection_worker(rank: int, world_size: int, port: int) -> None:
    _init_process_group(rank, world_size, port)
    try:
        gathered = _tp_peft(world_size, bridge=False)
        gathered.get_base_model()._tp_plan["q_proj"] = "colwise_gather_output"
        with pytest.raises(ValueError, match="only literal 'colwise' and 'rowwise'"):
            apply_tp_to_lora(gathered)

        mismatched = _tp_peft(world_size, bridge=False)
        mismatched.get_base_model()._tp_plan["q_proj"] = "rowwise"
        with pytest.raises(ValueError, match=r"requires Shard\(1\)"):
            apply_tp_to_lora(mismatched)

        unnamed = _tp_peft(world_size, bridge=False, named_mesh=False)
        with pytest.raises(ValueError, match="named one-dimensional TP mesh"):
            apply_tp_to_lora(unnamed)

        native = _tp_peft(world_size, bridge=False)
        native_layers = _lora_layers(native)
        mesh = native.get_base_model()._device_mesh
        for factor, shape, shard_dim in (
            (native_layers["q_proj"].lora_B["default"], (HIDDEN_FEATURES, LORA_RANK), 0),
            (native_layers["o_proj"].lora_A["default"], (LORA_RANK, HIDDEN_FEATURES), 1),
        ):
            local = factor.weight
            distributed = DTensor.from_local(
                local,
                mesh,
                [Shard(shard_dim)],
                run_check=False,
                shape=torch.Size(shape),
                stride=torch.empty(shape, device="meta").stride(),
            )
            factor._parameters["weight"] = nn.Parameter(distributed, requires_grad=local.requires_grad)
        with pytest.raises(ValueError, match="DTensor factors not created by this bridge"):
            apply_tp_to_lora(native)

        endpoint = _tp_peft(world_size, bridge=False)
        endpoint_base = endpoint.get_base_model()
        endpoint_base.get_output_embeddings = lambda: _lora_layers(endpoint)["q_proj"]
        with pytest.raises(ValueError, match="lm_head targets"):
            apply_tp_to_lora(endpoint)
    finally:
        dist.destroy_process_group()


def test_unsupported_style_placement_and_output_head_are_rejected():
    world_size = 2
    mp.start_processes(
        _target_rejection_worker,
        args=(world_size, free_port()),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )


def test_multiple_adapters_are_rejected():
    model = get_peft_model(_TinyTPModel(), _lora_config())
    model.add_adapter("second", _lora_config())

    with pytest.raises(ValueError, match="supports one adapter"):
        apply_tp_to_lora(model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lora_dropout", 0.1),
        ("use_dora", True),
        ("lora_bias", True),
        ("modules_to_save", ["head"]),
        ("trainable_token_indices", [0]),
        ("target_parameters", ["q_proj.weight"]),
    ],
)
def test_unsupported_lora_config_is_rejected_before_tp_mutation(field, value):
    model = get_peft_model(_TinyTPModel(), _lora_config())
    setattr(model.peft_config["default"], field, value)

    with pytest.raises(ValueError, match=field):
        apply_tp_to_lora(model)


@pytest.mark.parametrize("initialization", ["pissa", "olora", "loftq", "eva", "corda"])
def test_full_weight_or_input_initializers_are_rejected(initialization):
    model = get_peft_model(_TinyTPModel(), _lora_config())
    model.peft_config["default"].init_lora_weights = initialization

    with pytest.raises(ValueError, match="does not support init_lora_weights"):
        apply_tp_to_lora(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
