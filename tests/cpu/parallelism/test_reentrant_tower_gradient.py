#!/usr/bin/env python
"""A reentrant checkpoint's outputs carry a gradient only if a positional input does, so a tower block behind a frozen
patch embedding would keep its trainable parameters at no gradient. On every composite a reentrant step's tower gradient
matches a non-reentrant run of the same step, in one process and on gloo ranks under per-layer FSDP2 with fused and split
experts, and a run training nothing there is bit for bit what the stock layer call gives."""

import datetime
import functools
import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.utils.checkpoint
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoModelForImageTextToText, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from trl import SFTConfig

from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.distributed import fsdp
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers, ep_claimed_blocks
from halo.distributed.fsdp import IdentityParamSet
from halo.entrypoints import data_setup
from halo.models import gradient_checkpointing
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.introspection.module_tree import (
    backbone_with_layers,
    decoder_layers,
    embedding_module,
    find_module_path,
    media_module_names,
)
from halo.models.moe.family_specs import find_moe_family_spec
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.composite_rows import COMPOSITE_FAMILIES, composite_model_and_rows, text_row
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model, word_level_tokenizer

PG_TIMEOUT = datetime.timedelta(seconds=120)
# Long enough for an 8-way xdist host to reach the reduce, short enough to wait out.
HANG_TIMEOUT = datetime.timedelta(seconds=30)
FP32_ROUNDING = 2e-5
BF16_ROUNDING = 1e-2
FAMILIES = sorted(COMPOSITE_FAMILIES)
_STOCK_CALL = GradientCheckpointingLayer.__call__.__wrapped__
_BF16_COMPUTE = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
_MESH_NAMES = {1: ("dp_shard",), 2: ("dp_replicate", "dp_shard")}
_ADAPTER_SHAPES = ("adapters_in_blocks", "adapters_in_last_block", "adapters_on_tower")
_FINE_TUNE_SHAPES = ("fine_tune_blocks", "fine_tune_last_block", "fine_tune_tower")
_NOTHING_TRAINED_IN_THE_TOWER = ("text_adapters", "text_fine_tune")
_TP_FAMILIES = ("qwen3_vl", "qwen3_5_moe")


def _checkpointed_tower_layers(model) -> list[str]:
    return [
        f"{tower}.{name}"
        for tower in media_module_names(model)
        for name, module in model.get_submodule(tower).named_modules()
        if isinstance(module, GradientCheckpointingLayer)
    ]


def _linears_under(model, root: str) -> list[str]:
    return [
        f"{root}.{name}"
        for name, module in model.get_submodule(root).named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def _block_linears(model) -> list[str]:
    return [linear for block in _checkpointed_tower_layers(model) for linear in _linears_under(model, block)]


def _lora(model, targets: list[str]):
    # Off the identity init, so lora_A's gradient is as real as lora_B's on the first step.
    torch.manual_seed(1)
    return get_peft_model(model, LoraConfig(r=2, target_modules=targets, init_lora_weights=False))


def _train_only(model, prefixes: list[str]):
    roots = tuple(f"{prefix}." for prefix in prefixes)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(roots)
    return model


def _shape_model(model, shape: str):
    """``shape`` on ``model``: the family's blocks where it checkpoints some, else its tower's own linears."""
    towers = media_module_names(model)
    blocks = _checkpointed_tower_layers(model)
    text_layers = find_module_path(model, decoder_layers(backbone_with_layers(model)))
    tower_linears = [linear for tower in towers for linear in _linears_under(model, tower)]
    block_linears = _block_linears(model) or tower_linears
    text_linears = _linears_under(model, text_layers)
    adapters = {
        "adapters_in_blocks": block_linears,
        "adapters_in_last_block": _linears_under(model, blocks[-1]) if blocks else tower_linears,
        "adapters_on_tower": tower_linears,
        "adapters_in_blocks_and_layers": block_linears + text_linears,
        "text_adapters": text_linears,
    }
    fine_tunes = {
        "fine_tune_blocks": blocks or towers,
        "fine_tune_blocks_and_layers": [*(blocks or towers), text_layers],
        "fine_tune_last_block": blocks[-1:] or towers,
        "fine_tune_tower": towers,
        "text_fine_tune": [text_layers],
    }
    if shape in adapters:
        return _lora(model, adapters[shape])
    if shape in fine_tunes:
        return _train_only(model, fine_tunes[shape])
    assert shape == "whole_model", shape
    return model


def _grads(model) -> dict[str, torch.Tensor]:
    return {
        name: (param.grad.full_tensor() if hasattr(param.grad, "full_tensor") else param.grad).clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def _tower_names(model) -> tuple[str, ...]:
    towers = media_module_names(model)
    return tuple(f"{tower}." for tower in towers) + tuple(f"base_model.model.{tower}." for tower in towers)


def _step(family: str, shape: str, use_reentrant: bool | None, *, row: str = "image"):
    """One seeded forward and backward: ``(wrapped model, loss, gradients by name)``."""
    torch.manual_seed(0)
    model, image_row, plain_row = composite_model_and_rows(family)
    shaped = _shape_model(model, shape)
    if use_reentrant is not None:
        shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    loss = shaped(**(image_row if row == "image" else plain_row)).loss
    loss.backward()
    return shaped, loss.detach(), _grads(shaped)


def _worst_error(run: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], scale: float = 1.0) -> float:
    """Relative, over the tensors both hold; the caller checks the two name sets."""
    return max(
        (
            float((run[name] - grad * scale).abs().max() / (grad.abs().max() * scale + 1e-9))
            for name, grad in expected.items()
            if name in run
        ),
        default=0.0,
    )


def _assert_same_gradients(run: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], context) -> None:
    assert run.keys() == reference.keys(), (context, sorted(run.keys() ^ reference.keys()))
    assert _worst_error(run, reference) < FP32_ROUNDING, context


@pytest.mark.parametrize("shape", _ADAPTER_SHAPES + _FINE_TUNE_SHAPES)
@pytest.mark.parametrize("family", FAMILIES)
def test_a_reentrant_step_takes_a_non_reentrant_steps_gradient(family, shape):
    model, loss, reentrant = _step(family, shape, True)
    _, reference_loss, non_reentrant = _step(family, shape, False)
    assert torch.equal(loss, reference_loss), (family, shape)
    tower = _tower_names(model)
    trained = [name for name, grad in non_reentrant.items() if name.startswith(tower) and grad.abs().max() > 0]
    assert trained, (family, shape, "the reference reached nothing in the tower")
    _assert_same_gradients(reentrant, non_reentrant, (family, shape))


def _declares_an_input_embedding(model, tower: str) -> bool:
    module = model.get_submodule(tower)
    if not isinstance(module, PreTrainedModel):
        return False
    try:
        return module.get_input_embeddings() is not None
    except NotImplementedError:
        return False


def _block_adapters_lose_their_gradient(family: str) -> bool:
    with patch.object(GradientCheckpointingLayer, "__call__", _STOCK_CALL):
        model, _, grads = _step(family, "adapters_in_blocks", True)
    tower = _tower_names(model)
    return not any(name.startswith(tower) for name in grads)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_stock_call_cuts_a_checkpointed_block_behind_no_declared_embedding(family):
    """What the fix answers, against the installed transformers: its input-grad hook reaches a tower only through
    ``get_input_embeddings``, a tower with no checkpointed block has nothing to cut, and Llama 4's blocks take their
    hidden state by keyword, which no reentrant checkpoint sees."""
    torch.manual_seed(0)
    model, _, _ = composite_model_and_rows(family)
    has_blocks = bool(_checkpointed_tower_layers(model))
    declared = any(_declares_an_input_embedding(model, tower) for tower in media_module_names(model))
    expected = has_blocks and (not declared or family == "llama4")
    assert _block_adapters_lose_their_gradient(family) is expected, (has_blocks, declared)


@pytest.mark.parametrize("family", ["gemma4", "nemotron_h_omni"])
def test_a_patch_projection_adapter_alone_opens_the_path(family):
    """Why these two looked covered: an adapter on the projection before the blocks gave the blocks a grad-carrying
    input, whichever call ran."""
    torch.manual_seed(0)
    model, image_row, _ = composite_model_and_rows(family)
    tower = media_module_names(model)[0]
    linears = _linears_under(model, tower)
    block = _checkpointed_tower_layers(model)[0]
    first_outside = next(linear for linear in linears if not linear.startswith(f"{tower}.encoder"))
    first_inside = next(linear for linear in linears if linear.startswith(f"{block}."))
    shaped = _lora(model, [first_outside, first_inside])
    shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    with patch.object(GradientCheckpointingLayer, "__call__", _STOCK_CALL):
        shaped(**image_row).loss.backward()
    assert any(first_inside.rpartition(".")[2] in name and block in name for name in _grads(shaped)), family


def _recording_new_inputs(new_inputs: list[str], calls: list[str] | None = None):
    """A positional tensor the call did not hold is the alias only a trained layer behind frozen modules should take."""
    real = gradient_checkpointing._with_gradient_carrying_input

    def spy(layer, args, kwargs):
        if calls is not None:
            calls.append(type(layer).__name__)
        returned_args, returned_kwargs = real(layer, args, kwargs)
        held = {id(value) for value in (*args, *kwargs.values())}
        if any(isinstance(arg, torch.Tensor) and id(arg) not in held for arg in returned_args):
            new_inputs.append(type(layer).__name__)
        return returned_args, returned_kwargs

    return patch.object(gradient_checkpointing, "_with_gradient_carrying_input", spy)


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().numpy().tobytes()


def _assert_stock_step(family: str, shape: str, use_reentrant: bool | None, row: str) -> None:
    new_inputs: list[str] = []
    with _recording_new_inputs(new_inputs):
        _, loss, grads = _step(family, shape, use_reentrant, row=row)
    with patch.object(GradientCheckpointingLayer, "__call__", _STOCK_CALL):
        _, stock_loss, stock_grads = _step(family, shape, use_reentrant, row=row)
    assert not new_inputs, (family, shape, row, new_inputs)
    assert _bytes(loss) == _bytes(stock_loss), (family, shape, row)
    assert grads.keys() == stock_grads.keys(), (family, shape, row)
    for name, grad in grads.items():
        assert _bytes(grad) == _bytes(stock_grads[name]), (family, shape, row, name)


@pytest.mark.parametrize("use_reentrant", [True, False, None], ids=["reentrant", "non_reentrant", "off"])
@pytest.mark.parametrize("row", ["image", "text"])
@pytest.mark.parametrize("shape", _NOTHING_TRAINED_IN_THE_TOWER)
@pytest.mark.parametrize("family", FAMILIES)
def test_a_tower_nothing_reaches_is_bit_for_bit_the_stock_step(family, shape, row, use_reentrant):
    _assert_stock_step(family, shape, use_reentrant, row)


@pytest.mark.parametrize("row", ["image", "text"])
@pytest.mark.parametrize("family", sorted(set(FAMILIES) - {"llama4"}))
def test_a_full_fine_tune_is_bit_for_bit_the_stock_step(family, row):
    """The patch embedding trains, so each block's input carries a gradient already. Llama 4's tower takes its hidden
    state by keyword, which the stock call cuts: :func:`test_a_reentrant_step_takes_a_non_reentrant_steps_gradient`
    holds it instead."""
    _assert_stock_step(family, "whole_model", True, row)


@pytest.mark.parametrize("use_reentrant", [False, None], ids=["non_reentrant", "off"])
@pytest.mark.parametrize("family", FAMILIES)
def test_only_a_reentrant_checkpoint_takes_a_new_input(family, use_reentrant):
    _assert_stock_step(family, "adapters_in_blocks", use_reentrant, "image")


_TEXT_ONLY = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
# Refused at load, by a class patch that outlives the load in a shared process.
_REFUSES_CHECKPOINTING = {
    name for name in TINY_MOE_FAMILIES if not find_moe_family_spec(name).supports_gradient_checkpointing
}


@pytest.mark.parametrize("family", sorted(set(_TEXT_ONLY) - set(COMPOSITE_FAMILIES) - _REFUSES_CHECKPOINTING))
def test_a_text_model_takes_no_new_input_behind_a_frozen_embedding(family):
    """The input-grad hook reaches every text layer, so the fix stays out of the stack that carries a run's cost."""
    torch.manual_seed(0)
    model = tiny_family_model(_TEXT_ONLY[family]).train()
    embedding_module(model, "input").requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    new_inputs: list[str] = []
    calls: list[str] = []
    with _recording_new_inputs(new_inputs, calls):
        model(**text_row([1, 2, 3, 4, 5, 6])).loss.backward()
    # A family checkpointing its layers from the model's own forward takes no such call.
    assert bool(calls) is any(isinstance(module, GradientCheckpointingLayer) for module in model.modules()), family
    assert not new_inputs, (family, new_inputs)


class _Block(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, hidden_states=None, ids=None):
        return self.linear(hidden_states if hidden_states is not None else ids.float())


def _checkpointed_block() -> _Block:
    block = _Block().train()
    block.gradient_checkpointing = True
    block._gradient_checkpointing_func = gradient_checkpointing.scoped_checkpoint_func(
        functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=True)
    )
    return block


def test_a_layer_with_no_float_to_carry_the_gradient_is_refused():
    with pytest.raises(ValueError, match="takes no floating tensor"):
        _checkpointed_block()(ids=torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="takes no floating tensor"):
        _checkpointed_block()(None, torch.ones(1, 2, dtype=torch.long))


def test_a_hidden_state_by_keyword_reaches_the_checkpoint_as_its_first_argument():
    upstream = torch.nn.Parameter(torch.ones(1, 2))
    _checkpointed_block().requires_grad_(False)(hidden_states=upstream * 2).sum().backward()
    assert upstream.grad is not None
    block = _checkpointed_block()
    block(hidden_states=torch.ones(1, 2)).sum().backward()
    assert block.linear.weight.grad is not None


def test_a_frozen_layer_keeps_its_call():
    block = _checkpointed_block().requires_grad_(False)
    new_inputs: list[str] = []
    with _recording_new_inputs(new_inputs):
        assert not block(torch.ones(1, 2)).requires_grad
        assert not block(hidden_states=torch.ones(1, 2)).requires_grad
    assert not new_inputs


class _KeywordFedBlock(GradientCheckpointingLayer):
    """A frozen block whose gradient comes from a module upstream, in an argument beside its hidden state."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.requires_grad_(False)

    def forward(self, hidden_states=None, extra=None):
        return self.linear(hidden_states + (extra[0] if isinstance(extra, tuple) else extra))


def _keyword_fed_block(use_reentrant: bool) -> _KeywordFedBlock:
    block = _KeywordFedBlock().train()
    block.gradient_checkpointing = True
    block._gradient_checkpointing_func = functools.partial(
        torch.utils.checkpoint.checkpoint, use_reentrant=use_reentrant
    )
    return block


@pytest.mark.parametrize("nest", [lambda tensor: tensor, lambda tensor: (tensor,)], ids=["tensor", "in_a_tuple"])
def test_a_gradient_arriving_by_keyword_crosses_a_frozen_layer(nest):
    def upstream_gradient(use_reentrant: bool) -> torch.Tensor:
        torch.manual_seed(0)
        upstream = torch.nn.Parameter(torch.ones(1, 2))
        _keyword_fed_block(use_reentrant)(torch.ones(1, 2), extra=nest(upstream * 2)).sum().backward()
        return upstream.grad

    expected = upstream_gradient(False)
    assert expected.abs().max() > 0
    assert torch.equal(upstream_gradient(True), expected)


def test_a_gradient_no_positional_float_can_carry_is_refused():
    upstream = torch.nn.Parameter(torch.ones(1, 2))
    with pytest.raises(ValueError, match="takes no floating tensor"):
        _keyword_fed_block(True)(extra=upstream * 2)
    with pytest.raises(ValueError, match="takes no floating tensor"):
        _keyword_fed_block(True)((upstream * 2,))


def test_the_caller_keeps_its_own_tensor_out_of_the_graph():
    hidden = torch.ones(1, 2)
    output = _checkpointed_block()(hidden)
    output.sum().backward()
    assert output.requires_grad and not hidden.requires_grad and hidden.grad is None


def test_an_eval_or_no_grad_forward_takes_no_new_input():
    block = _checkpointed_block()
    new_inputs: list[str] = []
    with _recording_new_inputs(new_inputs), torch.no_grad():
        assert not block(torch.ones(1, 2)).requires_grad
    block.eval()
    with _recording_new_inputs(new_inputs):
        block(torch.ones(1, 2))
    assert not new_inputs


@pytest.mark.parametrize("family", ["qwen4_exp", "nemotron_h_omni", "step3p7"])
def test_every_nth_layer_and_a_scoped_enable_keep_the_gradient(family):
    """Some tower blocks stay unchecked, and the checkpoint function is the recompute scope's wrap."""

    def run(use_reentrant: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model, image_row, _ = composite_model_and_rows(family)
        shaped = _shape_model(model, "adapters_in_blocks")
        enable_scoped_gradient_checkpointing(
            shaped, gradient_checkpointing_kwargs={"use_reentrant": use_reentrant, "every_n_layers": 2}
        )
        shaped(**image_row).loss.backward()
        return _grads(shaped)

    _assert_same_gradients(run(True), run(False), family)


def test_several_forwards_per_backward_recompute_each_row():
    def run(use_reentrant: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model, image_row, plain_row = composite_model_and_rows("qwen3_vl")
        shaped = _shape_model(model, "adapters_in_blocks")
        shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        losses = [shaped(**row).loss for row in (image_row, plain_row, image_row)]
        sum(losses).backward()
        return _grads(shaped)

    _assert_same_gradients(run(True), run(False), "three rows")


def _mixed_step(family: str, case: dict, rank: int, mesh, *, use_reentrant: bool) -> dict[str, torch.Tensor]:
    """The tower's gradient after this rank's microsteps: its row holds an image where ``case`` says so."""
    torch.manual_seed(0)
    model, image_row, plain_row = composite_model_and_rows(family)
    shaped = _shape_model(model, case["shape"])
    shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    policy = _BF16_COMPUTE if case["bf16"] else MixedPrecisionPolicy()
    tower = _tower_names(shaped)
    fsdp.apply_fsdp2_per_layer(shaped, mesh, policy, case["reshard"], IdentityParamSet())
    data_setup.carry_image_refusal_above_adapters(shaped)
    for step in range(case["steps"]):
        last = step == case["steps"] - 1
        if case["window"]:
            shaped.set_requires_gradient_sync(last)
        if case["hold_gathered"]:
            for module in fsdp.fsdp2_modules(shaped):
                module.set_reshard_after_backward(last, recurse=False)
        holds = rank in case["image_ranks"] and (not case["window"] or step == case["image_step"])
        rows = [image_row if holds else plain_row, *[plain_row] * (case["forwards"] - 1)]
        sum(shaped(**row).loss for row in rows).backward()
        fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(shaped))
    return {name: grad for name, grad in _grads(shaped).items() if name.startswith(tower)}


def _mixed_worker(rank: int, out_dir: str, case: dict) -> None:
    """Every family in one gloo run: a reentrant mixed step against a non-reentrant run of the same step."""
    mesh_shape = tuple(case["mesh"])
    mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=_MESH_NAMES[len(mesh_shape)])
    verdicts = {}
    for family in case["families"]:
        expected = _mixed_step(family, case, rank, mesh, use_reentrant=False)
        run = _mixed_step(family, case, rank, mesh, use_reentrant=True)
        verdicts[family] = {
            "reached": sum(float(grad.abs().max()) > 0 for grad in expected.values()),
            "lost": sorted(expected.keys() - run.keys()),
            "extra": sorted(run.keys() - expected.keys()),
            "worst": _worst_error(run, expected),
        }
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(verdicts, handle)


_MIXED_CASE = {
    "families": FAMILIES,
    "shape": "adapters_in_blocks_and_layers",
    "bf16": False,
    "tolerance": FP32_ROUNDING * 10,
    "image_ranks": [0],
    "mesh": [2],
    "reshard": False,
    "steps": 1,
    "window": False,
    "image_step": 0,
    "hold_gathered": False,
    "forwards": 1,
}


def _assert_tower_gradient(tmp_path, world: int, case: dict) -> None:
    run_gloo_ranks(_mixed_worker, world, str(tmp_path), case, pg_timeout=PG_TIMEOUT)
    for rank in range(world):
        verdicts = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert sorted(verdicts) == sorted(case["families"])
        for family, verdict in verdicts.items():
            assert verdict["reached"] and not verdict["lost"] and not verdict["extra"], (family, verdict)
            assert verdict["worst"] < case["tolerance"], (family, verdict)


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize("shape", ["adapters_in_blocks_and_layers", "adapters_in_blocks", "fine_tune_blocks"])
def test_a_mixed_step_takes_a_non_reentrant_steps_tower_gradient(tmp_path, shape, reshard):
    """A text rank holds no tower gradient until the agreement before the reduce, on every tree shape."""
    _assert_tower_gradient(tmp_path, 2, {**_MIXED_CASE, "shape": shape, "reshard": reshard})


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
def test_an_hsdp_mesh_reduces_the_tower_across_both_dims(tmp_path, reshard):
    case = {
        **_MIXED_CASE,
        "families": ["qwen3_vl", "nemotron_h_omni", "qwen3_5_moe"],
        "mesh": [2, 2],
        "reshard": reshard,
    }
    _assert_tower_gradient(tmp_path, 4, case)


@pytest.mark.parametrize("image_step", [0, 1], ids=["first", "last"])
def test_an_image_on_one_microstep_of_a_window_reaches_the_reduce(tmp_path, image_step):
    case = {**_MIXED_CASE, "families": ["qwen3_vl", "gemma4"], "steps": 2, "window": True, "image_step": image_step}
    _assert_tower_gradient(tmp_path, 2, case)


def test_a_group_held_gathered_past_its_reduce_keeps_the_gradient(tmp_path):
    """``fsdp_reshard_after_backward: false``: every microstep reduces, and only the last reshards."""
    case = {**_MIXED_CASE, "families": ["qwen3_vl", "gemma4", "nemotron_h_omni"], "steps": 2, "hold_gathered": True}
    _assert_tower_gradient(tmp_path, 2, case)


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
def test_several_forwards_per_backward_keep_the_gradient(tmp_path, reshard):
    """Each row recomputes its own layers inside one backward, as env-GRPO's and the log-prob scorers' do."""
    case = {
        **_MIXED_CASE,
        "families": ["qwen3_vl", "qwen3_5_moe", "nemotron_h_omni"],
        "forwards": 2,
        "reshard": reshard,
    }
    _assert_tower_gradient(tmp_path, 2, case)


def test_bf16_compute_takes_the_gradient_within_its_rounding(tmp_path):
    case = {
        **_MIXED_CASE,
        "families": ["qwen3_vl", "qwen3_vl_deepstack", "qwen3_5_moe", "gemma4", "nemotron_h_omni"],
        "bf16": True,
        "tolerance": BF16_ROUNDING,
    }
    _assert_tower_gradient(tmp_path, 2, case)


def test_an_image_on_every_rank_reduces_the_whole_gradient(tmp_path):
    case = {**_MIXED_CASE, "families": ["qwen3_vl", "minimax_m3_vl"], "image_ranks": [0, 1]}
    _assert_tower_gradient(tmp_path, 2, case)


def _unagreed_worker(rank: int, out_dir: str, family: str) -> None:
    """Accelerate-managed FSDP2 has no agreement before the reduce: the image rank enters with the tower's adapters."""
    torch.manual_seed(0)
    model, image_row, plain_row = composite_model_and_rows(family)
    shaped = _shape_model(model, "adapters_in_blocks_and_layers")
    shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=_MESH_NAMES[1])
    with patch.object(fsdp, "_agree_grad_presence_before_reducing", lambda _module: None):
        fsdp.apply_fsdp2_per_layer(shaped, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
    loss = shaped(**(image_row if rank == 0 else plain_row)).loss
    with open(os.path.join(out_dir, f"forward{rank}"), "w"):
        pass
    loss.backward()
    fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(shaped))


def test_with_no_agreement_a_checkpointed_mixed_step_cannot_pair(tmp_path):
    """Why a torchrun launch is the remedy the refusal under ``accelerate launch`` names, with checkpointing on too:
    the step's reduces, past both forwards, never pair."""
    with pytest.raises(Exception, match=r"(?i)reduce_scatter|terminated with signal|timed out"):
        run_gloo_ranks(_unagreed_worker, 2, str(tmp_path), "qwen3_vl", pg_timeout=HANG_TIMEOUT)
    assert (tmp_path / "forward0").exists() and (tmp_path / "forward1").exists()


def _tp_step(family: str, grid, *, use_reentrant: bool, reshard: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model, image_row, plain_row = composite_model_and_rows(family)
    shaped = _shape_model(model, "fine_tune_blocks_and_layers")
    plan = {f"self_attn.{name}": ColwiseParallel() for name in ("q_proj", "k_proj", "v_proj")}
    plan["self_attn.o_proj"] = RowwiseParallel()
    for layer in decoder_layers(backbone_with_layers(shaped)):
        parallelize_module(layer, grid["tp"], plan)
    shaped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    tower = _tower_names(shaped)
    fsdp.apply_fsdp2_per_layer(shaped, grid["dp_shard"], MixedPrecisionPolicy(), reshard, IdentityParamSet())
    holds_image = grid.get_coordinate()[0] == 0
    shaped(**(image_row if holds_image else plain_row)).loss.backward()
    fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(shaped))
    return {name: grad for name, grad in _grads(shaped).items() if name.startswith(tower)}


def _tp_worker(rank: int, out_dir: str, reshard: bool) -> None:
    grid = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))
    verdicts = {}
    for family in _TP_FAMILIES:
        expected = _tp_step(family, grid, use_reentrant=False, reshard=reshard)
        run = _tp_step(family, grid, use_reentrant=True, reshard=reshard)
        verdicts[family] = {
            "tensors": len(expected),
            "lost": sorted(expected.keys() - run.keys()),
            "worst": _worst_error(run, expected),
        }
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(verdicts, handle)


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
def test_tensor_parallel_text_layers_leave_the_towers_gradient_alone(tmp_path, reshard):
    """TP refuses adapters, so a fine-tune: the tower stays plain tensors, replicated over the TP peers."""
    run_gloo_ranks(_tp_worker, 4, str(tmp_path), reshard, pg_timeout=PG_TIMEOUT)
    for rank in range(4):
        verdicts = json.loads((tmp_path / f"rank{rank}.json").read_text())
        for family, verdict in verdicts.items():
            assert verdict["tensors"] and not verdict["lost"], (family, verdict)
            assert verdict["worst"] < FP32_ROUNDING, (family, verdict)


def _ep_config(experts: str) -> EPConfig:
    """COLLECTIVE. No DeepEP here: ep1 keeps the experts fused, an expert-TP axis stores gate and up apart."""
    expert_tp_size = 2 if experts == "split" else 1
    return EPConfig(
        ep_size=1,
        expert_tp_size=expert_tp_size,
        stage_world_size=dist.get_world_size(),
        nvlink_domain_size=dist.get_world_size(),
        use_grouped_gemm=False,
        experts_fsdp_managed=experts == "fused_sharded",
    )


def _ep_model(family: str, experts: str, use_reentrant: bool):
    torch.manual_seed(0)
    model, image_row, plain_row = composite_model_and_rows(family)
    apply_ep_layers(model, _ep_config(experts))
    shaped = _shape_model(model, "adapters_in_blocks")
    enable_scoped_gradient_checkpointing(shaped, gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    return shaped, model, image_row, plain_row


def _ep_worker(rank: int, out_dir: str, case: dict) -> None:
    """DP over the first mesh dim; expert-TP peers share a row, as they do in a run."""
    world = dist.get_world_size()
    expert_tp = 2 if case["experts"] == "split" else 1
    grid = init_device_mesh("cpu", (world // expert_tp, expert_tp), mesh_dim_names=("dp_shard", "etp"))
    dp_mesh = grid["dp_shard"]
    holds_image = grid.get_coordinate()[0] == 0
    verdicts = {}
    for family in case["families"]:
        reference, _, image_row, _ = _ep_model(family, case["experts"], False)
        reference(**image_row).loss.backward()
        tower = _tower_names(reference)
        expected = {name: grad for name, grad in _grads(reference).items() if name.startswith(tower)}
        del reference
        shaped, model, image_row, plain_row = _ep_model(family, case["experts"], True)
        layers = [layer for _, layer in named_ep_layers(model)]
        ignored = [] if case["experts"] == "fused_sharded" else [p for layer in layers for p in layer.parameters()]
        fsdp.apply_fsdp2_per_layer(shaped, dp_mesh, MixedPrecisionPolicy(), case["reshard"], IdentityParamSet(ignored))
        shaped(**(image_row if holds_image else plain_row)).loss.backward()
        fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(shaped))
        run = {name: grad for name, grad in _grads(shaped).items() if name.startswith(tower)}
        verdicts[family] = {
            "fused": layers[0]._stores_fused_glu,
            "tensors": len(expected),
            "lost": sorted(expected.keys() - run.keys()),
            "worst": _worst_error(run, expected, scale=expert_tp / world),
        }
        del shaped, model
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(verdicts, handle)


def _ep_families() -> list[str]:
    families = []
    for family in FAMILIES:
        torch.manual_seed(0)
        model, _, _ = composite_model_and_rows(family)
        if ep_claimed_blocks(model):
            families.append(family)
    return families


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize(
    ("experts", "world"),
    [("fused_sharded", 2), ("fused_ignored", 2), ("split", 4)],
    ids=["fused_sharded", "fused_ignored", "split"],
)
def test_expert_layers_leave_the_towers_gradient_alone(tmp_path, experts, world, reshard):
    families = _ep_families()
    assert len(families) >= 8, families
    case = {"families": families, "experts": experts, "reshard": reshard}
    run_gloo_ranks(_ep_worker, world, str(tmp_path), case, pg_timeout=PG_TIMEOUT)
    for rank in range(world):
        verdicts = json.loads((tmp_path / f"rank{rank}.json").read_text())
        for family, verdict in verdicts.items():
            assert verdict["tensors"] and not verdict["lost"], (family, verdict)
            assert verdict["worst"] < 1e-3, (family, verdict)
    stored_fused = {family: verdict["fused"] for family, verdict in verdicts.items()}
    # Nemotron-H stores its latent experts apart at any axis.
    assert any(stored_fused.values()) is (experts != "split"), stored_fused
    assert not all(stored_fused.values()), stored_fused


def _single_rank_trainer(model, image_row: dict, output_dir, *, use_reentrant: bool, tokenizer, **args):
    keys = sorted(image_row)
    return DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            use_cpu=True,
            bf16=False,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            use_liger_kernel=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            eval_strategy="no",
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
            **{"max_steps": 1, "save_strategy": "no", **args},
        ),
        train_dataset=Dataset.from_list([{key: image_row[key].tolist() for key in keys}] * 2),
        data_collator=lambda batch: {key: torch.tensor(batch[0][key]) for key in keys},
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=2, target_modules=_block_linears(model), init_lora_weights=False, task_type="CAUSAL_LM"
        ),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        moe_balancing="none",
    )


def _tower_adapter_weights(model) -> dict[str, torch.Tensor]:
    tower = _tower_names(model)
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(tower)
    }


@pytest.mark.parametrize(
    ("family", "configured"),
    [("qwen3_vl", True), ("qwen3_5_moe", False), ("step3p7", False)],
    ids=["dense_reentrant", "moe_forced_reentrant", "step_forced_reentrant"],
)
def test_the_trainers_own_step_moves_the_towers_adapters(tmp_path, family, configured):
    """The route a run takes: the trainer wraps the adapters and enables checkpointing, a MoE forced reentrant whatever
    the config says, and one optimizer step moves each tower adapter off its init."""
    torch.manual_seed(0)
    model, image_row, _ = composite_model_and_rows(family)
    tokenizer = word_level_tokenizer(("hello", "world"))
    trainer = _single_rank_trainer(model, image_row, tmp_path, use_reentrant=configured, tokenizer=tokenizer)
    before = _tower_adapter_weights(trainer.model)
    assert before
    trainer.train()
    after = dict(trainer.model.named_parameters())
    unmoved = [name for name, start in before.items() if torch.equal(after[name], start)]
    assert not unmoved, (family, unmoved)


def test_a_resumed_run_replays_the_towers_adapters_bit_for_bit(tmp_path):
    def trainer_at(output_dir, **args):
        torch.manual_seed(0)
        model, image_row, _ = composite_model_and_rows("qwen3_vl")
        tokenizer = word_level_tokenizer(("hello", "world"))
        return _single_rank_trainer(
            model, image_row, output_dir, use_reentrant=True, tokenizer=tokenizer, lr_scheduler_type="constant", **args
        )

    unbroken = trainer_at(tmp_path / "unbroken", max_steps=2)
    unbroken.train()
    interrupted = trainer_at(tmp_path / "resumed", max_steps=1, save_strategy="steps", save_steps=1)
    interrupted.train()
    moved = _tower_adapter_weights(interrupted.model)
    resumed = trainer_at(tmp_path / "resumed", max_steps=2)
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed" / "checkpoint-1"))
    expected = _tower_adapter_weights(unbroken.model)
    assert expected.keys() == moved.keys() and expected
    for name, weight in _tower_adapter_weights(resumed.model).items():
        assert torch.equal(weight, expected[name]), name
        assert not torch.equal(weight, moved[name]), name


def test_a_trained_tower_adapter_reaches_the_merged_export(tmp_path):
    """Train, save, merge: the merged tower holds what the step trained, and scores as the adapters did."""
    torch.manual_seed(0)
    model, image_row, _ = composite_model_and_rows("qwen3_vl")
    tokenizer = word_level_tokenizer(("hello", "world"))
    base_dir, adapter_dir, merged_dir = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.float32).train()
    trainer = _single_rank_trainer(base, image_row, tmp_path / "run", use_reentrant=True, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    trainer.model.eval()
    with torch.no_grad():
        adapted = trainer.model(**image_row).logits
    merge_peft_adapter(adapter_dir=str(adapter_dir), output_dir=str(merged_dir), dtype=torch.float32)
    merged = AutoModelForImageTextToText.from_pretrained(merged_dir, dtype=torch.float32).eval()
    block = _checkpointed_tower_layers(model)[0]
    assert not all(
        torch.equal(param, model.get_submodule(block).get_parameter(name))
        for name, param in merged.get_submodule(block).named_parameters()
    )
    with torch.no_grad():
        # The adapter stores its factors at bf16.
        assert (merged(**image_row).logits - adapted).abs().max() < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
