#!/usr/bin/env python
"""A composite under the FSDP2 wrap, on real gloo, with one rank's batch holding an image and the others' text. Every
rank must enter each group's reduce with the same parameters, or the reduce-scatter faults or hangs; the tower gradient
must equal the image rank's single-process gradient over the world, and a frozen tower must stay out of it. Where the
reduces still could not pair, a batch holding no image is refused, an adapter on the tower included."""

import copy
import datetime
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from trl import SFTConfig

from halo.distributed import fsdp
from halo.distributed.fsdp import IdentityParamSet
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import apply_vision_options
from halo.entrypoints.script import run_trainer
from halo.models.introspection.module_tree import (
    backbone_with_layers,
    decoder_layers,
    embedding_module,
    media_module_names,
)
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.composite_rows import COMPOSITE_FAMILIES, composite_model_and_rows, text_row
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model
from tests.common.tiny_processors import TINY_PROCESSOR_MODEL_TYPES, tiny_processor

# Without the agreement a rank waits in a reduce-scatter its peer entered with another list, until this ends it.
PG_TIMEOUT = datetime.timedelta(seconds=60)
_GC_KINDS = {"off": None, "reentrant": True, "non_reentrant": False}
_TEXT_FAMILIES = {"qwen3": TINY_DENSE_FAMILIES["qwen3"], "qwen3_moe": TINY_MOE_FAMILIES["qwen3_moe"]}
_MESH_NAMES = {1: ("dp_shard",), 2: ("dp_replicate", "dp_shard")}
_NEMOTRON = "nemotron_h_omni"
_ACCELERATE_FSDP = {"ACCELERATE_USE_FSDP": "true"}
# Where an adapter lands: the tower, the decoder layers, the head.
_ADAPTER_SPOTS = {
    "tower_and_layers": ("tower", "layers"),
    "tower_and_head": ("tower", "head"),
    "layers": ("layers",),
}


def _model_and_rows(name: str):
    if name in _TEXT_FAMILIES:
        return tiny_family_model(_TEXT_FAMILIES[name]).train(), None, text_row()
    return composite_model_and_rows(name)


def _full(grad: torch.Tensor) -> torch.Tensor:
    return grad.full_tensor() if hasattr(grad, "full_tensor") else grad


def _apply_options(model, *, freeze: bool, peft: bool = False, is_vlm: bool = True, targets=None) -> None:
    apply_vision_options(
        type("Options", (), {"freeze_vision_tower": freeze, "min_pixels": None, "max_pixels": None}),
        model,
        type("ModelArgs", (), {"use_peft": peft, "lora_target_modules": targets}),
        None,
        is_vlm=is_vlm,
    )


def _linears(model, root: str) -> list[str]:
    return [
        f"{root}.{name}"
        for name, module in model.get_submodule(root).named_modules()
        if isinstance(module, torch.nn.Linear)
    ]


def _wrap_adapters(model, gc: str, spots: tuple[str, ...], *, embedding_trains: bool = False):
    """As a script does: the options, then the wrap the trainer takes. Checkpointing is on by the first forward."""
    layers = next(
        name for name, module in model.named_modules() if module is decoder_layers(backbone_with_layers(model))
    )
    named = {
        # Enough of the tower that an image row's gradient reaches an adapter on every family.
        "tower": _linears(model, media_module_names(model)[0])[:4],
        "layers": _linears(model, layers),
        "head": [name for name, _ in model.named_modules() if name.rpartition(".")[2] == "lm_head"],
    }
    targets = [target for spot in spots for target in named[spot]]
    _apply_options(model, freeze=False, peft=True, targets=targets)
    embedding = next(name for name, module in model.named_modules() if module is embedding_module(model, "input"))
    # A fixed draw of the adapters, on every rank and for the reference copy alike.
    torch.manual_seed(1)
    wrapped = get_peft_model(
        model, LoraConfig(r=2, target_modules=targets, modules_to_save=[embedding] if embedding_trains else None)
    )
    if _GC_KINDS[gc] is not None:
        wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": _GC_KINDS[gc]})
    return wrapped


def _adapter_grads(model) -> dict[str, torch.Tensor]:
    return {
        name.replace("_checkpoint_wrapped_module.", ""): _full(param.grad)
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def _worker(rank: int, out_dir: str, case: dict) -> None:
    torch.manual_seed(0)
    model, image_row, text_row = _model_and_rows(case["family"])
    image_rows = set(case["image_ranks"])
    reference = copy.deepcopy(model)
    if image_rows:
        reference(**image_row).loss.backward()
    single_rank = {name: param.grad.clone() for name, param in reference.named_parameters() if param.grad is not None}
    del reference

    if case["freeze_outside_layers"]:
        kept = {id(param) for param in decoder_layers(backbone_with_layers(model)).parameters()}
        for param in model.parameters():
            param.requires_grad = id(param) in kept
    if case["freeze_embedding"]:
        embedding_module(model, "input").requires_grad_(False)
    _apply_options(model, freeze=case["freeze"])
    if _GC_KINDS[case["gc"]] is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": _GC_KINDS[case["gc"]]})
    mesh_shape = tuple(case["mesh"])
    mesh = init_device_mesh("cpu", mesh_shape, mesh_dim_names=_MESH_NAMES[len(mesh_shape)])
    # A reduce dtype moves a finished microstep's gradient to the accumulator, so only that says the tower has one.
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float64) if case["window"] else MixedPrecisionPolicy()
    towers = tuple(f"{tower}." for tower in media_module_names(model))
    fsdp.apply_fsdp2_per_layer(model, mesh, policy, case["reshard"], IdentityParamSet())

    agreements = 0
    real_all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *args, **kwargs):
        nonlocal agreements
        agreements += tensor.dtype == torch.uint8
        return real_all_reduce(tensor, *args, **kwargs)

    with patch.object(dist, "all_reduce", counting_all_reduce):
        for step in range(case["steps"]):
            last = step == case["steps"] - 1
            # An accumulation window: one microstep holds the image, and only the last reduces.
            if case["window"]:
                model.set_requires_gradient_sync(last)
            # Every microstep reduces, and only the last reshards.
            if case["hold_gathered"]:
                for module in fsdp.fsdp2_modules(model):
                    module.set_reshard_after_backward(last, recurse=False)
            holds_image = rank in image_rows and (not case["window"] or step == case["image_step"])
            model(**(image_row if holds_image else text_row)).loss.backward()
            fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(model))

    world = dist.get_world_size()
    tower = [(name, param) for name, param in model.named_parameters() if towers and name.startswith(towers)]
    unreached = sum(name.replace("_checkpoint_wrapped_module.", "") not in single_rank for name, _ in tower)
    worst, compared, absent = 0.0, 0, 0
    for name, param in tower:
        if param.grad is None:
            absent += 1
            continue
        key = name.replace("_checkpoint_wrapped_module.", "")
        image_steps = 1 if case["window"] else case["steps"]
        expected = single_rank[key] * image_steps * len(image_rows) / world
        worst = max(worst, float((_full(param.grad) - expected).abs().max() / (expected.abs().max() + 1e-9)))
        compared += 1
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(
            {
                "tower": len(tower),
                "absent": absent,
                "unreached": unreached,
                "compared": compared,
                "worst": worst,
                "agreements": agreements,
            },
            handle,
        )


def _tower_worker(rank: int, out_dir: str) -> None:
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=_MESH_NAMES[1])
    verdicts = {}
    for name, family in sorted(COMPOSITE_FAMILIES.items()):
        torch.manual_seed(0)
        model = tiny_family_model(family)
        towers = tuple(f"{tower}." for tower in media_module_names(model))
        fsdp.apply_fsdp2_per_layer(model, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
        # The wrap rebinds each name to the group's sharded parameter.
        tower_params = {id(param) for param_name, param in model.named_parameters() if param_name.startswith(towers)}
        held = [
            "post_backward" in vars(group)
            for group in fsdp.fsdp2_param_groups(model)
            for fsdp_param in group.fsdp_params
            if id(fsdp_param.sharded_param) in tower_params
        ]
        verdicts[name] = {"tower": len(tower_params), "held": len(held), "agreeing": sum(held)}
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(verdicts, handle)


def _refusal_worker(rank: int, out_dir: str, family: str, freeze_embedding: bool) -> None:
    """Every rank takes the text row, so the refused forward strands no peer in the unshard before it."""
    torch.manual_seed(0)
    model, image_row, text_row = _model_and_rows(family)
    if freeze_embedding:
        embedding_module(model, "input").requires_grad_(False)
    _apply_options(model, freeze=False)
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=_MESH_NAMES[1])
    fsdp.apply_fsdp2_per_layer(model, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
    refused = ""
    try:
        model(**text_row)
    except ValueError as error:
        refused = str(error)
    # The eval forward a run takes on text rows reduces nothing.
    model.eval()
    with torch.no_grad():
        model(**text_row)
    model.train()
    # Each rank holding an image pairs its reduce.
    model(**image_row).loss.backward()
    fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(model))
    towers = tuple(f"{tower}." for tower in media_module_names(model))
    reached = [name for name, param in model.named_parameters() if name.startswith(towers) and param.grad is not None]
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump({"refused": refused, "reached": len(reached)}, handle)


def _adapter_refusal_worker(
    rank: int, out_dir: str, family: str, gc: str, spots: tuple[str, ...], reshard: bool = False
) -> None:
    """Every rank takes the text row, so the refused forward strands no peer in the reduce past it."""
    torch.manual_seed(0)
    model, _, text_row = _model_and_rows(family)
    wrapped = _wrap_adapters(model, gc, spots)
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=_MESH_NAMES[1])
    fsdp.apply_fsdp2_per_layer(wrapped, mesh, MixedPrecisionPolicy(), reshard, IdentityParamSet())
    data_setup.carry_image_refusal_above_adapters(wrapped)
    refused = ""
    try:
        wrapped(**text_row)
    except ValueError as error:
        refused = str(error)
    # The eval forward a run takes on text rows reduces nothing.
    wrapped.eval()
    with torch.no_grad():
        wrapped(**text_row)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump({"refused": refused}, handle)


def _adapter_worker(rank: int, out_dir: str, case: dict) -> None:
    torch.manual_seed(0)
    model, image_row, text_row = _model_and_rows(case["family"])
    wrapped = _wrap_adapters(
        model, case["gc"], _ADAPTER_SPOTS[case["spots"]], embedding_trains=case["embedding_trains"]
    )
    reference = copy.deepcopy(wrapped)
    reference(**image_row).loss.backward()
    single_rank = _adapter_grads(reference)
    del reference
    towers = tuple(f"base_model.model.{tower}." for tower in media_module_names(model))
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=_MESH_NAMES[1])
    fsdp.apply_fsdp2_per_layer(wrapped, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
    data_setup.carry_image_refusal_above_adapters(wrapped)
    image_rows = set(case["image_ranks"])
    wrapped(**(image_row if rank in image_rows else text_row)).loss.backward()
    fsdp.complete_fsdp2_post_backward(fsdp.fsdp2_param_groups(wrapped))
    world = dist.get_world_size()
    # A tower adapter only an image row reaches: the image ranks' share of its single-rank gradient.
    run_grads = _adapter_grads(wrapped)
    tower_grads = [name for name in single_rank if name.startswith(towers)]
    worst = 0.0
    for name in set(tower_grads) & set(run_grads):
        expected = single_rank[name] * len(image_rows) / world
        worst = max(worst, float((run_grads[name] - expected).abs().max() / (expected.abs().max() + 1e-9)))
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(
            {
                "tower_grads": len(tower_grads),
                "lost": sorted(set(tower_grads) - set(run_grads)),
                "reached": len(single_rank),
                "worst": worst,
            },
            handle,
        )


def _run(tmp_path, world: int, **case) -> list[dict]:
    run_gloo_ranks(_worker, world, str(tmp_path), case, pg_timeout=PG_TIMEOUT)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(world)]


_CASE = {
    "family": "qwen3_vl",
    "image_ranks": [0],
    "freeze": False,
    "gc": "off",
    "mesh": [2],
    "steps": 1,
    "window": False,
    "image_step": 0,
    "reshard": False,
    "hold_gathered": False,
    "freeze_outside_layers": False,
    "freeze_embedding": False,
}


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize("gc", list(_GC_KINDS))
def test_a_mixed_step_reduces_the_tower_to_the_image_ranks_gradient_over_the_world(tmp_path, gc, reshard):
    for result in _run(tmp_path, 2, **{**_CASE, "gc": gc, "reshard": reshard}):
        assert result["tower"] and result["absent"] == 0
        assert result["compared"] == result["tower"]
        assert result["worst"] < 1e-5, result


@pytest.mark.parametrize("family", TINY_PROCESSOR_MODEL_TYPES)
def test_each_tree_shape_agrees_on_its_own_processors_row(tmp_path, family):
    """Where the tower sits differs by family: under the embedding backbone, or on an unwrapped composite's root."""
    for result in _run(tmp_path, 2, **{**_CASE, "family": family}):
        assert result["tower"] and result["absent"] == result["unreached"] < result["tower"], result
        assert result["worst"] < 1e-5, result


@pytest.mark.parametrize("mesh", [[2], [2, 2]], ids=["fsdp", "hsdp"])
@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize("gc", list(_GC_KINDS))
def test_nemotron_h_tower_takes_the_mixed_path(tmp_path, gc, reshard, mesh):
    """A composite holding ``language_model`` itself, so its embedding and tower share the root group."""
    world = 2 if mesh == [2] else 4
    results = _run(tmp_path, world, **{**_CASE, "family": _NEMOTRON, "gc": gc, "reshard": reshard, "mesh": mesh})
    for result in results:
        assert result["absent"] == result["unreached"] < result["tower"], result
        assert result["worst"] < 1e-5, result


def test_a_moe_composite_agrees_under_reentrant_checkpointing(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "family": "qwen3_5_moe", "gc": "reentrant"}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


def test_every_composites_tower_sits_in_a_group_that_agrees(tmp_path):
    """A tower in a group left out would reduce a shorter list on a text rank, whichever family it is."""
    run_gloo_ranks(_tower_worker, 2, str(tmp_path), pg_timeout=PG_TIMEOUT)
    verdicts = json.loads((tmp_path / "rank0.json").read_text())
    assert set(verdicts) == set(COMPOSITE_FAMILIES)
    for name, verdict in verdicts.items():
        assert verdict["tower"] and verdict["held"] == verdict["agreeing"] == verdict["tower"], (name, verdict)


def test_mixed_steps_accumulate_across_backwards(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "steps": 2}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


@pytest.mark.parametrize("image_step", [0, 1], ids=["first", "last"])
def test_an_image_on_one_microstep_of_a_window_counts_at_its_reduce(tmp_path, image_step):
    """On the first, the reducing microstep holds text on every rank, so only the accumulator says the tower has a
    gradient. On the last, the tower's gradient has not reached the reduce dtype its neighbours' already hold."""
    for result in _run(tmp_path, 2, **{**_CASE, "steps": 2, "window": True, "image_step": image_step}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


def test_mixed_steps_held_gathered_reduce_each_backward(tmp_path):
    """``fsdp_reshard_after_backward: false``: a group stays gathered past its reduce, so the backward's completion
    visits it again on every rank."""
    for result in _run(tmp_path, 2, **{**_CASE, "steps": 2, "hold_gathered": True}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


@pytest.mark.parametrize("reshard", [False, True], ids=["zero2", "zero3"])
def test_an_hsdp_mesh_agrees_across_both_dims(tmp_path, reshard):
    """A lone image rank: its row and its column each reach only part of the world."""
    for result in _run(tmp_path, 4, **{**_CASE, "mesh": [2, 2], "reshard": reshard}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


@pytest.mark.parametrize(
    ("family", "world", "mesh", "extra", "agreements"),
    [
        ("qwen3", 2, [2], {}, 0),
        ("qwen3_moe", 2, [2], {}, 0),
        ("qwen3_vl", 2, [2], {}, 2),
        ("qwen3_vl", 4, [2, 2], {}, 4),
        ("qwen3_vl", 2, [2], {"steps": 2, "window": True}, 2),
        ("qwen3_vl", 2, [2], {"freeze_outside_layers": True}, 0),
    ],
    ids=["dense", "moe", "composite", "composite_hsdp", "window", "nothing_trainable"],
)
def test_only_a_composite_pays_for_the_agreement(tmp_path, family, world, mesh, extra, agreements):
    """One all-reduce per group and reducing backward on a composite's two groups, one more per group under HSDP; none
    on a group with nothing trainable, as an adapter's embedding and root are."""
    for result in _run(tmp_path, world, **{**_CASE, "family": family, "image_ranks": [], "mesh": mesh, **extra}):
        assert result["agreements"] == agreements, result


def test_an_all_text_run_never_writes_its_tower(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "image_ranks": []}):
        assert result["tower"] and result["absent"] == result["tower"], result


@pytest.mark.parametrize("family", ["qwen3_vl", _NEMOTRON])
def test_a_frozen_tower_stays_out_of_the_reduce(tmp_path, family):
    for result in _run(tmp_path, 2, **{**_CASE, "family": family, "freeze": True}):
        assert result["tower"] and result["absent"] == result["tower"], result


def test_all_image_rows_reduce_as_before(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "image_ranks": [0, 1]}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


@pytest.mark.parametrize("family", ["qwen3_vl", _NEMOTRON])
@pytest.mark.parametrize("gc", ["reentrant", "non_reentrant"])
def test_a_frozen_embedding_pairs_under_checkpointing(tmp_path, family, gc):
    """Checkpointing's input-grad hook gives the embedding's output a gradient on a text row too."""
    for result in _run(tmp_path, 2, **{**_CASE, "family": family, "gc": gc, "freeze_embedding": True}):
        assert result["absent"] == result["unreached"] < result["tower"], result
        assert result["worst"] < 1e-5, result


def test_every_composite_checkpoints_through_the_input_grad_hook():
    """What the refusal reads a checkpointed frozen embedding by: a composite the hook skips would hang unrefused."""
    for name, family in sorted(COMPOSITE_FAMILIES.items()):
        model = tiny_family_model(family)
        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing and model._require_grads_hooks, name
        assert next(embedding_module(model, "input").parameters(), None) is not None, name


@pytest.mark.parametrize(
    ("family", "freeze_embedding", "env", "named"),
    [
        ("qwen3_vl", True, {}, "frozen input embedding"),
        (_NEMOTRON, True, {}, "frozen input embedding"),
        ("qwen3_vl", False, _ACCELERATE_FSDP, "torchrun"),
        (_NEMOTRON, False, _ACCELERATE_FSDP, "torchrun"),
    ],
    ids=["frozen_embedding", "nemotron_frozen_embedding", "accelerate", "nemotron_accelerate"],
)
def test_a_batch_holding_no_image_is_refused_where_the_reduces_cannot_pair(
    tmp_path, family, freeze_embedding, env, named
):
    run_gloo_ranks(_refusal_worker, 2, str(tmp_path), family, freeze_embedding, pg_timeout=PG_TIMEOUT, env=env)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert named in result["refused"] and "freeze_vision_tower" in result["refused"], result
        assert result["reached"], "the image rows after the refusal must reach the tower"


def _no_layer_reason_worker(rank: int, out_dir: str, reshard: bool) -> None:
    """A fine-tune training the tower and head with the layers and embedding frozen, its text-row verdict by ZeRO level."""
    torch.manual_seed(0)
    model, _, _ = _model_and_rows("qwen3_vl")
    towers = tuple(f"{tower}." for tower in media_module_names(model))
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith((*towers, "lm_head"))
    _apply_options(model, freeze=False)
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=_MESH_NAMES[1])
    fsdp.apply_fsdp2_per_layer(model, mesh, MixedPrecisionPolicy(), reshard, IdentityParamSet())
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump({"reason": data_setup._why_a_batch_holding_no_image_cannot_pair(model)}, handle)


@pytest.mark.parametrize(("reshard", "refused"), [(False, False), (True, True)], ids=["zero2", "zero3"])
def test_a_fine_tune_with_no_layer_training_pairs_under_zero2_alone(tmp_path, reshard, refused):
    run_gloo_ranks(_no_layer_reason_worker, 2, str(tmp_path), reshard, pg_timeout=PG_TIMEOUT)
    for rank in range(2):
        assert bool(json.loads((tmp_path / f"rank{rank}.json").read_text())["reason"]) is refused


@pytest.mark.parametrize(
    "options",
    [{"freeze": True}, {"freeze": False, "peft": True, "targets": []}, {"freeze": False, "is_vlm": False}],
    ids=["frozen_tower", "expert_adapters_alone", "text_run"],
)
def test_a_run_with_no_trained_tower_or_no_images_registers_no_refusal(options):
    model, _, _ = _model_and_rows(_NEMOTRON)
    # An expert-only adapter run has no wrap, and its base is frozen by now.
    model.requires_grad_(not options.get("peft", False))
    _apply_options(model, **options)
    assert data_setup._refuse_a_batch_holding_no_image not in model._forward_pre_hooks.values()


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5_moe", _NEMOTRON])
def test_an_adapter_on_the_tower_is_refused_where_its_reduces_cannot_pair(tmp_path, family):
    """The frozen embedding an adapter run always has, with no checkpointing to give its output a gradient."""
    run_gloo_ranks(
        _adapter_refusal_worker,
        2,
        str(tmp_path),
        family,
        "off",
        _ADAPTER_SPOTS["tower_and_layers"],
        pg_timeout=PG_TIMEOUT,
    )
    for rank in range(2):
        refused = json.loads((tmp_path / f"rank{rank}.json").read_text())["refused"]
        assert "frozen input embedding" in refused and "gradient checkpointing" in refused, refused
        assert "lora_target_modules" in refused and "lora_" in refused.split("reached ")[1], refused
        # An adapter run refuses this knob, and trains adapters already.
        assert "freeze_vision_tower" not in refused and "train adapters" not in refused, refused


@pytest.mark.parametrize("family", ["qwen3_vl", _NEMOTRON])
def test_a_tower_with_no_layer_adapter_is_refused_under_zero3(tmp_path, family):
    """ZeRO-2 pairs this, as only end-of-backward groups reduce. Under ZeRO-3 the tower's group re-gathers on an image
    rank alone, and the step hangs."""
    run_gloo_ranks(
        _adapter_refusal_worker,
        2,
        str(tmp_path),
        family,
        "off",
        _ADAPTER_SPOTS["tower_and_head"],
        True,
        pg_timeout=PG_TIMEOUT,
    )
    for rank in range(2):
        assert "frozen input embedding" in json.loads((tmp_path / f"rank{rank}.json").read_text())["refused"]


def test_an_adapter_on_the_tower_is_refused_under_accelerate_with_checkpointing_on(tmp_path):
    run_gloo_ranks(
        _adapter_refusal_worker,
        2,
        str(tmp_path),
        "qwen3_vl",
        "reentrant",
        _ADAPTER_SPOTS["tower_and_layers"],
        pg_timeout=PG_TIMEOUT,
        env=_ACCELERATE_FSDP,
    )
    for rank in range(2):
        assert "torchrun" in json.loads((tmp_path / f"rank{rank}.json").read_text())["refused"]


@pytest.mark.parametrize(
    ("gc", "spots"),
    [("off", "layers"), ("off", "tower_and_head"), ("reentrant", "tower_and_layers")],
    ids=["adapters_off_the_tower", "no_layer_adapter", "checkpointed"],
)
def test_an_adapter_run_whose_reduces_pair_takes_no_refusal(tmp_path, gc, spots):
    """A text forward on every rank, which each of these reduces the same way on its peers."""
    run_gloo_ranks(
        _adapter_refusal_worker, 2, str(tmp_path), "qwen3_vl", gc, _ADAPTER_SPOTS[spots], pg_timeout=PG_TIMEOUT
    )
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text())["refused"] == ""


_ADAPTER_CASE = {
    "family": "qwen3_vl",
    "gc": "off",
    "spots": "tower_and_layers",
    "image_ranks": [0],
    "embedding_trains": False,
}


@pytest.mark.parametrize(
    "case",
    [
        {"gc": "reentrant"},
        {"family": _NEMOTRON, "gc": "reentrant"},
        {"family": _NEMOTRON, "gc": "non_reentrant"},
        {"family": "qwen3_5_moe", "gc": "reentrant"},
        {"embedding_trains": True},
        {"image_ranks": [0, 1]},
        {"spots": "layers"},
        {"spots": "tower_and_head"},
    ],
    ids=[
        "reentrant",
        "nemotron_reentrant",
        "nemotron_non_reentrant",
        "moe_reentrant",
        "trained_embedding",
        "an_image_on_every_rank",
        "off_the_tower",
        "no_layer_adapter",
    ],
)
def test_an_adapter_run_whose_reduces_pair_finishes_a_mixed_step(tmp_path, case):
    """A tower adapter's gradient, where one reaches it, is the image ranks' share of its own single-rank gradient."""
    run_gloo_ranks(_adapter_worker, 2, str(tmp_path), {**_ADAPTER_CASE, **case}, pg_timeout=PG_TIMEOUT)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["reached"] and not result["lost"] and result["worst"] < 1e-5, result


@pytest.mark.parametrize("gc", ["reentrant", "non_reentrant"])
def test_a_tower_adapter_takes_its_share_of_the_gradient_under_checkpointing(tmp_path, gc):
    run_gloo_ranks(_adapter_worker, 2, str(tmp_path), {**_ADAPTER_CASE, "gc": gc}, pg_timeout=PG_TIMEOUT)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["tower_grads"] == 8 and not result["lost"] and result["worst"] < 1e-5, result


def test_a_training_step_through_the_trainers_own_wrap_takes_the_refusal(tmp_path):
    """TRL wraps inside the trainer's constructor, so only ``run_trainer`` holds the model the loop calls. One process
    stands in for the data-parallel axis, whose mesh is all the refusal reads of it."""
    torch.manual_seed(0)
    model, _, text_row = _model_and_rows(_NEMOTRON)
    targets = _linears(model, media_module_names(model)[0])[:4] + ["q_proj"]
    _apply_options(model, freeze=False, peft=True, targets=targets)
    tokenizer = tiny_processor(_NEMOTRON).tokenizer
    ids = text_row["input_ids"][0].tolist()
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            max_length=len(ids),
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
        train_dataset=Dataset.from_dict({"input_ids": [ids] * 2, "labels": [ids] * 2}),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=2, target_modules=targets, task_type="CAUSAL_LM"),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        moe_balancing="none",
    )
    assert isinstance(trainer.model, PeftModel)
    context = SimpleNamespace(
        parallelism_config=trainer.parallelism_config, mode_suffix="test", resume_checkpoint=None
    )
    wide_mesh = SimpleNamespace(mesh_info=SimpleNamespace(mesh=SimpleNamespace(size=lambda: 2)))
    with (
        patch.object(data_setup, "fsdp2_param_groups", return_value=[wide_mesh]),
        pytest.raises(ValueError, match="an adapter reached"),
    ):
        run_trainer(trainer, context, method_name="SFT")


def test_one_rank_takes_no_refusal_under_an_adapter(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        model, _, text_row = _model_and_rows("qwen3_vl")
        wrapped = _wrap_adapters(model, "off", _ADAPTER_SPOTS["tower_and_layers"])
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=_MESH_NAMES[1])
        fsdp.apply_fsdp2_per_layer(wrapped, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
        data_setup.carry_image_refusal_above_adapters(wrapped)
        assert data_setup._refuse_a_batch_holding_no_image in wrapped._forward_pre_hooks.values()
        assert wrapped(**text_row).loss.isfinite()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("freeze_embedding", [False, True], ids=["trained_embedding", "frozen_embedding"])
def test_one_rank_reduces_nothing_and_keeps_the_row(tmp_path, monkeypatch, freeze_embedding):
    """A data-parallel axis of one, where every peer of a rank holds its batch."""
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        model, _, text_row = _model_and_rows(_NEMOTRON)
        if freeze_embedding:
            embedding_module(model, "input").requires_grad_(False)
        _apply_options(model, freeze=False)
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=_MESH_NAMES[1])
        fsdp.apply_fsdp2_per_layer(model, mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
        assert data_setup._refuse_a_batch_holding_no_image in model._forward_pre_hooks.values()
        assert model(**text_row).loss.isfinite()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
