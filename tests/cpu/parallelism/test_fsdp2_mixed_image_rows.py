#!/usr/bin/env python
"""A composite under the FSDP2 wrap, on real gloo, with one rank's batch holding an image and the others' text. Every
rank must enter each group's reduce with the same parameters, or the reduce-scatter faults or hangs; the tower gradient
must equal the image rank's single-process gradient over the world, and a frozen tower must stay out of it."""

import copy
import datetime
import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from halo.distributed import fsdp
from halo.distributed.fsdp import IdentityParamSet
from halo.entrypoints.data_setup import apply_vision_options
from halo.models.module_tree import backbone_with_layers, decoder_layers, media_module_names
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_LLAMA4_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_QWEN3_VL_FAMILY,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    tiny_family_model,
)
from tests.common.tiny_processors import TINY_PROCESSOR_MODEL_TYPES, tiny_image_batch, tiny_processor

# Without the agreement a rank waits in a reduce-scatter its peer entered with another list, until this ends it.
PG_TIMEOUT = datetime.timedelta(seconds=60)
_IMAGE_TOKEN, _VISION_START = 5, 4
_TEXT_ROW = "Describe the image . A square of noise ."
_GC_KINDS = {"off": None, "reentrant": True, "non_reentrant": False}
_FAMILIES = {"qwen3_vl": TINY_QWEN3_VL_FAMILY, **TINY_MOE_VLM_FAMILIES, **TINY_UNWRAPPED_COMPOSITE_FAMILIES}
_TEXT_FAMILIES = {"qwen3": TINY_DENSE_FAMILIES["qwen3"], "qwen3_moe": TINY_MOE_FAMILIES["qwen3_moe"]}
_COMPOSITES = {
    **_FAMILIES,
    "glm5_next": TINY_MOE_FAMILIES["glm5_next"],
    "llama4": TINY_LLAMA4_FAMILY,
    "step3p7": TINY_MOE_FAMILIES["step3p7"],
}
_MESH_NAMES = {1: ("dp_shard",), 2: ("dp_replicate", "dp_shard")}


def _image_row(config) -> dict[str, torch.Tensor]:
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grid = (1, 2 * merge, 2 * merge)
    patches = grid[0] * grid[1] * grid[2]
    ids = torch.tensor([[6, _VISION_START] + [_IMAGE_TOKEN] * (patches // merge**2) + [8, 9]])
    patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "pixel_values": torch.randn(patches, patch_width, generator=torch.Generator().manual_seed(1)),
        "image_grid_thw": torch.tensor([grid]),
        "mm_token_type_ids": (ids == _IMAGE_TOKEN).long(),
    }


def _text_row(ids: list[int] | None = None) -> dict[str, torch.Tensor]:
    ids = torch.tensor([ids or [6, 7, 8, 9, 10, 11, 12, 13]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _model_and_rows(name: str):
    """The family's own processor builds its rows where it has a local one, so its tower really reads them."""
    if name in _TEXT_FAMILIES:
        return tiny_family_model(_TEXT_FAMILIES[name]).train(), None, _text_row()
    if name not in TINY_PROCESSOR_MODEL_TYPES:
        model = tiny_family_model(_FAMILIES[name]).train()
        model.config.image_token_id, model.config.vision_start_token_id = _IMAGE_TOKEN, _VISION_START
        return model, _image_row(model.config), _text_row()
    processor = tiny_processor(name)
    model = tiny_family_model(_FAMILIES[name], processor.tokenizer).train()
    model.config.image_token_id = processor.image_token_id
    # The composites that read the id once, at construction.
    if hasattr(model, "image_token_id"):
        model.image_token_id = processor.image_token_id
    return model, tiny_image_batch(processor), _text_row(processor.tokenizer(_TEXT_ROW)["input_ids"])


def _full(grad: torch.Tensor) -> torch.Tensor:
    return grad.full_tensor() if hasattr(grad, "full_tensor") else grad


def _worker(rank: int, out_dir: str, case: dict) -> None:
    torch.manual_seed(0)
    model, image_row, text_row = _model_and_rows(case["family"])
    image_rows = set(case["image_ranks"])
    reference = copy.deepcopy(model)
    if image_rows:
        reference(**image_row).loss.backward()
    single_rank = {name: param.grad.clone() for name, param in reference.named_parameters() if param.grad is not None}
    del reference

    if case["freeze"]:
        apply_vision_options(
            type("Options", (), {"freeze_vision_tower": True, "min_pixels": None, "max_pixels": None}),
            model,
            type("ModelArgs", (), {"use_peft": False}),
            None,
            is_vlm=True,
        )
    if case["freeze_outside_layers"]:
        kept = {id(param) for param in decoder_layers(backbone_with_layers(model)).parameters()}
        for param in model.parameters():
            param.requires_grad = id(param) in kept
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
    for name, family in sorted(_COMPOSITES.items()):
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


def test_a_moe_composite_agrees_under_reentrant_checkpointing(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "family": "qwen3_5_moe", "gc": "reentrant"}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


def test_every_composites_tower_sits_in_a_group_that_agrees(tmp_path):
    """A tower in a group left out would reduce a shorter list on a text rank, whichever family it is."""
    run_gloo_ranks(_tower_worker, 2, str(tmp_path), pg_timeout=PG_TIMEOUT)
    verdicts = json.loads((tmp_path / "rank0.json").read_text())
    assert set(verdicts) == set(_COMPOSITES)
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


def test_a_frozen_tower_stays_out_of_the_reduce(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "freeze": True}):
        assert result["tower"] and result["absent"] == result["tower"], result


def test_all_image_rows_reduce_as_before(tmp_path):
    for result in _run(tmp_path, 2, **{**_CASE, "image_ranks": [0, 1]}):
        assert result["absent"] == 0 and result["worst"] < 1e-5, result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
