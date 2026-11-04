#!/usr/bin/env python
"""A lazy load of Nemotron-H, the only load its size allows: the fuser stacks a gateless expert's ``up_proj`` alone, each
rank takes its own experts on both routes, and what the release does not store is drawn by the tower that owns it."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

import halo.model_loading.lazy_safetensors.materialization as materialization
import halo.model_loading.parallel.ep as ep_lazy_loader
from halo.checkpoint.tools.lowp_export import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, _should_quantize
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.fp32_masters import restore_fp32_master_parameters
from halo.model_loading.parallel.lazy_plan import ExpertKeyLayout, is_lazy_loadable
from halo.models.nemotron_h import apply_nemotron_h_patches
from tests.common.models import TINY_NEMOTRON_H_CONFIG, TINY_RADIO_VISION_CONFIG
from tests.common.nemotron_h import BIAS_SLOT, write_tiny_hub_checkpoint, write_tiny_lightning_checkpoint
from tests.common.numerics import tensors_equal_at_narrower_dtype
from tests.common.parallelism import single_process_ep_config, stubbed_rank_ep_config
from tests.common.tiny_processors import tiny_image_batch, tiny_processor

RUN_DTYPE = torch.bfloat16
E = TINY_NEMOTRON_H_CONFIG["n_routed_experts"]
_BANKS = ("experts.up_proj", "experts.down_proj")
_LAYER_SCALE = "lambda1"
_ROUTES = {
    "composite": (AutoModelForImageTextToText, lambda config: config),
    "text": (AutoModelForCausalLM, lambda config: config.text_config),
}
_SENTINEL = -7.0


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_nemotron_h_patches()


@pytest.fixture(scope="module")
def hub_checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    directory = tmp_path_factory.mktemp("nemotron_h_hub")
    tensors = write_tiny_hub_checkpoint(directory, weights_dtype=RUN_DTYPE)
    return str(directory), tensors


def _eager(path: str, route: str) -> torch.nn.Module:
    model_class, narrow = _ROUTES[route]
    return model_class.from_pretrained(path, config=narrow(AutoConfig.from_pretrained(path)), dtype=RUN_DTYPE)


def _lazy(path: str, route: str, ep_config) -> torch.nn.Module:
    model_class, narrow = _ROUTES[route]
    return load_ep_model_lazy(
        path,
        ep_config,
        narrow(AutoConfig.from_pretrained(path)),
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=model_class,
    )


def _no_ep_layers(monkeypatch) -> None:
    """EP patching needs DeepEP at ``ep_size > 1``, and the banks are compared as the loader leaves them."""
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)


@pytest.fixture
def sentinel_layer_scale(monkeypatch):
    """Empty storage reads as a value no init writes, so a draw that never ran shows."""
    empty = materialization.materialize_empty_like

    def with_sentinel(model, key, device, dtype):
        empty(model, key, device, dtype)
        model.get_submodule(key.rpartition(".")[0]).get_buffer(_LAYER_SCALE).fill_(_SENTINEL)

    monkeypatch.setattr(materialization, "materialize_empty_like", with_sentinel)


def _rewrite(path: str, directory: Path, edit) -> str:
    """A copy of the checkpoint with its tensors passed through ``edit``."""
    directory.mkdir()
    shutil.copy2(Path(path) / "config.json", directory / "config.json")
    save_file(edit(load_file(Path(path) / "model.safetensors")), directory / "model.safetensors", {"format": "pt"})
    return str(directory)


def test_the_gate_admits_the_release_layout(hub_checkpoint):
    path, tensors = hub_checkpoint
    assert is_lazy_loadable(path) is True
    assert ExpertKeyLayout.detect(dict.fromkeys(tensors, "model.safetensors")) is ExpertKeyLayout.PER_EXPERT


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize(("ep_size", "rank"), [(1, 0), (2, 0), (2, 1)], ids=["ep1", "ep2_rank0", "ep2_rank1"])
def test_each_rank_loads_what_from_pretrained_does_with_its_own_experts(
    hub_checkpoint, monkeypatch, route, ep_size, rank
):
    path, _ = hub_checkpoint
    _no_ep_layers(monkeypatch)
    reference = _eager(path, route).state_dict()
    lazy = _lazy(path, route, stubbed_rank_ep_config(ep_size, ep_size, rank, E)).state_dict()

    assert lazy.keys() == reference.keys()
    local = slice(rank * E // ep_size, (rank + 1) * E // ep_size)
    banks = {key for key in reference if key.endswith(_BANKS)}
    assert len(banks) == 2 * TINY_NEMOTRON_H_CONFIG["layers_block_type"].count("moe")
    stale = [
        key
        for key, tensor in reference.items()
        if not tensors_equal_at_narrower_dtype(lazy[key], tensor[local] if key in banks else tensor)
    ]
    assert not stale, f"{len(stale)} tensors differ from from_pretrained: {stale[:5]}"


@pytest.mark.parametrize(("ep_size", "rank"), [(1, 0), (2, 0), (2, 1)], ids=["ep1", "ep2_rank0", "ep2_rank1"])
def test_each_rank_of_a_text_only_release_loads_what_from_pretrained_does(tmp_path, monkeypatch, ep_size, rank):
    """Lightning's layout: its tower at the root, no latent, and a multi-token head on disk the class does not build."""
    stored = write_tiny_lightning_checkpoint(tmp_path)
    path = str(tmp_path)
    assert is_lazy_loadable(path) is True
    assert ExpertKeyLayout.detect(dict.fromkeys(stored, "model.safetensors")) is ExpertKeyLayout.PER_EXPERT
    _no_ep_layers(monkeypatch)
    reference = AutoModelForCausalLM.from_pretrained(path, dtype=RUN_DTYPE).state_dict()
    lazy = load_ep_model_lazy(
        path,
        stubbed_rank_ep_config(ep_size, ep_size, rank, E),
        AutoConfig.from_pretrained(path),
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
    ).state_dict()

    assert lazy.keys() == reference.keys()
    assert not any("mtp" in key for key in lazy)
    local = slice(rank * E // ep_size, (rank + 1) * E // ep_size)
    banks = {key for key in reference if key.endswith(_BANKS)}
    assert len(banks) == 2 * TINY_NEMOTRON_H_CONFIG["layers_block_type"].count("moe")
    stale = [
        key
        for key, tensor in reference.items()
        if not tensors_equal_at_narrower_dtype(lazy[key], tensor[local] if key in banks else tensor)
    ]
    assert not stale, f"{len(stale)} tensors differ from from_pretrained: {stale[:5]}"
    assert all(value.dtype == torch.float32 for key, value in lazy.items() if key.endswith(BIAS_SLOT))


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_the_release_bias_keeps_its_stored_fp32(hub_checkpoint, monkeypatch, route):
    """A bf16 cast rounds it by a step wider than its spread, and selection follows."""
    path, tensors = hub_checkpoint
    _no_ep_layers(monkeypatch)
    state = _lazy(path, route, stubbed_rank_ep_config(2, 2, 0, E)).state_dict()
    stored = {key: value for key, value in tensors.items() if key.endswith(BIAS_SLOT)}
    assert stored and all(value.dtype == torch.float32 for value in stored.values()), "premise: stored in fp32"
    assert all(not torch.equal(value, value.to(RUN_DTYPE).float()) for value in stored.values()), (
        "premise: a bf16 cast changes it"
    )
    live = [value for key, value in state.items() if key.endswith(BIAS_SLOT)]
    assert len(live) == len(stored)
    for value in live:
        assert value.dtype == torch.float32
        assert any(torch.equal(value, written) for written in stored.values())


def test_the_loaded_banks_take_the_configured_fp32_experts(hub_checkpoint, monkeypatch):
    """Their masters go through the fuser, which reads the keep list by the bank's own name."""
    path, _ = hub_checkpoint
    _no_ep_layers(monkeypatch)
    state = _lazy(path, "text", stubbed_rank_ep_config(2, 2, 0, E, fp32_experts=True)).state_dict()
    banks = {key: value for key, value in state.items() if key.endswith(_BANKS)}
    assert banks and all(value.dtype == torch.float32 for value in banks.values())
    shared = next(key for key in state if ".shared_experts.up_proj." in key)
    assert state[shared].dtype == RUN_DTYPE, "shared experts stay at the run dtype"


def test_an_eager_load_rereads_the_configured_experts_at_fp32(tmp_path):
    """The reread takes its banks through the fuser too, by the names the eager load recorded."""
    tensors = write_tiny_hub_checkpoint(tmp_path)
    model = _eager(str(tmp_path), "text")
    banks = {key: value for key, value in model.state_dict(keep_vars=True).items() if key.endswith(_BANKS)}
    assert banks and all(value.dtype == RUN_DTYPE for value in banks.values()), "premise: the load rounded them"

    config = single_process_ep_config(E, fp32_experts=True, experts_fsdp_managed=False)
    restore_fp32_master_parameters(model, str(tmp_path), config, keep_non_ep=False, strict=True)

    state = model.state_dict()
    for key in banks:
        prefix, _, name = key.rpartition(".")
        stored = [
            tensors[f"language_model.{prefix.replace('model.', 'backbone.', 1)}.{e}.{name}.weight"] for e in range(E)
        ]
        assert state[key].dtype == torch.float32 and torch.equal(state[key], torch.stack(stored)), key


def test_the_layer_scale_is_drawn_by_its_tower_through_a_real_load(hub_checkpoint, monkeypatch, sentinel_layer_scale):
    """The release stores none, so every one of them runs through the absent-module sweep and the coverage gate."""
    path, tensors = hub_checkpoint
    assert not [key for key in tensors if _LAYER_SCALE in key], "premise: the release stores no layer scale"
    _no_ep_layers(monkeypatch)
    state = _lazy(path, "composite", single_process_ep_config()).state_dict()
    layer_scales = [value for key, value in state.items() if key.endswith(_LAYER_SCALE)]
    assert len(layer_scales) == 2 * TINY_RADIO_VISION_CONFIG["num_hidden_layers"]
    expected = AutoConfig.from_pretrained(path).vision_config.layerscale_value
    assert all(torch.all(value.float() == expected) for value in layer_scales)


def test_the_root_init_would_leave_the_layer_scale_at_raw_memory(hub_checkpoint, monkeypatch, sentinel_layer_scale):
    """The control: with the composite's own ``_init_weights`` the sentinel survives, with no error anywhere."""
    path, _ = hub_checkpoint
    monkeypatch.setattr(materialization, "_nearest_init_weights", lambda model, _owner: model._init_weights)
    _no_ep_layers(monkeypatch)
    state = _lazy(path, "composite", single_process_ep_config()).state_dict()
    assert all(torch.all(value.float() == _SENTINEL) for key, value in state.items() if key.endswith(_LAYER_SCALE))


@torch.no_grad()
def test_the_lazy_composite_computes_what_the_eager_one_does_on_an_image(tmp_path):
    """Both under the real EP layers, and the pixels move the loss, or the tower sat out the comparison."""
    processor = tiny_processor("nemotron_h_omni")
    batch = tiny_image_batch(processor)
    write_tiny_hub_checkpoint(tmp_path, weights_dtype=RUN_DTYPE)
    config = json.loads((tmp_path / "config.json").read_text())
    config["image_token_id"] = processor.image_token_id
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert batch["input_ids"].max() < TINY_NEMOTRON_H_CONFIG["vocab_size"], "premise: the processor's ids are in range"
    path = str(tmp_path)
    batch = batch | {"pixel_values": batch["pixel_values"].to(RUN_DTYPE)}

    eager = apply_ep_layers(_eager(path, "composite"), single_process_ep_config(E)).eval()
    lazy = _lazy(path, "composite", single_process_ep_config()).eval()
    assert named_ep_layers(lazy) and all(type(layer) is EPNemotronHMoELayer for _, layer in named_ep_layers(lazy))

    expected = eager(**batch, use_cache=False)
    actual = lazy(**batch, use_cache=False)
    assert torch.equal(actual.logits, expected.logits)
    assert torch.equal(actual.loss, expected.loss)
    blank = lazy(**batch | {"pixel_values": torch.zeros_like(batch["pixel_values"])}, use_cache=False).loss
    assert not torch.equal(blank, actual.loss), "the images never reached the loss"


def _fused(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The expert keys as one bank per layer: what a tool writing the module tree's spelling leaves behind."""
    out = {key: value for key, value in tensors.items() if ".backbone." not in key or ".mixer.experts." not in key}
    layers = TINY_NEMOTRON_H_CONFIG["layers_block_type"]
    for index in (position for position, kind in enumerate(layers) if kind == "moe"):
        prefix = f"language_model.backbone.layers.{index}.mixer.experts"
        for name in ("up_proj", "down_proj"):
            out[f"{prefix}.{name}"] = torch.stack([tensors[f"{prefix}.{e}.{name}.weight"] for e in range(E)])
    return out


@pytest.mark.parametrize("rank", [0, 1])
def test_a_fused_checkpoint_slices_both_banks(hub_checkpoint, tmp_path, monkeypatch, rank):
    """Planned whole, ``up_proj`` would reach the layer as rank 0's slice on every rank while ``down_proj`` is its own."""
    path, _ = hub_checkpoint
    _no_ep_layers(monkeypatch)
    reference = _eager(path, "composite").state_dict()
    fused = _rewrite(path, tmp_path / "fused", _fused)
    state = _lazy(fused, "composite", stubbed_rank_ep_config(2, 2, rank, E)).state_dict()
    local = slice(rank * E // 2, (rank + 1) * E // 2)
    for key, tensor in reference.items():
        if key.endswith(_BANKS):
            assert tensors_equal_at_narrower_dtype(state[key], tensor[local]), key


@pytest.mark.parametrize(
    ("name", "shape", "scoped"),
    [
        ("language_model.backbone.layers.1.mixer.experts.3.up_proj.weight", (32, 16), True),
        ("language_model.backbone.layers.1.mixer.experts.3.down_proj.weight", (16, 32), True),
        ("language_model.backbone.layers.1.mixer.shared_experts.up_proj.weight", (48, 64), True),
        ("language_model.backbone.layers.1.mixer.fc1_latent_proj.weight", (16, 64), False),
        ("language_model.backbone.layers.1.mixer.fc2_latent_proj.weight", (64, 16), False),
        ("language_model.backbone.layers.1.mixer.gate.weight", (8, 64), False),
    ],
)
def test_a_low_precision_export_reproduces_the_forwards_scope(name, shape, scoped):
    """The trainer converts expert banks and the shared MLP; the latent projections keep bf16 there, so they must here."""
    include, exclude = re.compile(DEFAULT_INCLUDE), re.compile(DEFAULT_EXCLUDE)
    assert _should_quantize(name, torch.zeros(shape), include, exclude) is scoped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
