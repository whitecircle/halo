#!/usr/bin/env python
"""An EP-gathered save writes every tensor outside the expert banks in its hub checkpoint's namespace. transformers
reads several checkpoints through renames its module tree does not keep, and the serving engines load hub names."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest
import torch
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import PrefixChange, WeightConverter, WeightRenaming, rename_source_key
from transformers.integrations.mxfp4 import Mxfp4Dequantize
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

import halo.checkpoint.parallel.ep as ep_save
from halo.checkpoint.hub_layout import ModuleLayoutView
from halo.checkpoint.parallel.ep import (
    reject_unmergeable_ep_sharded_save,
    save_ep_gathered_checkpoint,
)
from halo.checkpoint.parallel.save import save_fsdp2_checkpoint, save_tp_checkpoint
from halo.checkpoint.tools.expert_unfuse import unfuse_checkpoint
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.lazy_safetensors.conversion import Concat
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.external_tables import named_external_tables
from halo.models.moe.family_specs import expert_weight_roots, experts_container_attrs, moe_model_types
from halo.models.param_roles import persistent_buffers
from halo.trainers.grpo.rollout.weight_push import _dense_params, gather_and_send_weights
from tests.common.checkpoint_io import loading_problems, safetensors_state_dict, save_context
from tests.common.ep_stubs import StubEPLayerBase, bare_ep_layer
from tests.common.models import TINY_STEP3P7_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, TinyFamily, tiny_family_model

FAMILIES: dict[str, TinyFamily] = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}


@dataclass
class _Loaded:
    family: TinyFamily
    model: torch.nn.Module
    hub: dict[str, torch.Tensor]
    source: dict[str, torch.Tensor]
    saved: dict[str, torch.Tensor]
    ep_dir: str
    ep_layers: dict[str, EPMoELayerBase]


def _write_hub_checkpoint(family: TinyFamily, directory: str) -> None:
    """transformers drops every ``PrefixChange`` when it saves a config-built model, so a sub-model's own is kept;
    DeepSeek-V4's root keys keep the ``model.`` prefix."""
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    model._weight_conversions = [
        conversion
        for conversion in get_model_conversion_mapping(model, add_legacy=False)
        if not (isinstance(conversion, PrefixChange) and not conversion.scope_prefix)
    ]
    model.save_pretrained(directory)


def _load(family: TinyFamily, directory: str) -> torch.nn.Module:
    return family.load_class.from_pretrained(
        directory, dtype=torch.bfloat16, trust_remote_code=family.trust_remote_code
    )


def _loaded(name: str, tmp_path) -> _Loaded:
    family = FAMILIES[name]
    hub_dir, ep_dir = str(tmp_path / "hub"), str(tmp_path / "ep")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)
    source = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    apply_ep_layers(model, single_process_ep_config())
    ep_layers = {path: module for path, module in model.named_modules() if isinstance(module, EPMoELayerBase)}
    assert ep_layers, f"{name}: the EP patch wrapped nothing, so no assertion here would reach a gather"
    save_ep_gathered_checkpoint(save_context(model), ep_dir)
    return _Loaded(
        family, model, safetensors_state_dict(hub_dir), source, safetensors_state_dict(ep_dir), ep_dir, ep_layers
    )


def _live_name(model: torch.nn.Module, key: str) -> str:
    conversions = get_model_conversion_mapping(model, add_legacy=True)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    live = dict.fromkeys(model.state_dict(), True)
    return rename_source_key(key, renamings, converters, model.base_model_prefix, live)[0]


def _in_expert_bank(loaded: _Loaded, key: str) -> bool:
    """The one place the save's layout may differ from the hub's; the HF block's spelling need not match the EP
    layer's, so every family's roots are checked."""
    live = _live_name(loaded.model, key)
    for path in loaded.ep_layers:
        if live.startswith(f"{path}."):
            segments = live[len(path) + 1 :].split(".")
            if segments[0] in experts_container_attrs():
                segments = segments[2:] if segments[1].isdigit() else segments[1:]
            return segments[0] in expert_weight_roots()
    return False


def _outside_expert_banks(loaded: _Loaded, keys) -> set[str]:
    return {key for key in keys if not _in_expert_bank(loaded, key)}


@pytest.mark.parametrize("name", list(FAMILIES))
def test_gathered_save_writes_the_hub_namespace_outside_the_expert_banks(name, tmp_path):
    loaded = _loaded(name, tmp_path)
    hub, saved = _outside_expert_banks(loaded, loaded.hub), _outside_expert_banks(loaded, loaded.saved)

    assert set(loaded.hub) - hub and set(loaded.saved) - saved, f"{name}: no key classified as an expert bank"
    assert saved == hub, (
        f"{name}: the gathered save spells tensors outside the expert banks differently from the hub: "
        f"only saved={sorted(saved - hub)[:5]}, only hub={sorted(hub - saved)[:5]}"
    )


@pytest.mark.parametrize("name", list(FAMILIES))
def test_gathered_save_reloads_into_the_saved_tensors(name, tmp_path):
    loaded = _loaded(name, tmp_path)
    reloaded, info = loaded.family.load_class.from_pretrained(
        loaded.ep_dir,
        dtype=torch.bfloat16,
        trust_remote_code=loaded.family.trust_remote_code,
        output_loading_info=True,
    )

    assert loading_problems(info) == {}
    state = reloaded.state_dict()
    assert set(state) == set(loaded.source)
    differing = [key for key, tensor in loaded.source.items() if not torch.equal(state[key], tensor)]
    assert differing == [], f"{name}: tensors changed through the gathered save: {differing[:5]}"


_WRAPPERLESS_SAVERS = {"fsdp2": save_fsdp2_checkpoint, "tp": save_tp_checkpoint}


@pytest.mark.parametrize("saver", _WRAPPERLESS_SAVERS)
@pytest.mark.parametrize("name", list(FAMILIES))
def test_wrapperless_save_writes_the_hub_checkpoint(name, saver, tmp_path):
    """These writers revert the per-expert merges too."""
    family = FAMILIES[name]
    hub_dir, out_dir = str(tmp_path / "hub"), str(tmp_path / "out")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)

    _WRAPPERLESS_SAVERS[saver](save_context(model), out_dir)

    written, hub = set(safetensors_state_dict(out_dir)), set(safetensors_state_dict(hub_dir))
    assert written == hub, f"{name}: only written={sorted(written - hub)[:5]}, only hub={sorted(hub - written)[:5]}"


@pytest.mark.parametrize("name", ["cohere2_vision", "lfm2_vl"])
def test_a_model_without_a_load_record_writes_the_tower_level(name, tmp_path):
    """A lazily loaded or config-built model carries no record, so its save reverts the family's registry mapping."""
    family = FAMILIES[name]
    hub_dir, out_dir = str(tmp_path / "hub"), str(tmp_path / "out")
    _write_hub_checkpoint(family, hub_dir)
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    assert getattr(model, "_weight_conversions", None) is None, "premise: no load recorded conversions"

    save_tp_checkpoint(save_context(model), out_dir)

    assert set(safetensors_state_dict(out_dir)) == set(safetensors_state_dict(hub_dir))


class _RecordingSender:
    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert name not in self.sent, f"{name} forwarded twice"
        self.sent[name] = weights

    def finish_weight_update(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _gather_on_cpu(layer: EPMoELayerBase):
    """The sync asks for the GPU, and the family's assembly is what is under test."""

    def gather(device, **kwargs):
        return type(layer).start_gather_expert_state_dict(layer, **{**kwargs, "device": "cpu"})

    return gather


@pytest.mark.parametrize("name", list(FAMILIES))
def test_weight_sync_forwards_what_the_gathered_save_writes(name, tmp_path):
    """The sync and save invert one list, all but the persistent buffers the push leaves out: every one but those a
    family declares (``engine_loaded_buffers``)."""
    loaded = _loaded(name, tmp_path)
    for layer in loaded.ep_layers.values():
        layer.start_gather_expert_state_dict = _gather_on_cpu(layer)
    sender = _RecordingSender()
    gather_and_send_weights(loaded.model, sender)

    declared = {
        f"{path}.{attr}"
        for path, layer in loaded.ep_layers.items()
        if layer.FAMILY.engine_loaded_buffers is not None
        for attr in layer.FAMILY.engine_loaded_buffers.attrs
    }
    buffers = {buffer_name for buffer_name, _ in persistent_buffers(loaded.model)} - declared
    unsynced = {key for key in loaded.saved if _live_name(loaded.model, key) in buffers}
    assert set(sender.sent) == set(loaded.saved) - unsynced, (
        f"{name}: only sync={sorted(set(sender.sent) - set(loaded.saved))[:5]}, "
        f"only save={sorted(set(loaded.saved) - unsynced - set(sender.sent))[:5]}"
    )
    for key, tensor in sender.sent.items():
        assert torch.equal(tensor.to(loaded.saved[key].dtype), loaded.saved[key]), f"{name}: {key} differs"


def test_save_and_sync_split_experts_by_the_layers_own_roots():
    """GptOss's ``gate_up_proj_bias`` is this layer's replicated state: the sync must push it densely."""
    layer = bare_ep_layer(
        StubEPLayerBase,
        gate_up_proj=torch.zeros(2, 4, 8),
        down_proj=torch.zeros(2, 4, 4),
        gate_up_proj_bias=torch.ones(8),
    )
    model = torch.nn.Module()
    model.moe = layer

    assert "moe.gate_up_proj_bias" in layer.gather_layer_state_dict("moe")
    assert [name for name, _ in _dense_params(model, {"moe": layer})] == ["moe.gate_up_proj_bias"]
    assert layer.is_expert_param("gate_up_proj") and not layer.is_expert_param("gate_up_proj_bias")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_sharded_save_refused_exactly_where_the_merge_cannot_reproduce_it(name, tmp_path, monkeypatch):
    """``merge_ep_shards.py`` writes live names through ``export_key_renames`` alone plus each layer's gather."""
    loaded = _loaded(name, tmp_path)
    layer_cls = type(next(iter(loaded.ep_layers.values())))
    live = (key for key, _ in itertools.chain(loaded.model.named_parameters(), persistent_buffers(loaded.model)))
    merged = {
        layer_cls.FAMILY.to_hub_key(key)
        for key in live
        if not any(key.startswith(f"{path}.") for path in loaded.ep_layers)
    }
    for path, layer in loaded.ep_layers.items():
        merged |= set(layer.gather_layer_state_dict(path))

    reproducible = merged == set(loaded.saved)
    # The family the live layers name, whatever the config's model_type resolves to.
    monkeypatch.setattr(ep_save, "find_moe_family_spec", lambda model_type: layer_cls.FAMILY)
    try:
        reject_unmergeable_ep_sharded_save(loaded.model, world_size=1)
        refused = False
    except ValueError as refusal:
        assert "key-by-key stream cannot" in str(refusal), refusal
        refused = True
    assert refused != reproducible, (
        f"{name}: the merge {'would' if reproducible else 'would not'} reproduce the gathered save, "
        f"and the sharded save is {'refused' if not reproducible else 'allowed'} the other way"
    )


@pytest.mark.parametrize("name", list(FAMILIES))
def test_unfuse_turns_the_gathered_save_into_the_hub_checkpoint(name, tmp_path):
    """SGLang 0.5.22's per-expert loaders drop a fused pair; a family whose hub is fused is refused by the script."""
    loaded = _loaded(name, tmp_path)
    unfused = str(tmp_path / "unfused")
    try:
        unfuse_checkpoint(loaded.ep_dir, unfused)
    except ValueError as refusal:
        assert "does not store one tensor per expert" in str(refusal), refusal
        written = set(loaded.saved)
    else:
        written = set(safetensors_state_dict(unfused))
    hub = set(loaded.hub)
    assert written == hub, f"{name}: only written={sorted(written - hub)[:5]}, only hub={sorted(hub - written)[:5]}"


@pytest.mark.parametrize("name", list(FAMILIES))
def test_lazy_loader_maps_the_gathered_save(name, tmp_path):
    """Excludes the expert banks, which the fuser owns, and the external tables, which the load attaches."""
    loaded = _loaded(name, tmp_path)
    stock = tiny_family_model(loaded.family)
    apply_family_patches(stock.config, "sdpa")
    shell = type(stock)(stock.config)
    tables = {
        f"{path}.{table.part_key(index)}"
        for path, table in named_external_tables(shell)
        for index in range(table.part_count)
    }
    disk_to_model, fanout = family_key_mapping(shell, list(loaded.saved))

    live = set(shell.state_dict())
    # GLM-5 Next's k/v short convolutions are read through the fan-in's first source.
    fan_in = {
        sibling
        for targets in fanout.values()
        for _, ops in targets
        for op in ops
        if isinstance(op, Concat)
        for sibling in op.siblings
    }
    unmapped = sorted(
        key
        for key in _outside_expert_banks(loaded, loaded.saved) - tables
        if disk_to_model[key] not in live and key not in fan_in
    )
    assert unmapped == [], f"{name}: the lazy loader maps no live tensor for {unmapped[:5]}"


@pytest.mark.parametrize("name", list(FAMILIES))
def test_resume_reads_the_gathered_save_back(name, tmp_path):
    """Every live tensor but a tied alias comes back equal."""
    loaded = _loaded(name, tmp_path)
    shell = tiny_family_model(loaded.family).to(torch.bfloat16)
    tied = {key for key, _ in shell.named_parameters(remove_duplicate=False)} - dict(shell.named_parameters()).keys()

    read = dict(ModuleLayoutView(shell, list(loaded.saved)).items(loaded.saved.__getitem__))

    assert set(shell.state_dict()) - tied - set(read) == set(), f"{name}: the resume reads nothing for them"
    differing = [
        key for key, tensor in read.items() if not torch.equal(tensor.to(loaded.source[key].dtype), loaded.source[key])
    ]
    assert differing == [], f"{name}: {differing[:5]}"


def test_a_dequantizing_load_record_is_not_reverted_as_a_hub_layout(tmp_path):
    """The quantizer's converters would respell every expert bank to a bare ``gate_up_proj``, so the save reads the
    registry mapping."""
    family = FAMILIES["gpt_oss"]
    hub_dir, ep_dir = str(tmp_path / "hub"), str(tmp_path / "ep")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)
    model._weight_conversions = [
        *model._weight_conversions,
        *(
            WeightConverter(
                source_patterns=[f"{proj}_blocks", f"{proj}_scales"],
                target_patterns=f"{proj}$",
                operations=[Mxfp4Dequantize(hf_quantizer=None)],
            )
            for proj in ("gate_up_proj", "down_proj")
        ),
    ]
    apply_ep_layers(model, single_process_ep_config())

    save_ep_gathered_checkpoint(save_context(model), ep_dir)

    assert set(safetensors_state_dict(ep_dir)) == set(safetensors_state_dict(hub_dir))


def test_a_failing_revert_fails_the_save(tmp_path, monkeypatch):
    """Otherwise the chunk lands in the module spelling beside hub-spelled ones."""
    family = FAMILIES["cohere2_vision"]
    hub_dir, ep_dir = str(tmp_path / "hub"), str(tmp_path / "ep")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)
    apply_ep_layers(model, single_process_ep_config())

    def failing_revert(*args, **kwargs):
        raise ValueError("no reverse for this chunk")

    monkeypatch.setattr(ep_save, "revert_conversions", failing_revert)
    with pytest.raises(RuntimeError, match="no reverse for this chunk"):
        save_ep_gathered_checkpoint(save_context(model), ep_dir)


def test_the_sharded_refusal_is_wired_into_the_family_check(tmp_path):
    loaded = _loaded("cohere2_vision", tmp_path)
    with pytest.raises(ValueError, match="key-by-key stream cannot"):
        reject_unmergeable_ep_sharded_save(loaded.model, world_size=1)


def test_command_a_plus_writes_the_tower_vllm_loads(tmp_path):
    """vLLM 0.32.0 raises on a flattened tower key and on ``lm_head``; the experts stay the fused pair it reads."""
    saved = _loaded("cohere2_vision", tmp_path).saved

    assert "model.vision_tower.vision_model.embeddings.patch_embedding.weight" in saved
    assert not [key for key in saved if key.startswith("model.vision_tower.") and ".vision_model." not in key]
    assert "lm_head.weight" not in saved
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in saved


def test_step3p7_writes_the_release_spelling(tmp_path):
    """Pinned against the real ``Step-3.7-Flash`` index."""
    saved = _loaded("step3p7", tmp_path).saved
    for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]):
        if kind != "sparse":
            continue
        for leaf in ("gate.weight", "router_bias", "gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            assert f"model.layers.{i}.moe.{leaf}" in saved, f"layer {i}: hub key moe.{leaf} missing"
        for proj in ("gate", "up", "down"):
            assert f"model.layers.{i}.share_expert.{proj}_proj.weight" in saved
    module_spellings = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector")
    assert not [key for key in saved if any(spelling in key for spelling in module_spellings)]


def test_the_rosters_cover_every_multimodal_model_type_the_ep_registry_claims():
    multimodal = moe_model_types() & set(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
    assert multimodal, "premise: the registry claims multimodal wrappers"
    assert multimodal <= set(FAMILIES), f"no tiny model for {sorted(multimodal - set(FAMILIES))}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
