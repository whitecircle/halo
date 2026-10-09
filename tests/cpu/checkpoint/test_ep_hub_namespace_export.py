#!/usr/bin/env python
"""An EP-gathered save writes every tensor outside the expert banks in its hub checkpoint's namespace.

transformers reads several checkpoints through renames its module tree does not keep — a SigLIP
tower's ``vision_model`` level (Command A+, LFM2-VL), a vendor namespace (Inkling's ``model.llm.*``,
DeepSeek-V4's ``layers.N.attn.*``, GLM-5 Next's hyper-connections, Step-3.7's ``moe.*``), Mistral 3's
legacy llava prefixes — while the serving engines load hub names (vLLM 0.26.0 raises on Command A+'s
flattened tower). The gathered save therefore reverts what the load recorded, as ``save_pretrained``
does, on every streamed chunk; only the expert banks keep the layout their family's gather emits (fused
or per expert), which ``from_pretrained`` and both engines read either way.

Swept over every tiny EP family and every multimodal wrapper one ships under, each loaded from a tiny
checkpoint in the spelling its family's whole conversion mapping reverts to
(:func:`_write_hub_checkpoint`; the hub spelling, as far as transformers' own revert restores it) and
saved through the real
``save_ep_model`` after an ``ep_size=1`` EP patch. Per family: the save's keys outside the expert banks
are the hub checkpoint's; the artifact reloads, with no missing or unexpected key, into the tensors it
was saved from (so the per-chunk revert is the whole one); the wrapper-less writer writes the hub
checkpoint's keys exactly; ``unfuse_moe_experts.py`` turns the save into the hub checkpoint where the
hub stores experts one by one; a resume and the EP lazy loader read the save back; the weight sync
forwards what the save writes; and the sharded EP save is refused exactly where
``merge_ep_shards.py``'s key-by-key stream could not reproduce the save.

Run: ``python tests/cpu/checkpoint/test_ep_hub_namespace_export.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import PrefixChange, WeightConverter, WeightRenaming, rename_source_key
from transformers.integrations.mxfp4 import Mxfp4Dequantize
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

PartialState()  # save_ep_model logs through accelerate's logger

import src.distributed.checkpoint.ep_save as ep_save
from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint
from src.checkpoint.format import ModuleLayoutView
from src.distributed.checkpoint.ep_save import (
    _check_ep_merge_family_supported,
    _keys_merge_cannot_respell,
    save_ep_model,
)
from src.distributed.checkpoint.save import save_fsdp2_checkpoint
from src.distributed.checkpoint.tp_save import save_tp_model
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_classes,
    experts_container_attrs,
    gather_ep_layer_weights,
    is_expert_weight_attr,
    to_hub_layer_key,
)
from src.distributed.expert_parallel.lazy_loader import build_family_key_mapping
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.models.loading.lazy_safetensors.conversion import Concat
from src.models.structure import persistent_buffers
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights
from tests.common.checkpoint_io import loading_problems
from tests.common.models import TINY_STEP3P7_CONFIG
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, TinyFamily, tiny_family_model
from tests.common.utils import safetensors_state_dict

FAMILIES: dict[str, TinyFamily] = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}


@dataclass
class _Loaded:
    """One family loaded from its hub checkpoint, EP-patched, and what its gathered save wrote."""

    family: TinyFamily
    model: torch.nn.Module
    hub: dict[str, torch.Tensor]
    source: dict[str, torch.Tensor]
    saved: dict[str, torch.Tensor]
    ep_dir: str
    ep_layers: dict[str, EPMoELayerBase]


def _write_hub_checkpoint(family: TinyFamily, directory: str) -> None:
    """``family``'s seeded tiny model through transformers' save-side revert of the family's whole
    conversion mapping — the hub spelling, as far as that revert restores it (DeepSeek-V4's root keys
    keep the ``model.`` prefix the load adds).

    transformers drops every ``PrefixChange`` when it saves a model built from its config, which would
    write a SigLIP tower without the ``vision_model`` level the Command A+ and LFM2-VL indices carry, so
    a sub-model's own ``PrefixChange`` is kept. A model-level one is left out: it reads a text-only class
    out of its multimodal sibling's checkpoint, and such a class has no hub of its own.
    """
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
    patch_moe_model_for_ep(model, EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False))
    ep_layers = {path: module for path, module in model.named_modules() if isinstance(module, EPMoELayerBase)}
    assert ep_layers, f"{name}: the EP patch wrapped nothing, so no assertion here would reach a gather"
    save_ep_model(model, ep_dir)
    return _Loaded(
        family, model, safetensors_state_dict(hub_dir), source, safetensors_state_dict(ep_dir), ep_dir, ep_layers
    )


def _live_name(model: torch.nn.Module, key: str) -> str:
    """The live name checkpoint ``key`` loads into, by transformers' own load-side respelling."""
    conversions = get_model_conversion_mapping(model, add_legacy=True)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    live = dict.fromkeys(model.state_dict(), True)
    return rename_source_key(key, renamings, converters, model.base_model_prefix, live)[0]


def _in_expert_bank(loaded: _Loaded, key: str) -> bool:
    """Whether checkpoint ``key`` loads into an EP layer's expert weights — the one place the save's
    layout may differ from the hub's (fused against per expert)."""
    live = _live_name(loaded.model, key)
    for path in loaded.ep_layers:
        if live.startswith(f"{path}."):
            segments = live[len(path) + 1 :].split(".")
            if segments[0] in experts_container_attrs():
                segments = segments[2:] if segments[1].isdigit() else segments[1:]
            return is_expert_weight_attr(".".join(segments))
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


def _save_fsdp2(model: torch.nn.Module, output_dir: str) -> None:
    ctx = SimpleNamespace(
        model=model, is_save_rank=True, max_shard_size="5GB", training_checkpoint=False, tokenizer=None
    )
    save_fsdp2_checkpoint(ctx, output_dir)


_WRAPPERLESS_SAVERS = {"fsdp2": _save_fsdp2, "tp": save_tp_model}


@pytest.mark.parametrize("saver", _WRAPPERLESS_SAVERS)
@pytest.mark.parametrize("name", list(FAMILIES))
def test_wrapperless_save_writes_the_hub_checkpoint(name, saver, tmp_path):
    """The FSDP2/CP/TP writers revert the per-expert merges too, so they write the hub keys exactly."""
    family = FAMILIES[name]
    hub_dir, out_dir = str(tmp_path / "hub"), str(tmp_path / "out")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)

    _WRAPPERLESS_SAVERS[saver](model, out_dir)

    written, hub = set(safetensors_state_dict(out_dir)), set(safetensors_state_dict(hub_dir))
    assert written == hub, f"{name}: only written={sorted(written - hub)[:5]}, only hub={sorted(hub - written)[:5]}"


@pytest.mark.parametrize("name", ["cohere2_vision", "lfm2_vl"])
def test_a_model_without_a_load_record_writes_the_tower_level(name, tmp_path):
    """A lazily loaded (or config-built) model carries no record, so its save reverts the family's
    registry mapping, which must keep a sub-model's own ``PrefixChange``."""
    family = FAMILIES[name]
    hub_dir, out_dir = str(tmp_path / "hub"), str(tmp_path / "out")
    _write_hub_checkpoint(family, hub_dir)
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    assert getattr(model, "_weight_conversions", None) is None, "premise: no load recorded conversions"

    save_tp_model(model, out_dir)

    assert set(safetensors_state_dict(out_dir)) == set(safetensors_state_dict(hub_dir))


class _RecordingSender:
    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert name not in self.sent, f"{name} forwarded twice"
        self.sent[name] = weights

    def reset_prefix_cache(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _gather_on_cpu(layer: EPMoELayerBase):
    """The layer's own gather on the host: the sync asks for the GPU, and the family's assembly is what
    is under test."""

    def gather(device, **kwargs):
        return type(layer).gather_expert_state_dict(layer, "cpu", **kwargs)

    return gather


@pytest.mark.parametrize("name", list(FAMILIES))
def test_weight_sync_forwards_what_the_gathered_save_writes(name, tmp_path):
    """The sync and the save invert one list: the engine receives the checkpoint's names, all but the
    persistent buffers a parameter payload never carries."""
    loaded = _loaded(name, tmp_path)
    for layer in loaded.ep_layers.values():
        layer.gather_expert_state_dict = _gather_on_cpu(layer)
    sender = _RecordingSender()
    gather_and_send_weights(loaded.model, sender)

    buffers = {buffer_name for buffer_name, _ in persistent_buffers(loaded.model)}
    unsynced = {key for key in loaded.saved if _live_name(loaded.model, key) in buffers}
    assert set(sender.sent) == set(loaded.saved) - unsynced, (
        f"{name}: only sync={sorted(set(sender.sent) - set(loaded.saved))[:5]}, "
        f"only save={sorted(set(loaded.saved) - unsynced - set(sender.sent))[:5]}"
    )
    for key, tensor in sender.sent.items():
        assert torch.equal(tensor.to(loaded.saved[key].dtype), loaded.saved[key]), f"{name}: {key} differs"


@pytest.mark.parametrize("name", list(FAMILIES))
def test_sharded_save_refused_exactly_where_the_merge_cannot_reproduce_it(name, tmp_path):
    """``merge_ep_shards.py`` writes the live names through the family's ``_EXPORT_KEY_RENAMES`` alone,
    plus each layer's gather: where that is not what the gathered save writes, the sharded save must
    be refused, and only there."""
    loaded = _loaded(name, tmp_path)
    layer_cls = type(next(iter(loaded.ep_layers.values())))
    live = (key for key, _ in itertools.chain(loaded.model.named_parameters(), persistent_buffers(loaded.model)))
    merged = {
        to_hub_layer_key(key, layer_cls)
        for key in live
        if not any(key.startswith(f"{path}.") for path in loaded.ep_layers)
    }
    for path, layer in loaded.ep_layers.items():
        merged |= set(gather_ep_layer_weights(path, layer))

    reproducible = merged == set(loaded.saved)
    assert bool(_keys_merge_cannot_respell(loaded.model, layer_cls)) != reproducible, (
        f"{name}: the merge {'would' if reproducible else 'would not'} reproduce the gathered save, "
        f"and the sharded save is {'refused' if not reproducible else 'allowed'} the other way"
    )


@pytest.mark.parametrize("name", list(FAMILIES))
def test_unfuse_turns_the_gathered_save_into_the_hub_checkpoint(name, tmp_path):
    """Where the hub stores one tensor per expert and the gather writes the fused pair,
    ``unfuse_moe_experts.py`` is the documented step to the hub layout (SGLang 0.5.17's per-expert
    loaders drop a fused pair); it must land on the hub checkpoint's keys exactly. A family whose hub
    is fused is refused by the script, and its gathered save must already be the hub checkpoint."""
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
    """Continuing from an export runs the EP lazy loader over it: every key outside the expert banks
    (whose per-expert form the expert fuser owns) must land on a live tensor, the SigLIP towers'
    ``vision_model`` level included."""
    loaded = _loaded(name, tmp_path)
    shell = tiny_family_model(loaded.family)
    disk_to_model, fanout = build_family_key_mapping(shell, list(loaded.saved))

    live = set(shell.state_dict())
    # A fan-in's other sources (GLM-5 Next's k/v short convolutions) are read through its first one.
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
        for key in _outside_expert_banks(loaded, loaded.saved)
        if disk_to_model[key] not in live and key not in fan_in
    )
    assert unmapped == [], f"{name}: the lazy loader maps no live tensor for {unmapped[:5]}"


@pytest.mark.parametrize("name", list(FAMILIES))
def test_resume_reads_the_gathered_save_back(name, tmp_path):
    """A resume reads a checkpoint through ``ModuleLayoutView``: every live tensor but a tied alias must
    come back from the save, equal to what was saved."""
    loaded = _loaded(name, tmp_path)
    shell = tiny_family_model(loaded.family).to(torch.bfloat16)
    tied = {key for key, _ in shell.named_parameters(remove_duplicate=False)} - dict(shell.named_parameters()).keys()

    read = dict(ModuleLayoutView(shell, list(loaded.saved), loaded.saved.__getitem__).items())

    assert set(shell.state_dict()) - tied - set(read) == set(), f"{name}: the resume reads nothing for them"
    differing = [
        key for key, tensor in read.items() if not torch.equal(tensor.to(loaded.source[key].dtype), loaded.source[key])
    ]
    assert differing == [], f"{name}: {differing[:5]}"


def test_a_dequantizing_load_record_is_not_reverted_as_a_hub_layout(tmp_path):
    """A dequantizing MXFP4 gpt-oss load records the quantizer's converters, whose reverse respells
    every expert bank to a bare ``gate_up_proj`` with no layer prefix. The save reads the family's
    registry mapping instead and still writes the hub checkpoint."""
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
    patch_moe_model_for_ep(model, EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False))

    save_ep_model(model, ep_dir)

    assert set(safetensors_state_dict(ep_dir)) == set(safetensors_state_dict(hub_dir))


def test_a_failing_revert_fails_the_save(tmp_path, monkeypatch):
    """A chunk the revert cannot respell must fail the save, not land in the module spelling beside
    hub-spelled ones; the failure travels through the save's deferred guard."""
    family = FAMILIES["cohere2_vision"]
    hub_dir, ep_dir = str(tmp_path / "hub"), str(tmp_path / "ep")
    _write_hub_checkpoint(family, hub_dir)
    model = _load(family, hub_dir)
    patch_moe_model_for_ep(model, EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False))

    def failing_revert(*args, **kwargs):
        raise ValueError("no reverse for this chunk")

    monkeypatch.setattr(ep_save, "revert_conversions", failing_revert)
    with pytest.raises(RuntimeError, match="no reverse for this chunk"):
        save_ep_model(model, ep_dir)


def test_the_sharded_refusal_is_wired_into_the_family_check(tmp_path):
    loaded = _loaded("cohere2_vision", tmp_path)
    with pytest.raises(ValueError, match="key-by-key stream cannot"):
        _check_ep_merge_family_supported(loaded.model)


def test_command_a_plus_writes_the_tower_vllm_loads(tmp_path):
    """vLLM 0.26.0's ``Cohere2VisionForConditionalGeneration`` holds SigLIP under a ``vision_model``
    child and raises on a flattened tower key; its composite has no ``lm_head`` and raises on one too.
    The experts stay the fused pair vLLM reads."""
    saved = _loaded("cohere2_vision", tmp_path).saved

    assert "model.vision_tower.vision_model.embeddings.patch_embedding.weight" in saved
    assert not [key for key in saved if key.startswith("model.vision_tower.") and ".vision_model." not in key]
    assert "lm_head.weight" not in saved
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in saved


def test_step3p7_writes_the_release_spelling(tmp_path):
    """The spellings the serving engines read for Step-3.7, pinned against the real ``Step-3.7-Flash``
    index; no module-tree spelling may survive."""
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
    claimed = {model_type for cls in ep_layer_classes() for model_type in cls.HF_MODEL_TYPES}
    multimodal = claimed & set(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)
    assert multimodal, "premise: the registry claims multimodal wrappers"
    assert multimodal <= set(FAMILIES), f"no tiny model for {sorted(multimodal - set(FAMILIES))}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
