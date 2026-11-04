#!/usr/bin/env python
"""A source-schema family's ``config.json`` is the source repo's schema, since transformers absorbs the vendor keys
at load and engines read only those."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.errors import StrictDataclassFieldValidationError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    NemotronH_Omni_Reasoning_V3,
    NemotronHForCausalLM,
    PretrainedConfig,
)
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from halo.checkpoint import config_export
from halo.checkpoint.config_export import export_source_config_schema, save_model_config
from halo.checkpoint.tools.offline_export import save_full_checkpoint
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from halo.model_loading.model_preparation import auto_load_model
from halo.models.moe.family_specs import MOE_FAMILY_SPECS
from tests.common.models import (
    NEMOTRON_H_RELEASE_TEXT_CONFIG,
    NEMOTRON_H_RELEASE_VISION_CONFIG,
    TINY_NEMOTRON_H_CONFIG,
    TINY_RADIO_VISION_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.nemotron_h import (
    nemotron_h_lightning_config,
    nemotron_h_release_config,
    tiny_lightning_config,
    write_tiny_hub_checkpoint,
    write_tiny_lightning_checkpoint,
)
from tests.common.rosters import ep_layer_class_by_model_type

# The release's key spellings at the roster's geometry; ``test_the_source_config_parses_to_the_roster_geometry`` pins
# it against ``TINY_STEP3P7_CONFIG``.
TINY_STEP3P7_SOURCE_CONFIG = {
    "architectures": ["Step3p7ForConditionalGeneration"],
    "auto_map": {
        "AutoConfig": "configuration_step3p7.Step3p7Config",
        "AutoProcessor": "processing_step3.Step3VLProcessor",
        "AutoModelForCausalLM": "modeling_step3p7.Step3p7ForConditionalGeneration",
    },
    "model_type": "step3p7",
    "image_token_id": 2000,
    "vision_config": {
        "model_type": "perception_encoder",
        "width": TINY_STEP3P7_VISION_CONFIG["hidden_size"],
        "layers": TINY_STEP3P7_VISION_CONFIG["num_hidden_layers"],
        "heads": TINY_STEP3P7_VISION_CONFIG["num_attention_heads"],
        "image_size": TINY_STEP3P7_VISION_CONFIG["image_size"],
        "patch_size": TINY_STEP3P7_VISION_CONFIG["patch_size"],
        "mlp_ratio": TINY_STEP3P7_VISION_CONFIG["mlp_ratio"],
        "max_position_embeddings": TINY_STEP3P7_VISION_CONFIG["max_position_embeddings"],
    },
    "text_config": {
        "model_type": "step3p5",
        "vocab_size": TINY_STEP3P7_CONFIG["vocab_size"],
        "hidden_size": TINY_STEP3P7_CONFIG["hidden_size"],
        "intermediate_size": TINY_STEP3P7_CONFIG["intermediate_size"],
        "moe_intermediate_size": TINY_STEP3P7_CONFIG["moe_intermediate_size"],
        "share_expert_dim": TINY_STEP3P7_CONFIG["share_expert_dim"],
        "num_hidden_layers": TINY_STEP3P7_CONFIG["num_hidden_layers"],
        "num_attention_heads": TINY_STEP3P7_CONFIG["num_attention_heads"],
        "num_attention_groups": TINY_STEP3P7_CONFIG["num_key_value_heads"],
        "head_dim": TINY_STEP3P7_CONFIG["head_dim"],
        "attention_other_setting": {"num_attention_heads": TINY_STEP3P7_CONFIG["num_sliding_attention_heads"]},
        "layer_types": list(TINY_STEP3P7_CONFIG["layer_types"]),
        "sliding_window": TINY_STEP3P7_CONFIG["sliding_window"],
        "moe_num_experts": TINY_STEP3P7_CONFIG["n_routed_experts"],
        "moe_top_k": TINY_STEP3P7_CONFIG["num_experts_per_tok"],
        "moe_router_scaling_factor": TINY_STEP3P7_CONFIG["moe_router_scaling_factor"],
        "moe_layers_enum": ",".join(
            str(i) for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"
        ),
        "swiglu_limits": list(TINY_STEP3P7_CONFIG["swiglu_limits"]),
        "swiglu_limits_shared": list(TINY_STEP3P7_CONFIG["swiglu_limits_shared"]),
        "max_position_embeddings": TINY_STEP3P7_CONFIG["max_position_embeddings"],
        "tie_word_embeddings": TINY_STEP3P7_CONFIG["tie_word_embeddings"],
    },
}

# A composite in the pinned transformers whose release spells its config in the legacy field names the class absorbs, and
# an infinite ``time_step_limit`` through JSON's float tag.
TINY_NEMOTRON_SOURCE_CONFIG = {
    "architectures": ["NemotronH_Omni_Reasoning_V3"],
    "auto_map": {
        "AutoConfig": "configuration_nemotron_h_omni.NemotronH_Omni_Reasoning_V3_Config",
        "AutoModelForCausalLM": "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3",
    },
    "model_type": "nemotron_h_omni",
    "img_context_token_id": 18,
    "vit_hidden_size": TINY_RADIO_VISION_CONFIG["hidden_size"],
    "projector_hidden_size": TINY_NEMOTRON_H_CONFIG["hidden_size"],
    "video_pruning_rate": 0.7,
    "tie_word_embeddings": False,
    "llm_config": {**TINY_NEMOTRON_H_CONFIG, "time_step_limit": [0.0, {"__float__": "Infinity"}]},
    "vision_config": dict(TINY_RADIO_VISION_CONFIG),
}
# The export copies them and never imports them; the modeling entry names siblings to pin the transitive copy.
_NEMOTRON_MODULES = {
    "configuration_nemotron_h_omni.py": "from .configuration_radio import RadioConfig\n",
    "configuration_radio.py": "class RadioConfig: pass\n",
    "modeling_nemotron_h_omni.py": (
        "from .configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config\n"
        "from .modeling_radio import RadioModel\n"
    ),
    "modeling_radio.py": "from .configuration_radio import RadioConfig\n",
}

# The source's, which the pinned engines read, against the native-only ones they have no class for.
_SOURCE_SPELLINGS = ("moe_num_experts", "moe_top_k", "moe_layers_enum", "attention_other_setting")
_NATIVE_ONLY_SPELLINGS = ("n_routed_experts", "num_experts_per_tok", "mlp_layer_types", "per_layer_config")

_SOURCE_MODULES = {
    "configuration_step3p7.py": "class Step3p7Config: pass\n",
    "processing_step3.py": "class Step3VLProcessor: pass\n",
    "modeling_step3p7.py": "from .vision_encoder import VisionEncoder\n\n\nclass Step3p7ForConditionalGeneration: pass\n",
    "vision_encoder.py": "class VisionEncoder: pass\n",
}


@dataclass(frozen=True)
class _SourceFamily:
    """The source checkpoint a declaring family's export carries forward, and where its spellings sit."""

    source_config: dict
    modules: dict[str, str]
    model_class: type
    native_config: Callable[[], PretrainedConfig]
    text_section: str
    # The section holding the spellings below, "" for the top level.
    spelled_section: str
    source_spellings: tuple[str, ...]
    native_only_spellings: tuple[str, ...]

    def spelled(self, payload: dict) -> dict:
        return payload[self.spelled_section] if self.spelled_section else payload


def _native_step3p7_config() -> PretrainedConfig:
    return Step3p7Config(
        text_config=dict(TINY_STEP3P7_CONFIG),
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=TINY_STEP3P7_SOURCE_CONFIG["image_token_id"],
    )


def _native_nemotron_config() -> PretrainedConfig:
    return NemotronH_Omni_Reasoning_V3.config_class(
        text_config=dict(TINY_NEMOTRON_H_CONFIG),
        vision_config=dict(TINY_RADIO_VISION_CONFIG),
        vision_hidden_size=TINY_RADIO_VISION_CONFIG["hidden_size"],
        projector_hidden_size=TINY_NEMOTRON_H_CONFIG["hidden_size"],
        image_token_id=TINY_NEMOTRON_SOURCE_CONFIG["img_context_token_id"],
    )


FIXTURES = {
    EPStep3p7MoELayer: _SourceFamily(
        TINY_STEP3P7_SOURCE_CONFIG,
        _SOURCE_MODULES,
        Step3p7ForConditionalGeneration,
        _native_step3p7_config,
        text_section="text_config",
        spelled_section="text_config",
        source_spellings=_SOURCE_SPELLINGS,
        native_only_spellings=_NATIVE_ONLY_SPELLINGS,
    ),
    EPNemotronHMoELayer: _SourceFamily(
        TINY_NEMOTRON_SOURCE_CONFIG,
        _NEMOTRON_MODULES,
        NemotronH_Omni_Reasoning_V3,
        _native_nemotron_config,
        text_section="llm_config",
        spelled_section="",
        source_spellings=("llm_config", "vit_hidden_size", "img_context_token_id"),
        native_only_spellings=("text_config", "vision_hidden_size", "image_token_id"),
    ),
}
_IDS = [cls.__name__ for cls in FIXTURES]


def _write_source(directory, payload: dict, modules: dict[str, str]) -> str:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    for name, body in modules.items():
        with open(os.path.join(directory, name), "w") as handle:
            handle.write(body)
    return str(directory)


def _model_from(cls, source: str):
    """Loaded config, source recorded by the loader."""
    torch.manual_seed(0)
    model = FIXTURES[cls].model_class(AutoConfig.from_pretrained(source)).to(torch.bfloat16)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, source)
    return model


def _exported(tmp_path, cls, mutate=None) -> tuple[dict, str, object]:
    family = FIXTURES[cls]
    source = _write_source(tmp_path / "source", family.source_config, family.modules)
    export = str(tmp_path / "export")
    os.makedirs(export, exist_ok=True)
    model = _model_from(cls, source)
    if mutate is not None:
        mutate(model)
    save_model_config(model, export)
    with open(os.path.join(export, "config.json")) as handle:
        return json.load(handle), export, model


def test_every_declaring_family_has_a_source_schema_fixture():
    declaring = {
        cls for cls in set(ep_layer_class_by_model_type().values()) if cls.FAMILY.exports_source_config_schema
    }
    assert declaring == set(FIXTURES), (
        f"{sorted(c.__name__ for c in declaring ^ set(FIXTURES))}: a family declaring "
        f"exports_source_config_schema must pin its carried export schema here"
    )


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_source_config_parses_to_the_roster_geometry(cls, tmp_path):
    """Otherwise every assertion below is about a different config than the suite trains."""
    family = FIXTURES[cls]
    parsed = AutoConfig.from_pretrained(_write_source(tmp_path, family.source_config, family.modules)).to_dict()
    native = family.native_config().to_dict()
    for section in ("text_config", "vision_config"):
        assert parsed[section] == native[section], f"the {section} stand-in drifted from the roster config"


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_export_carries_the_source_schema_and_its_modules(cls, tmp_path):
    family = FIXTURES[cls]
    payload, export, _model = _exported(tmp_path, cls)
    spelled = family.spelled(payload)
    assert all(key in spelled for key in family.source_spellings), (
        f"the export dropped the source spellings the pinned engines read: "
        f"{[key for key in family.source_spellings if key not in spelled]}"
    )
    assert not [key for key in family.native_only_spellings if key in spelled], (
        f"the export kept transformers-only spellings the pinned engines have no class for: "
        f"{[key for key in family.native_only_spellings if key in spelled]}"
    )
    assert payload["auto_map"] == family.source_config["auto_map"]
    missing = [name for name in family.modules if not os.path.isfile(os.path.join(export, name))]
    assert not missing, f"the export ships no {missing} — the directory does not load without them"


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_export_reloads_to_the_config_that_was_trained(cls, tmp_path):
    """This transformers accepts the source's spellings, so the export reloads exactly."""
    _payload, export, model = _exported(tmp_path, cls)
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_a_patched_vocab_size_reaches_the_carried_config(cls, tmp_path):
    """Carried verbatim, it would serve the source's vocabulary against a resized embedding, an off-by-N no reader
    can detect."""
    family = FIXTURES[cls]
    patched = family.source_config[family.text_section]["vocab_size"] + 128

    def resize(model):
        model.resize_token_embeddings(patched)

    payload, export, model = _exported(tmp_path, cls, mutate=resize)
    assert all(key in family.spelled(payload) for key in family.source_spellings), "not the carried schema"
    assert payload[family.text_section]["vocab_size"] == patched
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_standalone_writer_carries_the_schema_too(cls, tmp_path):
    """The vocab patch is where a Step-3.7 run starts, through ``save_full_checkpoint``."""
    family = FIXTURES[cls]
    source = _write_source(tmp_path / "source", family.source_config, family.modules)
    export = str(tmp_path / "export")
    patched = family.source_config[family.text_section]["vocab_size"] + 128
    model = _model_from(cls, source)
    model.resize_token_embeddings(patched)

    save_full_checkpoint(model, export, source_dir=source)

    with open(os.path.join(export, "config.json")) as handle:
        payload = json.load(handle)
    assert all(key in family.spelled(payload) for key in family.source_spellings)
    assert payload[family.text_section]["vocab_size"] == patched
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


def test_a_change_the_source_schema_swallows_is_refused(tmp_path):
    """A field spelled through a legacy key parses back to the source's value, a geometry the trainer never had."""

    def widen_sliding_heads(model):
        model.config.text_config.num_sliding_attention_heads = TINY_STEP3P7_CONFIG["num_attention_heads"]

    with pytest.raises(ValueError, match="cannot express this run's config"):
        _exported(tmp_path, EPStep3p7MoELayer, mutate=widen_sliding_heads)


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_a_source_without_remote_code_leaves_the_native_schema(cls, tmp_path, caplog):
    """A source that was never servable has no schema to hand on."""
    family = FIXTURES[cls]
    source_config = {k: v for k, v in family.source_config.items() if k != "auto_map"}
    source = _write_source(tmp_path / "source", source_config, {})
    export = str(tmp_path / "export")
    os.makedirs(export, exist_ok=True)
    with caplog.at_level("WARNING"):
        save_model_config(_model_from(cls, source), export)
    with open(os.path.join(export, "config.json")) as handle:
        payload = json.load(handle)
    assert "auto_map" not in payload
    assert family.native_only_spellings[0] in family.spelled(payload)
    assert "declares no auto_map" in caplog.text


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_only_an_unreadable_source_degrades_to_the_native_schema(cls, tmp_path, monkeypatch, caplog):
    """Any other read failure is a bug and must surface rather than ship an unservable config."""
    family = FIXTURES[cls]
    source = _write_source(tmp_path / "source", family.source_config, family.modules)
    model = _model_from(cls, source)
    export = str(tmp_path / "export")
    os.makedirs(export, exist_ok=True)

    def _unreachable(*_args, **_kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(config_export, "cached_file", _unreachable)
    with caplog.at_level("WARNING"):
        save_model_config(model, export)
    assert "could not read config.json" in caplog.text

    def _bug(*_args, **_kwargs):
        raise RuntimeError("not a read failure")

    monkeypatch.setattr(config_export, "cached_file", _bug)
    with pytest.raises(RuntimeError, match="not a read failure"):
        save_model_config(model, export)


@pytest.fixture
def nemotron_source(tmp_path) -> str:
    return _write_source(tmp_path / "source", TINY_NEMOTRON_SOURCE_CONFIG, _NEMOTRON_MODULES)


@pytest.fixture
def release_source(tmp_path) -> str:
    """The release's own ``config.json``, with every field the roster's stand-in leaves out."""
    return _write_source(tmp_path / "source", nemotron_h_release_config(), _NEMOTRON_MODULES)


def _carried(tmp_path, config, source: str) -> tuple[dict, str]:
    export = str(tmp_path / "export")
    config.save_pretrained(export)
    export_source_config_schema(config, export, source=source)
    with open(os.path.join(export, "config.json")) as handle:
        return json.load(handle), export


def test_an_unchanged_export_is_the_source_config(tmp_path, nemotron_source):
    """Through the float tag ``from_dict`` needs decoded and the ``llm_config`` section ``text_config`` is spelled as."""
    config = AutoConfig.from_pretrained(nemotron_source)
    payload, export = _carried(tmp_path, config, nemotron_source)
    assert payload == TINY_NEMOTRON_SOURCE_CONFIG
    assert all(os.path.isfile(os.path.join(export, name)) for name in _NEMOTRON_MODULES)


def test_the_release_config_comes_back_whole(tmp_path, release_source):
    config = AutoConfig.from_pretrained(release_source)
    payload, export = _carried(tmp_path, config, release_source)
    assert payload == nemotron_h_release_config()
    assert all(os.path.isfile(os.path.join(export, name)) for name in _NEMOTRON_MODULES)


def test_a_live_release_shell_exports_the_release_config(tmp_path, release_source):
    """What a save writes on the real geometry: the stamps a load leaves, the class name and the generation config."""
    with torch.device("meta"):
        model = NemotronH_Omni_Reasoning_V3(AutoConfig.from_pretrained(release_source))
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, release_source)
    export = str(tmp_path / "export")
    save_model_config(model, export)
    with open(os.path.join(export, "config.json")) as handle:
        assert json.load(handle) == nemotron_h_release_config()
    assert os.path.isfile(os.path.join(export, "generation_config.json"))


def test_the_release_fixture_is_the_roster_release():
    release = nemotron_h_release_config()
    assert release["llm_config"]["layers_block_type"] == NEMOTRON_H_RELEASE_TEXT_CONFIG["layers_block_type"]
    assert release["vision_config"]["hidden_size"] == NEMOTRON_H_RELEASE_VISION_CONFIG["hidden_size"]


def test_the_release_tag_needs_decoding_before_the_class_reads_it():
    """Raw, the carry would crash on the release's own ``time_step_limit``, so the decode is load-bearing here."""
    release = nemotron_h_release_config()
    assert release["llm_config"]["time_step_limit"] == [0.0, {"__float__": "Infinity"}]
    config_class = NemotronH_Omni_Reasoning_V3.config_class
    with pytest.raises(StrictDataclassFieldValidationError, match="time_step_limit"):
        config_class.from_dict(copy.deepcopy(release))
    assert config_class.from_dict(config_class._decode_special_floats(release)).text_config.time_step_limit == [
        0.0,
        float("inf"),
    ]


def test_a_trainers_cache_write_on_the_composite_root_does_not_fail_the_save(tmp_path, nemotron_source):
    """A trainer writes ``use_cache`` on the root of a composite whose class does not hold it there. It raised out of
    the first save of an image run, as a field the source schema could not express."""
    config = AutoConfig.from_pretrained(nemotron_source)
    config.use_cache = False
    payload, _ = _carried(tmp_path, config, nemotron_source)
    assert "use_cache" not in payload


def test_a_run_change_lands_under_the_legacy_spelling(tmp_path, nemotron_source):
    config = AutoConfig.from_pretrained(nemotron_source)
    config.text_config.use_cache = False
    config.video_pruning_rate = 0.0
    config.vision_hidden_size = TINY_RADIO_VISION_CONFIG["hidden_size"] * 2
    payload, export = _carried(tmp_path, config, nemotron_source)
    assert "text_config" not in payload and payload["llm_config"]["use_cache"] is False
    assert payload["video_pruning_rate"] == 0.0
    assert "vision_hidden_size" not in payload and payload["vit_hidden_size"] == config.vision_hidden_size
    assert payload["llm_config"]["time_step_limit"] == [0.0, {"__float__": "Infinity"}], "the tag must stay on disk"
    assert AutoConfig.from_pretrained(export).to_diff_dict() == config.to_diff_dict()


def test_a_text_only_export_of_a_composite_keeps_its_own_schema(tmp_path, nemotron_source):
    """Carried, it would come out spelling the composite's ``model_type`` over the text tower's weights."""
    config = AutoConfig.from_pretrained(nemotron_source).text_config
    payload, export = _carried(tmp_path, config, nemotron_source)
    assert payload["model_type"] == "nemotron_h" and "auto_map" not in payload
    assert "llm_config" not in payload and "vision_config" not in payload
    assert not [name for name in _NEMOTRON_MODULES if os.path.isfile(os.path.join(export, name))]
    assert AutoConfig.from_pretrained(export).to_diff_dict() == config.to_diff_dict()


def test_a_reattached_text_export_speaks_the_release_schema_with_its_run_change(tmp_path):
    """The text export keeps transformers' schema, and the reattach hands the composite the release's back."""
    checkpoint, export, out = tmp_path / "checkpoint", tmp_path / "export", tmp_path / "out"
    write_tiny_hub_checkpoint(checkpoint)
    _write_source(checkpoint, TINY_NEMOTRON_SOURCE_CONFIG, _NEMOTRON_MODULES)
    model = auto_load_model(str(checkpoint), text_only=True, dtype=torch.bfloat16)
    model.config.use_cache = False
    save_full_checkpoint(model, str(export), source_dir=str(checkpoint))
    assert "llm_config" not in json.loads((export / "config.json").read_text())

    reattach_vision_tower(str(export), str(checkpoint), str(out), trust_remote_code=False)

    payload = json.loads((out / "config.json").read_text())
    expected = copy.deepcopy(TINY_NEMOTRON_SOURCE_CONFIG)
    expected["llm_config"] |= {"use_cache": False, "dtype": "bfloat16"}
    assert payload == expected, "a load's stamp on the export, such as its local path, must not reach the artifact"
    assert all((out / name).is_file() for name in _NEMOTRON_MODULES)


def test_a_text_only_load_of_the_composite_saves_the_text_class(tmp_path, caplog):
    """The whole path, from the composite's checkpoint: the save names the text class and reloads on its own."""
    checkpoint = tmp_path / "checkpoint"
    write_tiny_hub_checkpoint(checkpoint)
    _write_source(checkpoint, TINY_NEMOTRON_SOURCE_CONFIG, _NEMOTRON_MODULES)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=AutoConfig.from_pretrained(checkpoint).text_config, dtype=torch.bfloat16
    )
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, str(checkpoint))
    export = tmp_path / "export"
    export.mkdir()
    with caplog.at_level("WARNING"):
        save_model_config(model, str(export))
    with open(export / "config.json") as handle:
        payload = json.load(handle)
    assert payload["model_type"] == "nemotron_h" and payload["architectures"] == ["NemotronHForCausalLM"]
    assert not {"auto_map", "llm_config", "vision_config"} & set(payload)
    assert not [name for name in _NEMOTRON_MODULES if (export / name).exists()]
    assert not caplog.records, "a text-only export is the intended outcome, not a degraded one"
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.fixture
def lightning_source(tmp_path) -> str:
    return _write_source(tmp_path / "source", nemotron_h_lightning_config(), {})


def _exporter_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == config_export.logger.name and record.levelno >= logging.WARNING
    ]


def _saved_shell(source: str, export: str, mutate: Callable[[PretrainedConfig], None] | None = None) -> dict:
    """A save on the release's geometry, with no weights to allocate."""
    with torch.device("meta"):
        model = NemotronHForCausalLM(AutoConfig.from_pretrained(source))
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, source)
    if mutate is not None:
        mutate(model.config)
    save_model_config(model, export)
    with open(os.path.join(export, "config.json")) as handle:
        return json.load(handle)


def test_a_text_class_release_exports_its_own_config_with_no_warning(tmp_path, lightning_source, caplog):
    """No ``auto_map`` here is the release itself, so re-deriving from it, as the composite's warning advises, leads back
    to the same export."""
    with caplog.at_level("WARNING"):
        payload = _saved_shell(lightning_source, str(tmp_path / "export"))
    assert payload == nemotron_h_lightning_config()
    assert not _exporter_warnings(caplog)


def test_a_text_class_export_keeps_the_spellings_sglang_reads(tmp_path, lightning_source):
    """SGLang's own config class raises on transformers' ``linear_attention`` and ``full_attention``. The release
    declares no time-step limit, so the export writes no JSON float tag for a reader to decode."""
    export = str(tmp_path / "export")
    payload = _saved_shell(lightning_source, export)
    assert set(payload["layers_block_type"]) == {"mamba", "attention", "moe"}
    with open(os.path.join(export, "config.json")) as handle:
        assert "Infinity" not in handle.read()
    assert AutoConfig.from_pretrained(export).time_step_limit == (0.0, float("inf"))


def test_a_run_change_lands_on_the_text_class_release_schema(tmp_path, lightning_source):
    def change(config):
        config.use_cache = False
        config.vocab_size = 131200

    export = str(tmp_path / "export")
    payload = _saved_shell(lightning_source, export, change)
    assert payload == nemotron_h_lightning_config() | {"use_cache": False, "vocab_size": 131200}
    reloaded = AutoConfig.from_pretrained(export)
    assert reloaded.use_cache is False and reloaded.vocab_size == 131200


def test_a_change_of_the_layer_list_reaches_the_carried_config(tmp_path, lightning_source):
    """Parsed back, a cut model's layers are the ones it trained, whatever the carried count says."""

    def cut(config):
        config.layers_block_type = config.layers_block_type[:9]

    export = str(tmp_path / "export")
    _saved_shell(lightning_source, export, cut)
    assert AutoConfig.from_pretrained(export).num_hidden_layers == 9


def test_an_export_of_an_export_is_the_same_config(tmp_path, lightning_source, caplog):
    """A resume or a next stage reads the last export as its source."""
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    written = _saved_shell(lightning_source, first)
    with caplog.at_level("WARNING"):
        assert _saved_shell(first, second) == written
    assert not _exporter_warnings(caplog)


def test_an_export_off_a_composites_text_export_stays_in_transformers_schema(tmp_path, nemotron_source, caplog):
    """That source already speaks the text class's serialization, the float tag included, and the reattach is what
    serves it."""
    text_export = str(tmp_path / "text")
    with torch.device("meta"):
        model = NemotronHForCausalLM(AutoConfig.from_pretrained(nemotron_source).text_config)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, nemotron_source)
    save_model_config(model, text_export)
    with open(os.path.join(text_export, "config.json")) as handle:
        written = json.load(handle)
    assert "linear_attention" in written["layers_block_type"]

    with caplog.at_level("WARNING"):
        payload = _saved_shell(text_export, str(tmp_path / "next"))
    assert payload == written
    assert payload["time_step_limit"] == [0.0, {"__float__": "Infinity"}]
    assert not _exporter_warnings(caplog)


def test_the_no_auto_map_rule_reaches_the_text_class_alone():
    """Each other type a family declares reaches the engines through the modules it names, which the refusal above pins.
    A new family lands here, so its author says what its engines read."""
    reached = {
        model_type
        for spec in MOE_FAMILY_SPECS
        if spec.exports_source_config_schema
        for model_type in spec.model_types
        if config_export._read_without_remote_code(SimpleNamespace(model_type=model_type), {"model_type": model_type})
    }
    assert reached == {"nemotron_h"}


def test_no_auto_map_does_not_hand_on_a_source_of_another_type():
    assert not config_export._read_without_remote_code(SimpleNamespace(model_type="nemotron_h"), {})
    assert not config_export._read_without_remote_code(
        SimpleNamespace(model_type="nemotron_h"), {"model_type": "nemotron_h_omni"}
    )


@pytest.mark.parametrize(
    "model_type",
    [
        "qwen3",  # A dense family has no spec at all.
        *sorted(
            model_type
            for spec in MOE_FAMILY_SPECS
            if not spec.exports_source_config_schema
            for model_type in spec.model_types
        ),
    ],
)
def test_a_family_declaring_no_source_schema_keeps_transformers_serialization(model_type, tmp_path, caplog):
    """Its engines read what transformers writes, so a release's own spellings never reach its export."""
    source = _write_source(tmp_path / "source", {"model_type": model_type, "vendor_spelling": 1}, {})
    export = tmp_path / "export"
    export.mkdir()
    written = json.dumps({"model_type": model_type})
    (export / "config.json").write_text(written)
    with caplog.at_level("WARNING"):
        export_source_config_schema(SimpleNamespace(model_type=model_type), str(export), source=source)
    assert (export / "config.json").read_text() == written
    assert not _exporter_warnings(caplog)


# Run without the toolkit: it reads the export as anyone downstream does, with no conversion entry or patch of ours. The
# image's causal_conv1d runs on CUDA only, and without it upstream takes its torch path.
_STOCK_RELOAD = """
import json, sys
sys.modules["causal_conv1d"] = None
import torch
from transformers import AutoConfig, AutoModelForCausalLM
export, release = sys.argv[1:]
ids = torch.arange(1, 33).unsqueeze(0)
loaded = {}
for path in (export, release):
    model, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, output_loading_info=True)
    with torch.no_grad():
        logits = model.eval()(input_ids=ids, use_cache=False).logits
    loaded[path] = (model, info, logits)
model, info, logits = loaded[export]
print(json.dumps({
    "toolkit_imported": any(name == "halo" or name.startswith("halo.") for name in sys.modules),
    "class": type(model).__name__,
    "problems": {kind: list(info[kind]) for kind in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
    "tensors_equal": all(
        torch.equal(tensor, loaded[release][0].state_dict()[key]) for key, tensor in model.state_dict().items()
    ),
    "logits_equal": torch.equal(logits, loaded[release][2]),
    "layer_types": AutoConfig.from_pretrained(export).layer_types,
}))
"""


def test_stock_transformers_reloads_the_text_class_export_as_the_release(tmp_path):
    """A process that never imports the toolkit, on the toolkit's own writer. An unread weight would stay random, and
    only the comparison against the release would say so."""
    release, export = tmp_path / "release", tmp_path / "export"
    stored = write_tiny_lightning_checkpoint(release)
    assert any(key.startswith("mtp.") for key in stored), "premise: the release stores a head the export drops"
    model = auto_load_model(str(release), dtype=torch.bfloat16)
    save_full_checkpoint(model, str(export), source_dir=str(release))
    written = json.loads((export / "config.json").read_text())
    assert written == tiny_lightning_config()

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-c", _STOCK_RELOAD, str(export), str(release)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["toolkit_imported"] is False
    assert report["class"] == "NemotronHForCausalLM"
    assert not any(report["problems"].values()), report["problems"]
    assert report["tensors_equal"] and report["logits_equal"]
    assert set(report["layer_types"]) == {"linear_attention", "moe", "full_attention"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
