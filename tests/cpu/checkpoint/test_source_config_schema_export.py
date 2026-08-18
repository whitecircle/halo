#!/usr/bin/env python
"""A source-schema-exporting family's ``config.json`` must be the SOURCE repo's schema, not transformers'.

``_EXPORTS_SOURCE_CONFIG_SCHEMA`` (Step-3.7 Flash) marks a family whose serving engines have no
config class of their own and read it only through the source repo's ``auto_map`` modules. Those
modules spell the config in the vendor's own keys, which transformers 5.16 ABSORBS at load
(``Step3p7TextConfig.attribute_map`` plus the ``kwargs.pop`` derivations in ``__post_init__``) and
never re-emits — so a transformers-written config is a schema no pinned engine parses, and nothing
on the save side can invert it. Every toolkit config write therefore carries the source's own
``config.json`` and modules forward with this run's changes applied.

Three properties carry that, and each fails when the export breaks: the written config keeps the
source's spellings and none of the native-only ones, it reloads in the training image's transformers
to exactly the config that was trained, and a value this run changed (a ``patch_vocab.py``
``vocab_size``) reaches it. The roster is derived from the class hierarchy, so a family cannot start
declaring the flag without its own fixture landing here.

Run: ``python tests/cpu/checkpoint/test_source_config_schema_export.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR, save_model_config
from src.checkpoint.tool_io import save_full_checkpoint
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from tests.common.models import TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG

# The tiny stand-in for ``stepfun-ai/Step-3.7-Flash``'s own ``config.json``: the release's key
# spellings at the roster's geometry. Every MoE/attention field is the vendor's, not transformers' —
# ``moe_num_experts``/``moe_top_k``/``num_attention_groups`` are ``attribute_map`` aliases, and
# ``moe_layers_enum``/``attention_other_setting`` are derived away into ``mlp_layer_types`` /
# ``num_sliding_attention_heads`` + ``per_layer_config``. ``_parses_to_the_roster_geometry`` below
# pins it against ``TINY_STEP3P7_CONFIG``, so it cannot drift into a different model.
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

# Key spellings that decide whether the artifact is servable: the source's, which the pinned engines
# read, against the native-only ones transformers writes and they have no class for.
_SOURCE_SPELLINGS = ("moe_num_experts", "moe_top_k", "moe_layers_enum", "attention_other_setting")
_NATIVE_ONLY_SPELLINGS = ("n_routed_experts", "num_experts_per_tok", "mlp_layer_types", "per_layer_config")

# Stand-ins for the release's remote-code modules. The export copies them; it never imports them, and
# the modeling entry names a sibling to pin the transitive copy the ``auto_map`` never mentions.
_SOURCE_MODULES = {
    "configuration_step3p7.py": "class Step3p7Config: pass\n",
    "processing_step3.py": "class Step3VLProcessor: pass\n",
    "modeling_step3p7.py": "from .vision_encoder import VisionEncoder\n\n\nclass Step3p7ForConditionalGeneration: pass\n",
    "vision_encoder.py": "class VisionEncoder: pass\n",
}

# wrapper class -> the source checkpoint a declaring family's export must carry forward
FIXTURES = {EPStep3p7MoELayer: (TINY_STEP3P7_SOURCE_CONFIG, _SOURCE_MODULES)}
_IDS = [cls.__name__ for cls in FIXTURES]


def _write_source(directory, payload: dict, modules: dict[str, str]) -> str:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    for name, body in modules.items():
        with open(os.path.join(directory, name), "w") as handle:
            handle.write(body)
    return str(directory)


def _model_from(source: str):
    """The tiny model as a training run holds it: loaded config, source recorded by the loader."""
    torch.manual_seed(0)
    model = Step3p7ForConditionalGeneration(AutoConfig.from_pretrained(source)).to(torch.bfloat16)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, source)
    return model


def _exported(tmp_path, cls, mutate=None) -> tuple[dict, str, object]:
    source_config, modules = FIXTURES[cls]
    source = _write_source(tmp_path / "source", source_config, modules)
    export = str(tmp_path / "export")
    os.makedirs(export, exist_ok=True)
    model = _model_from(source)
    if mutate is not None:
        mutate(model)
    save_model_config(model, export)
    with open(os.path.join(export, "config.json")) as handle:
        return json.load(handle), export, model


def test_every_declaring_family_has_a_source_schema_fixture():
    declaring = {cls for cls in ep_layer_classes() if cls._EXPORTS_SOURCE_CONFIG_SCHEMA}
    assert declaring == set(FIXTURES), (
        f"{sorted(c.__name__ for c in declaring ^ set(FIXTURES))}: a family declaring "
        f"_EXPORTS_SOURCE_CONFIG_SCHEMA must pin its carried export schema here"
    )


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_source_config_parses_to_the_roster_geometry(cls):
    """The stand-in source must be the roster model in the vendor's spellings — otherwise every
    assertion below is about a different config than the rest of the suite trains."""
    source_config, _modules = FIXTURES[cls]
    parsed = Step3p7Config.from_dict(json.loads(json.dumps(source_config))).to_dict()
    native = Step3p7Config(
        text_config=dict(TINY_STEP3P7_CONFIG),
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=source_config["image_token_id"],
    ).to_dict()
    for section in ("text_config", "vision_config"):
        assert parsed[section] == native[section], f"the {section} stand-in drifted from the roster config"


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_export_carries_the_source_schema_and_its_modules(cls, tmp_path):
    payload, export, _model = _exported(tmp_path, cls)
    text = payload["text_config"]
    assert all(key in text for key in _SOURCE_SPELLINGS), (
        f"the export dropped the source spellings the pinned engines read: "
        f"{[key for key in _SOURCE_SPELLINGS if key not in text]}"
    )
    assert not [key for key in _NATIVE_ONLY_SPELLINGS if key in text], (
        f"the export kept transformers-only spellings the pinned engines have no class for: "
        f"{[key for key in _NATIVE_ONLY_SPELLINGS if key in text]}"
    )
    assert payload["auto_map"] == FIXTURES[cls][0]["auto_map"]
    missing = [name for name in FIXTURES[cls][1] if not os.path.isfile(os.path.join(export, name))]
    assert not missing, f"the export ships no {missing} — the directory does not load without them"


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_export_reloads_to_the_config_that_was_trained(cls, tmp_path):
    """The carry must cost nothing on the training side: this transformers accepts the source's
    spellings (that is how it loads the release), so the export reloads to the live config exactly."""
    _payload, export, model = _exported(tmp_path, cls)
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_a_patched_vocab_size_reaches_the_carried_config(cls, tmp_path):
    """``patch_vocab.py``'s one config-visible effect. Carried verbatim, it would serve the source's
    vocabulary against a resized embedding — the checkpoint-wide off-by-N no reader can detect."""
    patched = TINY_STEP3P7_CONFIG["vocab_size"] + 128

    def resize(model):
        model.resize_token_embeddings(patched)

    payload, export, model = _exported(tmp_path, cls, mutate=resize)
    assert all(key in payload["text_config"] for key in _SOURCE_SPELLINGS), "not the carried schema"
    assert payload["text_config"]["vocab_size"] == patched
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_the_standalone_writer_carries_the_schema_too(cls, tmp_path):
    """``save_full_checkpoint`` is the other config writer — the one ``patch_vocab.py`` and the
    after-training tools use. An export servable only when it came from a parallel save is half a
    path: the vocab patch is exactly where a Step-3.7 run starts."""
    source_config, modules = FIXTURES[cls]
    source = _write_source(tmp_path / "source", source_config, modules)
    export = str(tmp_path / "export")
    patched = TINY_STEP3P7_CONFIG["vocab_size"] + 128
    model = _model_from(source)
    model.resize_token_embeddings(patched)

    save_full_checkpoint(model, export, source_dir=source)

    with open(os.path.join(export, "config.json")) as handle:
        payload = json.load(handle)
    assert all(key in payload["text_config"] for key in _SOURCE_SPELLINGS)
    assert payload["text_config"]["vocab_size"] == patched
    assert AutoConfig.from_pretrained(export).to_diff_dict() == model.config.to_diff_dict()


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_a_change_the_source_schema_swallows_is_refused(cls, tmp_path):
    """The carry is only sound while the rewritten config re-parses to the live values. A field the
    source spells through a legacy key the walk cannot update parses back to the SOURCE's value, and
    the served model would then run a geometry the trainer never had — so the export refuses."""

    def widen_sliding_heads(model):
        model.config.text_config.num_sliding_attention_heads = TINY_STEP3P7_CONFIG["num_attention_heads"]

    with pytest.raises(ValueError, match="cannot express this run's config"):
        _exported(tmp_path, cls, mutate=widen_sliding_heads)


@pytest.mark.parametrize("cls", list(FIXTURES), ids=_IDS)
def test_a_source_without_remote_code_leaves_the_native_schema(cls, tmp_path, caplog):
    """A source that was never servable has no schema to hand on; the export says so and writes
    transformers' own rather than inventing one."""
    source_config = {k: v for k, v in FIXTURES[cls][0].items() if k != "auto_map"}
    source = _write_source(tmp_path / "source", source_config, {})
    export = str(tmp_path / "export")
    os.makedirs(export, exist_ok=True)
    with caplog.at_level("WARNING"):
        save_model_config(_model_from(source), export)
    with open(os.path.join(export, "config.json")) as handle:
        payload = json.load(handle)
    assert "auto_map" not in payload
    assert "n_routed_experts" in payload["text_config"]
    assert "declares no auto_map" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
