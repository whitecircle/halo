"""reattach_vision_tower rebuilds the servable wrapper layout from a text-only export, on a tiny real Qwen3.5-MoE
composite."""

import json
import logging
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForImageTextToText, Glm5NextConfig, Glm5NextForConditionalGeneration
from transformers.models.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

from halo.checkpoint import config_export
from halo.checkpoint.tools import vision_reattach
from halo.checkpoint.tools.offline_export import save_full_checkpoint
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.model_loading.model_preparation import auto_load_model
from halo.models.moe.family_specs import NEMOTRON_H, CompositeTextLayout, MoEFamilySpec
from halo.models.mtp.spec import find_mtp_head_spec
from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, TINY_UNWRAPPED_COMPOSITE_FAMILIES, tiny_family_model

_COMPOSITES = {**TINY_UNWRAPPED_COMPOSITE_FAMILIES, "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}
# Composites that keep their text tower under ``language_model.`` whole, with what each declares.
_DECLARED_LAYOUTS = {
    "minimax_m3_vl": CompositeTextLayout(
        composite_tower="language_model.model.", composite_head="language_model.lm_head."
    ),
    "nemotron_h_omni": CompositeTextLayout(
        tower="backbone.",
        composite_tower="language_model.backbone.",
        composite_head="language_model.lm_head.",
        composite_mtp_tail="language_model.mtp.",
    ),
}


def _tiny_composite_config() -> Qwen3_5MoeConfig:
    text = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "vocab_size": 64,
        "max_position_embeddings": 64,
    }
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 32}
    return Qwen3_5MoeConfig(text_config=text, vision_config=vision)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("reattach")
    base_dir, export_dir = str(root / "base"), str(root / "export")
    composite = _tiny_composite_config()
    torch.manual_seed(0)
    Qwen3_5MoeForConditionalGeneration(composite).save_pretrained(base_dir)

    export = Qwen3_5MoeForCausalLM(composite.text_config)
    # A value the base cannot contain, proving the text weights come from the export.
    with torch.no_grad():
        export.model.embed_tokens.weight.fill_(3.5)
    export.save_pretrained(export_dir)
    return base_dir, export_dir


def _weight_map(directory: str) -> dict[str, str]:
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f)["weight_map"]

    with safe_open(os.path.join(directory, "model.safetensors"), framework="pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")


def test_reattach_rebuilds_wrapper_layout(artifacts, tmp_path):
    base_dir, export_dir = artifacts
    out = str(tmp_path / "out")
    reattach_vision_tower(export_dir, base_dir, out, trust_remote_code=False)

    weight_map = _weight_map(out)
    keys = set(weight_map)
    assert any(k.startswith("model.language_model.") for k in keys), sorted(keys)[:5]
    assert any(k.startswith("model.visual.") for k in keys), "vision tower not carried over"
    assert not any(k.startswith("model.embed_tokens") for k in keys), "text keys left un-prefixed"

    config = AutoConfig.from_pretrained(out)
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert getattr(config, "vision_config", None) is not None
    assert config.text_config.vocab_size == 64

    embed_key = "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(out, weight_map[embed_key]), framework="pt") as f:
        assert torch.all(f.get_tensor(embed_key) == 3.5), "trained export weights were not the ones written"


def test_wrapper_layout_input_is_refused(artifacts, tmp_path):
    base_dir, _ = artifacts
    with pytest.raises(ValueError, match="not a text-only export"):
        reattach_vision_tower(base_dir, base_dir, str(tmp_path / "out2"), trust_remote_code=False)


def _relabelled_export(export_dir: str, target: str, relabel) -> str:
    """With every key passed through ``relabel``."""
    os.makedirs(target)
    tensors = load_file(os.path.join(export_dir, "model.safetensors"))
    save_file({relabel(key): value for key, value in tensors.items()}, os.path.join(target, "model.safetensors"))
    AutoConfig.from_pretrained(export_dir).save_pretrained(target)
    return target


@pytest.mark.parametrize(
    ("relabel", "match"),
    [
        (lambda key: key.replace("model.", "model.language_model.", 1), "would nest the wrapper prefix"),
        (
            lambda key: key.replace("model.norm.", "model.visual.norm.", 1),
            "address no module of the base's text tower",
        ),
    ],
    ids=["wrapper_prefixed", "stray_module"],
)
def test_an_export_whose_keys_do_not_match_its_config_is_refused(artifacts, tmp_path, relabel, match):
    """Re-prefixed onto no slot of the wrapper, the served checkpoint reloads its text tower randomly initialized."""
    base_dir, export_dir = artifacts
    relabelled = _relabelled_export(export_dir, str(tmp_path / "relabelled"), relabel)
    out = tmp_path / "out5"
    with pytest.raises(ValueError, match=match):
        reattach_vision_tower(relabelled, base_dir, str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


def test_a_base_storing_its_text_tower_under_another_prefix_is_refused(artifacts, tmp_path):
    """A base keeping a vendor namespace has nothing under ``model.language_model.``, so two text towers would
    collide on load."""
    base_dir, export_dir = artifacts
    vendor_base = tmp_path / "vendor_base"
    vendor_base.mkdir()
    tensors = load_file(os.path.join(base_dir, "model.safetensors"))
    save_file(
        {key.replace("model.language_model.", "model.llm.", 1): value for key, value in tensors.items()},
        str(vendor_base / "model.safetensors"),
        metadata={"format": "pt"},
    )
    AutoConfig.from_pretrained(base_dir).save_pretrained(vendor_base)
    out = tmp_path / "out4"
    with pytest.raises(ValueError, match="stores no text tower under model.language_model"):
        reattach_vision_tower(export_dir, str(vendor_base), str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


def test_an_output_aimed_at_the_base_is_refused(artifacts):
    """The writer's close sweeps every ``model*.safetensors`` it did not write, replacing the base."""
    base_dir, export_dir = artifacts
    before = sorted(os.listdir(base_dir))
    with pytest.raises(ValueError, match="same path"):
        reattach_vision_tower(export_dir, base_dir, base_dir, trust_remote_code=False)
    assert sorted(os.listdir(base_dir)) == before, "a refused re-attachment touched the base directory"


_BASE_TAIL = {"mtp.fc.weight": 1.5, "mtp.layers.0.mlp.gate.weight": 1.5, "mtp.norm.weight": 1.5}


def _with_tail(source: str, target, tail: dict[str, float]) -> str:
    """``source``'s tensors and config beside an MTP tail filled with each key's value."""
    target.mkdir()
    tensors = load_file(os.path.join(source, "model.safetensors"))
    tensors.update({key: torch.full((2, 2), value) for key, value in tail.items()})
    save_file(tensors, str(target / "model.safetensors"), metadata={"format": "pt"})
    AutoConfig.from_pretrained(source).save_pretrained(target)
    return str(target)


def _head_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "MTP head" in record.getMessage()]


def test_a_base_head_carried_beside_a_trained_trunk_warns_with_its_count_and_stays_byte_equal(
    artifacts, tmp_path, caplog
):
    base_dir, export_dir = artifacts
    base = _with_tail(base_dir, tmp_path / "base_tail", _BASE_TAIL)
    out = str(tmp_path / "out")
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(export_dir, base, out, trust_remote_code=False)
    written = load_file(os.path.join(out, "model.safetensors"))
    for key, value in _BASE_TAIL.items():
        assert torch.equal(written[key], torch.full((2, 2), value)), key
    warned = _head_warnings(caplog)
    assert len(warned) == 1 and f"{len(_BASE_TAIL)} tensor(s)" in warned[0], warned
    assert "mtp_train_scope: head_only" in warned[0], warned


def test_a_base_with_no_tail_reattaches_silently(artifacts, tmp_path, caplog):
    base_dir, export_dir = artifacts
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(export_dir, base_dir, str(tmp_path / "out"), trust_remote_code=False)
    assert not _head_warnings(caplog)


def test_a_head_the_export_trained_supersedes_the_bases_whole(artifacts, tmp_path, caplog):
    """The export's own head is the one to serve; a base-only tail key beside it would reload as a mixed head."""
    base_dir, export_dir = artifacts
    base = _with_tail(base_dir, tmp_path / "base_tail", {**_BASE_TAIL, "mtp.layers.0.mlp.shared_expert.weight": 1.5})
    trained = _with_tail(export_dir, tmp_path / "trained", {key: 2.5 for key in _BASE_TAIL})
    out = str(tmp_path / "out")
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(trained, base, out, trust_remote_code=False)
    written = load_file(os.path.join(out, "model.safetensors"))
    assert {key for key in written if key.startswith("mtp.")} == set(_BASE_TAIL)
    for key in _BASE_TAIL:
        assert torch.equal(written[key], torch.full((2, 2), 2.5)), key
    assert any(key.startswith("model.visual.") for key in written), "the vision tower still comes from the base"
    assert not _head_warnings(caplog), "a head the export trained is not stale"


def test_a_trained_head_of_another_width_than_the_bases_is_refused(artifacts, tmp_path):
    """A trained tail passes the key check and still meets the shape check."""
    base_dir, export_dir = artifacts
    base = _with_tail(base_dir, tmp_path / "base_tail", _BASE_TAIL)
    trained = _with_tail(export_dir, tmp_path / "trained", {key: 2.5 for key in _BASE_TAIL})
    tensors = load_file(os.path.join(trained, "model.safetensors"))
    tensors["mtp.fc.weight"] = torch.zeros(3, 3)
    save_file(tensors, os.path.join(trained, "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="another model of this family"):
        reattach_vision_tower(trained, base, str(tmp_path / "out"), trust_remote_code=False)


def test_a_stray_key_beside_a_trained_head_is_still_refused(artifacts, tmp_path):
    """Only what the wrapper class names as a tail joins the text layout."""
    base_dir, export_dir = artifacts
    trained = _with_tail(export_dir, tmp_path / "trained", {"mtp.fc.weight": 2.5, "draft.fc.weight": 2.5})
    with pytest.raises(ValueError, match="address no module of the base's text tower"):
        reattach_vision_tower(trained, base_dir, str(tmp_path / "out"), trust_remote_code=False)


def test_a_base_of_another_family_is_refused(artifacts, tmp_path):
    """The graft is a plain attribute assignment, so a mismatched tower loads and serves garbage."""
    _, export_dir = artifacts
    other_base = str(tmp_path / "glm5_base")
    Glm5NextForConditionalGeneration(
        Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG))
    ).save_pretrained(other_base)
    out = tmp_path / "out3"
    with pytest.raises(ValueError, match="wraps a 'glm5_next_text' text tower"):
        reattach_vision_tower(export_dir, other_base, str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


@pytest.fixture(params=sorted(_DECLARED_LAYOUTS))
def declared_composite(request, tmp_path, monkeypatch):
    """A composite saved in its hub spelling, with a tail no text tower owns, and its text-only export: the same
    tensors at the root, a marker on the head, under the text sub-config. The spec is claimed for this test alone."""
    name = request.param
    torch.manual_seed(0)
    composite = tiny_family_model(_COMPOSITES[name])
    spec = MoEFamilySpec(
        model_types=(composite.config.model_type, composite.config.text_config.model_type),
        text_only_layout=_DECLARED_LAYOUTS[name],
    )

    def claim(model_type):
        return spec if model_type in spec.model_types else None

    monkeypatch.setattr(vision_reattach, "find_moe_family_spec", claim)
    monkeypatch.setattr(config_export, "find_moe_family_spec", claim)

    base_dir, export_dir = tmp_path / "base", tmp_path / "export"
    composite.save_pretrained(base_dir)
    tensors = load_file(base_dir / "model.safetensors")
    tensors["language_model.mtp.layers.0.norm.weight"] = torch.ones(2)
    save_file(tensors, base_dir / "model.safetensors", metadata={"format": "pt"})

    trained = {
        key[len("language_model.") :]: value.clone()
        for key, value in tensors.items()
        if key.startswith(("language_model.backbone.", "language_model.model.", "language_model.lm_head."))
    }
    trained["lm_head.weight"].fill_(3.5)
    export_dir.mkdir()
    save_file(trained, export_dir / "model.safetensors", metadata={"format": "pt"})
    composite.config.text_config.save_pretrained(export_dir)
    return name, composite, tensors, str(base_dir), str(export_dir)


def test_a_declared_layout_restores_the_hub_spelling(declared_composite, tmp_path):
    name, _composite, base_tensors, base_dir, export_dir = declared_composite
    out = tmp_path / "out"
    reattach_vision_tower(export_dir, base_dir, str(out), trust_remote_code=False)
    written = load_file(out / "model.safetensors")
    assert set(written) == set(base_tensors), sorted(set(written) ^ set(base_tensors))[:8]
    assert torch.all(written["language_model.lm_head.weight"] == 3.5), "the trained head was not the one written"
    assert "language_model.mtp.layers.0.norm.weight" in written, "a tail no text tower owns is carried from the base"
    reloaded = AutoConfig.from_pretrained(out)
    assert reloaded.model_type == name and reloaded.vision_config is not None


def test_a_carried_heads_warning_offers_a_fit_only_on_a_family_with_a_spec(declared_composite, tmp_path, caplog):
    """``mtp_loss_weight`` refuses a family with no spec, so the warning cannot send a reader to it."""
    _name, composite, _base_tensors, base_dir, export_dir = declared_composite
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(export_dir, base_dir, str(tmp_path / "out"), trust_remote_code=False)
    warned = _head_warnings(caplog)
    assert len(warned) == 1 and "1 tensor(s)" in warned[0], warned
    assert ("mtp_train_scope" in warned[0]) == (find_mtp_head_spec(composite.config) is not None), warned


def test_only_some_declared_layouts_have_a_head_to_fit():
    """Both sides of the check above stay in the sweep."""
    specs = {name: find_mtp_head_spec(tiny_family_model(_COMPOSITES[name]).config) for name in _DECLARED_LAYOUTS}
    assert specs["nemotron_h_omni"] is not None and specs["minimax_m3_vl"] is None


def test_a_declared_layout_reloads_as_the_composite_with_the_trained_head(declared_composite, tmp_path):
    name, composite, _base_tensors, base_dir, export_dir = declared_composite
    out = tmp_path / "out"
    reattach_vision_tower(export_dir, base_dir, str(out), trust_remote_code=False)
    reloaded = _COMPOSITES[name].load_class.from_pretrained(out, dtype=torch.float32)
    assert type(reloaded) is type(composite)
    # Nemotron's composite exposes no output embedding of its own.
    head = reloaded.get_output_embeddings() or reloaded.language_model.get_output_embeddings()
    assert torch.all(head.weight == 3.5)
    assert torch.equal(reloaded.get_input_embeddings().weight, composite.get_input_embeddings().weight)


def test_the_roster_declares_nemotrons_layout():
    assert NEMOTRON_H.text_only_layout == _DECLARED_LAYOUTS["nemotron_h_omni"]


def test_a_text_only_load_exports_and_reattaches_to_the_release_spelling(tmp_path):
    """The whole path on Nemotron's real spec and conversions: the composite's level dropped, the head trained, a
    text-only export in the tower's spelling, then the composite again."""
    torch.manual_seed(0)
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])
    base_dir, export_dir, out = str(tmp_path / "base"), str(tmp_path / "export"), str(tmp_path / "out")
    composite.save_pretrained(base_dir)
    text_only = auto_load_model(base_dir, text_only=True, dtype=torch.float32)
    assert type(text_only).__name__ == "NemotronHForCausalLM"
    assert torch.equal(text_only.model.embeddings.weight, composite.language_model.model.embeddings.weight)
    with torch.no_grad():
        text_only.lm_head.weight.fill_(3.5)
    save_full_checkpoint(text_only, export_dir, source_dir=base_dir)
    exported = load_file(os.path.join(export_dir, "model.safetensors"))
    assert {key.split(".")[0] for key in exported} == {"backbone", "lm_head"}, sorted(exported)[:6]
    assert AutoConfig.from_pretrained(export_dir).model_type == "nemotron_h"

    reattach_vision_tower(export_dir, base_dir, out, trust_remote_code=False)

    assert set(load_file(os.path.join(out, "model.safetensors"))) == set(
        load_file(os.path.join(base_dir, "model.safetensors"))
    )
    reloaded = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32)
    assert torch.all(reloaded.language_model.lm_head.weight == 3.5)
    for key, value in composite.state_dict().items():
        if key != "language_model.lm_head.weight":
            assert torch.equal(reloaded.state_dict()[key], value), key


def test_the_graft_lands_in_the_attribute_holding_the_text_config(artifacts, monkeypatch):
    """A release's remote config keeps it under its own name, with ``text_config`` a read-only view of that."""

    class _RemoteConfig(Qwen3_5MoeConfig):
        @property
        def text_config(self):
            return self.llm_config

        @text_config.setter
        def text_config(self, value):
            raise AttributeError("read-only")

        def get_text_config(self, decoder=None, encoder=None):
            return self.llm_config

    base_dir, export_dir = artifacts
    remote = object.__new__(_RemoteConfig)
    remote.__dict__.update(AutoConfig.from_pretrained(base_dir).__dict__)
    remote.__dict__["llm_config"] = remote.__dict__.pop("text_config")
    monkeypatch.setattr(vision_reattach.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: remote)
    trained = Qwen3_5MoeConfig(text_config=_tiny_composite_config().text_config).text_config
    assert vision_reattach._grafted_composite_config(base_dir, trained, True) is remote
    assert remote.llm_config is trained


def test_an_undeclared_composite_of_that_layout_is_refused(declared_composite, tmp_path, monkeypatch):
    """The default prefixes address nothing in such a base, so its keys would carry over beside the export's."""
    _, _composite, _tensors, base_dir, export_dir = declared_composite
    monkeypatch.setattr(
        vision_reattach,
        "find_moe_family_spec",
        lambda model_type: MoEFamilySpec(model_types=(model_type,), text_only_layout=None)
        if model_type == "nemotron_h"
        else None,
    )
    with pytest.raises(ValueError, match="not a text-only export|stores no text tower"):
        reattach_vision_tower(export_dir, base_dir, str(tmp_path / "out"), trust_remote_code=False)


def test_an_export_of_another_tower_spelling_is_refused(declared_composite, tmp_path):
    """Keys the declared tower prefix does not address land on no slot of the composite."""
    _, _composite, _tensors, base_dir, export_dir = declared_composite
    tensors = load_file(os.path.join(export_dir, "model.safetensors"))
    tensors["transformer.stray.weight"] = torch.zeros(2)
    save_file(tensors, os.path.join(export_dir, "model.safetensors"), metadata={"format": "pt"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="address no module of the base's text tower"):
        reattach_vision_tower(export_dir, base_dir, str(out), trust_remote_code=False)
    assert not out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
