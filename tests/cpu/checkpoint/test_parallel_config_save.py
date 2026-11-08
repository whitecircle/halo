#!/usr/bin/env python
"""The config half of every parallel gathered save, which never calls ``save_pretrained``. A missing
``modeling_*.py`` makes every consumer raise; ``to_dict`` reads an empty ``model_type`` on Bailing/Ling."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
from transformers import AutoConfig, AutoModelForCausalLM, LagunaConfig, PretrainedConfig

from halo.checkpoint.config_export import drop_unshipped_auto_map, finalize_exported_config, save_model_config
from halo.checkpoint.weights_source import config_schema_source
from halo.models.introspection.model_classes import REMOTE_CODE_MODEL_CLASS_ATTR
from tests.common.models import TINY_LAGUNA_CONFIG

# What ``poolside/Laguna-S-2.1``'s config.json declares beside the in-library ``laguna`` model type.
_LAGUNA_HUB_AUTO_MAP = {
    "AutoConfig": "configuration_laguna.LagunaConfig",
    "AutoModelForCausalLM": "modeling_laguna.LagunaForCausalLM",
}


class _VendorConfig(PretrainedConfig):
    """No class-level ``model_type``, as Bailing/Ling ship."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _VendorModel:
    """Stands in for a dynamically-loaded model class (``transformers_modules.*``)."""

    def __init__(self, config):
        self.config = config


def _dynamic_model(tmp_path, monkeypatch, module_name: str = "modeling_vendor_moe"):
    """Only that module is bound in ``sys.modules``: a stand-in package would break every later remote-code load."""
    tmp_path.mkdir(parents=True)
    source = tmp_path / f"{module_name}.py"
    source.write_text("class VendorForCausalLM:\n    def __init__(self, config):\n        self.config = config\n")
    spec = importlib.util.spec_from_file_location(f"transformers_modules.vendor.{module_name}", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    config = _VendorConfig()
    config.model_type = "vendor_moe"
    config.auto_map = {"AutoModelForCausalLM": f"{module_name}.VendorForCausalLM"}
    return module.VendorForCausalLM(config)


def test_remote_code_modules_travel_with_the_checkpoint(tmp_path, monkeypatch):
    """``from_pretrained`` resolves ``auto_map`` and raises on the missing file, after a full training run."""
    model = _dynamic_model(tmp_path / "src", monkeypatch)
    out = tmp_path / "out"
    out.mkdir()

    save_model_config(model, str(out))

    written = sorted(p.name for p in out.iterdir())
    assert "modeling_vendor_moe.py" in written, f"auto_map names a module the save did not write: {written}"
    assert json.loads((out / "config.json").read_text())["auto_map"] == {
        "AutoModelForCausalLM": "modeling_vendor_moe.VendorForCausalLM"
    }, "the entry naming a shipped module must survive the export"


def test_remote_code_modules_travel_from_an_fsdp2_sharded_model(tmp_path, monkeypatch):
    """``fully_shard`` swaps in a ``FSDP<Name>`` subclass with torch's ``__module__``, so a live-class check skips
    the copy on every sharded run."""
    model = _dynamic_model(tmp_path / "src", monkeypatch)
    original_cls = type(model)
    # How torch.distributed.fsdp._fully_shard._fsdp_init installs its subclass.
    model.__class__ = type(f"FSDP{original_cls.__name__}", (original_cls,), {})
    assert not type(model).__module__.startswith("transformers_modules"), (
        "fixture no longer reproduces the FSDP2 class swap that hides the defining module"
    )
    out = tmp_path / "out"
    out.mkdir()

    save_model_config(model, str(out))

    written = sorted(p.name for p in out.iterdir())
    assert "modeling_vendor_moe.py" in written, f"auto_map names a module the save did not write: {written}"


def test_remote_code_modules_travel_from_a_stamped_stand_in(tmp_path, monkeypatch):
    """A pipeline stage is no remote-code class itself; the class stamped at construction is what is copied."""
    source = _dynamic_model(tmp_path / "src", monkeypatch)
    stage = _VendorModel(source.config)
    setattr(stage, REMOTE_CODE_MODEL_CLASS_ATTR, type(source))
    out = tmp_path / "out"
    out.mkdir()

    save_model_config(stage, str(out))

    written = sorted(p.name for p in out.iterdir())
    assert "modeling_vendor_moe.py" in written, f"the stamped remote-code module was not written: {written}"


def test_model_type_survives_a_vendor_config_with_no_class_attribute(tmp_path, monkeypatch):
    model = _dynamic_model(tmp_path / "src", monkeypatch)
    assert type(model.config).model_type == "", "fixture no longer reproduces the empty class attribute"
    out = tmp_path / "out"
    out.mkdir()

    save_model_config(model, str(out))

    payload = json.loads((out / "config.json").read_text())
    assert payload.get("model_type") == "vendor_moe", (
        f"config.json carries no family ({payload.get('model_type')!r}); the shard merge and the hub "
        f"key renames both key on it and would silently see nothing"
    )


def test_a_declared_model_type_is_left_alone(tmp_path):
    class _DeclaredConfig(PretrainedConfig):
        model_type = "declared_moe"

    out = tmp_path / "out"
    out.mkdir()
    save_model_config(_VendorModel(_DeclaredConfig()), str(out))

    assert json.loads((out / "config.json").read_text())["model_type"] == "declared_moe"


def test_a_plain_model_writes_no_remote_code(tmp_path):
    class _PlainConfig(PretrainedConfig):
        model_type = "plain"

    out = tmp_path / "out"
    out.mkdir()
    save_model_config(_VendorModel(_PlainConfig()), str(out))

    assert not [p.name for p in out.iterdir() if p.suffix == ".py"]


def _in_library_laguna(source_dir):
    """Without ``trust_remote_code``: in-library classes, off a config.json whose ``auto_map`` names the release's
    remote modules."""
    source_dir.mkdir()
    payload = LagunaConfig(**TINY_LAGUNA_CONFIG).to_dict() | {"auto_map": _LAGUNA_HUB_AUTO_MAP}
    (source_dir / "config.json").write_text(json.dumps(payload))
    config = AutoConfig.from_pretrained(str(source_dir))
    assert type(config) is LagunaConfig and config.auto_map == _LAGUNA_HUB_AUTO_MAP, (
        "fixture no longer reproduces an in-library load that keeps the hub auto_map"
    )
    return AutoModelForCausalLM.from_config(config)


def _save_pretrained_export(model, output_dir: str) -> None:
    """The single-GPU / tool path."""
    model.save_pretrained(output_dir)
    finalize_exported_config(model.config, output_dir, source=config_schema_source(model))


@pytest.mark.parametrize("export", [save_model_config, _save_pretrained_export], ids=["parallel", "save_pretrained"])
def test_an_in_library_load_exports_no_auto_map_naming_unshipped_modules(tmp_path, export):
    """transformers serializes the hub ``auto_map`` without shipping the modules, so a ``trust_remote_code`` reload
    raises."""
    model = _in_library_laguna(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()

    export(model, str(out))

    assert "auto_map" not in json.loads((out / "config.json").read_text())
    assert type(AutoConfig.from_pretrained(str(out), trust_remote_code=True)) is LagunaConfig


def test_only_the_unshipped_auto_map_entries_are_dropped(tmp_path):
    """Only the dangling entry goes."""
    (tmp_path / "modeling_vendor.py").write_text("")
    (tmp_path / "tokenization_vendor.py").write_text("")
    kept = {
        "AutoModelForCausalLM": "modeling_vendor.VendorForCausalLM",
        "AutoModel": "org/vendor-1.0--modeling_shared.SharedModel",
        "AutoTokenizer": ["tokenization_vendor.VendorTokenizer", None],
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "vendor", "auto_map": kept | {"AutoConfig": "configuration_vendor.VendorConfig"}})
    )

    drop_unshipped_auto_map(str(tmp_path))

    assert json.loads((tmp_path / "config.json").read_text())["auto_map"] == kept


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
