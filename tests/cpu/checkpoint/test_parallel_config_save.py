#!/usr/bin/env python
"""CPU tests for the config half of every parallel (EP/TP/FSDP2 gathered) save.

The parallel save paths write the weights by hand and therefore never call ``save_pretrained`` — so
everything else ``save_pretrained`` would have emitted has to be reproduced by ``save_model_config``.
Two of those things are load-bearing and go missing silently when they are not written:

* a remote-code model's ``modeling_*.py``, which its own ``auto_map`` names — without it the saved
  directory raises ``OSError: does not appear to have a file named modeling_<x>.py`` for every
  consumer (resume, export, the rollout servers);
* ``model_type``, which ``PretrainedConfig.to_dict`` reads off the CLASS — empty for the vendor
  config classes Bailing/Ling ship, so the key vanishes and every model-type-keyed reader
  downstream (the sharded-EP merge, the hub key renames) sees no family at all.

Run: ``pytest -m cpu tests/cpu/checkpoint/test_parallel_config_save.py``
"""

from __future__ import annotations

import json
import sys

import pytest
from transformers import PretrainedConfig

from src.checkpoint.config_export import save_model_config


class _VendorConfig(PretrainedConfig):
    """A remote-code config exactly as Bailing/Ling ships one: no class-level ``model_type``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _VendorModel:
    """Stands in for a dynamically-loaded model class (``transformers_modules.*``)."""

    def __init__(self, config):
        self.config = config


def _dynamic_model(tmp_path, module_name: str = "modeling_vendor_moe"):
    """A model whose class lives in a ``transformers_modules`` package on disk, as remote code does."""
    package = tmp_path / "transformers_modules" / "vendor"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / f"{module_name}.py").write_text(
        "class VendorForCausalLM:\n    def __init__(self, config):\n        self.config = config\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = __import__(f"transformers_modules.vendor.{module_name}", fromlist=["VendorForCausalLM"])
    finally:
        sys.path.remove(str(tmp_path))

    config = _VendorConfig()
    config.model_type = "vendor_moe"
    config.auto_map = {"AutoModelForCausalLM": f"{module_name}.VendorForCausalLM"}
    return module.VendorForCausalLM(config)


def test_remote_code_modules_travel_with_the_checkpoint(tmp_path):
    """The module the config's ``auto_map`` names must be written beside it.

    Without this the directory is loadable by nothing: ``from_pretrained`` resolves ``auto_map`` and
    raises on the missing file, which lands after a full training run.
    """
    model = _dynamic_model(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()

    save_model_config(model, str(out))

    written = sorted(p.name for p in out.iterdir())
    assert "modeling_vendor_moe.py" in written, f"auto_map names a module the save did not write: {written}"


def test_remote_code_modules_travel_from_an_fsdp2_sharded_model(tmp_path):
    """The same, after FSDP2 has rewritten ``model.__class__`` — the only way these models are saved.

    ``fully_shard`` swaps in a dynamic ``FSDP<Name>`` subclass whose ``__module__`` is torch's, so a
    check on the live class sees no remote code and skips the copy on every sharded run — i.e. exactly
    the runs that produce real checkpoints.
    """
    model = _dynamic_model(tmp_path / "src")
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


def test_model_type_survives_a_vendor_config_with_no_class_attribute(tmp_path):
    """``to_dict`` reads ``model_type`` off the class, which these vendor configs leave empty."""
    model = _dynamic_model(tmp_path / "src")
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
    """Anti-over-reach: the repair must not rewrite a config that serialized its family correctly."""

    class _DeclaredConfig(PretrainedConfig):
        model_type = "declared_moe"

    out = tmp_path / "out"
    out.mkdir()
    save_model_config(_VendorModel(_DeclaredConfig()), str(out))

    assert json.loads((out / "config.json").read_text())["model_type"] == "declared_moe"


def test_a_plain_model_writes_no_remote_code(tmp_path):
    """A first-party model has no ``auto_map``, so nothing extra may be emitted for it."""

    class _PlainConfig(PretrainedConfig):
        model_type = "plain"

    out = tmp_path / "out"
    out.mkdir()
    save_model_config(_VendorModel(_PlainConfig()), str(out))

    assert not [p.name for p in out.iterdir() if p.suffix == ".py"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
