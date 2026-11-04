#!/usr/bin/env python
"""Every save path finalizes ``config.json`` alike: live ``model_type`` (Bailing/Ling), source schema (Step-3.x).
Dropping either ships a directory the merge tools and the pinned server cannot read, with no error at train time."""

import json
import types
from pathlib import Path

import pytest
import torch
from transformers.utils import CONFIG_NAME

from halo.checkpoint.config_export import finalize_exported_config
from halo.models.moe.family_specs import find_moe_family_spec
from halo.trainers.checkpointing import CheckpointingMixin

BAILING = "bailing_moe"
SOURCE_SCHEMA_FAMILY = "step3p7"


def _written_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / CONFIG_NAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_family_less_written_config_gets_the_live_model_type_back(tmp_path):
    """Bailing/Ling declare no class-level ``model_type``, so ``to_dict`` omits it and its readers see no family."""
    path = _written_config(tmp_path, {"hidden_size": 8})

    finalize_exported_config(types.SimpleNamespace(model_type=BAILING), str(tmp_path), source=None)

    assert json.loads(path.read_text())["model_type"] == BAILING


def test_an_existing_model_type_is_left_alone(tmp_path):
    path = _written_config(tmp_path, {"model_type": "qwen3", "hidden_size": 8})

    finalize_exported_config(types.SimpleNamespace(model_type=BAILING), str(tmp_path), source=None)

    assert json.loads(path.read_text())["model_type"] == "qwen3"


def test_a_family_less_live_config_warns_rather_than_writing_a_wrong_family(tmp_path, caplog):
    """A silent pass would leave a checkpoint no tool can key."""
    with caplog.at_level("WARNING"):
        finalize_exported_config(types.SimpleNamespace(model_type=""), str(tmp_path), source=None)

    assert any("model_type" in record.message for record in caplog.records)


def test_the_source_schema_step_engages_only_for_the_families_that_declare_it(tmp_path, caplog):
    """Step-3.x has no config class in the pinned server, so an export without a source is unservable and must warn."""
    assert find_moe_family_spec(SOURCE_SCHEMA_FAMILY).exports_source_config_schema, "the family spec moved"
    assert find_moe_family_spec("qwen3") is None
    _written_config(tmp_path, {"model_type": SOURCE_SCHEMA_FAMILY})

    with caplog.at_level("WARNING"):
        finalize_exported_config(types.SimpleNamespace(model_type=SOURCE_SCHEMA_FAMILY), str(tmp_path), source=None)
    assert any("source" in record.message for record in caplog.records), (
        "a source-schema family exported with no source must warn — its config.json is unservable"
    )

    caplog.clear()
    _written_config(tmp_path, {"model_type": "qwen3"})
    with caplog.at_level("WARNING"):
        finalize_exported_config(types.SimpleNamespace(model_type="qwen3"), str(tmp_path), source=None)
    assert not any("source" in record.message for record in caplog.records)


class _BaseTrainer:
    def __init__(self):
        self.saved_to = None

    def save_model(self, output_dir, _internal_call=False):
        self.saved_to = output_dir
        _written_config(Path(output_dir), {"hidden_size": 8})  # no model_type, as Bailing serializes


class _Trainer(CheckpointingMixin, _BaseTrainer):
    def __init__(self, output_dir):
        super().__init__()
        self.args = types.SimpleNamespace(output_dir=output_dir, should_save=True)
        # A real Module: the base path's save wrappers unwrap it.
        self.model = torch.nn.Module()
        self.model.config = types.SimpleNamespace(model_type=BAILING)
        self.model._name_or_path = ""
        self._pristine_special_token_ids = []
        self.parallelism_config = types.SimpleNamespace(is_pp_mode=False)

    def _top_level_model(self):
        # The mixin reshards FSDP2 modules before writing; a bare Module makes that a no-op here.
        return torch.nn.Module()

    def _checkpoint_context(self):
        return types.SimpleNamespace(model=self.model, tokenizer=None)


def test_the_base_save_path_finalizes_the_config_it_just_wrote(tmp_path, monkeypatch):
    module = "halo.trainers.checkpointing"
    # False = no parallelism strategy claimed the write, so the base Trainer's save runs.
    monkeypatch.setattr(f"{module}.save_checkpoint", lambda ctx, output_dir: False)
    monkeypatch.setattr(f"{module}.fs_aware_makedirs", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(f"{module}.restore_config_fields", lambda ids: None)
    monkeypatch.setattr(f"{module}.save_balancing_biases", lambda *args, **kwargs: None)

    trainer = _Trainer(str(tmp_path))
    trainer.save_model()

    assert trainer.saved_to == str(tmp_path), "the base Trainer's save must still run"
    written = json.loads((tmp_path / CONFIG_NAME).read_text())
    assert written["model_type"] == BAILING, (
        "the base save path shipped a family-less config: the shard merge refuses it and the hub "
        "key renames degrade to 'no renames', with nothing failing at train time"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
