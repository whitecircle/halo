#!/usr/bin/env python
"""CPU test: every save path finalizes ``config.json`` the same way.

The EP/CP/TP writers run three rewrites transformers does not — the live ``model_type`` restored for
the vendor classes that declare none (Bailing/Ling), the flat legacy per-layer keys, and the source
repo's own schema for the families whose serving engines have no config class (Step-3.x). The base
(single-GPU / DDP / accelerate-FSDP) save path used to run only the middle one, so the same model
trained without parallelism shipped a directory the merge tools and the pinned server cannot read —
and nothing at train time says so. These tests fail if that path drops any of the three again.

    python tests/cpu/trainers/test_base_save_config_finalization.py
"""

import json
import types
from pathlib import Path

import pytest
import torch

import src.distributed.expert_parallel.layers.roster  # noqa: F401  — registers the source-schema families
from src.checkpoint.config_export import CONFIG_NAME, finalize_exported_config
from src.models.moe_balancing import exports_source_config_schema
from src.trainers.mixins.checkpointing import CheckpointingMixin

BAILING = "bailing_moe"
SOURCE_SCHEMA_FAMILY = "step3p7"


def _written_config(tmp_path: Path, payload: dict) -> Path:
    """What ``save_pretrained`` leaves behind — the state the finalizer has to repair."""
    path = tmp_path / CONFIG_NAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_family_less_written_config_gets_the_live_model_type_back(tmp_path):
    """Bailing/Ling declare no class-level ``model_type``, so ``to_dict`` omits it entirely and every
    model-type-keyed reader downstream (shard merge, hub renames) sees no family."""
    path = _written_config(tmp_path, {"hidden_size": 8})

    finalize_exported_config(types.SimpleNamespace(model_type=BAILING), str(tmp_path), source=None)

    assert json.loads(path.read_text())["model_type"] == BAILING


def test_an_existing_model_type_is_left_alone(tmp_path):
    """Anti-overreach: the restore must not overwrite a family transformers did serialize."""
    path = _written_config(tmp_path, {"model_type": "qwen3", "hidden_size": 8})

    finalize_exported_config(types.SimpleNamespace(model_type=BAILING), str(tmp_path), source=None)

    assert json.loads(path.read_text())["model_type"] == "qwen3"


def test_a_family_less_live_config_warns_rather_than_writing_a_wrong_family(caplog):
    """Nothing correct to write — but a silent pass would leave a checkpoint no tool can key."""
    with caplog.at_level("WARNING"):
        finalize_exported_config(types.SimpleNamespace(model_type=""), "/nonexistent", source=None)

    assert any("model_type" in record.message for record in caplog.records)


def test_the_source_schema_step_engages_only_for_the_families_that_declare_it(tmp_path, caplog):
    """Step-3.x has no config class in the pinned server; without a source to carry, the export is
    unservable and must say so. Every other family must not be touched by that step at all."""
    assert exports_source_config_schema(SOURCE_SCHEMA_FAMILY), "the family registry moved"
    assert not exports_source_config_schema("qwen3")
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
    """Stands in for ``transformers.Trainer``: writes the config the way its ``save_model`` does."""

    def __init__(self):
        self.saved_to = None

    def save_model(self, output_dir, _internal_call=False):
        self.saved_to = output_dir
        _written_config(Path(output_dir), {"hidden_size": 8})  # no model_type, as Bailing serializes


class _Trainer(CheckpointingMixin, _BaseTrainer):
    """The real mixin over a fake base — the base branch of ``save_model`` is what is under test."""

    def __init__(self, output_dir):
        super().__init__()
        self.args = types.SimpleNamespace(output_dir=output_dir, should_save=True)
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(model_type=BAILING), _name_or_path="")
        self._pristine_special_token_ids = []

    def _persist_router_balancing_biases(self, output_dir):
        pass

    def _top_level_model(self):
        # The mixin reshards FSDP2 modules before writing; a bare Module makes that a no-op here.
        return torch.nn.Module()

    def _checkpoint_context(self):
        return types.SimpleNamespace(
            model=self.model,
            tokenizer=None,
            has_expert_lora=False,
            merge_expert_lora_on_save=False,
            accelerate_manages_fsdp=False,
        )


def test_the_base_save_path_finalizes_the_config_it_just_wrote(tmp_path, monkeypatch):
    """The wiring, end to end: no parallelism strategy claims the save, so the base Trainer writes —
    and the mixin owes that directory the same three rewrites the parallel writers apply."""
    module = "src.trainers.mixins.checkpointing"
    monkeypatch.setattr(f"{module}.find_peft_model", lambda model: None)
    # False = no parallelism strategy claimed the write, so the base Trainer's save runs.
    monkeypatch.setattr(f"{module}.save_checkpoint", lambda ctx, output_dir: False)
    monkeypatch.setattr(f"{module}.fs_aware_makedirs", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(f"{module}.restore_special_token_ids", lambda ids: None)

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
