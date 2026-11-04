"""The adapter writer's ``.bin`` fallback must remove a leftover safetensors file, which every reader prefers.
A leftover would shadow the state just saved, so a resume or merge reads the previous adapter."""

import os

import pytest
import torch
from safetensors.torch import load_file

import halo.checkpoint.adapters as adapters_mod
from halo.checkpoint.format import ADAPTER_BIN_FILE, ADAPTER_SAFETENSORS_FILE


def _failing_save(*args, **kwargs):
    raise OSError("No space left on device")


def test_bin_fallback_removes_the_stale_safetensors(tmp_path, monkeypatch):
    stale = tmp_path / ADAPTER_SAFETENSORS_FILE
    stale.write_bytes(b"previous save's adapter")
    monkeypatch.setattr(adapters_mod, "save_safetensors_file", _failing_save)

    adapters_mod.write_adapter_file({"w": torch.ones(2)}, str(tmp_path))

    assert (tmp_path / ADAPTER_BIN_FILE).exists()
    assert not stale.exists(), (
        "the stale adapter_model.safetensors survived the .bin fallback — every reader prefers it, "
        "so resume/merge would load the PREVIOUS adapter"
    )


def test_a_non_contiguous_adapter_tensor_still_writes_safetensors(tmp_path):
    """safetensors refuses a non-contiguous tensor, and one view would cost the whole adapter the ``.bin`` fallback."""
    adapters_mod.write_adapter_file({"w": torch.ones(4, 2).t()}, str(tmp_path))

    assert (tmp_path / ADAPTER_SAFETENSORS_FILE).exists()
    assert not (tmp_path / ADAPTER_BIN_FILE).exists(), "a view fell back to the .bin format"


def test_tensors_sharing_a_storage_still_write_safetensors(tmp_path):
    """A tied embedding under both names is one storage, which safetensors refuses; each key reads back a copy."""
    shared = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    adapters_mod.write_adapter_file({"embed": shared, "head": shared, "lora": torch.ones(2)}, str(tmp_path))

    assert (tmp_path / ADAPTER_SAFETENSORS_FILE).exists()
    assert not (tmp_path / ADAPTER_BIN_FILE).exists(), "a tied pair fell back to the .bin format"
    read = load_file(tmp_path / ADAPTER_SAFETENSORS_FILE)
    assert torch.equal(read["embed"], shared) and torch.equal(read["head"], shared)


def test_bin_fallback_sweep_failure_is_loud_and_keeps_the_bin(tmp_path, monkeypatch, caplog):
    stale = tmp_path / ADAPTER_SAFETENSORS_FILE
    stale.write_bytes(b"previous save's adapter")
    monkeypatch.setattr(adapters_mod, "save_safetensors_file", _failing_save)

    real_remove = os.remove

    def failing_remove(path):
        if str(path).endswith(ADAPTER_SAFETENSORS_FILE):
            raise OSError("Operation not permitted")
        real_remove(path)

    monkeypatch.setattr(adapters_mod.os, "remove", failing_remove)
    with caplog.at_level("WARNING", logger=adapters_mod.logger.name):
        adapters_mod.write_adapter_file({"w": torch.ones(2)}, str(tmp_path))

    assert (tmp_path / ADAPTER_BIN_FILE).exists(), "a failed sweep must not lose the .bin that carries the state"
    assert any("stale" in record.message for record in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
