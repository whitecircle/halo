"""CPU tests for the adapter writer's ``.bin`` fallback.

Every reader prefers ``adapter_model.safetensors`` over ``adapter_model.bin`` — the loader's
``restore_adapters``, ``assert_no_expert_lora_adapter`` and ``PeftModel.from_pretrained`` all probe
the safetensors name first. So when the safetensors save fails and the writer falls back to the
``.bin``, a leftover safetensors file (this attempt's partial write, or a previous save's complete
one) would silently shadow the state that was just saved: a resume or merge then reads the PREVIOUS
adapter. The fallback must therefore remove the safetensors leftover after the ``.bin`` lands —
the same contract the gathered-checkpoint writer holds for its own fallback.

    python tests/cpu/peft/test_adapter_bin_fallback_sweep.py
"""

import os
import sys

import pytest
import torch

import src.distributed.checkpoint.peft as peft_mod
from src.checkpoint.format import ADAPTER_BIN_FILE, ADAPTER_SAFETENSORS_FILE
from src.distributed.checkpoint.peft import PeftAdapterSaver


def _failing_save(*args, **kwargs):
    raise OSError("No space left on device")


def test_bin_fallback_removes_the_stale_safetensors(tmp_path, monkeypatch):
    stale = tmp_path / ADAPTER_SAFETENSORS_FILE
    stale.write_bytes(b"previous save's adapter")
    monkeypatch.setattr(peft_mod, "safetensors_save_file", _failing_save)

    PeftAdapterSaver._write_adapter_state_dict({"w": torch.ones(2)}, str(tmp_path))

    assert (tmp_path / ADAPTER_BIN_FILE).exists()
    assert not stale.exists(), (
        "the stale adapter_model.safetensors survived the .bin fallback — every reader prefers it, "
        "so resume/merge would load the PREVIOUS adapter"
    )


def test_a_non_contiguous_adapter_tensor_still_writes_safetensors(tmp_path):
    """safetensors refuses a non-contiguous tensor, and the adapter file is written as one dict —
    so a single view (a gathered DTensor adapter, a ``modules_to_save`` clone) would cost the whole
    adapter the ``.bin`` fallback, which every reader then has to be told about."""
    PeftAdapterSaver._write_adapter_state_dict({"w": torch.ones(4, 2).t()}, str(tmp_path))

    assert (tmp_path / ADAPTER_SAFETENSORS_FILE).exists()
    assert not (tmp_path / ADAPTER_BIN_FILE).exists(), "a view fell back to the .bin format"


def test_bin_fallback_sweep_failure_is_loud_and_keeps_the_bin(tmp_path, monkeypatch, caplog):
    stale = tmp_path / ADAPTER_SAFETENSORS_FILE
    stale.write_bytes(b"previous save's adapter")
    monkeypatch.setattr(peft_mod, "safetensors_save_file", _failing_save)

    real_remove = os.remove

    def failing_remove(path):
        if str(path).endswith(ADAPTER_SAFETENSORS_FILE):
            raise OSError("Operation not permitted")
        real_remove(path)

    monkeypatch.setattr(peft_mod.os, "remove", failing_remove)
    with caplog.at_level("WARNING", logger=peft_mod.logger.name):
        PeftAdapterSaver._write_adapter_state_dict({"w": torch.ones(2)}, str(tmp_path))

    assert (tmp_path / ADAPTER_BIN_FILE).exists(), "a failed sweep must not lose the .bin that carries the state"
    assert any("stale" in record.message for record in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
