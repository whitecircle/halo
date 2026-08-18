#!/usr/bin/env python
"""CPU tests for the after-training tools' RAM/disk preflight.

The preflight WARNS and continues — the figures are estimates, so aborting on them would refuse
conversions that fit. What these tests pin: the warning fires (with numbers) when an estimate
exceeds the host, stays silent when the host is ample or unreadable, and the merge scripts actually
route through the shared helper before their heavy phase.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import src.checkpoint.tool_io as tool_io
from scripts.after_training.merge_ep_shards import merge_ep_shards
from src.checkpoint.tool_io import preflight_model_load_resources, preflight_resource_warning
from src.hardware import available_host_ram_bytes

H = 32


def test_ram_warning_fires_with_numbers(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1_000_000)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=None, ram_bytes=2_000_000_000)
    out = capsys.readouterr().out
    assert "WARNING" in out and "tool_x" in out and "RAM" in out
    assert "2.0 GB" in out and "0.0 GB" in out  # both sides of the comparison are named


def test_disk_warning_walks_up_to_an_existing_ancestor(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tool_io.shutil, "disk_usage", lambda path: SimpleNamespace(total=10, used=9, free=1_000))
    # The output dir does not exist yet — every tool preflights before creating it.
    preflight_resource_warning("tool_x", str(tmp_path / "a" / "b"), disk_bytes=1_000_000, ram_bytes=None)
    out = capsys.readouterr().out
    assert "WARNING" in out and "free" in out and "ENOSPC" in out


def test_silent_when_resources_are_ample(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 10**15)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=1, ram_bytes=1)
    assert capsys.readouterr().out == ""


def test_silent_and_no_raise_when_meminfo_is_unreadable(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: None)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=None, ram_bytes=10**18)
    assert capsys.readouterr().out == ""


def test_available_host_ram_reads_meminfo():
    # The toolkit only runs on Linux hosts/images, where /proc/meminfo carries MemAvailable.
    available = available_host_ram_bytes()
    assert available is not None and available > 0


def _tiny_checkpoint(directory):
    os.makedirs(directory, exist_ok=True)
    save_file({"w": torch.zeros(H, H)}, os.path.join(directory, "model.safetensors"))


def test_model_load_preflight_estimates_from_local_checkpoint_bytes(monkeypatch, capsys, tmp_path):
    src_dir = tmp_path / "src"
    _tiny_checkpoint(str(src_dir))
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)  # any load is too big
    preflight_model_load_resources(str(src_dir), str(tmp_path / "out"), tool="convert_to_bf16")
    assert "WARNING: convert_to_bf16" in capsys.readouterr().out


def test_model_load_preflight_is_silent_for_a_hub_id(monkeypatch, capsys, tmp_path):
    """A hub id resolves to no local directory — there is nothing to measure, and probing must not
    turn the preflight into a crash or a download."""
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)
    preflight_model_load_resources("Qwen/Qwen3-4B", str(tmp_path), tool="convert_to_bf16")
    assert capsys.readouterr().out == ""


def test_model_load_preflight_skips_ram_when_device_map_offloads(monkeypatch, capsys, tmp_path):
    """With --device_map the weights land on devices, not host RAM — no spurious RAM warning."""
    src_dir = tmp_path / "src"
    _tiny_checkpoint(str(src_dir))
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)
    monkeypatch.setattr(tool_io.shutil, "disk_usage", lambda path: SimpleNamespace(total=1, used=0, free=10**18))
    preflight_model_load_resources(str(src_dir), str(tmp_path / "out"), tool="convert_to_bf16", device_map="auto")
    assert capsys.readouterr().out == ""


def _write_tiny_ep_checkpoint(input_dir):
    """A one-layer gpt_oss EP-sharded checkpoint (2 ranks), just enough to drive the real merge."""
    E = 4
    with open(os.path.join(input_dir, "config.json"), "w") as f:
        json.dump({"model_type": "gpt_oss", "num_local_experts": E}, f)
    shards = [
        {
            "model.layers.0.mlp.gate_up_proj.shard_0": torch.randn(E // 2, H, 2 * H),
            "model.layers.0.mlp.down_proj.shard_0": torch.randn(E // 2, H, H),
            "model.layers.0.mlp.router.weight": torch.randn(E, H),
        },
        {
            "model.layers.0.mlp.gate_up_proj.shard_1": torch.randn(E // 2, H, 2 * H),
            "model.layers.0.mlp.down_proj.shard_1": torch.randn(E // 2, H, H),
        },
    ]
    weight_map = {}
    for rank, shard in enumerate(shards):
        fname = f"model-{rank:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(input_dir, fname))
        weight_map.update(dict.fromkeys(shard, fname))
    index = {"metadata": {"ep_size": 2, "format": "ep_sharded"}, "weight_map": weight_map}
    with open(os.path.join(input_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def test_merge_ep_shards_routes_through_the_preflight_and_still_completes(monkeypatch, capsys, tmp_path):
    """The streamed merge must warn about RAM/disk BEFORE its heavy phase — silence here means the
    script stopped calling the shared helper — and the warning must not abort the merge."""
    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    os.makedirs(input_dir)
    _write_tiny_ep_checkpoint(str(input_dir))

    merge_ep_shards(str(input_dir), str(output_dir), verbose=False)

    out = capsys.readouterr().out
    assert "WARNING: merge_ep_shards" in out and "RAM" in out
    merged = load_file(os.path.join(output_dir, "model.safetensors"))
    assert "model.layers.0.mlp.experts.gate_up_proj" in merged


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
