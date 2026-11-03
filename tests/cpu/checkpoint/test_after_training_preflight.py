#!/usr/bin/env python
"""The after-training tools' RAM/disk preflight warns and continues, since aborting on estimates would refuse
conversions that fit."""

import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import halo.checkpoint.tools.ep_shard_merge as ep_shard_merge
import halo.checkpoint.tools.offline_export as offline_export
from halo.checkpoint.format import OPTIMIZER_META_FILE, optimizer_shard_filename
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.checkpoint.tools.offline_export import (
    cross_device_bytes,
    preflight_model_load_resources,
    preflight_resource_warning,
)
from halo.hardware import available_host_ram_bytes

H = 32
# A tmpfs, so never the filesystem of a test's tmp_path on disk.
SECOND_FILESYSTEM = "/dev/shm"


def _warnings(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_ram_warning_fires_with_numbers(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1_000_000)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=None, ram_bytes=2_000_000_000)
    out = _warnings(caplog)
    assert "tool_x" in out and "RAM" in out
    assert "2.0 GB" in out and "0.0 GB" in out


def test_disk_warning_walks_up_to_an_existing_ancestor(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(
        offline_export.shutil, "disk_usage", lambda path: SimpleNamespace(total=10, used=9, free=1_000)
    )
    # Every tool preflights before creating the output dir.
    preflight_resource_warning("tool_x", str(tmp_path / "a" / "b"), disk_bytes=1_000_000, ram_bytes=None)
    out = _warnings(caplog)
    assert "free" in out and "ENOSPC" in out


def test_silent_when_resources_are_ample(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 10**15)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=1, ram_bytes=1)
    assert _warnings(caplog) == ""


def test_silent_and_no_raise_when_meminfo_is_unreadable(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: None)
    preflight_resource_warning("tool_x", str(tmp_path), disk_bytes=None, ram_bytes=10**18)
    assert _warnings(caplog) == ""


def test_available_host_ram_reads_meminfo():
    # /proc/meminfo carries MemAvailable on the toolkit's Linux hosts.
    available = available_host_ram_bytes()
    assert available is not None and available > 0


def _tiny_checkpoint(directory):
    os.makedirs(directory, exist_ok=True)
    save_file({"w": torch.zeros(H, H)}, os.path.join(directory, "model.safetensors"))


def test_model_load_preflight_estimates_from_local_checkpoint_bytes(monkeypatch, caplog, tmp_path):
    src_dir = tmp_path / "src"
    _tiny_checkpoint(str(src_dir))
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)  # Any load is too big.
    preflight_model_load_resources(str(src_dir), str(tmp_path / "out"), tool="convert_to_bf16")
    assert "convert_to_bf16: estimated peak host RAM" in _warnings(caplog)


def test_model_load_preflight_is_silent_for_a_hub_id(monkeypatch, caplog, tmp_path):
    """There is nothing to measure, and probing must not crash or download."""
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)
    preflight_model_load_resources("Qwen/Qwen3-4B", str(tmp_path), tool="convert_to_bf16")
    assert _warnings(caplog) == ""


def test_model_load_preflight_skips_ram_when_device_map_offloads(monkeypatch, caplog, tmp_path):
    """The weights land on devices."""
    src_dir = tmp_path / "src"
    _tiny_checkpoint(str(src_dir))
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)
    monkeypatch.setattr(
        offline_export.shutil, "disk_usage", lambda path: SimpleNamespace(total=1, used=0, free=10**18)
    )
    preflight_model_load_resources(str(src_dir), str(tmp_path / "out"), tool="convert_to_bf16", device_map="auto")
    assert _warnings(caplog) == ""


def _write_tiny_ep_checkpoint(input_dir):
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


def test_merge_ep_shards_routes_through_the_preflight_and_still_completes(monkeypatch, caplog, tmp_path):
    """Silence means the script stopped calling the shared helper."""
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    os.makedirs(input_dir)
    _write_tiny_ep_checkpoint(str(input_dir))

    merge_ep_shards(str(input_dir), str(output_dir), verbose=False)

    out = _warnings(caplog)
    assert "merge_ep_shards: estimated peak host RAM" in out
    merged = load_file(os.path.join(output_dir, "model.safetensors"))
    assert "model.layers.0.mlp.experts.gate_up_proj" in merged


def test_merge_ep_shards_disk_estimate_leaves_out_the_resume_state_it_links(monkeypatch, tmp_path):
    """On one filesystem the shards are hard-linked; counting them would warn a merge off a volume it fits on."""
    estimates = []
    monkeypatch.setattr(
        ep_shard_merge,
        "preflight_resource_warning",
        lambda tool, output_dir, *, disk_bytes, ram_bytes: estimates.append(disk_bytes),
    )
    input_dir = tmp_path / "in"
    os.makedirs(input_dir)
    _write_tiny_ep_checkpoint(str(input_dir))
    weight_bytes = sum(path.stat().st_size for path in input_dir.glob("*.safetensors"))
    for rank in range(2):
        torch.save({"state": {0: {"exp_avg": torch.zeros(H * H)}}}, input_dir / optimizer_shard_filename(rank))
    torch.save({"num_ranks": 2}, input_dir / OPTIMIZER_META_FILE)

    merge_ep_shards(str(input_dir), str(tmp_path / "out"), verbose=False)

    assert estimates == [weight_bytes]


def test_cross_device_bytes_counts_only_the_files_a_link_cannot_reach(tmp_path):
    linkable = tmp_path / optimizer_shard_filename(0)
    linkable.write_bytes(bytes(100))
    with tempfile.TemporaryDirectory(dir=SECOND_FILESYSTEM) as other:
        assert os.stat(other).st_dev != os.stat(tmp_path).st_dev, f"premise: {SECOND_FILESYSTEM} is another filesystem"
        unlinkable = Path(other) / optimizer_shard_filename(1)
        unlinkable.write_bytes(bytes(7))
        # The merges preflight before they create the output directory.
        output_dir = str(tmp_path / "merged")

        assert cross_device_bytes([str(linkable)], output_dir) == 0
        assert cross_device_bytes([str(linkable), str(unlinkable)], output_dir) == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
