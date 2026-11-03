#!/usr/bin/env python
"""``resolve_checkpoint_source`` reads the Hub cache alone offline, since a fetching call lists a pinned commit through
the Hub and raises. Online it still fetches, without the vendor weight dumps no tool reads."""

import json
from pathlib import Path

import pytest
from huggingface_hub import constants as hub_constants
from huggingface_hub.utils import filter_repo_objects

from halo.checkpoint.tools import offline_export
from halo.checkpoint.tools.offline_export import resolve_checkpoint_source

REPO = "org/model"
COMMIT = "c" * 40


def _stage_snapshot(cache: Path) -> Path:
    """With its ref and no tree listing."""
    storage = cache / f"models--{REPO.replace('/', '--')}"
    (storage / "refs").mkdir(parents=True)
    (storage / "refs" / "main").write_text(COMMIT)
    snapshot = storage / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (snapshot / "model.safetensors").write_bytes(b"")
    assert not (storage / "trees").exists()
    return snapshot


@pytest.mark.parametrize("revision", [COMMIT, None], ids=["pinned", "main"])
def test_offline_a_cached_snapshot_resolves_without_its_tree_listing(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    snapshot = _stage_snapshot(tmp_path / "hub")
    assert resolve_checkpoint_source(REPO, revision) == str(snapshot)


def test_online_the_resolver_may_fetch(monkeypatch):
    calls = []
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(offline_export, "snapshot_download", lambda repo, **kwargs: calls.append(kwargs) or "dir")
    assert resolve_checkpoint_source(REPO, COMMIT) == "dir"
    assert [(call["revision"], call.get("local_files_only", False)) for call in calls] == [(COMMIT, False)]


def test_a_hub_fetch_leaves_the_vendor_weight_dumps_behind(monkeypatch):
    """Fetching gpt-oss's ``original/`` and ``metal/`` downloads the model three times over for nothing."""
    calls = []
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(offline_export, "snapshot_download", lambda repo, **kwargs: calls.append(kwargs) or "dir")
    resolve_checkpoint_source(REPO, COMMIT)

    top_level = ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"]
    # Exact directory names: an adapter's carried config shares the dump's prefix and is kept.
    kept_subfolder = ["original_adapter_config/adapter_config.json"]
    dumps = ["original/model.safetensors", "original/config.json", "metal/model.bin"]
    listing = [*top_level, *kept_subfolder, *dumps]
    assert list(filter_repo_objects(listing, ignore_patterns=calls[0].get("ignore_patterns"))) == [
        *top_level,
        *kept_subfolder,
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
