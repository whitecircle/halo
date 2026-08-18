#!/usr/bin/env python
"""Revision pinning on the EP / EP+TP / EP+CP lazy weight-loading paths.

``model_revision`` must reach every hub-snapshot resolution: ``resolve_hub_or_local_dir``
resolves the checkpoint dir the lazy safetensors loader reads, so dropping ``revision`` there
silently trains on hub ``main`` weights against the pinned config. The commit below is an opaque
sentinel — what is under test is that it arrives, not which repo it names.

Run: python tests/cpu/parallelism/test_ep_loading_revision.py
"""

import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from accelerate import PartialState

from src.distributed.context_parallel.loading import load_model_for_ep_cp
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.loading import load_ep_model, resolve_hub_or_local_dir
from src.distributed.loading.model_loading import _load_ep_tp_model

PartialState()  # the loaders' accelerate logger needs accelerate state initialized

_EP_LOADING = "src.distributed.expert_parallel.loading"
_CP_LOADING = "src.distributed.context_parallel.loading"
_MODEL_LOADING = "src.distributed.loading.model_loading"

_REVISION = "970cfc9f5e7e5a4f5f6f0645955928a9b6a98415"


def _make_ep_config() -> EPConfig:
    """EPConfig valid in single-process mode (no torch.distributed)."""
    return EPConfig(ep_size=2, world_size=2, gpus_per_node=2)


# The loaders take the dispatcher's already-loaded config; every load here is patched out, so what
# it is never matters — only that it is threaded through.
_MODEL_CONFIG = SimpleNamespace()


class _RecordingSnapshotDownload:
    """Stand-in for ``huggingface_hub.snapshot_download`` that records every call."""

    def __init__(self, fail_local_files_only: bool = False, snapshot_dir: str = "/fake/snapshot"):
        self.calls: list[dict] = []
        self._fail_local_files_only = fail_local_files_only
        self._snapshot_dir = snapshot_dir

    def __call__(self, repo_id, **kwargs):
        self.calls.append({"repo_id": repo_id, **kwargs})
        if self._fail_local_files_only and kwargs.get("local_files_only"):
            raise FileNotFoundError("not in cache")
        return self._snapshot_dir


# resolve_hub_or_local_dir


def test_resolver_passes_revision_to_cached_snapshot():
    """The cache-first (local_files_only) resolution must be pinned to the revision."""
    fake = _RecordingSnapshotDownload()
    with patch(f"{_EP_LOADING}.snapshot_download", fake):
        result = resolve_hub_or_local_dir("org/repo", revision=_REVISION)

    assert result == "/fake/snapshot"
    assert len(fake.calls) == 1
    assert fake.calls[0]["local_files_only"] is True
    assert fake.calls[0]["revision"] == _REVISION


def test_resolver_passes_revision_to_download_fallback():
    """A cache miss must download the pinned revision, not hub main."""
    fake = _RecordingSnapshotDownload(fail_local_files_only=True)
    with patch(f"{_EP_LOADING}.snapshot_download", fake):
        result = resolve_hub_or_local_dir("org/repo", revision=_REVISION)

    assert result == "/fake/snapshot"
    assert len(fake.calls) == 2
    assert all(call["revision"] == _REVISION for call in fake.calls)


def test_resolver_defaults_to_none_revision():
    """Without a pin, both snapshot_download calls receive revision=None (hub default)."""
    fake = _RecordingSnapshotDownload(fail_local_files_only=True)
    with patch(f"{_EP_LOADING}.snapshot_download", fake):
        resolve_hub_or_local_dir("org/repo")

    assert len(fake.calls) == 2
    assert all(call["revision"] is None for call in fake.calls)


def test_resolver_local_dir_bypasses_hub():
    """Local directories resolve to their absolute path without touching the hub."""
    fake = _RecordingSnapshotDownload()
    with tempfile.TemporaryDirectory() as local_dir:
        with patch(f"{_EP_LOADING}.snapshot_download", fake):
            result = resolve_hub_or_local_dir(local_dir, revision=_REVISION)
        assert result == local_dir
    assert fake.calls == []


def test_resolver_warns_when_the_hub_cannot_be_resolved():
    """A failed download is a degradation, not routine: every rank then materializes the whole
    checkpoint through ``from_pretrained``. The cache MISS above it stays INFO — it is the ordinary
    first-run path, and warning there would train operators to ignore the line that matters."""

    def _unreachable(repo_id, **kwargs):
        raise OSError("401 gated repo")

    with patch(f"{_EP_LOADING}.snapshot_download", _unreachable), patch(f"{_EP_LOADING}.logger") as log:
        assert resolve_hub_or_local_dir("org/repo") is None

    assert log.warning.call_count == 1, "the failed resolution must not hide at INFO"
    assert "skipping EP lazy loading" in log.warning.call_args.args[0]
    assert log.info.call_count == 1, "the cache miss stays INFO"


def test_lazy_fallback_level_separates_a_family_property_from_an_environment_failure():
    """``lazy_incompatible_expert_layout`` is class-declared and supported, so it logs at INFO;
    a missing snapshot means the lazy path was WANTED and lost, so it warns."""

    def _run(local_dir: str | None, *, lazy_loadable: bool, has_safetensors: bool):
        with (
            patch(f"{_EP_LOADING}.resolve_hub_or_local_dir", lambda *a, **k: local_dir),
            patch(f"{_EP_LOADING}.decide_lazy_loadable", lambda *a, **k: lazy_loadable),
            patch(f"{_EP_LOADING}.has_safetensors_checkpoint", lambda *a, **k: has_safetensors),
            patch(f"{_EP_LOADING}._load_ep_model_huggingface", lambda *a, **k: nn.Linear(2, 2)),
            patch(f"{_EP_LOADING}.logger") as log,
        ):
            load_ep_model("org/repo", _make_ep_config(), _MODEL_CONFIG)
        return log

    unmappable = _run("/fake/snapshot", lazy_loadable=False, has_safetensors=True)
    assert unmappable.warning.call_count == 0
    assert any("lazy_incompatible_expert_layout" in call.args[0] for call in unmappable.info.call_args_list)

    no_snapshot = _run(None, lazy_loadable=False, has_safetensors=False)
    assert no_snapshot.warning.call_count == 1
    assert "no_local_snapshot" in no_snapshot.warning.call_args.args[0]


# load_ep_model threading


def test_load_ep_model_threads_revision():
    """load_ep_model must pin both the snapshot resolution and the from_pretrained fallback."""
    seen = {}

    def fake_resolve(model_name_or_path, revision=None):
        seen["resolve_revision"] = revision
        return None  # no local snapshot -> HF from_pretrained fallback

    def fake_hf_load(model_name_or_path, ep_config, *args, revision=None, **kwargs):
        seen["hf_revision"] = revision
        return nn.Linear(2, 2)

    with (
        patch(f"{_EP_LOADING}.resolve_hub_or_local_dir", fake_resolve),
        patch(f"{_EP_LOADING}._load_ep_model_huggingface", fake_hf_load),
    ):
        load_ep_model("org/repo", _make_ep_config(), _MODEL_CONFIG, revision=_REVISION)

    assert seen["resolve_revision"] == _REVISION
    assert seen["hf_revision"] == _REVISION


def test_load_ep_model_lazy_path_reads_pinned_snapshot():
    """The lazy safetensors loader must read the revision-pinned snapshot dir; the revision kwarg
    itself is absorbed (a local dir needs no hub revision)."""
    seen = {}

    def fake_resolve(model_name_or_path, revision=None):
        seen["resolve_revision"] = revision
        return "/fake/pinned/snapshot"

    def fake_lazy(local_dir, ep_config, config, **kwargs):
        seen["lazy_dir"] = local_dir
        seen["lazy_kwargs"] = kwargs
        return nn.Linear(2, 2)

    with (
        patch(f"{_EP_LOADING}.resolve_hub_or_local_dir", fake_resolve),
        # The rank-0 verdict, not the probes behind it: this test's subject is the revision pin.
        patch(f"{_EP_LOADING}.decide_lazy_loadable", return_value=True),
        patch(f"{_EP_LOADING}.load_ep_model_lazy", fake_lazy),
    ):
        load_ep_model("org/repo", _make_ep_config(), _MODEL_CONFIG, revision=_REVISION)

    assert seen["resolve_revision"] == _REVISION
    assert seen["lazy_dir"] == "/fake/pinned/snapshot"
    assert "revision" not in seen["lazy_kwargs"]


# EP+CP and EP+TP call sites


def test_load_model_for_ep_cp_threads_revision():
    """load_model_for_ep_cp forwards the revision pin into load_ep_model."""
    seen = {}

    def fake_load_ep_model(model_name_or_path, ep_config, *args, revision=None, **kwargs):
        seen["revision"] = revision
        return nn.Linear(2, 2)

    with (
        patch(f"{_CP_LOADING}.load_ep_model", fake_load_ep_model),
        patch(f"{_CP_LOADING}.patch_model_for_cp", side_effect=lambda model, cp_config: model),
    ):
        load_model_for_ep_cp(
            "org/repo",
            SimpleNamespace(ep_size=2),
            SimpleNamespace(cp_size=2),
            SimpleNamespace(),  # config: required, like the EP and PP loaders
            revision=_REVISION,
        )

    assert seen["revision"] == _REVISION


def test_ep_tp_lazy_resolution_uses_revision():
    """_load_ep_tp_model resolves the lazy snapshot dir at the revision pinned in common_kwargs."""
    seen = {}

    def fake_resolve(model_name_or_path, revision=None):
        seen["resolve_revision"] = revision
        return "/fake/pinned/snapshot"

    def fake_lazy(local_dir, ep_config, config, **kwargs):
        seen["lazy_dir"] = local_dir
        return nn.Linear(2, 2)

    pc = SimpleNamespace(
        ep_lazy_loading=True,
        ep_size=2,
        tp_size=2,
        data_parallel_size=1,
        pp_size=1,
        create_ep_config=_make_ep_config,
    )
    with (
        patch(f"{_MODEL_LOADING}.resolve_hub_or_local_dir", fake_resolve),
        # The rank-0 verdict, not the probes behind it: this test's subject is the revision pin.
        patch(f"{_MODEL_LOADING}.decide_lazy_loadable", return_value=True),
        patch(f"{_MODEL_LOADING}.load_ep_model_lazy", fake_lazy),
        patch(f"{_MODEL_LOADING}._apply_attention_only_tp"),
    ):
        _load_ep_tp_model(
            "org/repo",
            pc,
            model_class=None,
            common_kwargs={"config": _MODEL_CONFIG, "dtype": None, "trust_remote_code": True, "revision": _REVISION},
            local_rank=0,
        )

    assert seen["resolve_revision"] == _REVISION
    assert seen["lazy_dir"] == "/fake/pinned/snapshot"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
