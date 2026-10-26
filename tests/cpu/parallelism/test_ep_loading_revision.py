#!/usr/bin/env python
"""Revision pinning on the EP / EP+TP / EP+CP weight-loading paths.

The lazy loaders read ``ModelSource.local_dir``, the snapshot ``resolve_model_source`` agreed at the
pinned revision, and the ``from_pretrained`` fallback must receive the same revision, or it silently
trains on hub ``main`` weights against the pinned config. The commit below is an opaque sentinel —
what is under test is that it arrives, not which repo it names.

Run: python tests/cpu/parallelism/test_ep_loading_revision.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from accelerate import PartialState

from src.distributed.expert_parallel.config import EPConfig
from src.model_loading.parallel.cp import load_ep_cp_model
from src.model_loading.parallel.ep import load_ep_model
from src.model_loading.parallel.model_source import ModelSource
from src.model_loading.parallel.tp import load_ep_tp_model

PartialState()  # the loaders' accelerate logger needs accelerate state initialized

_EP_LOADING = "src.model_loading.parallel.ep"
_CP_LOADING = "src.model_loading.parallel.cp"
_TP_LOADING = "src.model_loading.parallel.tp"

_REVISION = "970cfc9f5e7e5a4f5f6f0645955928a9b6a98415"
_SOURCE = ModelSource("org/repo", _REVISION, "/fake/pinned/snapshot")


def _make_ep_config() -> EPConfig:
    """EPConfig valid in single-process mode (no torch.distributed)."""
    return EPConfig(ep_size=2, stage_world_size=2, nvlink_domain_size=2, experts_fsdp_managed=False)


def _ep_pc(**overrides) -> SimpleNamespace:
    fields = {"ep_config": _make_ep_config, "ep_lazy_loading": True, "fp32_non_ep_params": False}
    return SimpleNamespace(**{**fields, "max_concurrent_loading": 1, **overrides})


# The loaders take the dispatcher's already-loaded config; every load here is patched out, so what
# it is never matters — only that it is threaded through.
_MODEL_CONFIG = SimpleNamespace()
_KWARGS = {"config": _MODEL_CONFIG, "revision": _REVISION}


def test_lazy_fallback_level_separates_a_family_property_from_an_environment_failure():
    """``lazy_incompatible_expert_layout`` is class-declared and supported, so it logs at INFO;
    missing safetensors means the lazy path was WANTED and lost, so it warns."""

    def _run(*, has_safetensors: bool):
        with (
            patch(f"{_EP_LOADING}.resolve_lazy_checkpoint", lambda *a, **k: False),
            patch(f"{_EP_LOADING}.has_whole_model_weight_file", lambda *a, **k: has_safetensors),
            patch(f"{_EP_LOADING}._load_ep_model_huggingface", lambda *a, **k: nn.Linear(2, 2)),
            patch(f"{_EP_LOADING}.logger") as log,
        ):
            load_ep_model(_SOURCE, _ep_pc(), None, _KWARGS, preserve_checkpoint_precision=False)
        return log

    unmappable = _run(has_safetensors=True)
    assert unmappable.warning.call_count == 0
    assert any("lazy_incompatible_expert_layout" in call.args[0] for call in unmappable.info.call_args_list)

    no_safetensors = _run(has_safetensors=False)
    assert no_safetensors.warning.call_count == 1
    assert "no_safetensors" in no_safetensors.warning.call_args.args[0]


def test_load_ep_model_threads_revision():
    """The ``from_pretrained`` fallback loads the agreed source at the pinned revision."""
    seen = {}

    def fake_hf_load(source, pc, ep_config, model_class, from_pretrained_kwargs, **kwargs):
        seen["hf"] = (source.name, from_pretrained_kwargs.get("revision"))
        return nn.Linear(2, 2)

    with (
        patch(f"{_EP_LOADING}.resolve_lazy_checkpoint", return_value=False),
        patch(f"{_EP_LOADING}._load_ep_model_huggingface", fake_hf_load),
    ):
        load_ep_model(_SOURCE, _ep_pc(), None, _KWARGS, preserve_checkpoint_precision=False)

    assert seen["hf"] == ("org/repo", _REVISION)


def test_load_ep_model_lazy_path_reads_pinned_snapshot():
    """The lazy safetensors loader reads the revision-pinned snapshot dir, not the source's name."""
    seen = {}

    def fake_lazy(local_dir, ep_config, config, **kwargs):
        seen["lazy_dir"] = local_dir
        return nn.Linear(2, 2)

    with (
        # The rank-0 verdict, not the probes behind it: this test's subject is the revision pin.
        patch(f"{_EP_LOADING}.resolve_lazy_checkpoint", return_value=True),
        patch(f"{_EP_LOADING}.load_ep_model_lazy", fake_lazy),
    ):
        load_ep_model(_SOURCE, _ep_pc(), None, _KWARGS, preserve_checkpoint_precision=False)

    assert seen["lazy_dir"] == "/fake/pinned/snapshot"


# EP+CP and EP+TP call sites


def test_load_ep_cp_model_threads_revision():
    """load_ep_cp_model forwards the source and its revision pin into load_ep_model."""
    seen = {}

    def fake_load_ep_model(source, pc, model_class, from_pretrained_kwargs, **kwargs):
        seen["load"] = (source, from_pretrained_kwargs.get("revision"))
        return nn.Linear(2, 2)

    pc = _ep_pc(ep_config=lambda: SimpleNamespace(ep_size=2), cp_config=lambda: SimpleNamespace(cp_size=2))
    with (
        patch(f"{_CP_LOADING}.load_ep_model", fake_load_ep_model),
        patch(f"{_CP_LOADING}.UlyssesModel", side_effect=lambda model, cp_config: model),
    ):
        load_ep_cp_model(
            _SOURCE,
            pc,
            None,
            _KWARGS,
            preserve_checkpoint_precision=False,
        )

    assert seen["load"] == (_SOURCE, _REVISION)


def test_ep_tp_lazy_path_reads_pinned_snapshot():
    """load_ep_tp_model streams the experts from the agreed source's snapshot directory."""
    seen = {}

    def fake_lazy(local_dir, ep_config, config, **kwargs):
        seen["lazy_dir"] = local_dir
        return nn.Linear(2, 2)

    pc = SimpleNamespace(
        ep_lazy_loading=True,
        ep_size=2,
        tp_size=2,
        data_parallel_size=1,
        pp_size=1,
        fp32_non_ep_params=False,
        ep_config=_make_ep_config,
    )
    with (
        # The rank-0 verdict, not the probes behind it: this test's subject is the revision pin.
        patch(f"{_TP_LOADING}.resolve_lazy_checkpoint", return_value=True),
        patch(f"{_TP_LOADING}.load_ep_model_lazy", fake_lazy),
        patch(f"{_TP_LOADING}._apply_attention_only_tp"),
        patch(f"{_TP_LOADING}.timed_global_load", lambda tag, method, load_fn: load_fn()),
    ):
        load_ep_tp_model(
            _SOURCE,
            pc,
            None,
            {"config": _MODEL_CONFIG, "dtype": None, "trust_remote_code": True, "revision": _REVISION},
            preserve_checkpoint_precision=False,
        )

    assert seen["lazy_dir"] == "/fake/pinned/snapshot"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
