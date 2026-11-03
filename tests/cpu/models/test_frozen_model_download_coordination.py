#!/usr/bin/env python
"""The frozen/teacher model's first hub contact is the coordinated source resolution.
On a cold cache every rank would hit the hub at once, and per-node caches could each answer with a different commit."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from halo.model_loading import model_preparation
from halo.model_loading.parallel import frozen_models
from halo.model_loading.parallel.model_source import ModelSource

AGREED_COMMIT = "1111111111111111111111111111111111111111"


def test_the_source_is_resolved_before_the_first_fetch_and_pins_every_read(monkeypatch):
    events: list[str] = []
    reads: dict[str, object] = {}

    def resolve(model_name_or_path, revision, *, tag):
        events.append(f"resolve:{tag}")
        return ModelSource(model_name_or_path, AGREED_COMMIT, model_name_or_path)

    def fetch_config(*args, **kwargs):
        events.append("config")
        reads["config"] = kwargs.get("revision")
        return MagicMock()

    def fetch_weights(*args, **kwargs):
        events.append("weights")
        reads["weights"] = kwargs.get("revision")
        return MagicMock()

    monkeypatch.setattr(frozen_models, "resolve_model_source", resolve)
    monkeypatch.setattr(model_preparation, "apply_remote_code_compat_shims", lambda: events.append("shims"))
    monkeypatch.setattr(model_preparation, "AutoConfig", MagicMock(from_pretrained=fetch_config))
    monkeypatch.setattr(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa")
    monkeypatch.setattr(frozen_models, "apply_family_patches", lambda config, attn: attn)
    monkeypatch.setattr(frozen_models, "auto_load_model", fetch_weights)
    with (
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(frozen_models, "apply_run_model_policies"),
        patch.object(frozen_models, "warm_attention_kernels"),
    ):
        frozen_models.load_frozen_auxiliary_model(
            "org/teacher", dtype=torch.float32, download_tag="teacher_model", trust_remote_code=False
        )

    assert events == ["resolve:teacher_model", "shims", "config", "weights"], (
        f"the coordinated resolution must precede every fetch, and the remote-code shims the config fetch: {events}"
    )
    assert reads == {"config": AGREED_COMMIT, "weights": AGREED_COMMIT}, (
        f"the config and weight fetches must read at the agreed revision: {reads}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
