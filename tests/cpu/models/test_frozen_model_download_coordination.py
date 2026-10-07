#!/usr/bin/env python
"""The frozen/teacher model's FIRST hub contact is the coordinated source resolution.

``load_frozen_auxiliary_model`` is the one loader whose repo nothing pre-populates (the policy path
resolves its own source before any per-rank read). Its ``AutoConfig.from_pretrained`` — enough to
pull config.json and import a remote modeling file — must not run before
``resolve_model_source``: on a cold cache every rank of every node then hits the hub at once, and
per-node caches would each answer with whatever commit they hold. Both the config and the weight
fetch must then read at the revision the resolution agreed, so a teacher scored on one node is the
same teacher as on every other.

    python tests/cpu/models/test_frozen_model_download_coordination.py
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.distributed.loading import frozen_models

AGREED_COMMIT = "1111111111111111111111111111111111111111"


def test_the_source_is_resolved_before_the_first_fetch_and_pins_every_read(monkeypatch):
    events: list[str] = []
    reads: dict[str, object] = {}

    def resolve(model_name_or_path, revision, *, tag):
        events.append(f"resolve:{tag}")
        return AGREED_COMMIT

    def fetch_config(*args, **kwargs):
        events.append("config")
        reads["config"] = kwargs.get("revision")
        return MagicMock()

    def fetch_weights(*args, **kwargs):
        events.append("weights")
        reads["weights"] = kwargs.get("revision")
        return MagicMock()

    monkeypatch.setattr(frozen_models, "resolve_model_source", resolve)
    monkeypatch.setattr(frozen_models, "apply_remote_code_compat_shims", lambda: events.append("shims"))
    monkeypatch.setattr(frozen_models, "AutoConfig", MagicMock(from_pretrained=fetch_config))
    monkeypatch.setattr(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa")
    monkeypatch.setattr(frozen_models, "apply_family_attention_patches", lambda config, attn: attn)
    monkeypatch.setattr(frozen_models, "auto_load_model", fetch_weights)
    with (
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(frozen_models, "finalize_run_model"),
        patch.object(frozen_models, "warm_attention_kernels"),
    ):
        frozen_models.load_frozen_auxiliary_model("org/teacher", dtype=torch.float32, download_tag="teacher_model")

    assert events == ["shims", "resolve:teacher_model", "config", "weights"], (
        f"the remote-code shims, then the coordinated resolution, must precede every fetch: {events}"
    )
    assert reads == {"config": AGREED_COMMIT, "weights": AGREED_COMMIT}, (
        f"the config and weight fetches must read at the agreed revision: {reads}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
