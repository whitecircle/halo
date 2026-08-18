#!/usr/bin/env python
"""The frozen/teacher model's FIRST hub contact must sit inside the coordinated download block.

``load_frozen_auxiliary_model`` is the one loader whose repo nothing pre-populates (the policy path
snapshots main-first before any per-rank read). Its ``AutoConfig.from_pretrained`` — enough to pull
config.json and import a remote modeling file — must not run before ``fs_aware_main_first``: on a
cold cache every rank of every node then hits the hub at once, the exact thundering herd the
coordination scheme exists to order, and a violation of the documented "wrap any download block"
rule.

    python tests/cpu/models/test_frozen_model_download_coordination.py
"""

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.distributed.loading import frozen_models


def test_config_fetch_happens_inside_the_coordinated_phase(monkeypatch):
    events: list[str] = []

    @contextlib.contextmanager
    def recording_main_first(tag):
        events.append("enter")
        yield
        events.append("exit")

    fake_auto_config = MagicMock()
    fake_auto_config.from_pretrained = MagicMock(side_effect=lambda *a, **k: events.append("config") or MagicMock())
    monkeypatch.setattr(frozen_models, "fs_aware_main_first", recording_main_first)
    monkeypatch.setattr(frozen_models, "apply_remote_code_compat_shims", lambda: events.append("shims"))
    monkeypatch.setattr(frozen_models, "AutoConfig", fake_auto_config)
    monkeypatch.setattr(frozen_models, "resolve_attn_implementation", lambda *a, **k: "sdpa")
    monkeypatch.setattr(frozen_models, "apply_family_attention_patches", lambda *a, **k: None)
    monkeypatch.setattr(frozen_models, "auto_load_model", lambda *a, **k: events.append("weights") or MagicMock())
    with (
        patch.object(frozen_models, "finalize_loaded_model"),
        patch.object(frozen_models, "finalize_run_model"),
    ):
        frozen_models.load_frozen_auxiliary_model("org/teacher", dtype=torch.float32, download_tag="teacher_model")

    assert "config" in events and "weights" in events, f"the load path changed shape: {events}"
    assert events.index("shims") < events.index("enter"), "remote-code shims must precede the first fetch"
    assert events.index("enter") < events.index("config") < events.index("exit"), (
        f"the config fetch must run inside fs_aware_main_first, got order {events}"
    )
    assert events.index("enter") < events.index("weights") < events.index("exit"), (
        f"the weight load must stay inside fs_aware_main_first, got order {events}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
