#!/usr/bin/env python
"""Lazy-load generation-config restore: an absent file is fine, an unreadable one raises.
Swallowed, it drops gpt-oss's harmony stop tokens from every later save, a model that never stops generating."""

import pytest
import torch.nn as nn
from transformers import GenerationConfig

from halo.model_loading.lazy_safetensors.meta_shell import _restore_checkpoint_generation_config


def _raising_from_pretrained(*args, **kwargs):
    raise OSError("stale NFS handle")


def test_unreadable_present_file_raises(monkeypatch, tmp_path):
    (tmp_path / "generation_config.json").write_text("{}")
    monkeypatch.setattr(GenerationConfig, "from_pretrained", staticmethod(_raising_from_pretrained))
    with pytest.raises(OSError, match="stale NFS"):
        _restore_checkpoint_generation_config(nn.Module(), str(tmp_path))


def test_absent_file_keeps_rebuilt_defaults(monkeypatch, tmp_path):
    """``from_config`` already rebuilt defaults on the shell, so the object must survive untouched."""
    monkeypatch.setattr(GenerationConfig, "from_pretrained", staticmethod(_raising_from_pretrained))
    model = nn.Module()
    rebuilt = GenerationConfig(eos_token_id=[200002, 199999], temperature=0.35)
    model.generation_config = rebuilt

    _restore_checkpoint_generation_config(model, str(tmp_path))

    assert model.generation_config is rebuilt, "the rebuilt defaults were replaced or cleared"
    assert model.generation_config.eos_token_id == [200002, 199999]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
