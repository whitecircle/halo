"""``finalize_exported_config`` refuses an unregistered EP export roster.

The family maps in ``moe_balancing`` fill at EP subclass definition; a process that reaches a config
writer without importing ``src.distributed.expert_parallel.layers.roster`` would read an empty
roster and silently drop the schema a family's servers require.
"""

import json

import pytest
from transformers import PretrainedConfig

from src.checkpoint.config_export import finalize_exported_config
from src.models import moe_balancing


def _write_config(tmp_path, model_type: str) -> PretrainedConfig:
    config = PretrainedConfig(model_type=model_type)
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    return config


def test_an_unregistered_roster_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(moe_balancing, "_EP_WRAPPED_MODEL_TYPES", set())
    config = _write_config(tmp_path, "step3p7")
    with pytest.raises(RuntimeError, match="no EP family registered"):
        finalize_exported_config(config, str(tmp_path), source=None)


def test_the_registered_roster_finalizes_a_family_that_needs_nothing(tmp_path):
    import src.distributed.expert_parallel.layers.roster  # noqa: F401  registers the roster, as every real writer path does

    assert moe_balancing.ep_roster_registered()
    config = _write_config(tmp_path, "qwen3")
    finalize_exported_config(config, str(tmp_path), source=None)
    assert json.loads((tmp_path / "config.json").read_text())["model_type"] == "qwen3"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
