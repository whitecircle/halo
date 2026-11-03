#!/usr/bin/env python
"""``TrainingArguments.ddp_timeout`` is refused. ``parse_script_args`` brings the process group up with
``DIST_NCCL_TIMEOUT_MINUTES`` as its watchdog before any config is built, so a YAML value would parse and never
apply."""

import pytest
from transformers import TrainingArguments

from halo.entrypoints import script


def _parse(monkeypatch, tmp_path, yaml_text: str):
    config = tmp_path / "config.yaml"
    config.write_text(f"output_dir: {tmp_path / 'out'}\nuse_cpu: true\nbf16: false\n{yaml_text}")
    monkeypatch.setattr(script, "init_distributed", lambda: None)
    monkeypatch.setattr(script, "install_log_tee", lambda output_dir: None)
    monkeypatch.setattr("sys.argv", ["train.py", str(config)])
    return script.parse_script_args(TrainingArguments)


def test_a_ddp_timeout_set_in_the_config_is_refused_naming_the_knob_that_applies(monkeypatch, tmp_path):
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "45")
    with pytest.raises(ValueError, match="ddp_timeout=7200 would never apply") as refused:
        _parse(monkeypatch, tmp_path, "ddp_timeout: 7200\n")
    assert "DIST_NCCL_TIMEOUT_MINUTES" in str(refused.value) and "45 min" in str(refused.value)


def test_a_config_leaving_ddp_timeout_at_its_default_parses(monkeypatch, tmp_path):
    """Only a changed value is a request."""
    default = TrainingArguments.ddp_timeout
    assert _parse(monkeypatch, tmp_path, "").ddp_timeout == default
    assert _parse(monkeypatch, tmp_path, f"ddp_timeout: {default}\n").ddp_timeout == default


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
