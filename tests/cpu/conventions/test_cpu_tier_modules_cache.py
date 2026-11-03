#!/usr/bin/env python
"""``tests/cpu/conftest.py`` sends the CPU tier's remote-code copies to a cache of the process's own. On
``$HF_HOME/modules`` they are shared by every xdist worker and session, and a read-only mount fails the first
remote-code load."""

import json
import sys
from pathlib import Path

import pytest
import transformers.dynamic_module_utils

import tests.cpu.conftest as tier
from tests.common.repo import probe_findings

MARKER = "FINDINGS:"

_CONFIG_SOURCE = (
    "from transformers import PretrainedConfig\n\n\nclass TierProbeConfig(PretrainedConfig):\n"
    '    model_type = "halo_tier_probe"\n'
)

_PROBE = """
import pathlib
import shutil

import tests.cpu.conftest as tier
from huggingface_hub.constants import HF_HOME
from transformers import AutoConfig

findings = []
AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
if tier._MODULES_CACHE is None:
    findings.append("the tier conftest set no module cache, so remote code went to " + HF_HOME)
else:
    cache = pathlib.Path(tier._MODULES_CACHE)
    if cache.resolve().is_relative_to(pathlib.Path(HF_HOME).resolve()):
        findings.append("the tier's module cache lies in the shared HF home: " + str(cache))
    if not list(cache.rglob("configuration_tier_probe.py")):
        findings.append("the remote config was not copied into the tier's module cache " + str(cache))
    shutil.rmtree(cache)
print(MARKER + "|".join(findings))
"""


def test_remote_code_copies_land_in_the_process_cache(tmp_path):
    (tmp_path / "configuration_tier_probe.py").write_text(_CONFIG_SOURCE)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "halo_tier_probe", "auto_map": {"AutoConfig": "configuration_tier_probe.TierProbeConfig"}}
        )
    )
    script = f"CHECKPOINT = {str(tmp_path)!r}\nMARKER = {MARKER!r}\n{_PROBE}"
    findings = probe_findings(script, MARKER)
    assert not findings, "\n".join(findings)


def test_this_session_loads_into_the_process_cache():
    if Path(getattr(sys.modules["__main__"], "__file__", "")).resolve() == Path(__file__).resolve():
        pytest.skip("a standalone run imports transformers ahead of the tier conftest")
    assert tier._MODULES_CACHE is not None, "transformers was imported before tests/cpu/conftest.py ran"
    assert transformers.dynamic_module_utils.HF_MODULES_CACHE == tier._MODULES_CACHE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
