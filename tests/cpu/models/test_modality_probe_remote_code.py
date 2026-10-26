"""The modality probe must refuse an untrusted remote-code config, not route around it.

``probe_checkpoint`` reads the checkpoint config under the run's own ``trust_remote_code``. A remote-code
architecture under ``trust_remote_code: false`` is refused by transformers; swallowed as a hub
failure it would pick an ``Auto*`` class off the name heuristic, and the model load would refuse the
same config only after the datasets are loaded. A genuinely unreadable config still falls back.
"""

import json

import pytest

from halo.model_loading.parallel.model_source import probe_checkpoint

_CONFIG_MODULE = """
from transformers import PretrainedConfig


class RemoteProbeConfig(PretrainedConfig):
    model_type = "halo_remote_probe"
"""


@pytest.fixture
def remote_code_checkpoint(tmp_path):
    (tmp_path / "configuration_remote_probe.py").write_text(_CONFIG_MODULE)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "halo_remote_probe",
                "auto_map": {"AutoConfig": "configuration_remote_probe.RemoteProbeConfig"},
            }
        )
    )
    return str(tmp_path)


def test_an_untrusted_remote_code_config_is_refused_naming_the_knob(remote_code_checkpoint):
    with pytest.raises(ValueError, match="Set trust_remote_code: true"):
        probe_checkpoint(remote_code_checkpoint, trust_remote_code=False)


def test_a_trusted_remote_code_config_is_read(remote_code_checkpoint):
    assert probe_checkpoint(remote_code_checkpoint, trust_remote_code=True)[1] is False


def test_an_unreadable_config_still_falls_back_to_the_name_heuristic(tmp_path):
    assert probe_checkpoint(str(tmp_path / "missing-qwen3-vl"), trust_remote_code=False)[1] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
