"""The modality probe must refuse an untrusted remote-code config, not route around it.

``is_vlm_model`` reads the checkpoint config under the run's own ``trust_remote_code``. A remote-code
architecture under ``trust_remote_code: false`` is refused by transformers; swallowed as a hub
failure it would pick an ``Auto*`` class off the name heuristic, and the model load would refuse the
same config only after the datasets are loaded. A genuinely unreadable config still falls back.
"""

import json

import pytest
import transformers.dynamic_module_utils

from src.models.modality import is_vlm_model

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
        is_vlm_model(remote_code_checkpoint, trust_remote_code=False)


def test_a_trusted_remote_code_config_is_read(remote_code_checkpoint, tmp_path_factory, monkeypatch):
    modules_cache = str(tmp_path_factory.mktemp("hf_modules"))
    monkeypatch.setattr(transformers.dynamic_module_utils, "HF_MODULES_CACHE", modules_cache)
    assert is_vlm_model(remote_code_checkpoint, trust_remote_code=True) is False


def test_an_unreadable_config_still_falls_back_to_the_name_heuristic(tmp_path):
    assert is_vlm_model(str(tmp_path / "missing-qwen3-vl"), trust_remote_code=False) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
