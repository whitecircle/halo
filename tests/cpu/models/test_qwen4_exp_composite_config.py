#!/usr/bin/env python
"""A saved Flash-Next composite config must keep the vision field the release's server-side tower reads."""

import json

import pytest
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

from tests.common.models import TINY_QWEN4_EXP_CONFIG, TINY_QWEN4_EXP_VISION_CONFIG


def test_a_saved_composite_config_carries_the_deepstack_indexes_the_server_reads(tmp_path):
    config = Qwen4ExpConfig(text_config=dict(TINY_QWEN4_EXP_CONFIG), vision_config=dict(TINY_QWEN4_EXP_VISION_CONFIG))
    config.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["vision_config"]["deepstack_visual_indexes"] == []
    assert Qwen4ExpConfig.from_pretrained(tmp_path).vision_config.deepstack_visual_indexes == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
