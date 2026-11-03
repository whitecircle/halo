#!/usr/bin/env python
"""A pipeline stage carries the generation config, or PP checkpoints serve with default sampling and EOS, silently.
The PP save hands ``save_model_config`` a stage, which must mirror ``generation_config`` and ``can_generate``."""

from __future__ import annotations

import json
import os

import pytest
import torch.nn as nn
from transformers import GenerationConfig, PretrainedConfig

from halo.checkpoint.config_export import save_model_config
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from tests.common.pp_fakes import hand_built_stage


def _stage() -> PipelineStageModule:
    """The split logic is covered elsewhere."""
    return hand_built_stage(nn.Linear(4, 4), None, is_first=True, is_last=False, layer_offset=0)


def test_stage_without_generation_config_is_not_generative():
    """As their unsplit models do."""
    stage = _stage()
    assert stage.generation_config is None
    assert stage.can_generate() is False


def test_pp_save_writes_generation_config(tmp_path):
    stage = _stage()
    stage.config = PretrainedConfig()
    stage.generation_config = GenerationConfig(eos_token_id=[199999, 200002], do_sample=True, temperature=0.7)

    save_model_config(stage, str(tmp_path))

    written = os.path.join(tmp_path, "generation_config.json")
    assert os.path.isfile(os.path.join(tmp_path, "config.json")), "config.json must still be written"
    assert os.path.isfile(written), (
        "generation_config.json missing — a PP checkpoint would serve with default sampling/EOS"
    )
    # The custom EOS set is the payload that lives nowhere else.
    with open(written) as f:
        assert json.load(f)["eos_token_id"] == [199999, 200002]


def test_non_generative_stage_writes_no_generation_config(tmp_path):
    stage = _stage()
    stage.config = PretrainedConfig()

    save_model_config(stage, str(tmp_path))

    assert os.path.isfile(os.path.join(tmp_path, "config.json"))
    assert not os.path.isfile(os.path.join(tmp_path, "generation_config.json")), (
        "a reward/classification stage must not ship a generation config it does not have"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
