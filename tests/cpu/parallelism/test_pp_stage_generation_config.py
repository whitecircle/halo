#!/usr/bin/env python
"""A pipeline stage must carry the generation config, or PP checkpoints ship unservable.

``save_model_config`` is the one home that writes ``config.json`` **and**
``generation_config.json`` together, because ``PretrainedConfig`` does not carry the latter and
custom ``eos_token_id`` sets (harmony/GPT-OSS), stop strings and sampling defaults live nowhere
else. The PP save path hands it a :class:`PipelineStageModule`, not the original model — so unless
the stage mirrors ``generation_config`` and ``can_generate``, every PP checkpoint is written with
``config.json`` alone and served with default sampling and EOS. It trains and resumes; it just
generates wrongly, silently.

Run: ``pytest -m cpu tests/cpu/parallelism/test_pp_stage_generation_config.py``
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch.nn as nn
from transformers import GenerationConfig, PretrainedConfig

from src.checkpoint.config_export import save_model_config
from src.distributed.pipeline_parallel.stage import PipelineStageModule


def _stage() -> PipelineStageModule:
    """A minimal stage; the split logic is covered elsewhere — this pins the config-carrying seam."""
    return PipelineStageModule(
        nn.Linear(4, 4),
        None,
        is_first=True,
        is_last=False,
        backbone_prefix="model",
        head_attr="lm_head",
        layer_attr="layers",
        layer_offset=0,
    )


def test_stage_without_generation_config_is_not_generative():
    """Reward/classification stages must report False, as their unsplit models do."""
    stage = _stage()
    assert stage.generation_config is None
    assert stage.can_generate() is False


def test_pp_save_writes_generation_config(tmp_path):
    """The writing end: save_model_config must emit BOTH files for a generative stage."""
    stage = _stage()
    stage.config = PretrainedConfig()
    stage.generation_config = GenerationConfig(eos_token_id=[199999, 200002], do_sample=True, temperature=0.7)

    save_model_config(stage, str(tmp_path))

    written = os.path.join(tmp_path, "generation_config.json")
    assert os.path.isfile(os.path.join(tmp_path, "config.json")), "config.json must still be written"
    assert os.path.isfile(written), (
        "generation_config.json missing — a PP checkpoint would serve with default sampling/EOS"
    )
    # The custom EOS set is the payload that lives nowhere else; assert it survived, not just the file.
    with open(written) as f:
        assert json.load(f)["eos_token_id"] == [199999, 200002]


def test_non_generative_stage_writes_no_generation_config(tmp_path):
    """Anti-vacuity: the writer must not emit the file for a stage that cannot generate."""
    stage = _stage()
    stage.config = PretrainedConfig()

    save_model_config(stage, str(tmp_path))

    assert os.path.isfile(os.path.join(tmp_path, "config.json"))
    assert not os.path.isfile(os.path.join(tmp_path, "generation_config.json")), (
        "a reward/classification stage must not ship a generation config it does not have"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
