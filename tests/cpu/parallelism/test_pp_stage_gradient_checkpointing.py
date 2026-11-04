#!/usr/bin/env python
"""A pipeline stage honors the trainer's gradient-checkpointing call as ``PreTrainedModel`` does.
Reentrant stays refused, since FSDP2 registers no pre-backward hooks under ``no_grad``."""

from __future__ import annotations

import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2


def _stage(pp_rank: int):
    cfg = Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1)
    return build_pipeline_stage(Qwen3ForCausalLM(cfg), pp_rank, PP_SIZE)


def _layer_flags(stage) -> list[bool]:
    return [m.gradient_checkpointing for m in stage.model.modules() if isinstance(m, GradientCheckpointingLayer)]


@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
def test_every_n_layers_counts_from_the_stages_first_layer(pp_rank):
    stage = _stage(pp_rank)
    stage.gradient_checkpointing_enable(every_n_layers=2)
    flags = _layer_flags(stage)
    assert flags and flags == [index % 2 == 0 for index in range(len(flags))], flags
    assert stage.is_gradient_checkpointing

    stage.gradient_checkpointing_enable()
    assert all(_layer_flags(stage))

    stage.gradient_checkpointing_disable()
    assert not any(_layer_flags(stage))
    assert not stage.is_gradient_checkpointing


def test_reentrant_checkpointing_is_refused():
    stage = _stage(0)
    with pytest.raises(ValueError, match="use_reentrant=True"):
        stage.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    assert not stage.is_gradient_checkpointing


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
