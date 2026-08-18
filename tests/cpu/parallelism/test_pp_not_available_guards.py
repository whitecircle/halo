#!/usr/bin/env python
"""The two guards that keep pipeline parallelism unlaunchable while its schedule engine is absent.

The PP seams ship — the config surface, the rank math, the validators, the stage/loss contracts —
but not the engine that would drive microbatches through the stages. Two raises stand between a
user and a run that would otherwise build a pipeline nothing can step:

* ``parallelism_config_from_args`` refuses ``pipeline_parallel_size > 1`` at the single production
  entry point, ahead of every rank-math and axis-set validator, so the refusal reaches the user
  before any process group or model load.
* ``PipelineRuntime`` refuses construction, so a hand-built config that skipped the entry point
  still cannot reach a schedule.

Both messages must name the release status and route to the doc page: a run rejected without a
remedy is as unusable as one that hangs.

Run: python tests/cpu/parallelism/test_pp_not_available_guards.py
"""

import pytest

from src.args.distributed_args import DistributedArguments
from src.distributed.pipeline_parallel.runtime import PipelineRuntime
from src.training.parallelism_args import parallelism_config_from_args

PP_DOC = "agent-docs/parallelism/pipeline-parallelism.md"


def test_pp_size_one_still_builds():
    """Anti-vacuity for the rejection below: the default path through the same builder is unaffected,
    so the raise is about ``pp_size > 1`` and not about the builder refusing everything."""
    config = parallelism_config_from_args(DistributedArguments())
    assert config.pp_size == 1
    assert not config.is_pp_mode


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"expert_parallel_size": 2},  # PP+EP is in SUPPORTED_AXIS_SETS — the release gate still wins
        {"tensor_parallel_size": 2},  # PP+TP is not — the release gate must answer first anyway
    ],
)
def test_pipeline_parallel_size_above_one_is_rejected_at_config_time(extra):
    """``supports_pp=True`` is the SFT spelling — the trainer-support gate is NOT what stops PP here,
    so the message must be the release one for every axis combination, allowlisted or not."""
    dist_args = DistributedArguments(pipeline_parallel_size=2, **extra)
    with pytest.raises(ValueError) as excinfo:
        parallelism_config_from_args(dist_args, supports_pp=True)

    message = str(excinfo.value)
    assert "not yet available in this release" in message, message
    assert "pipeline_parallel_size=1" in message, f"the message must name the remedy: {message}"
    assert PP_DOC in message, f"the message must route to the doc page: {message}"


def test_constructing_the_pipeline_runtime_raises():
    """The engine itself refuses, so the seam cannot be driven from a hand-built config. The guard is
    the constructor's first statement, so placeholder arguments reach it."""
    with pytest.raises(NotImplementedError) as excinfo:
        PipelineRuntime(
            stage_module=None,
            config=None,
            pp_group=None,
            device=None,
            n_microbatches=1,
        )

    message = str(excinfo.value)
    assert "not yet available in this release" in message, message
    assert PP_DOC in message, f"the message must route to the doc page: {message}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
