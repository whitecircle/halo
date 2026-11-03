#!/usr/bin/env python
"""The PP+EP stage-coverage gate is rank-local, since a hybrid MoE can give one stage only dense layers.
Raised on one rank alone, the MoE stages walk into the chain broadcast and the job waits out a full NCCL timeout."""

from types import SimpleNamespace
from unittest import mock

import pytest

from halo.trainers.parallelism.pipeline_rules import reject_uncovered_pp_ep_layers
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin
from tests.common.parallelism import make_parallelism_config

_EP_LAYERS = [object()]


def _config(**overrides):
    """Two 4-rank stages, each exactly one NVLink domain."""
    return make_parallelism_config(world_size=8, gpus_per_node=4, pp_size=2, **overrides)


def _joined_reason(config, ep_layers) -> str | None:
    """The joiner is stubbed, so nothing raises here."""
    with mock.patch("halo.trainers.parallelism.pipeline_rules.reject_across_ranks") as join:
        reject_uncovered_pp_ep_layers(config, ep_layers)
    join.assert_called_once()
    return join.call_args.args[0]


def test_a_stage_without_ep_layers_is_refused_and_the_message_names_the_stage():
    with pytest.raises(RuntimeError, match="PP stage .* carries no EP-patched modules") as err:
        reject_uncovered_pp_ep_layers(_config(ep_size=4), [])
    assert "pipeline_parallel_size" in str(err.value)


def test_a_stage_that_carries_ep_layers_passes():
    reject_uncovered_pp_ep_layers(_config(ep_size=4), _EP_LAYERS)


def test_a_pp_run_without_ep_is_not_gated_at_all():
    """The dense stages of a dense run are the normal case."""
    reject_uncovered_pp_ep_layers(_config(), [])


def test_the_verdict_is_handed_to_the_joiner_rather_than_raised():
    """A lone raise strands the MoE stages in the chain broadcast that follows."""
    assert "carries no EP-patched modules" in _joined_reason(_config(ep_size=4), [])
    assert _joined_reason(_config(ep_size=4), _EP_LAYERS) is None


@pytest.mark.parametrize("ep_layers", [[], _EP_LAYERS], ids=["dense_stage", "ep_stage"])
def test_the_stage_wrap_builds_the_stage_group_then_gates_then_wraps(ep_layers):
    """Gating the build on the rank-local layer list deadlocks inside ``dist.new_group`` before any raise prints."""
    config = _config(ep_size=4)
    calls = mock.Mock()
    trainer = SimpleNamespace(
        parallelism_config=config,
        _fsdp_exclusions=lambda: SimpleNamespace(params=()),
        _apply_dp_fsdp2=calls.wrap,
        _apply_ep_aware_dp_fsdp2=calls.wrap,
    )
    stage = object()
    with (
        mock.patch(f"{PipelineTrainerMixin.__module__}.build_stage_group", calls.build_stage_group),
        mock.patch(f"{PipelineTrainerMixin.__module__}.reject_uncovered_pp_ep_layers", calls.reject),
    ):
        PipelineTrainerMixin._wrap_pp_stage(trainer, stage, ep_layers)
    assert [name for name, _, _ in calls.mock_calls] == ["build_stage_group", "reject", "wrap"]
    assert calls.reject.call_args.args == (config, ep_layers)
    assert calls.wrap.call_args.args[0] is stage
    assert trainer._pp_stage_group is calls.build_stage_group.return_value


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
