#!/usr/bin/env python
"""The PP+EP stage-coverage gate: a rank-local verdict that must be raised world-uniformly.

A stage with no EP-patched modules runs a different collective program than its MoE peers, so the
setup refuses it. The verdict is RANK-LOCAL — a hybrid MoE with leading dense layers can give one
stage only dense layers — and the stages that do carry EP modules walk straight on into the chain
broadcast that pins the metric keys. Raised on one rank alone, that costs the job a full NCCL
timeout with the explaining traceback only on the dead ranks; joined, every rank reports the cause.

Run: ``python tests/cpu/parallelism/test_pp_ep_coverage_gate.py`` (or ``pytest -m cpu``).
"""

import ast
import inspect

import pytest

from src.trainers.mixins import pipeline as pipeline_mixin
from src.trainers.mixins.pipeline import pp_ep_coverage_reason
from tests.common.parallelism import make_parallelism_config

_EP_MODULES = [("model.layers.0.mlp", object())]


def _config(**overrides):
    """Two 4-rank stages, each exactly one NVLink domain — the shape a PP+EP run takes."""
    return make_parallelism_config(world_size=8, gpus_per_node=4, pp_size=2, **overrides)


def test_a_stage_without_ep_modules_is_refused_and_the_message_names_the_stage():
    reason = pp_ep_coverage_reason(_config(ep_size=4), [])
    assert reason is not None
    assert "PP stage" in reason and "pipeline_parallel_size" in reason


def test_a_stage_that_carries_ep_modules_passes():
    assert pp_ep_coverage_reason(_config(ep_size=4), _EP_MODULES) is None


def test_a_pp_run_without_ep_is_not_gated_at_all():
    """The dense stages of a dense run are the normal case, not a partition mistake."""
    assert pp_ep_coverage_reason(_config(), []) is None


def test_the_setup_hands_the_verdict_to_the_joiner_rather_than_raising_it():
    """Single-process, the two shapes are indistinguishable — a lone raise and a joined one both stop
    the caller — so the call site itself is the assertion. Every use of the reason inside the mixin
    must be an argument to ``reject_across_ranks``; a bare raise is what strands the MoE stages in the
    chain broadcast that follows two lines later."""
    tree = ast.parse(inspect.getsource(pipeline_mixin.PipelineTrainerMixin))
    joined = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "reject_across_ranks"
        and any(
            getattr(arg.func, "id", None) == "pp_ep_coverage_reason" for arg in node.args if isinstance(arg, ast.Call)
        )
    ]
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "pp_ep_coverage_reason"
    ]
    assert len(uses) == 1 and len(joined) == 1, (
        f"{len(uses)} use(s) of pp_ep_coverage_reason, {len(joined)} of them joined across ranks"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
