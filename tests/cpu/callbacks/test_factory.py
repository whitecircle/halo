#!/usr/bin/env python
"""The run's callback roster (:func:`halo.callbacks.factory.build_run_callbacks`): its order, the
balancing callbacks a resolved mode selects, and the refusal of an unresolved mode.

Run: python tests/cpu/callbacks/test_factory.py
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerCallback

from halo.callbacks.factory import build_run_callbacks
from tests.common.parallelism import make_parallelism_config

_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)


class _GenerateStandIn(TrainerCallback):
    pass


def _moe_model() -> torch.nn.Module:
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(num_experts=8, num_experts_per_tok=2, output_router_logits=False)
    return model


def _every_callback_args() -> SimpleNamespace:
    return SimpleNamespace(
        enable_efficiency_metrics=True,
        enable_moe_metrics=True,
        enable_torch_profiler=True,
        router_balancing_rate=1e-3,
        num_full_model_params=None,
        report_mfu_diagnostics=False,
        profiler_output_dir="unused",
        profiler_wait=1,
        profiler_warmup=1,
        profiler_active=1,
        profiler_ranks="0",
        profiler_record_memory_snapshot=False,
    )


def _roster(balancing_mode: str) -> list[str]:
    callbacks = build_run_callbacks(
        _every_callback_args(),
        SimpleNamespace(logging_steps=1, gradient_accumulation_steps=1),
        _moe_model(),
        _NO_PARALLELISM,
        balancing_mode=balancing_mode,
        generate_callback=_GenerateStandIn(),
    )
    return [cb.__name__ if isinstance(cb, type) else type(cb).__name__ for cb in callbacks]


@pytest.mark.parametrize(
    ("balancing_mode", "balancing_callback"),
    [
        ("aux_loss", "MoEMetricsCallback"),
        ("none", "MoEMetricsCallback"),
        ("bias_update", "BalancingBiasCallback"),
        ("bias_update_transient", "BalancingBiasCallback"),
    ],
)
def test_the_roster_order(balancing_mode, balancing_callback):
    assert _roster(balancing_mode) == [
        "ParameterStatsCallback",
        "_GenerateStandIn",
        "EfficiencyCallback",
        balancing_callback,
        "TorchProfilerCallback",
    ]


def test_an_unresolved_mode_is_refused():
    """``auto`` reaching the factory means the collective configure step was skipped."""
    with pytest.raises(ValueError, match="configure_moe_balancing"):
        _roster("auto")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
