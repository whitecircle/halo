"""Measurement provenance through real efficiency hooks, with only hardware answers scripted."""

import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from halo.callbacks import efficiency
from halo.callbacks.efficiency import EfficiencyCallback
from halo.diagnostics.profiling import CudaMemoryGiB
from tests.common.efficiency_observation import ObservedEfficiencyCallback
from tests.common.parallelism import make_parallelism_config
from tests.common.reporting import BENCH_DECIMALS, snapshot_efficiency

_MEMORY = CudaMemoryGiB(1.0, 2.0, 3.0, 4.0)
_PHASE_STEPS = (3, 1)


class _Hardware:
    def __init__(self):
        self.now = 0.0
        self.memory = _MEMORY
        self.read_calls = 0

    def perf_counter(self):
        return self.now

    def read_memory(self):
        self.read_calls += 1
        return self.memory


@pytest.fixture
def hardware(monkeypatch):
    device = _Hardware()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(efficiency, "time", SimpleNamespace(perf_counter=device.perf_counter))
    monkeypatch.setattr(efficiency, "read_cuda_memory", device.read_memory)
    return device


def _phase(hardware, start_step=0, *, tokens="measured", seconds=0.25, callback_type=ObservedEfficiencyCallback):
    callback = callback_type(make_parallelism_config(world_size=1, gpus_per_node=1), n_warmup_steps=1)
    args = SimpleNamespace(
        include_num_input_tokens_seen="all",
        logging_steps=1,
        max_length=32,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
        tf32=False,
    )
    callback.on_init_end(args, SimpleNamespace(), None)
    callback.on_train_begin(args, SimpleNamespace(global_step=start_step), None)
    for step in range(start_step + 1, 5):
        callback.on_step_begin(args, SimpleNamespace(global_step=step - 1), None)
        hardware.now += seconds
        count = {"measured": step * 32, "missing": None, "stuck": 32, "implausible": step * 10000}[tokens]
        callback.on_step_end(args, SimpleNamespace(global_step=step, num_input_tokens_seen=count), None)
    return callback


@pytest.mark.parametrize("memory", (_MEMORY, CudaMemoryGiB(0.001, 0.001, 0.001, 0.001)))
def test_both_phases_accept_real_samples_including_rounded_zero(hardware, memory):
    hardware.memory = memory
    for start_step, expected_steps in zip((0, 2), _PHASE_STEPS, strict=True):
        callback = _phase(hardware, start_step)
        assert all(callback.measurement_checks(expected_steps).values())
        assert callback.measured_step_times == (0.25,) * expected_steps
        assert callback.tps.avg_tokens_per_second == 128.0
        assert callback.memory_readings == [memory] * expected_steps
        assert callback.memory.peak_allocated_gb == round(memory.peak_allocated, BENCH_DECIMALS)
    assert hardware.read_calls == sum(_PHASE_STEPS)


def test_observation_preserves_the_base_callback_metrics(hardware):
    base = _phase(hardware, callback_type=EfficiencyCallback)
    observed = _phase(hardware)
    assert all(observed.measurement_checks(3).values())
    assert base.measured_step_times == observed.measured_step_times
    assert snapshot_efficiency(base) == snapshot_efficiency(observed)
    assert hardware.read_calls == 6


@pytest.mark.parametrize("bad_phase", (0, 1))
@pytest.mark.parametrize("tokens", ("missing", "stuck", "implausible"))
def test_either_phase_refuses_positive_throughput_from_token_estimates(hardware, bad_phase, tokens):
    for phase, (start_step, expected_steps) in enumerate(zip((0, 2), _PHASE_STEPS, strict=True)):
        callback = _phase(hardware, start_step, tokens=tokens if phase == bad_phase else "measured")
        checks = callback.measurement_checks(expected_steps)
        assert callback.tps.avg_tokens_per_second > 0
        assert checks == {"steps_measured": True, "tokens_measured": phase != bad_phase, "memory_measured": True}
        assert callback.token_estimate_calls == (expected_steps if phase == bad_phase else 0)


@pytest.mark.parametrize("bad_phase", (0, 1))
def test_either_phase_refuses_a_missing_memory_read_despite_default_zero(hardware, bad_phase):
    for phase, (start_step, expected_steps) in enumerate(zip((0, 2), _PHASE_STEPS, strict=True)):
        hardware.memory = None if phase == bad_phase else _MEMORY
        callback = _phase(hardware, start_step)
        checks = callback.measurement_checks(expected_steps)
        assert checks == {"steps_measured": True, "tokens_measured": True, "memory_measured": phase != bad_phase}
        if phase == bad_phase:
            assert callback.tps.avg_tokens_per_second > 0 and callback.memory.peak_allocated_gb == 0
            assert callback.memory_readings == []
    assert hardware.read_calls == sum(_PHASE_STEPS)


@pytest.mark.parametrize("bad_phase", (0, 1))
@pytest.mark.parametrize("fault", ("skip", "discard"))
def test_either_phase_refuses_skipped_or_discarded_base_memory_collection(hardware, bad_phase, fault):
    def collector(_callback):
        if fault == "discard":
            efficiency.read_cuda_memory()

    for phase, (start_step, expected_steps) in enumerate(zip((0, 2), _PHASE_STEPS, strict=True)):
        with patch.object(EfficiencyCallback, "_record_memory", collector) if phase == bad_phase else nullcontext():
            callback = _phase(hardware, start_step)
        checks = callback.measurement_checks(expected_steps)
        assert checks == {"steps_measured": True, "tokens_measured": True, "memory_measured": phase != bad_phase}
        assert len(callback.memory_readings) == (0 if phase == bad_phase and fault == "skip" else expected_steps)


@pytest.mark.parametrize("bad_phase", (0, 1))
@pytest.mark.parametrize("field", CudaMemoryGiB._fields)
@pytest.mark.parametrize("value", (math.nan, math.inf, -1.0))
def test_either_phase_refuses_invalid_raw_memory_samples(hardware, bad_phase, field, value):
    for phase, (start_step, expected_steps) in enumerate(zip((0, 2), _PHASE_STEPS, strict=True)):
        hardware.memory = _MEMORY._replace(**{field: value}) if phase == bad_phase else _MEMORY
        callback = _phase(hardware, start_step)
        assert callback.measurement_checks(expected_steps)["memory_measured"] == (phase != bad_phase)


@pytest.mark.parametrize("seconds", (-0.25, math.nan, math.inf))
@pytest.mark.parametrize("start_step,expected_steps", ((0, 3), (2, 1)))
def test_invalid_elapsed_time_cannot_qualify_as_a_measured_step(hardware, seconds, start_step, expected_steps):
    callback = _phase(hardware, start_step, seconds=seconds)
    checks = callback.measurement_checks(expected_steps)
    assert not checks["steps_measured"] and not checks["tokens_measured"]


@pytest.mark.parametrize("start_step", (0, 2))
def test_zero_elapsed_time_fails_in_the_base_callback(hardware, start_step):
    with pytest.raises(ZeroDivisionError):
        _phase(hardware, start_step, seconds=0.0).measurement_checks(1)


@pytest.mark.parametrize("expected_steps", (0, 2, 4))
@pytest.mark.parametrize("start_step", (0, 2))
def test_wrong_measurement_count_cannot_pass(hardware, expected_steps, start_step):
    assert not any(_phase(hardware, start_step).measurement_checks(expected_steps).values())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
