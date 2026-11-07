"""Observe real efficiency measurements without replacing their token, clock or memory calculations."""

import math
from unittest.mock import patch

from halo.callbacks import efficiency
from tests.common.reporting import BENCH_DECIMALS


class ObservedEfficiencyCallback(efficiency.EfficiencyCallback):
    """The real callback with per-phase witnesses for fallback tokens and successful memory reads."""

    def __init__(self, parallelism_config, *, n_warmup_steps):
        super().__init__(parallelism_config, n_warmup_steps=n_warmup_steps)
        self.token_estimate_calls = 0
        self.memory_readings = []

    def measurement_checks(self, expected_steps):
        """Verdicts for measured steps, token provenance and the actual samples behind rounded memory reports."""
        times = self.measured_step_times
        measured = (
            len(times) == expected_steps
            and expected_steps > 0
            and all(math.isfinite(seconds) and seconds > 0 for seconds in times)
        )
        memory = (
            len(self.memory_readings) == expected_steps
            and expected_steps > 0
            and all(math.isfinite(value) and value >= 0 for reading in self.memory_readings for value in reading)
        )
        if memory:
            memory = self.memory.peak_allocated_gb == round(
                self.memory_readings[-1].peak_allocated, BENCH_DECIMALS
            ) and self.memory.training_peak_allocated_gb == round(
                max(reading.peak_allocated for reading in self.memory_readings), BENCH_DECIMALS
            )
        return {
            "steps_measured": measured,
            "tokens_measured": measured and self.token_estimate_calls == 0,
            "memory_measured": memory,
        }

    def _fallback_step_tokens(self, world_size, measured_cluster):
        self.token_estimate_calls += 1
        return super()._fallback_step_tokens(world_size, measured_cluster)

    def _record_memory(self):
        """Observe the base collector's device read without adding a second hardware query."""
        reader = efficiency.read_cuda_memory

        def read_memory():
            """Preserve the reader's result and retain non-None samples for later validation."""
            reading = reader()
            if reading is not None:
                self.memory_readings.append(reading)
            return reading

        with patch.object(efficiency, "read_cuda_memory", side_effect=read_memory):
            super()._record_memory()
