"""Timing, operand rotation and the run record the single-GPU profiling benchmarks share: per-repeat samples, never
a bare mean, and the GPU's clocks, power and software stack beside them."""

import datetime
import functools
import importlib.metadata
import itertools
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence

import pynvml
import torch
from torch.profiler import ProfilerActivity, profile

PACKAGES = (
    "torch",
    "transformers",
    "flash-attn-4",
    "flash-attn",
    "nvidia-cutlass-dsl",
    "triton",
    "liger-kernel",
    "peft",
    "deep-gemm",
    "torchao",
)
# Read at import by the NGC base image: set, it lets cuBLAS take TF32 even at float32 matmul precision "highest".
TF32_OVERRIDE_VAR = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
# Operand copies span this many L2 caches, so each call reads its operands from HBM as a model's next layer would.
L2_ROTATION_FACTOR = 4


def summarize(samples: Sequence[float]) -> dict:
    """Median, extremes and ``spread`` ((max - min) / median) of per-repeat samples."""
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return {
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "spread": (ordered[-1] - ordered[0]) / median if median else 0.0,
        "n": len(ordered),
    }


def _warm(fn: Callable[[], object], calls: int) -> None:
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()


def time_device(
    fn: Callable[[], object], *, iters: int, repeats: int, warmup: int, cooldown: float = 0.0
) -> list[float]:
    """Seconds per call of ``fn``, one sample per repeat: CUDA events around ``iters`` back-to-back calls, so a call
    shorter than its host launch cost reads as that cost. With ``cooldown``, each repeat follows that many idle seconds
    and a fresh warm-up, so it starts below the board's power limit rather than throttled by the repeat before."""
    if not cooldown:
        _warm(fn, warmup)
    samples = []
    for _ in range(repeats):
        if cooldown:
            time.sleep(cooldown)
            _warm(fn, warmup)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3 / iters)
    return samples


def capture_graph(fn: Callable[[], object], calls: int, warmup: int) -> torch.cuda.CUDAGraph:
    """``calls`` back-to-back calls of ``fn`` recorded in one CUDA graph, after ``warmup`` calls on a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    return graph


def time_graph(
    fn: Callable[[], object], *, iters: int, repeats: int, warmup: int, cooldown: float = 0.0
) -> list[float]:
    """Seconds per call of ``fn`` with ``iters`` calls replayed as one CUDA graph: the device time, without the host's
    per-call launch cost. ``cooldown`` as in :func:`time_device`."""
    graph = capture_graph(fn, iters, warmup)
    samples = time_device(graph.replay, iters=1, repeats=repeats, warmup=2, cooldown=cooldown)
    return [seconds / iters for seconds in samples]


def calls_per_repeat(seconds_per_call: float, repeat_seconds: float, floor: int = 1, cap: int = 100_000) -> int:
    """Calls that fill a repeat of about ``repeat_seconds`` given a per-call estimate, kept within ``[floor, cap]``."""
    return min(cap, max(floor, math.ceil(repeat_seconds / seconds_per_call)))


def _peak_bytes_above(step: Callable[[], object]) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def peak_and_seconds(step: Callable[[], object], *, repeats: int, repeat_seconds: float) -> dict:
    """``step``'s allocator peak above what it held before the call (measured on a warm call, past any JIT compile and
    workspace allocation) and its seconds per call over ``repeats`` repeats of about ``repeat_seconds``; ``{"oom":
    message}`` when it does not fit."""
    try:
        step()
        peak = _peak_bytes_above(step)
        probe = summarize(time_device(step, iters=1, repeats=3, warmup=1))["median"]
        iters = calls_per_repeat(probe, repeat_seconds)
        samples = time_device(step, iters=iters, repeats=repeats, warmup=1)
    except torch.cuda.OutOfMemoryError as exc:
        return {"oom": str(exc).splitlines()[0]}
    return {
        "peak_bytes_above_inputs": peak,
        "calls_per_repeat": iters,
        "seconds": samples,
        "seconds_summary": summarize(samples),
    }


def profile_kernels(fn: Callable[[], object], calls: int = 20) -> dict:
    """Kernels one call of ``fn`` launches, read from a CUDA profiler trace over ``calls`` calls: count per call, each
    name's count per call with its launch grid and block, and every kernel's device microseconds."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "trace.json")
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as trace:
            events = json.load(trace)["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    names = Counter(event["name"] for event in kernels)
    geometry = {
        event["name"]: {"grid": event["args"].get("grid"), "block": event["args"].get("block")} for event in kernels
    }
    return {
        "kernels_per_call": len(kernels) / calls,
        "kernels": {name: {"per_call": count / calls, **geometry[name]} for name, count in names.items()},
        "kernel_us": sorted(event["dur"] for event in kernels),
    }


def rotation_count(set_bytes: int) -> int:
    """Copies of a ``set_bytes`` operand set that together exceed :data:`L2_ROTATION_FACTOR` L2 caches."""
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(1, math.ceil(L2_ROTATION_FACTOR * l2 / set_bytes))


def rotating(sets: Sequence[tuple]) -> Callable[[], tuple]:
    """Each call returns the next operand set, cycling."""
    cycle = itertools.cycle(sets)
    return lambda: next(cycle)


@functools.cache
def _nvml_handle():
    pynvml.nvmlInit()
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")


def gpu_state() -> dict:
    """Board power, temperature, clocks and active clock-event reasons of the current GPU, read now from NVML."""
    handle = _nvml_handle()
    return {
        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1e3,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
        "clock_event_reasons": pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle),
    }


class ClockSampler:
    """Samples the current GPU's SM clock and board power from NVML every ``period`` seconds on a thread, between
    ``__enter__`` and ``__exit__``; ``samples`` holds ``(unix_time, sm_clock_mhz, power_w)``."""

    def __init__(self, period: float):
        self.period = period
        self.samples: list[tuple[float, int, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ClockSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        handle = _nvml_handle()
        while not self._stop.is_set():
            self.samples.append(
                (
                    time.time(),
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1e3,
                )
            )
            time.sleep(self.period)


def run_record() -> dict:
    """The GPU, driver, software versions and command of this run, with the GPU's state at its start."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    handle = _nvml_handle()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "command": [sys.executable, *sys.argv],
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "compute_capability": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "memory_bytes": props.total_memory,
            "memory_bus_width_bits": props.memory_bus_width,
            "l2_bytes": props.L2_cache_size,
            "max_threads_per_sm": props.max_threads_per_multi_processor,
            "max_threads_per_block": props.max_threads_per_block,
            "registers_per_sm": props.regs_per_multiprocessor,
            "shared_memory_per_sm_bytes": props.shared_memory_per_multiprocessor,
            "shared_memory_per_block_optin_bytes": props.shared_memory_per_block_optin,
            "warp_size": props.warp_size,
            "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1e3,
            "max_sm_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "max_mem_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM),
        },
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "versions": versions,
        "env": {TF32_OVERRIDE_VAR: os.environ.get(TF32_OVERRIDE_VAR)},
        "state_at_start": gpu_state(),
    }


def write_record(path: str, record: dict) -> None:
    """Write ``record`` as JSON, replacing the file whole so a partial run leaves the last complete one."""
    partial = f"{path}.partial"
    with open(partial, "w", encoding="utf-8") as out:
        json.dump(record, out, indent=1)
    os.replace(partial, path)
