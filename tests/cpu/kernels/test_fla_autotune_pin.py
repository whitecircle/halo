"""The FLA autotune pin (``pin_fla_autotune_first_fit``, ``tests/common/fla_autotune.py``): every fla kernel runs
the first config that launches, past those exceeding sm_90/sm_10x shared memory."""

import gc
from types import SimpleNamespace

import pytest
import torch
import triton
from fla.ops.common import chunk_delta_h
from fla.ops.utils import cumsum
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources
from triton.runtime.jit import JITFunction

from halo.kernels.fla_launch import triton_autotuner
from tests.common import fla_autotune

_KERNEL = "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"
# A CPU import builds a shorter list.
_GPU_CONFIGS = [
    triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
    for num_warps in [2, 4]
    for num_stages in [4, 3, 2]
    for BV in [64, 32]
]
_FP32_UNLAUNCHABLE = (_GPU_CONFIGS[0], _GPU_CONFIGS[6])
_REQUIRED, _LIMIT = 271116, 232448
# Heads (value heads, keys repeated to them), key and value head dims.
_H, _K, _V = 4, 32, 32


def _tuner() -> Autotuner:
    return triton_autotuner(getattr(chunk_delta_h, _KERNEL))


def _launch_kernel() -> None:
    """On a 64-token row."""
    q, k, w = (torch.randn(1, 64, _H, _K) for _ in range(3))
    do, dv = (torch.randn(1, 64, _H, _V) for _ in range(2))
    chunk_delta_h.chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, g=torch.randn(1, 64, _H), scale=_K**-0.5)


@pytest.fixture
def launches(monkeypatch):
    """``unlaunchable`` configs raise, the rest are recorded."""
    tuner = _tuner()
    for name in ("configs", "cache", "cache_results", "_do_bench"):
        monkeypatch.setattr(tuner, name, getattr(tuner, name))
    tuner.configs, tuner.cache = list(_GPU_CONFIGS), {}
    state = SimpleNamespace(unlaunchable=(), launched=[])

    def run(self, *args, grid, warmup, **kwargs):
        config = next(c for c in _GPU_CONFIGS if c.all_kwargs().items() <= kwargs.items())
        if config in state.unlaunchable:
            raise OutOfResources(_REQUIRED, _LIMIT, "shared memory")
        state.launched.append(config)

    monkeypatch.setattr(JITFunction, "run", run)
    yield state
    vars(tuner).pop("do_bench", None)


@pytest.fixture
def pinned():
    finalizers = []
    fla_autotune.pin_fla_autotune_first_fit(SimpleNamespace(on_teardown=finalizers.append))
    yield
    for finalize in reversed(finalizers):
        finalize()


def _later_is_faster(launched):
    """Each config times faster than every config listed before it."""

    def timed(kernel_call, quantiles):
        kernel_call()
        return [-float(_GPU_CONFIGS.index(launched[-1]))] * len(quantiles)

    return timed


def test_the_single_config_pin_raises_on_the_fp32_configs(launches):
    """Non-vacuity: launching the first config alone raises the GPU row's error."""
    launches.unlaunchable = _FP32_UNLAUNCHABLE
    _tuner().configs = _GPU_CONFIGS[:1]
    with pytest.raises(OutOfResources, match=f"Required: {_REQUIRED}, Hardware limit: {_LIMIT}"):
        _launch_kernel()


def test_the_pin_skips_configs_that_cannot_launch(launches, pinned):
    launches.unlaunchable = _FP32_UNLAUNCHABLE
    _launch_kernel()
    assert _tuner().best_config == _GPU_CONFIGS[1]
    assert launches.launched[-1] == _GPU_CONFIGS[1]


def test_the_pin_keeps_the_first_config_where_it_launches(launches, pinned):
    _launch_kernel()
    assert launches.launched[-1] == _GPU_CONFIGS[0]


def test_timings_choose_without_the_pin(launches):
    """Control: the stub benchmarker is live and steers the stock autotuner to the last config."""
    tuner = _tuner()
    tuner.cache_results, tuner._do_bench = False, _later_is_faster(launches.launched)
    _launch_kernel()
    assert launches.launched[-1] == _GPU_CONFIGS[-1]


def test_timings_do_not_choose_under_the_pin(launches, pinned):
    _tuner()._do_bench = _later_is_faster(launches.launched)
    launches.unlaunchable = _FP32_UNLAUNCHABLE
    _launch_kernel()
    assert launches.launched[-1] == _GPU_CONFIGS[1]


def test_a_disk_cached_pick_is_not_reused_under_the_pin(launches, pinned, monkeypatch):
    def cached_pick(self, tuning_key, configs, bench_fn):
        self.cache[tuning_key] = configs[-1]
        return True

    monkeypatch.setattr(Autotuner, "check_disk_cache", cached_pick)
    _launch_kernel()
    assert launches.launched[-1] == _GPU_CONFIGS[0]


def test_a_kernel_with_no_launchable_config_still_raises(launches, pinned):
    launches.unlaunchable = tuple(_GPU_CONFIGS)
    with pytest.raises(OutOfResources):
        _launch_kernel()


def test_the_pin_reaches_every_fla_autotuner_the_process_holds(pinned):
    """Found through the garbage collector, since the pin finds kernels through fla's module namespaces."""
    alive = {
        id(obj): obj
        for obj in gc.get_objects()
        if issubclass(type(obj), Autotuner) and obj.base_fn.__module__.startswith("fla.")
    }
    decay = (cumsum.chunk_local_cumsum_scalar_kernel, cumsum.chunk_local_cumsum_vector_kernel)
    assert {id(triton_autotuner(kernel)) for kernel in decay} <= alive.keys()
    unpinned = [
        tuner.base_fn.__name__
        for tuner in alive.values()
        if vars(tuner).get("do_bench") is not fla_autotune._launch_once or tuner.cache_results
    ]
    assert not unpinned, f"fla autotuners the pin missed: {sorted(unpinned)}"


def test_teardown_restores_every_autotuner():
    tuners = fla_autotune._loaded_fla_autotuners()
    assert _tuner() in tuners
    before = [(tuner.cache, tuner.cache_results, "do_bench" in vars(tuner)) for tuner in tuners]
    finalizers = []
    fla_autotune.pin_fla_autotune_first_fit(SimpleNamespace(on_teardown=finalizers.append))
    assert all(vars(tuner)["do_bench"] is fla_autotune._launch_once for tuner in tuners)
    for finalize in finalizers:
        finalize()
    after = [(tuner.cache, tuner.cache_results, "do_bench" in vars(tuner)) for tuner in tuners]
    assert after == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
