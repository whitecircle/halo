"""FLA's GatedDeltaNet state kernel autotunes over ``num_warps=2`` configs only, once a load finishes on Blackwell.

Its ``num_warps=4`` configs give run-to-run different, partly wrong outputs on sm_10x, and the stock autotuner picks
them at Qwen3.5/3.6 shapes. These tests fail if ``finalize_loaded_model`` stops restricting the kernel on Blackwell
(whatever imported FLA: transformers' hub-kernel factory or a remote-code module), starts restricting it on another
architecture, or leaves an already-picked ``num_warps=4`` config in the autotuner's cache. The GPU behaviour itself
is pinned by ``tests/gpu/kernels/test_fla_blackwell_autotune.py``.
"""

import sys

import pytest
import torch
from fla.ops.common import chunk_delta_h

from halo.kernels import fla_autotune
from halo.kernels.fla_autotune import restrict_fla_blackwell_autotune
from halo.model_loading.buffer_repair import finalize_loaded_model


def _autotuner():
    return fla_autotune.triton_autotuner(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)


def _warps() -> set[int]:
    return {config.num_warps for config in _autotuner().configs}


@pytest.fixture
def gpu(monkeypatch):
    """Pretend to run on a GPU of the given major capability, and restore the kernel's configs and cache."""
    autotuner = _autotuner()
    configs, cache = list(autotuner.configs), dict(autotuner.cache)

    def as_major(major: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, 0))

    yield as_major
    autotuner.configs, autotuner.cache = configs, cache


def test_the_locked_fla_offers_num_warps_4_here():
    """Non-vacuity guard: without ``num_warps=4`` configs to drop, the restriction tests below would prove nothing."""
    assert 4 in _warps()


def test_blackwell_keeps_only_num_warps_2(gpu):
    gpu(10)
    restrict_fla_blackwell_autotune()
    assert _warps() == {2}


@pytest.mark.parametrize("major", [8, 9, 12])
def test_other_architectures_keep_every_config(gpu, major):
    before = _warps()
    gpu(major)
    restrict_fla_blackwell_autotune()
    assert _warps() == before


def test_cpu_only_process_keeps_every_config(gpu, monkeypatch):
    before = _warps()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restrict_fla_blackwell_autotune()
    assert _warps() == before


def test_process_without_fla_loaded_is_left_alone(gpu, monkeypatch):
    before = _warps()
    gpu(10)
    monkeypatch.delitem(sys.modules, "fla")
    restrict_fla_blackwell_autotune()
    assert _warps() == before


def test_finalizing_a_loaded_model_applies_the_restriction(gpu):
    gpu(10)
    finalize_loaded_model(torch.nn.Linear(2, 2))
    assert _warps() == {2}


def test_a_cached_num_warps_4_pick_is_dropped(gpu):
    autotuner = _autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 4)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    restrict_fla_blackwell_autotune()
    assert autotuner.cache == {}


def test_a_cached_num_warps_2_pick_survives(gpu):
    autotuner = _autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 2)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    restrict_fla_blackwell_autotune()
    assert autotuner.cache == {("shape",): picked}


def test_unautotuned_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.setattr(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", lambda *args: None)
    with pytest.raises(RuntimeError, match="no longer autotuned"):
        restrict_fla_blackwell_autotune()


def test_moved_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.delattr(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    with pytest.raises(RuntimeError, match="re-check the Blackwell gate"):
        restrict_fla_blackwell_autotune()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
