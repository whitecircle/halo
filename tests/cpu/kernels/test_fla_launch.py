"""FLA's launch options through ``apply_fla_launch_options``. ``num_warps=4`` configs give run-to-run different,
partly wrong outputs on sm_10x, and the stock autotuner picks them at Qwen3.5/3.6 shapes."""

import pytest
import torch
from fla.ops.common import chunk_delta_h
from triton.runtime.autotuner import Autotuner

from halo.kernels.fla_launch import apply_fla_launch_options, triton_autotuner
from halo.model_loading.finalization import finalize_loaded_model

_STATE_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def _state_autotuner() -> Autotuner:
    return triton_autotuner(getattr(chunk_delta_h, _STATE_KERNEL))


def _warps() -> set[int]:
    return {config.num_warps for config in _state_autotuner().configs}


@pytest.fixture
def gpu(monkeypatch):
    """Restores the state kernel's configs and cache."""
    autotuner = _state_autotuner()
    configs, cache = list(autotuner.configs), dict(autotuner.cache)

    def as_major(major: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, 0))

    yield as_major
    autotuner.configs, autotuner.cache = configs, cache


def test_the_locked_fla_offers_num_warps_4_here():
    """Non-vacuity: without ``num_warps=4`` configs to drop, the restriction tests prove nothing."""
    assert 4 in _warps()


def test_blackwell_keeps_only_num_warps_2(gpu):
    gpu(10)
    apply_fla_launch_options()
    assert _warps() == {2}


@pytest.mark.parametrize("major", [8, 9, 12])
def test_other_architectures_keep_every_config(gpu, major):
    before = _warps()
    gpu(major)
    apply_fla_launch_options()
    assert _warps() == before


def test_cpu_only_process_keeps_every_config(gpu, monkeypatch):
    before = _warps()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    apply_fla_launch_options()
    assert _warps() == before


def test_finalizing_a_loaded_model_applies_the_restriction(gpu):
    gpu(10)
    finalize_loaded_model(torch.nn.Linear(2, 2))
    assert _warps() == {2}


def test_a_cached_num_warps_4_pick_is_dropped(gpu):
    autotuner = _state_autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 4)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    apply_fla_launch_options()
    assert autotuner.cache == {}


def test_a_cached_num_warps_2_pick_survives(gpu):
    autotuner = _state_autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 2)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    apply_fla_launch_options()
    assert autotuner.cache == {("shape",): picked}


def test_unautotuned_state_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.setattr(chunk_delta_h, _STATE_KERNEL, lambda *args: None)
    with pytest.raises(RuntimeError, match="not an autotuned FLA kernel here; re-check the Blackwell restriction"):
        apply_fla_launch_options()


def test_moved_state_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.delattr(chunk_delta_h, _STATE_KERNEL)
    with pytest.raises(RuntimeError, match="re-check the Blackwell restriction"):
        apply_fla_launch_options()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
