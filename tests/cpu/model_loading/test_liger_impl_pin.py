"""The Liger implementation pin (``pin_liger_impls``): the CuTe DSL RoPE is wrong after the first launch of a shape
under ``CUTE_DSL_ENABLE_TVM_FFI``, so ``LIGER_KERNEL_IMPL_ROPE`` pins Triton."""

import os

import pytest
from liger_kernel.backends.capability import Capability
from liger_kernel.backends.dispatch import clear_available_cache, resolve_impl
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.kernels.jit_cache import ensure_fa4_kernel_cache_env, pin_liger_impls
from halo.model_loading.liger import dispatch
from halo.model_loading.liger.dispatch import apply_liger_kernel, sanitize_liger_config

ROPE_VAR = "LIGER_KERNEL_IMPL_ROPE"
LEGACY_ROPE_VAR = "LIGER_KERNEL_BACKEND_ROPE"


@pytest.fixture
def rope_unset(monkeypatch):
    # Set first so monkeypatch removes the pin's own write at teardown.
    for var in (ROPE_VAR, LEGACY_ROPE_VAR):
        monkeypatch.setenv(var, "unset")
        monkeypatch.delenv(var)


@pytest.fixture
def every_impl_available(monkeypatch):
    monkeypatch.setattr(Capability, "_evaluate", lambda self, device=None: (True, None))
    clear_available_cache()
    yield
    clear_available_cache()


def _sanitize() -> dict:
    return sanitize_liger_config({}, needs_ep_layers=False, model_config=CONFIG_MAPPING["qwen3"]())


def test_sanitize_pins_rope_to_triton(rope_unset):
    _sanitize()
    assert os.environ[ROPE_VAR] == "nvidia-triton"


def test_load_site_pins_rope(rope_unset, monkeypatch):
    monkeypatch.setattr(dispatch, "_run_resolved_applier", lambda *args, **kwargs: {})
    apply_liger_kernel(CONFIG_MAPPING["qwen3"]())
    assert os.environ[ROPE_VAR] == "nvidia-triton"


def test_enabling_tvm_ffi_pins_rope(rope_unset, monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    monkeypatch.setenv("CUTE_DSL_ENABLE_TVM_FFI", "1")
    ensure_fa4_kernel_cache_env()
    assert os.environ[ROPE_VAR] == "nvidia-triton"


def test_an_empty_value_is_pinned(rope_unset, monkeypatch):
    monkeypatch.setenv(ROPE_VAR, "")
    pin_liger_impls()
    assert os.environ[ROPE_VAR] == "nvidia-triton"


@pytest.mark.parametrize("var", [ROPE_VAR, LEGACY_ROPE_VAR])
@pytest.mark.parametrize("spelling", ["nvidia-triton", "triton"])
def test_an_explicit_triton_choice_is_kept(rope_unset, monkeypatch, var, spelling):
    monkeypatch.setenv(var, spelling)
    pin_liger_impls()
    assert os.environ[var] == spelling


@pytest.mark.parametrize("var", [ROPE_VAR, LEGACY_ROPE_VAR])
@pytest.mark.parametrize("spelling", ["nvidia-cutedsl", "cutedsl", "nvidia-cutile"])
def test_another_explicit_choice_raises(rope_unset, monkeypatch, var, spelling):
    monkeypatch.setenv(var, spelling)
    with pytest.raises(ValueError, match=f"{var}={spelling}"):
        _sanitize()


def test_liger_auto_selects_the_cutedsl_rope_unpinned(rope_unset, every_impl_available):
    assert resolve_impl("rope").impl_name == "nvidia-cutedsl"


def test_the_pin_resolves_rope_to_triton(rope_unset, every_impl_available):
    pin_liger_impls()
    assert resolve_impl("rope").impl_name == "nvidia-triton"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
