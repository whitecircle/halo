#!/usr/bin/env python
"""configure_float32_matmul_precision pins fp32 matmuls to 'highest' over the image's TF32 default, whose 10-bit
mantissa collapses RoPE positions past 2048."""

import os

import pytest
import torch

from halo.model_loading.dtype import configure_float32_matmul_precision


@pytest.fixture(autouse=True)
def _restore_precision():
    """Saves and restores the process-global precision and the knob."""
    original = torch.get_float32_matmul_precision()
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    saved_env = os.environ.pop("HALO_FP32_MATMUL_PRECISION", None)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(original)
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        os.environ.pop("HALO_FP32_MATMUL_PRECISION", None)
        if saved_env is not None:
            os.environ["HALO_FP32_MATMUL_PRECISION"] = saved_env


def test_defaults_to_highest_true_fp32():
    torch.set_float32_matmul_precision("high")  # Stands in for the inherited TF32 default.
    configure_float32_matmul_precision()
    assert torch.get_float32_matmul_precision() == "highest"


def test_env_knob_opts_back_into_tf32():
    """The escape hatch for an fp32-heavy job that tolerates TF32."""
    os.environ["HALO_FP32_MATMUL_PRECISION"] = "high"
    configure_float32_matmul_precision()
    assert torch.get_float32_matmul_precision() == "high"


def test_invalid_value_falls_back_to_highest():
    """A malformed knob must not silently select a wrong precision."""
    torch.set_float32_matmul_precision("high")
    os.environ["HALO_FP32_MATMUL_PRECISION"] = "not-a-precision"
    configure_float32_matmul_precision()
    assert torch.get_float32_matmul_precision() == "highest"


def test_cudnn_tf32_follows_the_same_knob():
    """``torch.backends.cudnn.allow_tf32`` is a separate switch defaulting True, so an fp32 Conv1d (Zaya) would keep
    running in TF32."""
    torch.backends.cudnn.allow_tf32 = True
    configure_float32_matmul_precision()
    assert torch.backends.cudnn.allow_tf32 is False

    os.environ["HALO_FP32_MATMUL_PRECISION"] = "high"
    configure_float32_matmul_precision()
    assert torch.backends.cudnn.allow_tf32 is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
