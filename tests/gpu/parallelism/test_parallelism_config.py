#!/usr/bin/env python
"""
Unit tests for ParallelismConfig validation logic.

Tests the ParallelismConfig dataclass from src.distributed.parallelism_config
for correctness of computed fields (dp_size, mode flags, mode_string) and
validation of supported/unsupported parallelism combinations.

Runs under torchrun to provide a distributed context (required by ParallelismConfig
for auto-detecting world_size and ranks), but the tests themselves are mostly
CPU-based math validation.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/test_parallelism_config.py
"""

import os
import traceback

import torch
import torch.distributed as dist

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.utils import log

# Individual Test Functions


def test_default_config() -> bool:
    """ParallelismConfig() with all defaults should have all sizes = 1."""
    try:
        config = ParallelismConfig()
        world_size = dist.get_world_size()

        assert config.ep_size == 1, f"Expected ep_size=1, got {config.ep_size}"
        assert config.cp_size == 1, f"Expected cp_size=1, got {config.cp_size}"
        assert config.tp_size == 1, f"Expected tp_size=1, got {config.tp_size}"
        assert config.expert_tp_size == 1, f"Expected expert_tp_size=1, got {config.expert_tp_size}"
        assert config.data_parallel_size == world_size, (
            f"Expected dp_size={world_size}, got {config.data_parallel_size}"
        )
        assert config.world_size == world_size, f"Expected world_size={world_size}, got {config.world_size}"
        assert not config.is_ep_mode, "Default config should not be EP mode"
        assert not config.is_cp_mode, "Default config should not be CP mode"
        assert not config.is_tp_mode, "Default config should not be TP mode"
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_ep_only() -> bool:
    """EP-only: dp_size should equal world_size (EP is orthogonal to DP)."""
    try:
        world_size = dist.get_world_size()
        config = ParallelismConfig(ep_size=world_size)

        assert config.is_ep_mode, "Should be EP mode"
        assert not config.is_cp_mode, "Should not be CP mode"
        assert not config.is_tp_mode, "Should not be TP mode"
        # EP is orthogonal to DP: dp_size = world_size
        assert config.data_parallel_size == world_size, (
            f"EP-only dp_size should be {world_size}, got {config.data_parallel_size}"
        )
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_tp_only() -> bool:
    """TP-only: dp_size should equal world_size / tp_size."""
    try:
        world_size = dist.get_world_size()
        tp_size = 2
        if world_size < tp_size:
            log("    SKIP: world_size < tp_size")
            return True

        config = ParallelismConfig(tp_size=tp_size)

        assert config.is_tp_mode, "Should be TP mode"
        assert not config.is_ep_mode, "Should not be EP mode"
        assert not config.is_cp_mode, "Should not be CP mode"
        expected_dp = world_size // tp_size
        assert config.data_parallel_size == expected_dp, (
            f"TP-only dp_size should be {expected_dp}, got {config.data_parallel_size}"
        )
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_cp_only() -> bool:
    """CP-only: dp_size should equal world_size / cp_size."""
    try:
        world_size = dist.get_world_size()
        cp_size = 2
        if world_size < cp_size:
            log("    SKIP: world_size < cp_size")
            return True

        config = ParallelismConfig(cp_size=cp_size)

        assert config.is_cp_mode, "Should be CP mode"
        assert not config.is_ep_mode, "Should not be EP mode"
        assert not config.is_tp_mode, "Should not be TP mode"
        expected_dp = world_size // cp_size
        assert config.data_parallel_size == expected_dp, (
            f"CP-only dp_size should be {expected_dp}, got {config.data_parallel_size}"
        )
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_ep_cp() -> bool:
    """EP+CP: both is_ep_mode and is_cp_mode should be True."""
    try:
        world_size = dist.get_world_size()
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

        # EP+CP requires ep_group_size == gpus_per_node for node-local EP
        ep_size = gpus_per_node
        cp_size = gpus_per_node

        config = ParallelismConfig(ep_size=ep_size, cp_size=cp_size)

        assert config.is_ep_mode, "Should be EP mode"
        assert config.is_cp_mode, "Should be CP mode"
        assert config.is_ep_cp_mode, "Should be EP+CP mode"
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_ep_tp() -> bool:
    """EP+TP: both is_ep_mode and is_tp_mode should be True."""
    try:
        world_size = dist.get_world_size()
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

        # TP must divide gpus_per_node, EP must also be valid
        tp_size = 2
        ep_size = gpus_per_node

        if gpus_per_node < tp_size:
            log("    SKIP: gpus_per_node < tp_size")
            return True

        config = ParallelismConfig(ep_size=ep_size, tp_size=tp_size)

        assert config.is_ep_mode, "Should be EP mode"
        assert config.is_tp_mode, "Should be TP mode"
        assert config.is_ep_tp_mode, "Should be EP+TP mode"
        expected_dp = world_size // tp_size
        assert config.data_parallel_size == expected_dp, (
            f"EP+TP dp_size should be {expected_dp}, got {config.data_parallel_size}"
        )
        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_invalid_tp_cp() -> bool:
    """TP+CP combination should raise ValueError (not supported)."""
    try:
        try:
            _config = ParallelismConfig(tp_size=2, cp_size=2)
            # If we reach here, the validation did NOT raise
            log("    ERROR: Expected ValueError for TP+CP, but no exception raised")
            return False
        except ValueError as ve:
            # Expected: TP+CP is not supported
            error_msg = str(ve).lower()
            if "tp" in error_msg and "cp" in error_msg:
                return True
            else:
                log(f"    ERROR: ValueError raised but message unexpected: {ve}")
                return False
    except Exception as e:
        log(f"    EXCEPTION (unexpected): {e}")
        return False


def test_mode_string() -> bool:
    """Verify mode_string returns correct strings for each mode."""
    try:
        world_size = dist.get_world_size()
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

        # No parallelism and no grouped GEMM -> mode_string is None
        config_default = ParallelismConfig(use_grouped_gemm=False)
        assert config_default.mode_string is None, (
            f"No-mode (grouped GEMM off) mode_string should be None, got '{config_default.mode_string}'"
        )

        # Grouped GEMM only (no parallelism) -> 'gmm'
        config_gmm = ParallelismConfig(use_grouped_gemm=True)
        assert config_gmm.mode_string == "gmm", (
            f"Grouped-GEMM-only mode_string should be 'gmm', got '{config_gmm.mode_string}'"
        )

        # EP-only
        config_ep = ParallelismConfig(ep_size=gpus_per_node)
        assert config_ep.mode_string == "ep", f"EP-only mode_string should be 'ep', got '{config_ep.mode_string}'"

        # TP-only
        config_tp = ParallelismConfig(tp_size=2)
        assert config_tp.mode_string == "tp", f"TP-only mode_string should be 'tp', got '{config_tp.mode_string}'"

        # CP-only
        config_cp = ParallelismConfig(cp_size=2)
        assert config_cp.mode_string == "cp", f"CP-only mode_string should be 'cp', got '{config_cp.mode_string}'"

        # EP+TP
        config_ep_tp = ParallelismConfig(ep_size=gpus_per_node, tp_size=2)
        assert "ep" in config_ep_tp.mode_string, (
            f"EP+TP mode_string should contain 'ep', got '{config_ep_tp.mode_string}'"
        )
        assert "tp" in config_ep_tp.mode_string, (
            f"EP+TP mode_string should contain 'tp', got '{config_ep_tp.mode_string}'"
        )

        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_summary() -> bool:
    """Verify summary() returns a non-empty string."""
    try:
        config = ParallelismConfig()
        summary_str = config.summary()
        assert isinstance(summary_str, str), "summary() should return a string"
        assert len(summary_str) > 0, "summary() should return non-empty string"
        assert "PARALLELISM" in summary_str, f"summary() should contain 'PARALLELISM', got: {summary_str[:100]}"

        # Also check summary for EP mode
        world_size = dist.get_world_size()
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        config_ep = ParallelismConfig(ep_size=gpus_per_node)
        summary_ep = config_ep.summary()
        assert "EP" in summary_ep, f"EP summary should contain 'EP', got: {summary_ep[:100]}"

        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


def test_needs_ep_wrappers() -> bool:
    """needs_ep_wrappers should be True when ep_size > 1 or use_grouped_gemm=True."""
    try:
        # Default: use_grouped_gemm=True by default, so wrappers are needed
        config_default = ParallelismConfig()
        assert config_default.needs_ep_wrappers, (
            "Default config should need EP wrappers (use_grouped_gemm defaults to True)"
        )

        # Explicitly disabled: no EP wrappers needed
        config_no_gmm = ParallelismConfig(use_grouped_gemm=False)
        assert not config_no_gmm.needs_ep_wrappers, (
            "Config with use_grouped_gemm=False and no EP should not need EP wrappers"
        )

        # EP enabled
        world_size = dist.get_world_size()
        gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        config_ep = ParallelismConfig(ep_size=gpus_per_node)
        assert config_ep.needs_ep_wrappers, "EP config should need EP wrappers"

        # Grouped GEMM only (no EP distribution)
        config_gmm = ParallelismConfig(use_grouped_gemm=True)
        assert config_gmm.needs_ep_wrappers, "Grouped GEMM config should need EP wrappers"

        return True
    except Exception as e:
        log(f"    EXCEPTION: {e}")
        return False


# Test Runner

ALL_TESTS = [
    ("test_default_config", test_default_config),
    ("test_ep_only", test_ep_only),
    ("test_tp_only", test_tp_only),
    ("test_cp_only", test_cp_only),
    ("test_ep_cp", test_ep_cp),
    ("test_ep_tp", test_ep_tp),
    ("test_invalid_tp_cp", test_invalid_tp_cp),
    ("test_mode_string", test_mode_string),
    ("test_summary", test_summary),
    ("test_needs_ep_wrappers", test_needs_ep_wrappers),
]


def run(ctx) -> dict:
    """Run every ParallelismConfig test, one check per entry in ALL_TESTS."""
    log(f"\n{'=' * 70}")
    log("  ParallelismConfig Validation Tests")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}\n")

    checks = {}
    for test_name, test_fn in ALL_TESTS:
        log(f"  [{test_name}] Running...")
        try:
            result = test_fn()
        except Exception as e:
            log(f"  [{test_name}] UNHANDLED EXCEPTION: {e}")
            if ctx.rank == 0:
                traceback.print_exc()
            result = False

        checks[test_name] = result
        log(f"  [{test_name}] {'PASS' if result else 'FAIL'}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="parallelism_config", partial_state=False)(run)

if __name__ == "__main__":
    main()
