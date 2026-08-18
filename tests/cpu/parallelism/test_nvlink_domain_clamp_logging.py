#!/usr/bin/env python
"""The single-node nvlink_domain_size clamp must be visible in the log.

A rack-wide ``NVLINK_DOMAIN_SIZE`` (e.g. 72, baked into a host environment) on a single-node job
is legal — the domain is clamped to the job's world — but a silent clamp leaves the summary showing
only the clamped value, so a user checking why their domain setting "did not take" has nothing to
find.

Run: python tests/cpu/parallelism/test_nvlink_domain_clamp_logging.py (or pytest -m cpu).
"""

from unittest.mock import patch

import pytest

from tests.common.parallelism import create_config
from tests.cpu.parallelism.test_parallelism_config import _MOD


def _messages(mock_logger) -> list[str]:
    return [str(call.args[0]) for call in mock_logger.info.call_args_list + mock_logger.warning.call_args_list]


def test_single_node_clamp_is_logged():
    with patch(f"{_MOD}.logger") as mock_logger:
        cfg = create_config(nvlink_domain_size=72, world_size=8, gpus_per_node=8)
    assert cfg.nvlink_domain_size == 8, "clamp semantics must not change"
    clamp_lines = [m for m in _messages(mock_logger) if "clamp" in m.lower() and "72" in m and "8" in m]
    assert clamp_lines, f"clamp 72 -> 8 was not logged; logged: {_messages(mock_logger)}"


def test_unclamped_domain_logs_nothing_about_clamping():
    with patch(f"{_MOD}.logger") as mock_logger:
        cfg = create_config(nvlink_domain_size=8, world_size=8, gpus_per_node=8)
    assert cfg.nvlink_domain_size == 8
    clamp_lines = [m for m in _messages(mock_logger) if "clamp" in m.lower()]
    assert not clamp_lines, f"no clamp happened, nothing to log; logged: {clamp_lines}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
