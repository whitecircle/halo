#!/usr/bin/env python
"""``nvlink_domain_size`` rules that are NOT the single-node clamp.

``test_nvlink_domain_clamp_logging.py`` covers the legal single-node clamp and its log line. Three
neighbouring behaviors of the same block have no coverage, and each one silently mis-shapes every
node-local group if it regresses:

* the **multi-node refusal** — a rack-wide ``NVLINK_DOMAIN_SIZE`` on a multi-node sub-job must
  RAISE, not clamp. Clamping there would be a lie about the fabric: the ranks may span racks, and
  every node-local EP/CP group would be sized for an NVLink domain that does not exist.
* the **partial trailing domain** — ``world_size % nvlink_domain_size != 0`` truncates
  ``num_nvlink_domains`` by floor division and orphans the trailing ranks from every node-local
  group. It must raise before any group is built.
* the clamp is **rank-gated and INFO** — a per-rank log line at world scale is noise, and the
  summary shows only the clamped value, so this line is the sole record of the adjustment.

Also pinned: the clamped value actually reaches the rank math, not just the attribute.

Run: python tests/cpu/parallelism/test_nvlink_domain_size_rules.py  (or pytest -m cpu)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.common.parallelism import create_config
from tests.cpu.parallelism.test_parallelism_config import _MOD

# ── A rack-wide domain on a MULTI-NODE sub-job must raise, never clamp ──────────


def test_a_domain_larger_than_a_multi_node_job_is_refused():
    """16 GPUs over 2 nodes with NVLINK_DOMAIN_SIZE=72: the job may span racks, so the toolkit must
    not quietly decide all 16 share a domain."""
    with pytest.raises(ValueError) as err:
        create_config(nvlink_domain_size=72, world_size=16, gpus_per_node=8)
    text = str(err.value)
    assert "NVLINK_DOMAIN_SIZE" in text, "the raise must name the env var the operator actually set"
    assert "72" in text and "16" in text, f"the raise must show both sizes: {text}"
    assert "NVLINK_DOMAIN_SIZE=16" in text, "the raise must give the one-rack fix"


def test_a_domain_equal_to_the_multi_node_world_is_accepted():
    """Anti-vacuity for the refusal: the documented one-rack spelling (domain == world) must pass."""
    cfg = create_config(nvlink_domain_size=16, world_size=16, gpus_per_node=8)
    assert cfg.nvlink_domain_size == 16
    assert cfg.num_nvlink_domains == 1


# ── A world that does not divide into whole domains must raise ──────────────────


def test_a_partial_trailing_domain_is_refused():
    """24 GPUs with a 16-GPU domain: floor division would report 1 domain and orphan 8 ranks."""
    with pytest.raises(ValueError) as err:
        create_config(nvlink_domain_size=16, world_size=24, gpus_per_node=8)
    text = str(err.value)
    assert "divisible by nvlink_domain_size" in text
    assert "orphan" in text, f"the raise must name the consequence, not just the arithmetic: {text}"


def test_a_domain_smaller_than_a_node_is_refused():
    """The neighbouring rule, on the other side: a domain must be a whole number of nodes."""
    with pytest.raises(ValueError, match="multiple of gpus_per_node"):
        create_config(nvlink_domain_size=4, world_size=16, gpus_per_node=8)


@pytest.mark.parametrize(("domain", "world"), [(8, 16), (16, 32), (8, 8)])
def test_worlds_that_divide_into_whole_domains_are_accepted(domain, world):
    """Anti-vacuity: the divisibility guard must not reject the shapes it exists to protect."""
    cfg = create_config(nvlink_domain_size=domain, world_size=world, gpus_per_node=8)
    assert cfg.num_nvlink_domains == world // domain


# ── The clamp is rank-gated, INFO, and reaches the rank math ───────────────────


def test_the_clamp_line_is_info_and_only_on_the_global_main_process():
    """A per-rank clamp line is noise at world scale; the gate is what keeps it to one."""
    with patch(f"{_MOD}.logger") as main_logger:
        create_config(nvlink_domain_size=72, world_size=8, gpus_per_node=8, rank=0)
    info_lines = [str(call.args[0]) for call in main_logger.info.call_args_list]
    assert any("clamping" in line for line in info_lines), f"the clamp must be INFO: {info_lines}"

    with patch(f"{_MOD}.logger") as worker_logger:
        cfg = create_config(nvlink_domain_size=72, world_size=8, gpus_per_node=8, rank=3)
    worker_lines = [
        str(call.args[0]) for call in worker_logger.info.call_args_list + worker_logger.warning.call_args_list
    ]
    assert not any("clamping" in line for line in worker_lines), f"rank 3 logged the clamp: {worker_lines}"
    assert cfg.nvlink_domain_size == 8, "the clamp itself must NOT be rank-gated, only its log line"


def test_the_clamped_value_is_what_the_rank_math_uses():
    """The clamp must land before the domain arithmetic — an unclamped 72 would report 0 domains
    (floor division) and take every node-local group's size with it."""
    cfg = create_config(nvlink_domain_size=72, world_size=8, gpus_per_node=8)
    assert cfg.nvlink_domain_size == 8
    assert cfg.num_nvlink_domains == 1, "the clamped domain must yield exactly one domain"


def test_an_ep_group_sized_to_the_unclamped_domain_would_have_been_rejected():
    """The clamp's downstream contract: ep8 on one clamped domain is the supported single-group
    shape, and stays accepted."""
    cfg = create_config(ep_size=8, nvlink_domain_size=72, world_size=8, gpus_per_node=8)
    assert cfg.nvlink_domain_size == 8
    assert cfg.is_racy_single_domain_multigroup_ep is False
    assert cfg.num_ep_groups == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
