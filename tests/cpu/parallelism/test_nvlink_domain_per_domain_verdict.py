#!/usr/bin/env python
"""``validate_nvlink_domain_against_fabric`` judges each NVLink domain, since every fabric-less GPU reports one
``NO_FABRIC`` sentinel. The clique half is in ``tests/cpu/parallelism/test_nvlink_fabric_check.py``."""

from unittest.mock import patch

import pytest

from halo.distributed.nvlink import NO_FABRIC, validate_nvlink_domain_against_fabric

_MOD = "halo.distributed.nvlink"


def _run(domain: int, world: int, cliques: list[int], node_widths: list[int], local: int = 0):
    """``local`` is this rank's own clique, the only per-rank input."""

    def fake_all_gather(out_list, _obj):
        # Ordinary block placement, so only the fabric legs judge.
        hosts = [f"node{rank // node_widths[0]}" for rank in range(len(cliques))]
        out_list[:] = list(zip(cliques, node_widths, hosts, strict=True))

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=len(cliques)),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=local),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        validate_nvlink_domain_against_fabric(domain, world, node_widths[0])


def test_a_fabricless_domain_is_judged_even_when_the_job_holds_a_fabric_elsewhere():
    """Ranks 8-15 are fabric-less trays; a job-wide rule sees rank 0's fabric and never judges them."""
    with pytest.raises(ValueError, match="every rank reports NO NVLink fabric"):
        _run(domain=8, world=16, cliques=[3] * 8 + [NO_FABRIC] * 8, node_widths=[4] * 16, local=3)


def test_an_nvl72_domain_of_fabricless_nodes_is_rejected():
    """World 144: rack A is a real clique, the rest nine 8-GPU fabric-less nodes; the symptom is a fraction of
    expected throughput, not a crash."""
    with pytest.raises(ValueError, match=r"domain\(s\) \[1\]"):
        _run(domain=72, world=144, cliques=[1] * 72 + [NO_FABRIC] * 72, node_widths=[8] * 144, local=1)


def test_the_per_domain_verdict_is_the_same_on_every_rank():
    """A raise on a subset of ranks is a hang, which is worse than the misconfiguration."""
    cliques = [1] * 72 + [NO_FABRIC] * 72
    verdicts = set()
    for rank_local in (1, NO_FABRIC):
        try:
            _run(domain=72, world=144, cliques=cliques, node_widths=[8] * 144, local=rank_local)
            verdicts.add("pass")
        except ValueError:
            verdicts.add("raise")
    assert verdicts == {"raise"}, f"ranks disagreed on the per-domain verdict: {verdicts}"


def test_fabricless_domains_that_fit_their_node_are_left_alone():
    """Every fabric-less domain sits inside one 4-GPU tray."""
    _run(domain=4, world=16, cliques=[3] * 8 + [NO_FABRIC] * 8, node_widths=[4] * 16, local=3)


def test_the_widen_the_domain_hint_never_names_a_domain_the_check_would_refuse():
    """Counting the sentinel as an 8-rank island advises a domain the per-domain rejection refuses; the advice is
    capped by the narrowest node."""
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=4, world=16, cliques=[7] * 8 + [NO_FABRIC] * 8, node_widths=[8] * 8 + [4] * 8, local=7)
    assert not logger.warning.called, f"advised a domain the check rejects: {logger.warning.call_args}"


def test_an_under_declared_domain_on_a_pure_fabric_job_still_warns():
    """72 GPUs share one clique while node-local axes are capped at 8; under-claiming is safe, and the warning is
    the point."""
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=8, world=72, cliques=[1] * 72, node_widths=[8] * 72, local=1)
    assert logger.warning.called, "leaving 64 of 72 NVLink GPUs unused should say so"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
