#!/usr/bin/env python
"""``validate_nvlink_domain_against_fabric`` judges each NVLink domain, not the job as a whole.

Every fabric-less GPU reports the same ``NO_FABRIC`` sentinel, so the clique-straddle test cannot
tell one fabric-less node from four of them: a domain laid across several of them shows exactly one
"clique" and passes. The gathered node widths are the only evidence such a domain cannot exist, and
reading them job-wide answers only for a job where NO rank has a fabric. A job that holds both — an
NVL72 rack beside plain NVL8 trays, or a mixed pool — never reaches that branch, and its fabric-less
domains go unjudged: their "node-local" EP/TP/CP groups run over the network while ``requires_rdma``
reports False, which is the multi-hour-into-training failure this check exists to prevent.

The clique/straddle half and the read of ``nvidia-smi`` itself are pinned in
``tests/cpu/parallelism/test_nvlink_fabric_check.py``; this file pins the per-domain verdict.

    python tests/cpu/parallelism/test_nvlink_domain_per_domain_verdict.py
"""

from unittest.mock import patch

import pytest

from src.distributed.nvlink import NO_FABRIC, validate_nvlink_domain_against_fabric

_MOD = "src.distributed.nvlink"


def _run(domain: int, world: int, cliques: list[int], node_widths: list[int], local: int = 0):
    """Drive the check over a fake topology: per-rank clique ids and per-rank ``gpus_per_node``.

    ``local`` is this rank's own clique — the only per-rank input, which the uniformity test varies.
    """

    def fake_all_gather(out_list, _obj):
        # Ordinary block placement: rank r on node r // gpus_per_node, so only the fabric legs judge.
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
    """NVL8×N beside an MNNVL island: 16 ranks on 4-GPU trays, domain 8.

    Ranks 0-7 share fabric clique 3, so their 8-wide domain legitimately spans two trays over MNNVL.
    Ranks 8-15 are fabric-less trays — that domain is two 4-GPU boxes with nothing but the network
    between them. The job-wide rule sees a fabric on rank 0 and never judges the second domain.
    """
    with pytest.raises(ValueError, match="every rank reports NO NVLink fabric"):
        _run(domain=8, world=16, cliques=[3] * 8 + [NO_FABRIC] * 8, node_widths=[4] * 16, local=3)


def test_an_nvl72_domain_of_fabricless_nodes_is_rejected():
    """NVL72 shape: world 144, ``NVLINK_DOMAIN_SIZE=72``.

    Rack A (ranks 0-71) is one real clique; the second 72 ranks are nine 8-GPU nodes with no fabric
    at all — Fabric Manager never brought them up, or they are plain HGX boxes. Declaring them a
    72-wide domain makes EP groups of 72 that leave NVLink at every 8-rank boundary. At this scale
    the symptom is a job that trains at a fraction of the expected throughput, not a crash.
    """
    with pytest.raises(ValueError, match=r"domain\(s\) \[1\]"):
        _run(domain=72, world=144, cliques=[1] * 72 + [NO_FABRIC] * 72, node_widths=[8] * 144, local=1)


def test_the_per_domain_verdict_is_the_same_on_every_rank():
    """A raise on a subset of ranks is a hang, which is worse than the misconfiguration.

    The verdict is a pure function of the gathered list, so drive it once per rank — with that
    rank's own local clique, the only per-rank input — and require one answer.
    """
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
    """Anti-over-rejection: the ordinary mixed pool must still run.

    Same topology, domain 4 — every fabric-less domain now sits inside one 4-GPU tray, which is
    exactly what a fabric-less node supports.
    """
    _run(domain=4, world=16, cliques=[3] * 8 + [NO_FABRIC] * 8, node_widths=[4] * 16, local=3)


def test_the_widen_the_domain_hint_never_names_a_domain_the_check_would_refuse():
    """``NO_FABRIC`` is not an island: it is one sentinel shared by every fabric-less node.

    8 ranks on a real clique plus 8 on fabric-less 4-GPU trays, domain 4. Counting the sentinel as
    one 8-rank island advises raising the domain to 8 — which the per-domain rejection above then
    refuses, because those trays hold 4 GPUs each. The advice is capped by the narrowest node.
    """
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=4, world=16, cliques=[7] * 8 + [NO_FABRIC] * 8, node_widths=[8] * 8 + [4] * 8, local=7)
    assert not logger.warning.called, f"advised a domain the check rejects: {logger.warning.call_args}"


def test_an_under_declared_domain_on_a_pure_fabric_job_still_warns():
    """Anti-over-correction: under-claiming a real island is always safe, and saying so is the point.

    NVL72 with ``NVLINK_DOMAIN_SIZE`` left at the OS-node width: 72 GPUs share one clique while
    node-local EP/TP/CP are capped at 8.
    """
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=8, world=72, cliques=[1] * 72, node_widths=[8] * 72, local=1)
    assert logger.warning.called, "leaving 64 of 72 NVLink GPUs unused should say so"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
