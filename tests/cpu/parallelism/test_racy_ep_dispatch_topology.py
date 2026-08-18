"""The racy-EP gate counts DISPATCH groups, not EP groups.

DeepEP builds its buffer on the dispatch group — ``ep_size`` ranks. Under expert-TP an
``ep_group_size``-wide EP group is split into ``expert_tp_size`` dispatch groups of ``ep_size``
(``EPConfig._create_expert_tp_groups``), so ``ep4 + etp2`` on one 8-GPU domain is the same
two-concurrent-4-rank-group topology as bare ``ep4`` — whose combine barriers race FSDP2's
domain-wide NCCL and hang intermittently. Keying the predicate on ``ep_group_size`` lets exactly that
shape through: ``ep_group_size == 8 == domain``.

``ParallelismConfig.is_racy_single_domain_multigroup_ep`` is topology-only — one NVLink domain AND
``ep_size > 2`` AND ``nvlink_domain_size > ep_group_size`` — so it reads the same for pure EP, EP+TP
and EP+ETP, and both halves are pinned here: every racy shape rejected, every reliable single-domain
shape still accepted.

Rejection must land at config time, before any model load: the alternative is a ~50%-of-runs hang on
an 8-GPU node. That covers the global-scope EP+ETP shape too, which EPConfig would otherwise raise on during
group construction — after the whole model has loaded.

    python tests/cpu/parallelism/test_racy_ep_dispatch_topology.py
"""

import sys

import pytest

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config


def _config(world: int, gpus_per_node: int = 8, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, **kwargs)


def test_two_large_dispatch_groups_in_one_domain_are_rejected():
    """Bare ep4 on 8 → two 4-rank dispatch groups, the shape that hangs."""
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(8, ep_size=4)


def test_ep_etp_forms_the_same_dispatch_topology_the_gate_rejects_for_bare_ep():
    """The ETP exemption is measured, not arithmetic luck — pinned so it cannot drift silently.

    ``EPConfig._create_expert_tp_groups`` splits the 8-rank EP group into dispatch groups of
    ``ep_size`` — for ep4+etp2 that is ``[0-3]`` and ``[4-7]``, byte-for-byte the two-4-rank-group
    topology this gate rejects for bare ep4. Both halves of the mechanism the rejection NAMES appear
    to hold for it too: ``EPConfig.is_deferred_dp`` needs more than one NVLink domain (and
    ``expert_tp_size == 1``), so on one domain ``_apply_ep_aware_dp_fsdp2`` shards over the whole DP
    world in BOTH shapes, and the combine spans a strict subset in both.

    Run on an 8-GPU node through the real trainer with the gate disabled, the two shapes still do NOT
    behave alike: bare ep4 fails with ``CUDA error: Invalid access of peer GPU memory over nvlink``
    on the elastic backend, on legacy, and with gradient checkpointing; ep4+etp2 trains 25 steps
    clean. So the exemption tracks something the ``ep_group_size``-vs-``ep_size`` distinction stands
    in for, and keying the predicate on ``ep_size`` would reject a shape that demonstrably works.
    """
    config = _config(8, ep_size=4, expert_tp_size=2)
    assert not config.is_racy_single_domain_multigroup_ep
    assert config.ep_group_size == config.nvlink_domain_size, "the exemption rests on this equality"
    assert config.nvlink_domain_size // config.ep_size == 2, "…while there are two dispatch groups"
    # Same single-domain topology as the rejected bare-ep4 shape: no deferral, so FSDP stays DP-wide.
    assert config.num_nvlink_domains == 1
    # EP+TP is not the escape hatch the docs suggest — it hits this very rejection.
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(8, ep_size=4, tp_size=2)


def test_ep_etp_racy_two_dispatch_groups_rejected():
    """The ETP exemption above is an equality, not a blanket pass — widen the domain and it lapses.

    The same ep4 × etp2, but on one 16-GPU NVLink domain (NVL72-style rack): ``ep_group_size`` is 8
    while the domain is 16, so two 8-rank dispatch groups again share a domain with FSDP2's
    domain-wide NCCL. Keying the gate on "ETP is on" rather than on the topology would miss it.
    """
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(16, 16, ep_size=4, expert_tp_size=2, ep_scope="node")


@pytest.mark.parametrize("world,gpn", [(8, 8), (16, 8), (32, 8), (4, 4)])
def test_every_accepted_multi_dispatch_group_shape_is_explained(world, gpn):
    """Completeness: enumerate what the validators ACCEPT and classify every >2-rank multi-dispatch
    -group-per-domain shape that survives.

    A shape survives only if one of the three documented reasons applies:
      * more than one NVLink domain — ``EPConfig.is_deferred_dp`` engages and FSDP shards over the
        EP group, so the reduce-scatter shares the combine's membership;
      * ``expert_tp_size > 1`` — the deliberate, GPU-unvalidated exemption above;
      * the dispatch group is 2 ranks — the documented-benign case.

    A new accept outside those three means the gate grew a hole nobody reasoned about, so this
    fails rather than letting it through. Enumeration, not a re-implementation of the predicate.
    """
    unexplained = []
    for ep in (1, 2, 3, 4, 8, 16):
        for etp in (1, 2, 4, 8):
            for tp in (1, 2, 4):
                for scope in ("node", "global"):
                    try:
                        cfg = _config(world, gpn, ep_size=ep, expert_tp_size=etp, tp_size=tp, ep_scope=scope)
                    except ValueError:
                        continue
                    if cfg.ep_size <= 2 or cfg.nvlink_domain_size // cfg.ep_size <= 1:
                        continue
                    if cfg.num_nvlink_domains > 1 or cfg.expert_tp_size > 1:
                        continue
                    unexplained.append((ep, etp, tp, scope))
    assert not unexplained, (
        f"accepted >2-rank multi-dispatch-group shapes with neither a deferral nor the ETP "
        f"exemption on world={world}/domain={gpn}: {unexplained}"
    )


@pytest.mark.parametrize(
    "kwargs,label",
    [
        ({"ep_size": 8}, "one 8-rank dispatch group fills the domain"),
        ({"ep_size": 2, "expert_tp_size": 4}, "2-rank dispatch groups are the documented-benign case"),
        ({"ep_size": 2}, "ep2 alone"),
        ({"expert_tp_size": 8}, "pure ETP has no dispatch group at all"),
        ({"ep_size": 8, "tp_size": 2, "ep_scope": "global"}, "EP+TP: one global dispatch group"),
    ],
)
def test_supported_single_domain_shapes_still_pass(kwargs, label):
    """Anti-over-rejection: the gate must not narrow the shapes the docs promise."""
    config = _config(8, **kwargs)
    assert not config.is_racy_single_domain_multigroup_ep, label


def test_multi_domain_ep_is_not_judged_by_this_gate():
    """Two domains of 8 with node-local ep8: one dispatch group per domain, nothing racing."""
    config = _config(16, ep_size=8)
    assert not config.is_racy_single_domain_multigroup_ep
    assert config.num_ep_groups == 2


def test_global_scope_ep_etp_on_one_domain_is_rejected_at_config_time():
    """EPConfig raises for this shape during group construction — after the model has loaded.
    The config gate must catch it first, so the run dies in a second rather than after the load."""
    with pytest.raises(ValueError, match="one ETP group per NVLink domain"):
        _config(8, ep_size=2, expert_tp_size=2, ep_scope="global")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
