"""The racy-EP gate counts dispatch groups, ``ep_size`` ranks each; keying on ``ep_group_size`` lets ``ep4 + etp2``
through. Rejection lands at config time, since the alternative is a ~50%-of-runs hang on an 8-GPU node."""

import pytest

from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config


def _config(world: int, gpus_per_node: int = 8, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, **kwargs)


def test_two_large_dispatch_groups_in_one_domain_are_rejected():
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(8, ep_size=4)


def test_ep_etp_forms_the_same_dispatch_topology_the_gate_rejects_for_bare_ep():
    """Run on an 8-GPU node with the gate off, bare ep4 faults with ``Invalid access of peer GPU memory over nvlink``
    while ep4+etp2 trains 25 steps clean. Keying on ``ep_size`` would reject a shape that works."""
    config = _config(8, ep_size=4, expert_tp_size=2)
    assert not config.is_racy_single_domain_multigroup_ep
    assert config.ep_group_size == config.nvlink_domain_size, "the exemption rests on this equality"
    assert config.nvlink_domain_size // config.ep_size == 2, "…while there are two dispatch groups"
    # Same topology as the rejected bare ep4: no deferral, so FSDP stays DP-wide.
    assert config.num_nvlink_domains == 1
    # EP+TP hits this very rejection.
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(8, ep_size=4, tp_size=2)


def test_ep_etp_racy_two_dispatch_groups_rejected():
    """The exemption is an equality: on a 16-GPU domain, two 8-rank dispatch groups again share a domain with
    FSDP2's NCCL."""
    with pytest.raises(ValueError, match="dispatch groups"):
        _config(16, 16, ep_size=4, expert_tp_size=2, ep_scope="node")


@pytest.mark.parametrize("world,gpn", [(8, 8), (16, 8), (32, 8), (4, 4)])
def test_every_accepted_multi_dispatch_group_shape_is_explained(world, gpn):
    """A survivor needs >1 domain (FSDP shards over the EP group), ``expert_tp_size > 1`` (unvalidated on GPU), or
    2-rank dispatch groups; any other accept fails here."""
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
    """The gate must not narrow the shapes the docs promise."""
    config = _config(8, **kwargs)
    assert not config.is_racy_single_domain_multigroup_ep, label


def test_multi_domain_ep_is_not_judged_by_this_gate():
    """One dispatch group per domain, nothing racing."""
    config = _config(16, ep_size=8)
    assert not config.is_racy_single_domain_multigroup_ep
    assert config.num_ep_groups == 2


def test_global_scope_ep_etp_on_one_domain_is_rejected_at_config_time():
    """EPConfig would raise during group construction, after the model has loaded."""
    with pytest.raises(ValueError, match="one ETP group per NVLink domain"):
        _config(8, ep_size=2, expert_tp_size=2, ep_scope="global")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
