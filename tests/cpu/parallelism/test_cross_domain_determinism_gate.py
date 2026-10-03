#!/usr/bin/env python
"""``full_determinism`` on an EP group spanning NVLink domains is refused while the entry script builds its
parallelism config, before any weight is read.

The dispatcher builds DeepEP's deterministic buffer under the mode ``full_determinism`` turns on, but
across domains DeepEP runs its hybrid RDMA kernels, which have no deterministic mode: they place received
tokens in atomic claim order, so every expert weight gradient is summed in a different order on every
step. A group inside one domain, and a run without ``full_determinism``, build normally.

    python tests/cpu/parallelism/test_cross_domain_determinism_gate.py
"""

from types import SimpleNamespace

import pytest

from src.args.distributed_args import DistributedArguments
from src.trainers.sft import DistributedSFTTrainer
from src.training.parallelism_args import parallelism_config_from_args
from tests.common.parallelism import simulated_world


def _built(*, world_size: int, ep_scope: str, full_determinism: bool):
    """The config the entry-script prologue builds for an ep8 run on ``world_size`` ranks of 8-GPU domains."""
    with simulated_world(world_size, 8):
        return parallelism_config_from_args(
            DistributedArguments(expert_parallel_size=8, ep_scope=ep_scope),
            training_config=SimpleNamespace(full_determinism=full_determinism),
            trainer_cls=DistributedSFTTrainer,
        )


@pytest.fixture(autouse=True)
def _no_host_domain_size(monkeypatch):
    """The host's ``NVLINK_DOMAIN_SIZE`` (an NVL72 box's 72) would change every domain verdict."""
    monkeypatch.delenv("NVLINK_DOMAIN_SIZE", raising=False)


def test_an_ep_group_spanning_domains_is_refused():
    with pytest.raises(ValueError, match="spanning NVLink domains"):
        _built(world_size=16, ep_scope="global", full_determinism=True)


@pytest.mark.parametrize(
    ("world_size", "ep_scope", "full_determinism"),
    [(16, "node", True), (8, "global", True), (16, "global", False)],
    ids=["node_local_groups", "global_group_inside_one_domain", "no_full_determinism"],
)
def test_runs_deepep_can_serve_build(world_size, ep_scope, full_determinism):
    config = _built(world_size=world_size, ep_scope=ep_scope, full_determinism=full_determinism)
    assert config.ep_size == 8
    assert config.requires_rdma == (world_size > 8 and ep_scope == "global")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
