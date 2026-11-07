#!/usr/bin/env python
"""``rollout_sync_topology`` picks per-server groups or one group over every rollout server; the keys a group cannot
honour are refused where the config is built, before any model loads."""

from typing import get_args

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.weight_sync.group_layout import SyncTopologySetting

_TWO = [{"url": "http://s0:8000"}, {"url": "http://s1:8000"}]


def _config(servers, **overrides) -> AsyncRolloutConfig:
    return AsyncRolloutConfig(rollout_server_configs=servers, **overrides)


def test_the_field_takes_exactly_the_topologies_the_pool_builds_and_defaults_to_per_server():
    assert get_args(AsyncRolloutConfig.__annotations__["rollout_sync_topology"]) == get_args(SyncTopologySetting)
    assert AsyncRolloutConfig().rollout_sync_topology == "per_server"


@pytest.mark.parametrize(
    ("servers", "remedy"),
    [
        ([{"url": "http://s0:8000", "group_port": 51216}, {"url": "http://s1:8000"}], "Set rollout_group_port once"),
        (
            [{"url": "http://s0:8000", "group_host": "10.0.0.1"}, {"url": "http://s1:8000", "group_host": "10.0.0.2"}],
            "Use one group_host",
        ),
    ],
    ids=["per-server-port", "two-group-hosts"],
)
def test_a_single_group_refuses_what_a_per_server_group_takes_and_names_the_remedy(servers, remedy):
    _config(servers)  # the same entries stay valid per server
    with pytest.raises(ValueError, match=remedy):
        _config(servers, rollout_sync_topology="single_group")


def test_one_group_host_and_node_labels_are_accepted_in_a_single_group():
    config = _config(
        [
            {"url": "http://s0:8000", "group_host": "10.0.0.1", "node": "rack0"},
            {"url": "http://s1:8000", "group_host": "10.0.0.1", "node": "rack1"},
        ],
        rollout_sync_topology="single_group",
    )
    assert config.rollout_sync_topology == "single_group"


@pytest.mark.parametrize("setting", ["per_server", "auto"])
def test_a_node_label_means_nothing_to_per_server_groups(setting):
    with pytest.raises(ValueError, match="rollout_sync_topology: single_group"):
        _config([{"url": "http://s0:8000", "node": "rack0"}, _TWO[1]], rollout_sync_topology=setting)


def test_auto_takes_what_a_per_server_group_takes():
    servers = [{"url": "http://s0:8000", "group_port": 51216}, {"url": "http://s1:8000", "group_host": "10.0.0.2"}]
    assert _config(servers, rollout_sync_topology="auto").rollout_sync_topology == "auto"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
