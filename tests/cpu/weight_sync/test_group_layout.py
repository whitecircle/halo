#!/usr/bin/env python
"""The one-group layout puts the trainer at rank 0 and each server's workers in a contiguous block, servers of one node
adjacent: the ranks over every server are exactly ``1 .. world_size - 1`` with none twice."""

import pytest

from halo.weight_sync.group_layout import (
    reject_mixed_localities,
    reject_pipeline_stages,
    reject_split_nodes,
    resolve_group_host,
    resolve_group_layout,
    resolve_node_label,
    validate_sync_topology,
)

# Engine worlds a server reports: TP1, TP2, TP4, an attn-dp SGLang server spanning 8 GPUs, a vLLM DP2 x TP2 x PP2.
_WORLDS = [1, 2, 4, 8, 8]


def _ranks(layout, worlds):
    return [
        rank
        for index in range(len(worlds))
        for rank in range(layout.rank_offsets[index], layout.rank_offsets[index] + worlds[index])
    ]


def test_ranks_over_every_server_are_exactly_the_group_with_no_collision():
    layout = resolve_group_layout(["a"] * len(_WORLDS), _WORLDS)

    assert sorted(_ranks(layout, _WORLDS)) == list(range(1, layout.world_size))
    assert layout.world_size == 1 + sum(_WORLDS)


def test_servers_of_one_node_are_one_contiguous_block_in_node_order():
    nodes = ["b", "a", "b", "a"]
    worlds = [2, 1, 4, 8]

    layout = resolve_group_layout(nodes, worlds)

    assert layout.order == (1, 3, 0, 2), "node a first, config order kept within a node"
    assert layout.rank_offsets == (10, 1, 12, 2)
    assert sorted(_ranks(layout, worlds)) == list(range(1, layout.world_size))


def test_negative_control_an_unsorted_or_mis_stepped_layout_fails_the_exact_offsets():
    nodes, worlds = ["b", "a", "b", "a"], [2, 1, 4, 8]
    expected = resolve_group_layout(nodes, worlds)
    unsorted = resolve_group_layout(["a"] * 4, worlds)
    assert unsorted.rank_offsets != expected.rank_offsets, "the node sort leaves no mark on the offsets"
    stepped = [1 + sum(worlds[:index]) + index for index in range(4)]
    assert tuple(stepped) != unsorted.rank_offsets, "a step of engine world + 1 would leave holes in the ranks"


def test_a_label_count_that_differs_from_the_servers_is_refused():
    with pytest.raises(ValueError, match="2 node labels for 3 servers"):
        resolve_group_layout(["a", "b"], [1, 1, 1])


def test_one_address_under_two_node_labels_is_refused_with_the_remedy():
    reject_split_nodes(["a", "a"], ["10.0.0.1", "10.0.0.1"], ["http://x:1", "http://y:1"])
    with pytest.raises(ValueError, match=r"carry node labels \['10.0.0.1', 'node1'\].*one 'node' label"):
        reject_split_nodes(["10.0.0.1", "node1"], ["10.0.0.1", "10.0.0.1"], ["http://10.0.0.1:1", "http://node1:1"])


def test_the_node_label_defaults_to_the_url_host():
    assert resolve_node_label({"url": "http://gpu7:8000"}) == "gpu7"
    assert resolve_node_label({"url": "http://gpu7:8000", "node": "rack2"}) == "rack2"


def test_one_group_takes_one_address_and_no_per_server_port():
    one_host = [{"url": "u"}, {"url": "v", "group_host": "10.0.0.9"}, {"url": "w", "group_host": "10.0.0.9"}]
    validate_sync_topology("single_group", one_host)
    assert resolve_group_host(one_host) == "10.0.0.9"
    assert resolve_group_host([{"url": "u"}]) is None
    with pytest.raises(ValueError, match="Set rollout_group_port once, or use rollout_sync_topology: per_server"):
        validate_sync_topology("single_group", [{"url": "u", "group_port": 1}])
    with pytest.raises(ValueError, match="one group_host, or rollout_sync_topology: per_server"):
        validate_sync_topology(
            "single_group", [{"url": "u", "group_host": "10.0.0.1"}, {"url": "v", "group_host": "10.0.0.2"}]
        )


def test_the_one_validator_refuses_an_unknown_topology_and_a_node_label_per_server_cannot_use():
    with pytest.raises(ValueError, match="Unknown rollout_sync_topology 'ring'"):
        validate_sync_topology("ring", [{"url": "u"}])
    with pytest.raises(ValueError, match="rollout_sync_topology: single_group"):
        validate_sync_topology("per_server", [{"url": "u", "node": "a"}])
    validate_sync_topology("per_server", [{"url": "u", "group_port": 1, "group_host": "h"}])


def test_local_and_remote_servers_share_one_address_only_where_the_user_named_it():
    urls = ["http://a:1", "http://b:1"]
    for found in ([True, True], [False, False], [True, False]):
        reject_mixed_localities(urls, found, "10.0.0.1")
    for found in ([True, True], [False, False]):
        reject_mixed_localities(urls, found, None)
    with pytest.raises(ValueError, match=r"http://a:1.*Set group_host"):
        reject_mixed_localities(urls, [True, False], None)


def test_a_pipeline_parallel_server_cannot_take_a_rank_block():
    reject_pipeline_stages("SGLang", ["u", "v"], [1, 1])
    with pytest.raises(RuntimeError, match=r"server v runs --pp-size 2"):
        reject_pipeline_stages("SGLang", ["u", "v"], [1, 2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
