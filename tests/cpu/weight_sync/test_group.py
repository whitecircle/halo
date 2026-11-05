#!/usr/bin/env python
"""One weight-sync group over several servers: the layout reaches every join, each chunk is declared on every server and
sent once, and a failure after the first declare is global while one before it is recoverable."""

from __future__ import annotations

import threading
import time
from collections import deque

import pytest
import torch

from halo.weight_sync.clients import base as client_base
from halo.weight_sync.clients.base import GroupMembershipLost
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.group import WeightSyncGroup
from halo.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync import Wire
from tests.common.weight_sync_ring import FakeRing, fake_servers, ring_client_class
from tests.cpu.weight_sync.test_pool import _bare_pool

_BUDGET = 4096
_ENGINES = ["sglang", "vllm"]
_DECLARE = {"sglang": "/update_weights_from_distributed", "vllm": "/update_weights"}
_JOIN = {"sglang": "/init_weights_update_group", "vllm": "/init_weight_transfer_engine"}
_PAUSE = {"sglang": "/pause_generation", "vllm": "/pause"}
_RESUME = {"sglang": "/continue_generation", "vllm": "/resume"}


@pytest.fixture
def rig(request):
    """``(engine, servers, ring, make_pool)`` for one engine and the engine worlds the test names."""
    opened: list = []

    def build(engine: str, worlds: list[int], *, ring: FakeRing | None = None, configs: list[dict] | None = None):
        servers = fake_servers(engine, worlds)
        ring = ring or FakeRing()
        entries = [{"url": server.url, **(configs[i] if configs else {})} for i, server in enumerate(servers)]
        pool = WeightSyncPool(
            server_configs=entries,
            connection_timeout=5.0,
            client_cls=ring_client_class(engine, ring),
            base_group_port=0,
            topology="single_group",
        )
        opened.append((pool, servers))
        return pool, servers, ring

    yield build
    for pool, servers in opened:
        pool.close_communicator()
        for server in servers:
            server.close()


def _join_body(engine, server):
    body = server.bodies[_JOIN[engine]]
    return body["init_info"] if engine == "vllm" else body


def _fail_declare(engine, server):
    """The next declare answers 500; on SGLang the membership probe, an empty declare, passes first."""
    server.replies[_DECLARE[engine]] = deque([200, 500] if engine == "sglang" else [500])


def _push(pool: WeightSyncPool, count: int = 10, *, rows: int = 128) -> list[str]:
    names = [f"w{index}" for index in range(count)]
    for name in names:
        pool.update_named_param(name, torch.zeros(rows, dtype=torch.float32))
    pool.finish_weight_update()
    return names


@pytest.mark.parametrize("engine", _ENGINES)
def test_every_join_carries_its_block_and_the_ranks_are_exactly_the_group(rig, engine):
    pool, servers, _ = rig(engine, [1, 2, 4])
    pool.init_communicator("cpu")

    bodies = [_join_body(engine, server) for server in servers]
    world = 1 + 1 + 2 + 4
    assert {body["world_size"] for body in bodies} == {world}
    ranks = [
        rank
        for body, size in zip(bodies, (1, 2, 4), strict=True)
        for rank in range(body["rank_offset"], body["rank_offset"] + size)
    ]
    assert sorted(ranks) == list(range(1, world)), f"ranks {sorted(ranks)} collide or leave a hole"
    if engine == "sglang":
        assert len({body["group_name"] for body in bodies}) == 1, "the servers joined under different group names"


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_push_declares_every_chunk_on_every_server_and_sends_each_once(rig, engine, monkeypatch):
    monkeypatch.setattr("halo.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    pool, servers, ring = rig(engine, [1, 2, 1])
    pool.init_communicator("cpu")

    names = _push(pool)

    wires = [Wire(), Wire(), Wire()]
    reference, _ = _bare_pool(3, wires=wires)
    for name in names:
        reference.update_named_param(name, torch.zeros(128, dtype=torch.float32))
    reference.finish_weight_update()
    assert len(wires[0].chunk_names) > 1, "the budget never cut a chunk, so the test would be vacuous"
    assert ring.chunks == wires[0].chunk_names, "the group cut its chunks differently from the per-server pool"
    for server in servers:
        bodies = [body for path, body in server.posted if path == _DECLARE[engine]]
        declared = [body["names"] if engine == "sglang" else body["update_info"]["names"] for body in bodies]
        probes = [names for names in declared if not names]
        declared = [names for names in declared if names]
        assert len(probes) == (1 if engine == "sglang" else 0), "SGLang takes one membership probe per sync, vLLM none"
        assert declared == ring.chunks, f"{server.url} was declared other chunks than the ring sent"
        for route in (_PAUSE[engine], _RESUME[engine]):
            assert server.count(f"POST {route}") == 1, f"{route} ran {server.count(f'POST {route}')} times"
        assert not server.paused


def test_negative_control_a_group_that_sent_per_server_fails_the_one_send_check(rig, monkeypatch):
    monkeypatch.setattr("halo.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    pool, _, ring = rig("sglang", [1, 1])
    pool.init_communicator("cpu")
    real = pool._group._send_on_leader

    def twice(chunk):
        real(chunk)
        real(chunk)

    monkeypatch.setattr(pool._group, "_send_on_leader", twice)
    names = _push(pool)

    with pytest.raises(AssertionError):
        assert len(ring.chunks) == len({tuple(chunk) for chunk in ring.chunks}), "a chunk crossed the ring twice"
    assert sorted(name for chunk in ring.chunks for name in chunk) != sorted(names)


def test_sorted_servers_take_their_blocks_in_node_order(rig, monkeypatch):
    monkeypatch.setattr("halo.weight_sync.group.reject_split_nodes", lambda *args: None)
    pool, servers, _ = rig("sglang", [2, 1], configs=[{"node": "b"}, {"node": "a"}])
    pool.init_communicator("cpu")

    assert _join_body("sglang", servers[1])["rank_offset"] == 1, "node a joins first"
    assert _join_body("sglang", servers[0])["rank_offset"] == 2


def test_two_node_labels_on_one_address_are_refused_before_any_join(rig):
    pool, servers, _ = rig("sglang", [1, 1], configs=[{"node": "a"}, {"node": "b"}])

    with pytest.raises(RuntimeError, match="node labels"):
        pool.init_communicator("cpu")

    assert all(server.count(f"POST {_JOIN['sglang']}") == 0 for server in servers)


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_failed_declare_after_a_peer_accepted_its_own_is_global_and_prompt(rig, engine):
    """The peers that accepted their declares are parked in the receive, which no trainer-side call frees, so the group
    aborts its half, leaves every server paused and unusable, and names the failed server first."""
    ring = FakeRing(ready=False)
    pool, servers, _ = rig(engine, [1, 1, 1], ring=ring)
    pool.init_communicator("cpu")
    _fail_declare(engine, servers[1])
    pool.update_named_param("w", torch.zeros(8, dtype=torch.float32))

    started = time.monotonic()
    with pytest.raises(
        RuntimeError, match=r"RESTART the .* servers: " + servers[1].url.replace(".", r"\.")
    ) as failure:
        pool.finish_weight_update()

    assert time.monotonic() - started < 15, "the trainer waited out a broadcast no peer could finish"
    assert ring.aborted.is_set(), "the trainer's half was not aborted"
    assert all(server.paused for server in servers), "a server was resumed with a half-updated model"
    assert all(server.count(f"POST {_RESUME[engine]}") == 0 for server in servers)
    assert all(member._unusable_reason for member in pool._group._members)
    with pytest.raises(RuntimeError, match="left mid-update"):
        pool._group._open_update()
    assert str(failure.value).index(servers[1].url) < str(failure.value).index(servers[0].url)


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_pause_that_fails_on_one_server_resumes_the_rest_and_leaves_the_sync_replayable(rig, engine):
    pool, servers, ring = rig(engine, [1, 1, 1])
    pool.init_communicator("cpu")
    servers[2].status[_PAUSE[engine]] = 400
    pool.update_named_param("w", torch.zeros(8, dtype=torch.float32))

    with pytest.raises(RuntimeError, match=servers[2].url.replace(".", r"\.")):
        pool.finish_weight_update()

    assert ring.chunks == [], "a chunk was declared with a server that never paused"
    assert not any(server.paused for server in servers[:2]), "a server that paused stayed paused"
    assert all(member._unusable_reason is None for member in pool._group._members)
    servers[2].status.clear()
    pool.update_named_param("w", torch.zeros(8, dtype=torch.float32))
    pool.finish_weight_update()
    assert ring.chunks == [["w"]]


def _restart(engine, server):
    if engine == "sglang":
        server.groups.clear()
    else:
        server.instance_id = "launch-1"


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_server_restarted_between_syncs_re_forms_the_group_once_and_the_sync_completes(rig, engine):
    pool, servers, ring = rig(engine, [1, 1])
    pool.init_communicator("cpu")
    first_name = _join_body(engine, servers[0]).get("group_name")
    _push(pool, 2)
    _restart(engine, servers[1])

    _push(pool, 2)

    assert all(server.count(f"POST {_JOIN[engine]}") == 2 for server in servers), "the group did not re-form once"
    if engine == "sglang":
        assert _join_body(engine, servers[0])["group_name"] != first_name, "a re-form reused a group name"
    assert ring.chunks == [["w0", "w1"], ["w0", "w1"]]
    assert not any(server.paused for server in servers)
    assert ring.released >= 1, "the old group's trainer half was never released"


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_group_that_stays_stale_after_its_one_re_form_raises_with_every_server_resumed(rig, engine, monkeypatch):
    pool, servers, ring = rig(engine, [1, 1])
    pool.init_communicator("cpu")
    if engine == "sglang":
        servers[1].status[_DECLARE[engine]] = 400
    else:

        def lost() -> None:
            raise GroupMembershipLost("restarted")

        for member in pool._group._members:
            monkeypatch.setattr(member, "verify_engine_identity", lost)
    pool.update_named_param("w", torch.zeros(8, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="still lacks"):
        pool.finish_weight_update()

    assert ring.chunks == []
    assert all(server.count(f"POST {_JOIN[engine]}") == 2 for server in servers), "the group did not re-form once"
    assert not any(server.paused for server in servers), "a failed re-form left a server paused"


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_re_form_that_fails_names_the_lost_group_on_the_next_sync(rig, engine):
    pool, servers, _ = rig(engine, [1, 1])
    pool.init_communicator("cpu")
    _push(pool, 2)
    _restart(engine, servers[1])
    servers[0].status[_JOIN[engine]] = 500
    pool.update_named_param("w", torch.zeros(8, dtype=torch.float32))

    with pytest.raises(RuntimeError, match="failed to join"):
        pool.finish_weight_update()

    with pytest.raises(RuntimeError, match="lost in a failed re-form"):
        pool._group._open_update()


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_resume_that_keeps_failing_after_a_good_push_is_retried_three_times_then_raises(rig, engine, monkeypatch):
    monkeypatch.setattr(client_base, "_RESUME_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    pool, servers, _ = rig(engine, [1, 1])
    pool.init_communicator("cpu")
    servers[1].status[_RESUME[engine]] = 500

    with pytest.raises(RuntimeError, match=r"stayed paused after a good weight push") as failure:
        _push(pool, 2)

    assert servers[1].url in str(failure.value)
    assert servers[1].count(f"POST {_RESUME[engine]}") == 4, "one resume and three retries"
    assert servers[0].count(f"POST {_RESUME[engine]}") == 1


def test_a_chunk_is_paced_by_the_slowest_server(rig, monkeypatch):
    monkeypatch.setattr("halo.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    pool, servers, ring = rig("vllm", [1, 1])
    pool.init_communicator("cpu")
    servers[1].hold.add("/update_weights")
    pusher = threading.Thread(target=_push, args=(pool,), daemon=True)

    pusher.start()
    servers[1].wait_for_request("POST /update_weights")
    time.sleep(0.5)

    assert len(ring.chunks) == 1, "the next chunk went out before the slowest server took the first"
    assert servers[0].count("POST /update_weights") == 1
    servers[1].release()
    pusher.join(30)
    assert not pusher.is_alive()
    assert len(ring.chunks) > 1


def test_a_group_refuses_a_per_server_port_and_an_unknown_topology_at_construction():
    cls = ring_client_class("sglang", FakeRing())
    with pytest.raises(ValueError, match="Set rollout_group_port once"):
        WeightSyncPool(
            [{"url": "http://a:1", "group_port": 5}],
            connection_timeout=0.0,
            client_cls=cls,
            base_group_port=0,
            topology="single_group",
        )
    with pytest.raises(ValueError, match="rollout_sync_topology"):
        WeightSyncPool(
            [{"url": "http://a:1"}], connection_timeout=0.0, client_cls=cls, base_group_port=0, topology="ring"
        )


def test_a_pipeline_parallel_sglang_server_is_refused_in_a_group_and_not_per_server(rig):
    pool, servers, _ = rig("sglang", [1, 1])
    servers[1].pp_size = 2

    with pytest.raises(RuntimeError, match=r"--pp-size 2.*rollout_sync_topology: per_server"):
        pool.init_communicator("cpu")

    assert SGLangWeightSyncClient(base_url=servers[1].url).fetch_engine_world_size() == 1


def test_the_group_leader_is_the_first_server_of_the_lowest_node(rig, monkeypatch):
    monkeypatch.setattr("halo.weight_sync.group.reject_split_nodes", lambda *args: None)
    pool, servers, _ = rig("sglang", [1, 1], configs=[{"node": "z"}, {"node": "a"}])
    pool.init_communicator("cpu")

    assert isinstance(pool._group, WeightSyncGroup)
    assert pool._group.leader.base_url == servers[1].url


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
