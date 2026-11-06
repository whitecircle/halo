#!/usr/bin/env python
"""``rollout_sync_topology: auto`` is one group for two or more servers that report the same served model and
weight-loading flags and that one group can rank, and per-server groups otherwise; an explicit setting always wins."""

from __future__ import annotations

import logging

import pytest

from halo.weight_sync.clients.base import ServerFlags
from halo.weight_sync.group_eligibility import ServerReading, find_fleet_disagreement, resolve_sync_topology
from halo.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync_ring import FakeRing, fake_servers, ring_client_class

_FLAGS = ServerFlags(toolkit_image=True, dtype="bfloat16", quantization="", moe_backend="triton")


def _reading(index: int, *, model="m", flags=_FLAGS, stages=1, local=True, node=None) -> ServerReading:
    return ServerReading(
        url=f"http://s{index}:8000",
        node=node or f"n{index}",
        host=f"10.0.0.{index}",
        is_local=local,
        model=model,
        flags=flags,
        pipeline_stages=stages,
    )


def _fleet(count: int, **overrides) -> list[ServerReading]:
    return [_reading(index, **overrides) for index in range(count)]


def _one_differs(**overrides) -> list[ServerReading]:
    return [_reading(0), _reading(1, **overrides), _reading(2)]


_TABLE = [
    ("auto", _fleet(1), None, "per_server"),
    ("auto", _fleet(2), None, "single_group"),
    ("auto", _fleet(8), None, "single_group"),
    ("auto", _one_differs(model="other"), None, "per_server"),
    ("auto", _one_differs(flags=_FLAGS._replace(dtype="float16")), None, "per_server"),
    ("auto", _one_differs(flags=_FLAGS._replace(quantization="fp8")), None, "per_server"),
    ("auto", _one_differs(flags=_FLAGS._replace(moe_backend="auto")), None, "per_server"),
    ("auto", _one_differs(flags=_FLAGS._replace(dtype=None)), None, "single_group"),
    ("auto", _one_differs(flags=None), None, "per_server"),
    ("auto", _one_differs(model=None), None, "per_server"),
    ("auto", _one_differs(stages=2), None, "per_server"),
    ("auto", [_reading(0), _reading(1, local=False)], None, "per_server"),
    ("auto", [_reading(0), _reading(1, local=False)], "10.0.0.9", "single_group"),
    ("auto", [_reading(0, node="a"), _reading(1, node="b")], None, "single_group"),
    ("single_group", _one_differs(model="other"), None, "single_group"),
    ("single_group", _fleet(1), None, "single_group"),
    ("per_server", _fleet(4), None, "per_server"),
]


@pytest.mark.parametrize(("setting", "readings", "group_host", "expected"), _TABLE)
def test_resolution_table(setting, readings, group_host, expected):
    topology, why = resolve_sync_topology(setting, readings, group_host)
    assert topology == expected
    assert why


def test_the_refusal_names_the_attribute_the_servers_disagree_on():
    topology, why = resolve_sync_topology("auto", _one_differs(flags=_FLAGS._replace(dtype="float16")), None)
    assert topology == "per_server"
    assert "dtype" in why and "float16" in why


def test_a_flag_a_server_does_not_report_is_not_compared():
    readings = _one_differs(flags=_FLAGS._replace(quantization=None))
    assert find_fleet_disagreement(*zip(*[(r.url, r.model, r.flags) for r in readings], strict=True)) is None


@pytest.fixture
def pooled():
    opened = []

    def build(engine: str, worlds: list[int], topology: str = "auto", configure=None):
        servers = fake_servers(engine, worlds)
        for index, server in enumerate(servers):
            if configure:
                configure(index, server)
        pool = WeightSyncPool(
            server_configs=[{"url": server.url} for server in servers],
            connection_timeout=5.0,
            client_cls=ring_client_class(engine, FakeRing()),
            base_group_port=0,
            topology=topology,
        )
        opened.append((pool, servers))
        return pool

    yield build
    for pool, servers in opened:
        pool.close_communicator()
        for server in servers:
            server.close()


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
@pytest.mark.parametrize(("worlds", "grouped"), [([1], False), ([1, 1], True), ([1, 2, 4], True)])
def test_the_pool_forms_one_group_for_a_homogeneous_fleet_and_logs_the_topology_once(
    pooled, caplog, engine, worlds, grouped
):
    pool = pooled(engine, worlds)
    with caplog.at_level(logging.INFO):
        pool.init_communicator("cpu")
    assert (pool.group is not None) is grouped
    verdicts = [record for record in caplog.records if "Weight-sync topology" in record.getMessage()]
    assert len(verdicts) == 1
    assert (f"topology {'single_group' if grouped else 'per_server'}") in verdicts[0].getMessage()


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_the_pool_keeps_a_mixed_fleet_per_server(pooled, engine):
    def odd_dtype(index, server):
        server.dtype = "float16" if index == 1 else "bfloat16"

    pool = pooled(engine, [1, 1], configure=odd_dtype)
    pool.init_communicator("cpu")
    assert pool.group is None
    assert len(pool._clients) == 2


def test_the_pool_keeps_a_pipeline_parallel_fleet_per_server(pooled):
    def staged(index, server):
        server.pp_size = 2 if index == 1 else 1

    pool = pooled("sglang", [1, 1], configure=staged)
    pool.init_communicator("cpu")
    assert pool.group is None


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_an_explicit_setting_wins_over_what_the_servers_report(pooled, engine):
    def odd_dtype(index, server):
        server.dtype = "float16" if index == 1 else "bfloat16"

    grouped = pooled(engine, [1, 1], topology="single_group", configure=odd_dtype)
    grouped.init_communicator("cpu")
    assert grouped.group is not None
    per_server = pooled(engine, [1, 1], topology="per_server")
    per_server.init_communicator("cpu")
    assert per_server.group is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
