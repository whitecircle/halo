#!/usr/bin/env python
"""A server restarted between syncs loses its group, and the next push fails before any chunk went out, so nothing of the
sync landed and it can be sent again. Only a run written as a one-entry list recovered: the same config spelled
``rollout_server_url`` raised on every rank."""

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from halo.trainers.grpo.environmental.async_rollouts import AsyncRolloutMixin
from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync import Wire

_URL = "http://127.0.0.1:8000"


class _Client(VLLMWeightSyncClient):
    """The engine side over :class:`Wire`, whose first client fails its first chunk as a restarted engine would."""

    built: ClassVar[list["_Client"]] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wire = Wire(fail_on_send=not self.built)
        self.wire.attach(self)
        self.built.append(self)

    def check_server(self, total_timeout: float = 0.0):
        """No server to probe."""

    def init_communicator(self, device=0) -> None:
        self._sync_device = resolve_sync_device(device)

    def close_communicator(self, *, at_exit: bool = False) -> None:
        self.abort_weight_update()


@pytest.fixture
def host(monkeypatch):
    _Client.built.clear()
    monkeypatch.setattr(
        "halo.trainers.grpo.environmental.async_rollouts.resolve_weight_sync_client", lambda backend: _Client
    )
    host = SimpleNamespace(
        _weight_sync_client=None,
        _adapter_sync=None,
        accelerator=SimpleNamespace(is_main_process=True, device=torch.device("cpu")),
        async_config=SimpleNamespace(
            rollout_backend="vllm",
            rollout_server_url=_URL,
            rollout_server_configs=None,
            rollout_connection_timeout=1.0,
            rollout_group_port=51216,
            rollout_sync_topology="per_server",
        ),
    )
    AsyncRolloutMixin._init_weight_sync_client(host)
    return host


def test_a_single_url_builds_the_pool(host):
    assert isinstance(host._weight_sync_client, WeightSyncPool)
    assert [config["url"] for config in host._weight_sync_client.server_configs] == [_URL]
    assert host._weight_sync_client._group_port(0) == 51216


def test_a_chunk_failed_before_anything_landed_is_sent_again_on_a_new_group(host):
    pool = host._weight_sync_client
    pool.update_named_param("a", torch.zeros(4, dtype=torch.bfloat16))
    pool.finish_weight_update()
    assert len(_Client.built) == 2, "no reconnect"
    assert _Client.built[1].wire.chunk_names == [["a"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
