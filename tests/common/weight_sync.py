"""Weight-sync stand-ins shared by the suites: an offline client, a recording wire, a recording sender,
a stock model."""

from typing import NamedTuple
from unittest.mock import patch

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import CONFIG_MAPPING, PretrainedConfig

from src.distributed.nccl.clients.base import BaseWeightSyncClient
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient


def offline_sglang_client(base_url: str = "http://localhost:30000") -> SGLangWeightSyncClient:
    """A SGLang client built by its real ``__init__`` minus the live-server probe (``check_server``)."""
    with patch.object(SGLangWeightSyncClient, "check_server"):
        return SGLangWeightSyncClient(base_url=base_url)


class StockModel(nn.Module):
    """A model with a config but no EP wrapper — the use_grouped_gemm: false module tree."""

    def __init__(self, model_type: str):
        super().__init__()
        if model_type in CONFIG_MAPPING:
            self.config = CONFIG_MAPPING[model_type]()
        else:  # remote-code spellings (Bailing family) have no in-library config class
            self.config = PretrainedConfig()
            self.config.model_type = model_type
        self.weight = nn.Parameter(torch.zeros(1))


class Wire:
    """The engine side of one client: records each chunk put on the wire, in order.

    Stubs the per-engine seams only (``_broadcast_chunk`` and the phase calls), so the client's own
    chunk accounting — the count ``can_replay_sync`` reads — runs as it does in production.
    """

    def __init__(self, fail_on_send: bool = False, retain: bool = True):
        self.chunks: list[list[tuple[str, torch.Tensor]]] = []
        self.opened = 0
        self.closed = 0
        self.fail_on_send = fail_on_send
        # A real engine copies what it receives into its own storage and keeps no reference to the
        # trainer's snapshot. The lifetime tests need that; the others want the values back.
        self.retain = retain
        self.client: BaseWeightSyncClient | None = None

    def attach(self, client: BaseWeightSyncClient) -> BaseWeightSyncClient:
        self.client = client
        client.begin_weight_update = self._begin
        client._broadcast_chunk = self._send
        client.end_weight_update = self._end
        return client

    def _begin(self):
        self.opened += 1

    def _send(self, named_params, final: bool = False):
        if self.fail_on_send:
            raise ConnectionError("server died mid-flush")
        self.chunks.append(list(named_params) if self.retain else [(name, None) for name, _ in named_params])

    def _end(self, tail):
        try:
            self.client.send_weights(tail, final=True)  # as both real clients close: through the seam
        finally:
            self.closed += 1  # the real clients close and resume in a finally too

    @property
    def sent(self) -> list[tuple[str, torch.Tensor]]:
        return [item for chunk in self.chunks for item in chunk]

    @property
    def chunk_names(self) -> list[list[str]]:
        return [[name for name, _ in chunk] for chunk in self.chunks]


class ForwardedParam(NamedTuple):
    """One tensor ``gather_and_send_weights`` forwarded: its shape is the global one for a DTensor."""

    name: str
    shape: tuple[int, ...]
    is_dtensor: bool
    value: torch.Tensor | None


class RecordingSender:
    """The engine end of ``gather_and_send_weights``, with no NCCL and no server: records every forward.

    ``keep_values`` also keeps a detached clone of each tensor, for a value comparison; leave it off on
    a full-size checkpoint, whose forwarded weights the clones would hold a second time.
    """

    def __init__(self, keep_values: bool = False):
        self.keep_values = keep_values
        self.params: list[ForwardedParam] = []

    def update_named_param(self, name: str, data: torch.Tensor) -> None:
        value = data.detach().clone() if self.keep_values else None
        self.params.append(ForwardedParam(name, tuple(data.shape), isinstance(data, DTensor), value))

    def reset_prefix_cache(self) -> None:
        pass

    @property
    def names(self) -> list[str]:
        return [param.name for param in self.params]
