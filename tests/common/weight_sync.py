"""Weight-sync stand-ins shared by the suites: an offline client, a recording wire, a recording sender,
a stock model, and the probes of what a LoRA push leaves behind."""

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NamedTuple
from unittest.mock import patch

import torch
from accelerate.utils import is_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import CONFIG_MAPPING, PretrainedConfig

from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.nccl.clients.base import BaseWeightSyncClient
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.runtime import DeferredRankFailure, broadcast_from_rank0, materialize_dtensor, to_local
from src.env import env_int
from src.trainers.grpo.rollout.weight_sync import _HubForwarder
from src.trainers.mixins.ep_introspection import named_ep_layers
from tests.common.ports import free_port


def weight_transfer_port(knob: str, offset: int = 0) -> int:
    """The trainer-side weight-transfer group port a GPU suite binds. COLLECTIVE when drawn.

    The ``HALO_TEST_*`` variable ``knob`` pins it, plus ``offset``. Unset, rank 0 draws one with
    :func:`~tests.common.ports.free_port` and every rank takes it: a fixed default would sit inside the
    kernel's ephemeral range, where an outbound connection elsewhere on the host can hold it. The engine
    learns the port from the trainer's init request, so the server needs no matching setting.
    """
    pinned = env_int(knob, None)
    if pinned is not None:
        return pinned + offset
    return broadcast_from_rank0(free_port())


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
    a full-size checkpoint, whose forwarded weights the clones would hold a second time. Carries the
    client calls ``sync_weights_to_client`` makes around the push, so the trainers' own entry point can
    drive it too.
    """

    def __init__(self, keep_values: bool = False):
        self.keep_values = keep_values
        self.params: list[ForwardedParam] = []

    def scope_co_load_groups(self, module_names) -> None:
        pass

    def update_named_param(self, name: str, data: torch.Tensor) -> None:
        value = data.detach().clone() if self.keep_values else None
        self.params.append(ForwardedParam(name, tuple(data.shape), isinstance(data, DTensor), value))

    def reset_prefix_cache(self) -> None:
        pass

    def abort_weight_update(self) -> None:
        pass

    @property
    def names(self) -> list[str]:
        return [param.name for param in self.params]


def local_parameters(
    model: nn.Module, keep: Callable[[str, nn.Parameter], bool] | None = None
) -> dict[str, torch.Tensor]:
    """Every parameter as this rank holds it, resharded first: the registration a sync reads. ``keep``
    narrows the copy to the ``(name, param)`` it accepts, for a policy too large to copy whole.
    Rank-local."""
    reshard_fsdp2_modules(model)
    return {
        name: to_local(param.data).detach().clone()
        for name, param in model.named_parameters()
        if keep is None or keep(name, param)
    }


def moved_parameters(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    """Names in ``before`` whose tensor in ``after`` is not bit-identical."""
    return [name for name, value in before.items() if not torch.equal(value, after[name])]


def merged_by_peft(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter of a copy of unsharded ``model`` after PEFT's own in-place ``merge_adapter``, by
    live name: the oracle a fold is held to bit for bit (:func:`as_pushed` spells it as a push)."""
    merged = copy.deepcopy(model)
    merged.merge_adapter()
    return {name: param.detach().clone() for name, param in merged.named_parameters()}


def lora_bases_and_merges(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per PEFT-LoRA'd weight, under its live name: the full frozen base, and the base with the delta
    added as PEFT's merge adds it (``w += delta`` in the weight's dtype). Collective on DTensor
    shards: every rank calls it."""
    reshard_fsdp2_modules(model)
    bases, merges = {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        key = f"{name}.base_layer.weight"
        bases[key] = materialize_dtensor(module.get_base_layer().weight.data).clone()
        merges[key] = bases[key].clone()
        for adapter in module.active_adapters:
            merges[key] += materialize_dtensor(module.get_delta_weight(adapter))
    return bases, merges


def as_pushed(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``tensors``, keyed by live-tree name, as the push hands them to the engine: through the sync's
    own forwarder, so the EP export renames, the PEFT name normalization and a hub-namespace revert
    (Step-3.7) spell them exactly as a push does. Rank-local."""
    recorder = RecordingSender(keep_values=True)
    guard = DeferredRankFailure("expected push")
    prefix = model.prefix if is_peft_model(model) else None
    forwarder = _HubForwarder(recorder, model, named_ep_layers(model), prefix, guard)
    for name, tensor in tensors.items():
        forwarder.send(name, tensor)
    forwarder.flush()
    if guard.reason is not None:
        raise RuntimeError(f"the sync's forwarder refused the expected tensors: {guard.reason}")
    return {param.name: param.value for param in recorder.params}


@contextmanager
def folded_in_place(peft_model: nn.Module) -> Iterator[None]:
    """PEFT's in-place merge for the body and its bf16 unmerge after, with nothing written back.

    A sync run inside it pushes the same weights (it folds no adapter a merge already carries), while
    the frozen base keeps the unmerge's rounding misses: the negative control for the suites asserting
    that pushes leave the base untouched, and the in-place baseline their memory is measured against.
    Resharded first, as the sync itself is, so the merge lands on the shards the sync reads.
    """
    reshard_fsdp2_modules(peft_model)
    peft_model.merge_adapter()
    try:
        yield
    finally:
        peft_model.unmerge_adapter()
