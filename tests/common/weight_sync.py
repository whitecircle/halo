"""Weight-sync stand-ins shared by the suites, and the probes of what a LoRA push leaves behind."""

import copy
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import NamedTuple
from unittest.mock import patch

import torch
from accelerate.utils import is_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import CONFIG_MAPPING, PretrainedConfig

from halo.distributed.expert_parallel.moe_layer.gather import EPGatherMixin
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.env import env_int, env_str
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.dtensor import materialize_dtensor, to_local
from halo.trainers.grpo.rollout.weight_push import _HubForwarder, gather_and_send_weights
from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.parallelism import single_process_ep_config
from tests.common.ports import free_port
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# ``or`` rather than an env_str default: an exported-but-empty variable passes the conftest gate, which reads it the
# same way.
SGLANG_SERVER_URL = env_str("SGLANG_SERVER_URL") or "http://localhost:30000"
# The MoE suites serve a different checkpoint on VLLM_SERVER_URL.
VLLM_DENSE_SERVER_URL = (
    env_str("HALO_TEST_VLLM_DENSE_SERVER_URL") or env_str("VLLM_SERVER_URL") or "http://localhost:8010"
)


def weight_transfer_port(knob: str, offset: int = 0) -> int:
    """The trainer-side group port a GPU suite binds. COLLECTIVE when drawn. Unset, rank 0 draws one: a fixed default
    sits in the kernel's ephemeral range, where an outbound connection can hold it."""
    pinned = env_int(knob, None)
    if pinned is not None:
        return pinned + offset
    return broadcast_from_rank0(free_port())


def offline_client[Client: WeightSyncClientBase](
    client_class: type[Client], base_url: str = "http://localhost", **kwargs
) -> Client:
    """A client built by its real ``__init__`` minus the live-server probe."""
    with patch.object(client_class, "check_server"):
        return client_class(base_url=base_url, **kwargs)


def push(client: WeightSyncClientBase, named_params: Iterable[tuple[str, torch.Tensor]]) -> None:
    """One sync as the trainer drives it: params buffered, chunks streamed as the budget fills, the tail sent. A
    failure aborts the update first."""
    try:
        for name, tensor in named_params:
            client.update_named_param(name, tensor)
        client.finish_weight_update()
    except BaseException:
        client.abort_weight_update()
        raise


class StockModel(nn.Module):
    """A model with a config but no EP layer, the use_grouped_gemm: false module tree."""

    def __init__(self, model_type: str):
        super().__init__()
        if model_type in CONFIG_MAPPING:
            self.config = CONFIG_MAPPING[model_type]()
        else:  # Remote-code spellings have no in-library config class.
            self.config = PretrainedConfig()
            self.config.model_type = model_type
        self.weight = nn.Parameter(torch.zeros(1))


class Wire:
    """The engine side of one client: records each chunk in order, so the client's own chunk accounting runs as in
    production."""

    def __init__(self, fail_on_send: bool = False, retain: bool = True):
        self.chunks: list[list[tuple[str, torch.Tensor]]] = []
        self.opened = 0
        self.closed = 0
        self.fail_on_send = fail_on_send
        # A real engine copies what it receives and keeps no reference to the trainer's snapshot; the lifetime tests
        # need that.
        self.retain = retain
        self.client: WeightSyncClientBase | None = None

    def attach(self, client: WeightSyncClientBase) -> WeightSyncClientBase:
        self.client = client
        client._start_engine_update = self._begin
        client._broadcast_chunk = self._send
        client._finish_engine_update = self._end
        client._lift_pause = lambda timeout, context: None
        return client

    def _begin(self):
        self.opened += 1

    def _send(self, named_params, final: bool):
        if self.fail_on_send:
            raise ConnectionError("server died mid-flush")
        self.chunks.append(list(named_params) if self.retain else [(name, None) for name, _ in named_params])

    def _end(self, tail):
        try:
            self.client._transmit_chunk(tail, final=True)
        finally:
            self.closed += 1

    @property
    def sent(self) -> list[tuple[str, torch.Tensor]]:
        return [item for chunk in self.chunks for item in chunk]

    @property
    def chunk_names(self) -> list[list[str]]:
        return [[name for name, _ in chunk] for chunk in self.chunks]


class ServerlessSGLangClient(SGLangWeightSyncClient):
    """The real SGLang client with no server: the engine side is a recording :class:`Wire`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wire = Wire()
        self.wire.attach(self)

    def check_server(self, total_timeout: float = 0.0):
        """No server to probe."""

    def init_communicator(self, device=0):
        self._sync_device = resolve_sync_device(device)


class ForwardedParam(NamedTuple):
    """One forwarded tensor; its shape is the global one for a DTensor."""

    name: str
    shape: tuple[int, ...]
    is_dtensor: bool
    value: torch.Tensor | None


class RecordingSender:
    """The engine end of ``gather_and_send_weights`` with no NCCL or server: records every forward. Leave
    ``keep_values`` off on a full-size checkpoint."""

    def __init__(self, keep_values: bool = False):
        self.keep_values = keep_values
        self.params: list[ForwardedParam] = []

    def scope_co_load_groups(self, module_names) -> None:
        pass

    def update_named_param(self, name: str, data: torch.Tensor, *, owned: bool = False) -> None:
        value = data.detach().clone() if self.keep_values else None
        self.params.append(ForwardedParam(name, tuple(data.shape), isinstance(data, DTensor), value))

    def finish_weight_update(self) -> None:
        pass

    def abort_weight_update(self) -> None:
        pass

    @property
    def names(self) -> list[str]:
        return [param.name for param in self.params]


def ep_model(family: str, overrides: dict | None = None) -> nn.Module:
    """A tiny model of ``family`` under single-process EP layers, in bf16 as the trainer holds it."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides).to(torch.bfloat16)
    apply_ep_layers(model, single_process_ep_config())
    assert named_ep_layers(model), f"{family}: no EP layer to gather"
    return model


def pushed_values(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every tensor one push forwards, by hub name; a name sent twice raises."""
    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    assert len(sender.names) == len(set(sender.names)), "a tensor was forwarded twice"
    return {param.name: param.value for param in sender.params}


def local_parameters(
    model: nn.Module, keep: Callable[[str, nn.Parameter], bool] | None = None
) -> dict[str, torch.Tensor]:
    """Every parameter as this rank holds it, resharded first; ``keep`` narrows the copy for a policy too large to copy
    whole. Rank-local."""
    reshard_fsdp2_modules(model)
    return {
        name: to_local(param.data).detach().clone()
        for name, param in model.named_parameters()
        if keep is None or keep(name, param)
    }


def moved_parameters(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    return [name for name, value in before.items() if not torch.equal(value, after[name])]


def merged_by_peft(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter of an unsharded copy after PEFT's own in-place ``merge_adapter``: the oracle a fold is held to
    bit for bit."""
    merged = copy.deepcopy(model)
    merged.merge_adapter()
    return {name: param.detach().clone() for name, param in merged.named_parameters()}


def lora_bases_and_merges(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per PEFT-LoRA'd weight: the frozen base, and base + delta as PEFT adds it (``w += delta`` in its dtype).
    Collective on DTensor shards."""
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
    """``tensors`` as a push hands them, through the sync's own forwarder, so renames and hub-namespace reverts match.
    Rank-local."""
    recorder = RecordingSender(keep_values=True)
    guard = DeferredRankFailure("expected push")
    prefix = model.prefix if is_peft_model(model) else None
    forwarder = _HubForwarder(recorder, model, dict(named_ep_layers(model)), prefix, guard)
    for name, tensor in tensors.items():
        forwarder.send(name, tensor)
    forwarder.flush()
    if guard.reason is not None:
        raise RuntimeError(f"the sync's forwarder refused the expected tensors: {guard.reason}")
    return {param.name: param.value for param in recorder.params}


@contextmanager
def gathering_on_the_host() -> Iterator[None]:
    """The push asks for its experts on the GPU, which the CPU tier does not have."""
    start = EPGatherMixin.start_gather_expert_state_dict

    def on_host(self, *, device="cpu", **kwargs):
        return start(self, device="cpu", **kwargs)

    with patch.object(EPGatherMixin, "start_gather_expert_state_dict", on_host):
        yield


@contextmanager
def folded_in_place(peft_model: nn.Module) -> Iterator[None]:
    """PEFT's in-place merge and its bf16 unmerge after, with nothing written back: the negative control and memory
    baseline for pushes. A sync inside pushes the same weights, while the frozen base keeps the unmerge's rounding
    misses."""
    reshard_fsdp2_modules(peft_model)
    peft_model.merge_adapter()
    try:
        yield
    finally:
        peft_model.unmerge_adapter()
