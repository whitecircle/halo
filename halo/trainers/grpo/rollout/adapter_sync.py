"""Adapter-only weight sync for LoRA runs: the engine keeps the frozen base the first push (and any restart) gave it, and
each sync loads the trainer's adapter under a new name that new requests select. What the engine cannot serve as an
adapter is refused here, before the run, with the reason."""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import requests
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from peft.tuners.lora.layer import Linear, LoraLayer
from transformers.core_model_loading import rename_source_key

from halo.checkpoint.hub_layout import export_conversions, reversed_export_transforms
from halo.checkpoint.parallel.adapters import gather_adapter_state
from halo.checkpoint.reading import find_hub_snapshot
from halo.checkpoint.weights_source import weights_read_from
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.dtype import has_packed_quantized_params
from halo.model_loading.tokenizer_setup import vocabulary_grown
from halo.models.peft_adapters import tuner_adapter_param_ids
from halo.models.unwrapping import (
    PEFT_BASE_MODEL_PREFIX,
    base_transformers_model,
    find_peft_model,
    unwrap_framework_wrappers,
)
from halo.runtime.collectives import broadcast_from_rank0, combine_across_ranks
from halo.runtime.consensus import DeferredRankFailure, rank0_verdict
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.rollout.engine_preflight import AdapterRequirements
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights
from halo.weight_sync.adapter_payload import (
    ADAPTER_WIRE_DTYPE,
    build_adapter_config,
    serialize_adapter_tensors,
    wire_adapter_tensors,
)
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.base import ServedBase
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# Every adapter version this trainer loads is named under it, which is how a restarted trainer finds the versions
# its predecessor left on an engine.
ADAPTER_NAME_PREFIX = "halo-adapter"


class AdapterServing(NamedTuple):
    """What the run's adapters are, as the engine will be asked to serve them."""

    rank: int
    # The trainer's ``lora_alpha / r`` (or rsLoRA's ``/ sqrt(r)``), folded into B on the wire.
    scaling: float
    # The suffixes of the adapted modules.
    modules: tuple[str, ...]


class BasePlan(NamedTuple):
    """What rank 0's probe of the servers decided, for every rank."""

    push_base: bool
    # Each probed server's adapters, ``None`` for one that did not answer.
    holdings: dict[str, list[str] | None]


def resolve_launch_base(model: torch.nn.Module, async_config: AsyncRolloutConfig) -> ServedBase | None:
    """The base the trainer's frozen weights are, spelled as an engine launched with them reports it: a hub id is
    resolved to its snapshot directory through the cache the trainer loaded from. ``None`` where they are not a
    checkpoint an engine could have launched with (a from-scratch init, a grown vocabulary, a hub id the cache lacks)."""
    live = unwrap_framework_wrappers(model)
    path = weights_read_from(live)
    if path is None or vocabulary_grown(live):
        return None
    revision = getattr(getattr(live, "config", None), "_commit_hash", None)
    if not os.path.isdir(path):
        try:
            path = snapshot_download(path, revision=revision, local_files_only=True)
        except (LocalEntryNotFoundError, HFValidationError):
            return None
    path = path.rstrip("/")
    snapshot = find_hub_snapshot(path)
    return ServedBase(
        model_id=async_config.model_name, path=path, revision=snapshot[1] if snapshot is not None else revision
    )


def serves_launch_base(served: ServedBase, launch: ServedBase) -> bool:
    """Whether a server reports the weights the trainer launched with, matched on what identifies them: the model id
    its requests name, then the repo and commit both paths take in the hub cache, or else the same real path. A path
    that is neither is no match, and nor is a revision the server reports that the trainer's does not equal."""
    if served.model_id != launch.model_id or served.path is None or launch.path is None:
        return False
    served_snapshot, launch_snapshot = find_hub_snapshot(served.path), find_hub_snapshot(launch.path)
    if served_snapshot is not None and launch_snapshot is not None:
        same = served_snapshot == launch_snapshot
    else:
        same = os.path.realpath(served.path) == os.path.realpath(launch.path)
    return same and served.revision in (None, launch.revision)


def resolve_live_versions(async_config: AsyncRolloutConfig) -> int:
    """The adapter versions an engine can hold at once: the newest, one per round in flight under prefetch, and the
    version being retired while the next loads."""
    return (async_config.prefetch_depth if async_config.prefetch_active() else 0) + 2


def reject_unadaptable_config(async_config: AsyncRolloutConfig) -> None:
    """Refuses what the config alone settles."""
    client_cls = resolve_weight_sync_client(async_config.rollout_backend)
    reasons = []
    if client_cls.ADAPTER_SYNC_REFUSAL is not None:
        reasons.append(f"rollout_backend={async_config.rollout_backend!r}: {client_cls.ADAPTER_SYNC_REFUSAL}")
    if async_config.routing_replay == "rollout":
        reasons.append(
            "routing_replay: rollout (R3) captures the routing of the engine's forward, and whether that capture "
            "still fires through a LoRA-wrapped MoE layer has not been verified"
        )
    if not async_config.model_name or ":" in async_config.model_name:
        reasons.append(
            f"the rollout model_name {async_config.model_name!r} must be set and hold no colon: SGLang reads a "
            f"request's 'model' as '<base>:<adapter>' and would split the name"
        )
    if reasons:
        raise ValueError(
            "rollout_adapter_sync cannot serve this run: " + "; ".join(reasons) + ". Set rollout_adapter_sync: false "
            "to push the adapters folded into the weights."
        )


def _adapted_modules(model: torch.nn.Module) -> dict[str, LoraLayer]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoraLayer)}


def _build_hub_respeller(model: torch.nn.Module) -> Callable[[str], str | None]:
    """Maps a module's path in the live tree to the hub checkpoint's spelling of it, ``None`` where a converter fuses or
    splits its weight, which no adapter can follow."""
    conversions = export_conversions(base_transformers_model(model))
    renamings, converters = reversed_export_transforms(conversions)

    def respell(module_name: str) -> str | None:
        path = module_name.removeprefix(PEFT_BASE_MODEL_PREFIX)
        if not conversions:
            return path
        renamed, claimed_by = rename_source_key(f"{path}.weight", renamings, converters, reverse=True)
        return None if claimed_by is not None else renamed.removesuffix(".weight")

    return respell


def _unadaptable_reasons(model: torch.nn.Module, parallelism_config: ParallelismConfig, client_cls) -> list[str]:
    reasons = []
    peft_model = find_peft_model(model)
    if peft_model is None:
        return ["the model carries no PEFT adapter (use_peft: true with a LoRA config)"]
    config = peft_model.peft_config["default"]
    if has_packed_quantized_params(model):
        reasons.append("QLoRA: the engine serves the unquantized base the trainer's quantized one is not")
    for mode, active in (
        (
            "tensor parallelism: a replicated A diverges and a sharded B is corrupted by the TP gradient sync",
            parallelism_config.is_tp_mode,
        ),
        ("pipeline parallelism: adapter state is not stage-aware", parallelism_config.is_pp_mode),
        (
            "context parallelism: the Ulysses wrapper respells the adapter keys, and the respelling is not validated",
            parallelism_config.is_cp_mode,
        ),
    ):
        if active:
            reasons.append(mode)
    if model_has_expert_lora(model):
        reasons.append("native expert LoRA: no engine layout has been validated for the grouped expert adapters")
    for flag, why in (
        ("use_dora", "DoRA: the engine refuses it"),
        ("lora_bias", "lora_bias: the engine refuses it"),
        ("modules_to_save", "modules_to_save: the engine has no route for a fully trained module, so it would drift"),
        ("rank_pattern", "rank_pattern: the engine reads one rank"),
        ("alpha_pattern", "alpha_pattern: the engine reads one alpha"),
        ("target_parameters", "target_parameters: PEFT's fused-parameter route has no engine counterpart"),
    ):
        if getattr(config, flag, None):
            reasons.append(why)
    if vocabulary_grown(model):
        reasons.append("a grown vocabulary: the engine's is fixed and refuses added-token adapters")
    adapter_ids = tuner_adapter_param_ids(model)
    drifting = sorted(
        name for name, param in model.named_parameters() if param.requires_grad and id(param) not in adapter_ids
    )
    if drifting and not model_has_expert_lora(model):
        reasons.append(
            f"trainable parameters other than the adapters ({', '.join(drifting[:3])}...): the engine's base would "
            f"stop matching the trainer's"
        )
    wrong_dtype = sorted(
        {
            str(param.dtype)
            for param in model.parameters()
            if id(param) in adapter_ids and param.dtype != ADAPTER_WIRE_DTYPE
        }
    )
    if wrong_dtype:
        reasons.append(
            f"adapters in {wrong_dtype}: the engine stores {ADAPTER_WIRE_DTYPE}, so it would score a rounded copy of "
            f"what the trainer trains (fp32_non_ep_params)"
        )
    for name, module in _adapted_modules(model).items():
        suffix = name.rsplit(".", 1)[-1]
        if type(module) is not Linear or suffix not in client_cls.ADAPTER_SERVABLE_MODULES:
            reasons.append(
                f"the adapted module {name!r} is not one {client_cls.ENGINE_RELEASE} serves "
                f"({', '.join(client_cls.ADAPTER_SERVABLE_MODULES)} as plain linears)"
            )
            break
    respell = _build_hub_respeller(model)
    for name in _adapted_modules(model):
        if respell(name) is None:
            reasons.append(
                f"the adapted module {name!r} is fused or split in the hub layout, which no adapter follows"
            )
            break
    return reasons


def reject_unadaptable_model(model: torch.nn.Module, parallelism_config: ParallelismConfig, client_cls) -> None:
    """Refuses at construction, every rank alike, a model whose adapters the engine cannot serve in place of folded
    weights; names every reason."""
    reasons = _unadaptable_reasons(unwrap_framework_wrappers(model), parallelism_config, client_cls)
    if reasons:
        raise ValueError(
            "rollout_adapter_sync cannot serve this model's adapters: "
            + "; ".join(reasons)
            + ". Set rollout_adapter_sync: false to push the adapters folded into the weights."
        )


def resolve_adapter_serving(model: torch.nn.Module) -> AdapterServing:
    peft_model = find_peft_model(unwrap_framework_wrappers(model))
    config = peft_model.peft_config["default"]
    scaling = config.lora_alpha / (math.sqrt(config.r) if config.use_rslora else config.r)
    modules = tuple(sorted({name.rsplit(".", 1)[-1] for name in _adapted_modules(peft_model)}))
    return AdapterServing(rank=config.r, scaling=scaling, modules=modules)


def build_adapter_requirements(serving: AdapterServing, async_config: AsyncRolloutConfig) -> AdapterRequirements:
    return AdapterRequirements(
        rank=serving.rank, modules=serving.modules, live_versions=resolve_live_versions(async_config)
    )


def _hub_adapter_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``state`` under the hub's module paths, which the engine reads layer ids and projections off."""
    respell = _build_hub_respeller(model)
    hub = {}
    for key, tensor in state.items():
        module_name, marker, leaf = key.rpartition(".lora_")
        path = respell(module_name) if marker else None
        if path is None:
            raise ValueError(f"The adapter tensor {key!r} has no hub spelling the engine could load it under.")
        hub[f"{PEFT_BASE_MODEL_PREFIX}{path}{marker}{leaf}"] = tensor
    return hub


def _union(gathered: list[list[str]]) -> frozenset[str]:
    return frozenset(name for names in gathered for name in names)


class AdapterSync:
    """Serves the policy as an adapter on the rollout servers. :meth:`sync` is COLLECTIVE; ``servers`` is bound on the
    forwarding rank alone."""

    def __init__(
        self,
        serving: AdapterServing,
        versions: AdapterVersions,
        routes: AdapterRoutes,
        launch_base: ServedBase | None,
    ):
        self.serving = serving
        self.versions = versions
        self.routes = routes
        # What a restarted server serves again when it launched with the trainer's base; ``None`` never matches.
        self.launch_base = launch_base
        self.servers: AdapterServers | None = None
        # Rank 0's, so no two trainers (or two runs of one) name a version alike on an engine.
        self._nonce: str = broadcast_from_rank0(uuid.uuid4().hex[:8])
        self._sequence = 0
        self._base_pushed = False
        # The versions rank 0 has loaded and not retired.
        self._loaded: list[str] = []
        # The newest version's load, kept on rank 0 to put it on a server that restarted before the next sync.
        self._newest_load: tuple[str, dict, str] | None = None

    def reset(self) -> None:
        """A second ``train()`` pushes the base again."""
        self._base_pushed = False
        self._loaded = []
        self._newest_load = None
        if self.servers is not None:
            self.servers.close()
            self.servers = None

    def sync(self, trainer, pool) -> None:
        """The base once, and again when a server lost the adapter it held (a restart) and no longer serves the
        trainer's base, then the adapter. A restarted server's client reconnects first: its old communicator has no
        peer, and a push into it waits out the broadcast deadline."""
        model = unwrap_framework_wrappers(trainer.model)
        plan = rank0_verdict(partial(self._plan_base_push, pool), "Adapter probe of the rollout servers")
        self._learn(plan.holdings)
        if plan.push_base:
            sync_trainer_weights(trainer, pool, fold_adapters=False)
            self._base_pushed = True
            self._retire_predecessor_versions()
        name = f"{ADAPTER_NAME_PREFIX}-{self._nonce}-{self._sequence}"
        self._sequence += 1
        self._load_adapter(model, name)
        self.versions.publish(name)
        self._retire_unheld(name)
        self.routes.forget_except(self._tracked_versions())

    def recover(self, pool) -> None:
        """COLLECTIVE, between syncs. A server found restarted gets the newest version at once when it serves the
        trainer's base; one that does not stays out of routing until the next sync pushes the base. Never raises on a
        server that is down or refuses the load: the next call tries again."""
        if self.versions.current is None:
            return
        holdings = rank0_verdict(
            partial(self._restore_restarted, pool), "Adapter recovery of restarted rollout servers"
        )
        self._learn(holdings)

    def _tracked_versions(self) -> set[str]:
        current = self.versions.current
        return self.versions.held() | ({current} if current is not None else set())

    def _learn(self, holdings: dict[str, list[str] | None]) -> None:
        self.routes.learn(holdings, self._tracked_versions())

    def _plan_base_push(self, pool) -> BasePlan:
        """Whether the base must be pushed; reconnects the clients of servers that lost the newest adapter first."""
        current = self.versions.current
        if not self._base_pushed or current is None:
            pool.reconnect_servers()
            return BasePlan(True, {})
        holdings = self._bound_servers().probe()
        silent = sorted(url for url, names in holdings.items() if names is None)
        if silent:
            raise RuntimeError(
                f"Rollout server(s) {silent} did not answer the adapter probe (connection refused or timed out): "
                f"they are down or still restarting. Restart them before the sync."
            )
        restarted = [url for url, names in holdings.items() if current not in names]
        if restarted and all(self._serves_trainers_base(url) for url in restarted):
            pool.mark_servers_restarted(restarted)
            return BasePlan(False, holdings)
        if restarted:
            pool.reconnect_servers(restarted)
        return BasePlan(bool(restarted), holdings)

    def _restore_restarted(self, pool) -> dict[str, list[str] | None]:
        servers = self._bound_servers()
        holdings = servers.probe()
        current = self.versions.current
        if self._newest_load is None or self._newest_load[0] != current:
            return holdings
        restarted = [url for url, names in holdings.items() if names is not None and current not in names]
        restorable = [url for url in restarted if self._serves_trainers_base(url)]
        if not restorable:
            return holdings
        name, config, payload = self._newest_load
        try:
            servers.load(name, config, payload, urls=restorable)
        except RuntimeError as error:
            logger.warning(f"Adapter {name} could not be put back on restarted server(s) {restorable}: {error}")
            return holdings
        pool.mark_servers_restarted(restorable)
        for url in restorable:
            holdings[url] = [*holdings[url], name]
        logger.info(f"Loaded adapter {name} on restarted rollout server(s) {restorable}; their base is the trainer's.")
        return holdings

    def _serves_trainers_base(self, url: str) -> bool:
        """Whether the server, restarted, serves the frozen base the trainer holds: it launched with the weights the
        trainer read, under the model id its requests name."""
        if self.launch_base is None:
            return False
        try:
            served = self._bound_servers().served_base(url)
        except requests.exceptions.RequestException:
            return False
        return serves_launch_base(served, self.launch_base)

    def _bound_servers(self) -> AdapterServers:
        if self.servers is None:
            raise RuntimeError("The adapter sync reached its forwarding rank with no rollout servers bound.")
        return self.servers

    def _retire_predecessor_versions(self) -> None:
        """Retires the versions an earlier trainer left on the engines, which nothing here leases."""

        def retire() -> None:
            servers = self._bound_servers()
            left = sorted(
                {name for names in servers.loaded() for name in names if name.startswith(ADAPTER_NAME_PREFIX)}
            )
            servers.unload(left)

        rank0_verdict(retire, "Retiring a predecessor's adapter versions")

    def _load_adapter(self, model: torch.nn.Module, name: str) -> None:
        """COLLECTIVE. The adapter is gathered whole on the forwarding rank, scaled and sent to every server."""
        reshard_fsdp2_modules(model)
        forwarding = is_global_main_process()
        guard = DeferredRankFailure(f"Adapter load of {name} onto the rollout servers")
        state = gather_adapter_state(find_peft_model(model), retain=forwarding)
        if forwarding:
            guard.run(partial(self._send, model, name, state))
        del state
        guard.reject()

    def _send(self, model: torch.nn.Module, name: str, state: dict[str, torch.Tensor]) -> None:
        wire = wire_adapter_tensors(_hub_adapter_state(model, state), self.serving.scaling)
        config = build_adapter_config(rank=self.serving.rank, target_modules=self.serving.modules)
        payload = serialize_adapter_tensors(wire)
        self._bound_servers().load(name, config, payload)
        self._newest_load = (name, config, payload)
        self._loaded.append(name)
        logger.info(f"Loaded adapter {name} on the rollout servers ({len(wire)} tensors)")

    def _retire_unheld(self, current: str) -> None:
        """COLLECTIVE. A version leaves once no rank's rollout manager leases it, so the ranks' held sets join first."""
        held = combine_across_ranks(sorted(self.versions.held()), _union, what="held adapter versions")

        def retire() -> None:
            stale = [name for name in self._loaded if name != current and name not in held]
            self._bound_servers().unload(stale)
            self._loaded = [name for name in self._loaded if name not in stale]

        rank0_verdict(retire, "Retiring stale adapter versions")
