"""Adapter-only weight sync for LoRA runs: the engine keeps the frozen base the first push (and any restart) gave it, and
each sync loads the trainer's adapter under a new name that new requests select. What the engine cannot serve as an
adapter is refused here, before the run or a saved adapter's export, with the reason."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Collection, Iterable, Mapping
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
from halo.checkpoint.weights_source import weights_read_at, weights_read_from
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.distributed.expert_parallel.moe_layer.expert_lora import (
    expert_lora_layers,
    gather_expert_lora_adapters,
    model_has_expert_lora,
)
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.dtype import has_packed_quantized_params
from halo.model_loading.tokenizer_setup import vocabulary_grown
from halo.models.introspection.config_access import config_model_types
from halo.models.introspection.unwrapping import (
    PEFT_BASE_MODEL_PREFIX,
    base_transformers_model,
    find_peft_model,
    unwrap_framework_wrappers,
)
from halo.models.moe.expert_layout import to_w13_w2_expert_adapter
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.peft_adapters import tuner_adapter_param_ids
from halo.runtime.collectives import broadcast_from_rank0, combine_across_ranks
from halo.runtime.consensus import DeferredRankFailure, rank0_verdict
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.rollout.engine_preflight import AdapterRequirements
from halo.trainers.grpo.rollout.weight_push import serves_family_as_adapter, sync_trainer_weights
from halo.weight_sync.adapter_directory import (
    publish_adapter_directory,
    remove_adapter_directories,
    remove_staged_adapter_directories,
)
from halo.weight_sync.adapter_payload import (
    ADAPTER_WIRE_DTYPE,
    build_adapter_config,
    serialize_adapter_tensors,
    wire_adapter_tensors,
)
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import ADAPTER_NAME_PREFIX, AdapterVersions
from halo.weight_sync.clients.base import ServedBase
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


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
    resolved to the snapshot of the commit the trainer read, through the cache it loaded from. ``None`` where they are
    not a checkpoint an engine could have launched with (a from-scratch init, a grown vocabulary, a hub id the cache
    lacks)."""
    live = unwrap_framework_wrappers(model)
    path = weights_read_from(live)
    if path is None or vocabulary_grown(live):
        return None
    revision = weights_read_at(live)
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


def engine_adapter_refusal(backend: str, model_config) -> str | None:
    """Why ``backend``'s engine serves no adapter of ``model_config``'s family, ``None`` where it does: an engine with
    an ``ADAPTER_SYNC_REFUSAL`` serves only the families validated as adapters on it."""
    client_cls = resolve_weight_sync_client(backend)
    if client_cls.ADAPTER_SYNC_REFUSAL is not None and not serves_family_as_adapter(model_config, backend):
        return f"rollout_backend={backend!r}: {client_cls.ADAPTER_SYNC_REFUSAL}"
    return None


def reject_unadaptable_config(async_config: AsyncRolloutConfig, model_config=None) -> None:
    """Refuses what the rollout config and the model's family (``model_config``) settle."""
    client_cls = resolve_weight_sync_client(async_config.rollout_backend)
    family_refusal = engine_adapter_refusal(async_config.rollout_backend, model_config)
    reasons = [family_refusal] if family_refusal is not None else []
    if client_cls.ADAPTER_NEEDS_DIRECTORY and async_config.rollout_adapter_dir is None:
        reasons.append(
            f"rollout_backend={async_config.rollout_backend!r} loads an adapter only from a directory: set "
            f"rollout_adapter_dir to one every server reads at the same path"
        )
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


def hub_spellings_absent_from_base(model: torch.nn.Module, module_names: Iterable[str]) -> dict[str, str | None]:
    """Each of ``module_names`` (as PEFT keys spell them) whose hub spelling names no weight the base's own checkpoint
    holds, with that spelling. ``model`` is the live model or the base's class built without weights; a module a later
    transformers moved or wrapped respells to a name the engine would find nothing under."""
    respell = _build_hub_respeller(model)
    base = base_transformers_model(model)
    # A live PEFT model holds an adapted module's weight under ``base_layer``.
    held = {
        respell(name.removesuffix(".weight").removesuffix(".base_layer"))
        for name, _ in base.named_parameters(remove_duplicate=False)
        if name.endswith(".weight")
    }
    return {name: path for name in module_names if (path := respell(name)) is None or path not in held}


def _attention_lora_spec(config: Mapping) -> ExpertLoraSpec:
    return ExpertLoraSpec(r=config["r"], alpha=config["lora_alpha"], use_rslora=bool(config.get("use_rslora")))


def _resolve_server_name(model_types: tuple[str, ...], client_cls) -> str:
    """The image ``client_cls`` serves a model of ``model_types`` from, or its engine release where it names none."""
    image = client_cls.image_for(model_types)
    return image.tag if image else client_cls.ENGINE_RELEASE


def _unservable_expert_lora(model: torch.nn.Module, config: Mapping, expert: ExpertLoraSpec, client_cls) -> list[str]:
    """Native grouped expert adapters ride beside the attention ones, so they need the engine's routed-expert LoRA for
    the model and the attention adapters' rank and scaling: one adapter config carries both."""
    reasons = []
    model_types = config_model_types(getattr(base_transformers_model(model), "config", None))
    if not client_cls.ADAPTER_EXPERT_MODEL_TYPES.intersection(model_types):
        reasons.append(
            f"native expert LoRA: {_resolve_server_name(model_types, client_cls)} serves no routed-expert adapter on "
            f"this model"
        )
    attention = _attention_lora_spec(config)
    if expert.r != attention.r or expert.scaling != attention.scaling:
        reasons.append("native expert LoRA at another rank or scaling than the attention adapters: one is served")
    return reasons


def _unservable_module_reason(name: str, model: torch.nn.Module, client_cls) -> str:
    model_types = config_model_types(getattr(base_transformers_model(model), "config", None))
    return (
        f"the adapted module {name!r} is not one {_resolve_server_name(model_types, client_cls)} serves "
        f"({', '.join(client_cls.ADAPTER_SERVABLE_MODULES)} as plain linears)"
    )


def unservable_adapter_reasons(
    config: Mapping,
    module_names: Collection[str],
    model: torch.nn.Module,
    client_cls,
    expert: ExpertLoraSpec | None = None,
) -> list[str]:
    """Why the engine of ``client_cls`` cannot serve the LoRA that ``config`` (a ``LoraConfig``'s fields) puts on the
    modules ``module_names`` of ``model``, with ``expert``'s native grouped expert adapters beside it: what the
    adapters declare, whether a live trainer or a saved adapter directory holds them."""
    reasons = _unservable_expert_lora(model, config, expert, client_cls) if expert is not None else []
    for flag, why in (
        ("use_dora", "DoRA: the engine refuses it"),
        ("lora_bias", "lora_bias: the engine refuses it"),
        ("modules_to_save", "modules_to_save: the engine has no route for a fully trained module, so it would drift"),
        ("rank_pattern", "rank_pattern: the engine reads one rank"),
        ("alpha_pattern", "alpha_pattern: the engine reads one alpha"),
        ("target_parameters", "target_parameters: PEFT's fused-parameter route has no engine counterpart"),
    ):
        if config.get(flag):
            reasons.append(why)
    unservable = next(
        (
            name
            for name in module_names
            if not any(f".{name}".endswith(f".{servable}") for servable in client_cls.ADAPTER_SERVABLE_MODULES)
        ),
        None,
    )
    if unservable is not None:
        reasons.append(_unservable_module_reason(unservable, model, client_cls))
    respell = _build_hub_respeller(model)
    unspelled = next((name for name in module_names if respell(name) is None), None)
    if unspelled is not None:
        reasons.append(
            f"the adapted module {unspelled!r} is fused or split in the hub layout, which no adapter follows"
        )
    return reasons


def _unadaptable_reasons(model: torch.nn.Module, parallelism_config: ParallelismConfig, client_cls) -> list[str]:
    reasons = []
    peft_model = find_peft_model(model)
    if peft_model is None:
        return ["the model carries no PEFT adapter (use_peft: true with a LoRA config)"]
    config = peft_model.peft_config["default"]
    adapted = _adapted_modules(model)
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
    layers = expert_lora_layers(model)
    expert = layers[0][1].ep_config.expert_lora if layers else None
    reasons.extend(unservable_adapter_reasons(config.to_dict(), list(adapted), model, client_cls, expert))
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
    not_linear = next((name for name, module in adapted.items() if type(module) is not Linear), None)
    if not_linear is not None and _unservable_module_reason(not_linear, model, client_cls) not in reasons:
        reasons.append(_unservable_module_reason(not_linear, model, client_cls))
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


def build_adapter_serving(config: Mapping, module_names: Iterable[str]) -> AdapterServing:
    """How the engine serves the LoRA that ``config`` (a ``LoraConfig``'s fields) puts on the modules ``module_names``."""
    modules = tuple(sorted({name.rsplit(".", 1)[-1] for name in module_names}))
    return AdapterServing(rank=config["r"], scaling=_attention_lora_spec(config).scaling, modules=modules)


def resolve_adapter_serving(model: torch.nn.Module) -> AdapterServing:
    peft_model = find_peft_model(unwrap_framework_wrappers(model))
    return build_adapter_serving(peft_model.peft_config["default"].to_dict(), _adapted_modules(peft_model))


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


def _engine_expert_adapters(model: torch.nn.Module, experts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The gathered grouped adapters (``<layer>.experts.<attr>.lora_{A,B}``) as the engine's 3D per-expert ones on the
    checkpoint's expert weights, under the hub's layer paths (:func:`to_w13_w2_expert_adapter`)."""
    respell = _build_hub_respeller(model)
    engine = {}
    for key, tensor in experts.items():
        layer, _, rest = key.partition(".experts.")
        attr, side = rest.split(".lora_")
        path = respell(layer)
        if path is None:
            raise ValueError(
                f"The expert layer {layer!r} has no hub spelling the engine could load its adapters under."
            )
        weight, matrix = to_w13_w2_expert_adapter(attr, side, tensor)
        engine[f"{PEFT_BASE_MODEL_PREFIX}{path}.experts.{weight}.lora_{side}.weight"] = matrix
    return engine


def build_engine_adapter(
    model: torch.nn.Module, serving: AdapterServing, state: dict[str, torch.Tensor], experts: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict]:
    """The tensors and ``adapter_config.json`` the engine loads: ``state`` (keyed as PEFT saves an adapter) and
    ``experts`` (keyed as :func:`gather_expert_lora_adapters` returns them and an adapter save writes them) under the
    hub's names and the engine's expert layout, the scaling folded into B in the wire dtype. ``model`` supplies the
    hub spelling: the live model, or the base's class built without weights. Raises on a tensor with no hub
    spelling."""
    engine_state = {**_hub_adapter_state(model, state), **_engine_expert_adapters(model, experts)}
    wire = wire_adapter_tensors(engine_state, serving.scaling)
    return wire, build_adapter_config(rank=serving.rank, target_modules=serving.modules)


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
        base_refusal: str | None = None,
        adapter_dir: str | None = None,
    ):
        self.serving = serving
        self.versions = versions
        self.routes = routes
        # What a restarted server serves again when it launched with the trainer's base; ``None`` never matches.
        self.launch_base = launch_base
        # Why no engine takes this model's full weights; the servers must then hold its launch base, never a push.
        self.base_refusal = base_refusal
        # Where each version is published for the servers to read; ``None`` sends it in the request body.
        self.adapter_dir = adapter_dir
        self.servers: AdapterServers | None = None
        # Rank 0's, so no two trainers (or two runs of one) name a version alike on an engine.
        self._nonce: str = broadcast_from_rank0(uuid.uuid4().hex[:8])
        self._sequence = 0
        self._base_pushed = False
        # Whether this run's first sync has retired the versions found on the engines.
        self._predecessors_retired = False
        # The versions rank 0 has loaded and not retired.
        self._loaded: list[str] = []
        # The newest version's load, kept on rank 0 to put it on a server that restarted before the next sync.
        self._newest_load: tuple[str, dict, str] | None = None

    def reset(self) -> None:
        """A second ``train()`` pushes the base again, and its first sync retires the versions this one leaves loaded."""
        self._base_pushed = False
        self._predecessors_retired = False
        self._loaded = []
        self._newest_load = None
        if self.servers is not None:
            self.servers.close()
            self.servers = None

    def sync(self, trainer, pool) -> None:
        """The base once, and again when a server lost the adapter it held (a restart) and no longer serves the
        trainer's base, then the adapter. A restarted server's client reconnects first: its old communicator has no
        peer, and a push into it waits out the broadcast deadline. ``pool`` is ``None`` for a base no push carries,
        which forms no group."""
        model = unwrap_framework_wrappers(trainer.model)
        plan = rank0_verdict(partial(self._plan_base_push, pool), "Adapter probe of the rollout servers")
        self._learn(plan.holdings)
        if plan.push_base:
            sync_trainer_weights(trainer, pool, fold_adapters=False)
            self._base_pushed = True
        if plan.push_base or not self._predecessors_retired:
            self._retire_predecessor_versions()
            self._predecessors_retired = True
        name = f"{ADAPTER_NAME_PREFIX}{self._nonce}-{self._sequence}"
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
        if self.base_refusal is not None:
            return self._plan_launch_base_only()
        if not self._base_pushed or current is None:
            pool.reconnect_servers()
            return BasePlan(True, {})
        holdings = self._bound_servers().require_all_answering()
        restarted = [url for url, names in holdings.items() if current not in names]
        if restarted and all(self._serves_trainers_base(url) for url in restarted):
            pool.mark_servers_restarted(restarted)
            return BasePlan(False, holdings)
        if restarted:
            pool.reconnect_servers(restarted)
        return BasePlan(bool(restarted), holdings)

    def _plan_launch_base_only(self) -> BasePlan:
        """For a base no push can carry, which forms no weight-sync group: every server must hold the trainer's launch
        base, and each takes the new version the sync loads."""
        holdings = self._bound_servers().require_all_answering()
        foreign = [url for url in holdings if not self._serves_trainers_base(url)]
        if foreign:
            launch = self.launch_base.path if self.launch_base is not None else "the checkpoint the trainer loaded"
            raise ValueError(
                f"Rollout server(s) {foreign} do not serve the trainer's base, and this model's base cannot be pushed "
                f"({self.base_refusal}). Launch them on {launch} under the rollout model_name."
            )
        return BasePlan(False, holdings)

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
        if pool is not None:
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
        """Retires the versions an earlier trainer, or an earlier ``train()`` of this one, left on the engines, which
        nothing here leases, and removes their directories, with those of versions no engine lists any more (a server
        restarted since, or an eviction past its loaded limit) and the stages of writes a killed trainer cut short,
        which would otherwise pile up run after run."""

        def retire() -> None:
            servers = self._bound_servers()
            left = {name for names in servers.loaded() for name in names if name.startswith(ADAPTER_NAME_PREFIX)}
            on_disk = self.adapter_dir is not None and os.path.isdir(self.adapter_dir)
            if on_disk:
                left.update(name for name in os.listdir(self.adapter_dir) if name.startswith(ADAPTER_NAME_PREFIX))
            # This run's versions may be leased on a server that kept them; they leave through _retire_unheld.
            left.difference_update(self._loaded)
            servers.unload(sorted(left))
            self._remove_directories(sorted(left))
            if on_disk:
                remove_staged_adapter_directories(self.adapter_dir, ADAPTER_NAME_PREFIX)

        rank0_verdict(retire, "Retiring a predecessor's adapter versions")

    def _load_adapter(self, model: torch.nn.Module, name: str) -> None:
        """COLLECTIVE. The adapter is gathered whole on the forwarding rank, scaled and sent to every server."""
        reshard_fsdp2_modules(model)
        forwarding = is_global_main_process()
        guard = DeferredRankFailure(f"Adapter load of {name} onto the rollout servers")
        state = gather_adapter_state(find_peft_model(model), retain=forwarding)
        experts = gather_expert_lora_adapters(model, retain=forwarding)
        if forwarding:
            guard.run(partial(self._send, model, name, state, experts))
        del state, experts
        guard.reject()

    def _send(
        self, model: torch.nn.Module, name: str, state: dict[str, torch.Tensor], experts: dict[str, torch.Tensor]
    ) -> None:
        wire, config = build_engine_adapter(model, self.serving, state, experts)
        if self.adapter_dir is None:
            payload = serialize_adapter_tensors(wire)
        else:
            payload = publish_adapter_directory(self.adapter_dir, name, wire, config)
        try:
            self._bound_servers().load(name, config, payload)
        except BaseException:
            # The load is all or none, so no server holds the version.
            self._remove_directories([name])
            raise
        self._newest_load = (name, config, payload)
        self._loaded.append(name)
        logger.info(f"Loaded adapter {name} on the rollout servers ({len(wire)} tensors)")

    def _retire_unheld(self, current: str) -> None:
        """COLLECTIVE. A version leaves once no rank's rollout manager leases it, so the ranks' held sets join first."""
        held = combine_across_ranks(sorted(self.versions.held()), _union, what="held adapter versions")

        def retire() -> None:
            stale = [name for name in self._loaded if name != current and name not in held]
            self._bound_servers().unload(stale)
            self._remove_directories(stale)
            self._loaded = [name for name in self._loaded if name not in stale]

        rank0_verdict(retire, "Retiring stale adapter versions")

    def _remove_directories(self, names: list[str]) -> None:
        """The published directories of versions no server holds."""
        if self.adapter_dir is not None:
            remove_adapter_directories(self.adapter_dir, names)
