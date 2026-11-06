"""Adapter-only weight sync for LoRA runs: the engine keeps the frozen base the first push (and any restart) gave it, and
each sync loads the trainer's adapter under a new name that new requests select. What the engine cannot serve as an
adapter is refused here, before the run, with the reason."""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import torch
from peft.tuners.lora.layer import Linear, LoraLayer
from transformers.core_model_loading import rename_source_key

from halo.checkpoint.hub_layout import export_conversions, reversed_export_transforms
from halo.checkpoint.parallel.adapters import gather_adapter_state
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.fsdp import reshard_fsdp2_modules
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
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.registry import resolve_weight_sync_client

if TYPE_CHECKING:
    from halo.configs.grpo_environmental import AsyncRolloutConfig
    from halo.distributed.parallelism_config import ParallelismConfig

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

    def __init__(self, serving: AdapterServing, versions: AdapterVersions):
        self.serving = serving
        self.versions = versions
        self.servers: AdapterServers | None = None
        # Rank 0's, so no two trainers (or two runs of one) name a version alike on an engine.
        self._nonce: str = broadcast_from_rank0(uuid.uuid4().hex[:8])
        self._sequence = 0
        self._base_pushed = False
        # The versions rank 0 has loaded and not retired.
        self._loaded: list[str] = []

    def reset(self) -> None:
        """A second ``train()`` pushes the base again."""
        self._base_pushed = False
        self._loaded = []
        if self.servers is not None:
            self.servers.close()
            self.servers = None

    def sync(self, trainer, pool) -> None:
        """The base once, and again when a server lost the adapter it held (a restart), then the adapter."""
        model = unwrap_framework_wrappers(trainer.model)
        if rank0_verdict(self._base_is_stale, "Adapter probe of the rollout servers"):
            sync_trainer_weights(trainer, pool, fold_adapters=False)
            self._base_pushed = True
            self._retire_predecessor_versions()
        name = f"{ADAPTER_NAME_PREFIX}-{self._nonce}-{self._sequence}"
        self._sequence += 1
        self._load_adapter(model, name)
        self.versions.publish(name)
        self._retire_unheld(name)

    def _base_is_stale(self) -> bool:
        if not self._base_pushed:
            return True
        current = self.versions.current
        return current is None or bool(self._bound_servers().lacking(current))

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
        self._bound_servers().load(name, config, serialize_adapter_tensors(wire))
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
