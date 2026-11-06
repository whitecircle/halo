"""The trainer side of the rollout-engine weight sync: every rank gathers the EP experts, the buffers a family declares
and the dense shards, and the forwarding rank streams them under hub names, PEFT folded out of place."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Protocol

import torch
from accelerate.utils import is_peft_model
from transformers.core_model_loading import rename_source_key

from halo.checkpoint.hub_layout import export_conversions, reversed_export_transforms, revert_conversions
from halo.diagnostics.profiling import log_cuda_memory
from halo.distributed.expert_parallel.families.roster import ep_layer_classes_for_config
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes, iter_hand_sliced_full
from halo.env import env_flag, env_non_negative_int
from halo.model_loading.dtype import has_packed_quantized_params
from halo.models.attention.gpt_oss_sinks import SinksPolicy, neutralized_gpt_oss_sinks, stamped_sinks_policy
from halo.models.config_access import config_model_types, get_config_field
from halo.models.moe.balancing import (
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    is_transient_balancing_router,
    iter_balancing_routers,
    resolve_balancing_slot,
)
from halo.models.moe.config_fields import has_routed_experts
from halo.models.peft_adapters import LoraFolds, lora_fold_targets, lora_folded_data
from halo.models.unwrapping import base_transformers_model, plain_peft_param_name, unwrap_framework_wrappers
from halo.runtime.collectives import collective_side_stream
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit
from halo.runtime.dtensor import materialize_dtensor
from halo.runtime.ranks import is_global_main_process
from halo.weight_sync.chunking import WEIGHT_SYNC_CHUNK_BYTES
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# The rank the push sends from: accelerate's main process, which `sync_weights_to_client` takes as the forwarding rank.
FORWARDING_RANK = 0


class WeightSyncSender(Protocol):
    """What a push sends through: one engine client, or a pool of them."""

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None: ...

    def update_named_param(self, name: str, weights: torch.Tensor) -> None: ...

    def finish_weight_update(self) -> None: ...

    def abort_weight_update(self) -> None: ...


def _dense_params(
    model: torch.nn.Module, ep_layers: dict[str, EPMoELayerBase]
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """All but the EP experts and the hand-sliced TP shards, which are not DTensors and go gathered through the
    drain."""
    hand_sliced = hand_sliced_suffixes(model)
    for name, param in model.named_parameters():
        expert = any(
            name.startswith(root + ".") and layer.is_expert_param(name[len(root) + 1 :])
            for root, layer in ep_layers.items()
        )
        if expert or (hand_sliced and name.endswith(hand_sliced)):
            continue
        yield name, param


def _reject_unpushed_folds(model: torch.nn.Module, folds: LoraFolds, pushed: set[int]) -> None:
    unpushed = folds.keys() - pushed
    if unpushed:
        names = sorted(name for name, param in model.named_parameters() if id(param) in unpushed)
        raise ValueError(
            f"PEFT LoRA adapts {names}, which the weight sync does not send through its dense push, the "
            f"only path that folds PEFT adapters: the served weights would miss their delta. EP expert "
            f"weights take native expert LoRA (list the expert projections in lora_target_modules)."
        )


def _reject_model_state(model: torch.nn.Module, ep_layers: dict[str, EPMoELayerBase]) -> None:
    if has_packed_quantized_params(model):
        raise ValueError(
            "QLoRA (quantized base weights) is not supported with rollout-engine weight sync: the sync forwards "
            "raw parameter storage under base-weight names, and bnb-packed non-floating-point tensors "
            "corrupt the served policy. Use plain LoRA (use_peft without load_in_4bit/load_in_8bit) "
            "or full fine-tuning."
        )
    _reject_unpushed_folds(
        model, lora_fold_targets(model), {id(param) for _name, param in _dense_params(model, ep_layers)}
    )
    # Refused rather than repaired: every checkpoint writer re-emits neutralized sinks, but this sync
    # forwards named_parameters and has no such seam.
    if neutralized_gpt_oss_sinks(model):
        raise ValueError(
            "GptOss weight sync with sinks removed by the flash_attention_2 reset_sinks reset: the "
            "sync sends named_parameters only, so the removed sinks are never pushed and the rollout "
            "engine keeps generating with the pretrained sinks while the trainer runs without them — "
            "permanently off-policy with no error at sync time. On-policy RL for GptOss requires "
            "reset_sinks: false with a sink-carrying attention implementation (flash_attention_4 / "
            "eager), which is what the shipped GRPO configs set."
        )
    if stamped_sinks_policy(model) is SinksPolicy.TRAINABLE:
        raise ValueError(
            "train_sinks: true is SFT-only: no rollout engine has a validated end-to-end sync for sinks "
            "that change every step, so the trainer would drift from its generator with no error at sync "
            "time. On-policy GptOss RL keeps the pretrained sinks live and frozen (reset_sinks: false)."
        )


def _engine_loaded_buffer_attrs(layer: EPMoELayerBase, config) -> tuple[str, ...]:
    buffers = layer.FAMILY.engine_loaded_buffers
    return buffers.attrs if buffers is not None and get_config_field(config, buffers.config_flag, False) else ()


def _engine_loaded_buffer(layer: EPMoELayerBase, attr: str) -> torch.Tensor:
    """Read through the owner: ``lora_modules_to_save`` may wrap it, and the wrapper answers with the copy that routes.
    A parameter is refused, as the dense push already sends it."""
    owner, _, leaf = attr.rpartition(".")
    buffer = getattr(layer.get_submodule(owner), leaf)
    if not isinstance(buffer, torch.Tensor) or isinstance(buffer, torch.nn.Parameter):
        raise AttributeError(f"{attr!r} is {type(buffer).__name__}, not a buffer")
    return buffer


def _reject_stock_tree(config, family_layers: list[str], client_cls) -> None:
    """A model with no live EP layer forwards its stock module tree, whose expert names no validated sync reads. Only a
    dense model with no MoE family's spelling passes."""
    model_types = sorted(config_model_types(config))
    routed = has_routed_experts(config)
    if family_layers and routed:
        raise ValueError(
            f"{', '.join(family_layers)} resolves for model_type {model_types}, but the model carries no live EP "
            f"layer (expert_parallel_size 1 with use_grouped_gemm: false): weight sync ships experts in the layout "
            f"the family's gather_expert_state_dict emits, and without an EP layer it would forward the stock "
            f"module tree's fused expert tensors under module names, which {client_cls.ENGINE_RELEASE}'s loader drops "
            f"with no error, so attention, norms and routers would sync while the experts keep serving launch "
            f"weights. Set use_grouped_gemm: true (the torchrun default), which installs the EP layers at "
            f"expert_parallel_size 1 too."
        )
    if family_layers:
        raise ValueError(
            f"model_type {model_types} is a spelling of the MoE family {', '.join(family_layers)}, but this checkpoint "
            f"has no routed experts: it is a dense tower under a composite the weight sync has no validated layout "
            f"for. Train it offline, or sync its dense text checkpoint."
        )
    if routed:
        raise ValueError(
            f"model_type {model_types} is a MoE with no Halo EP family, so no expert layout has been validated against "
            f"{client_cls.ENGINE_RELEASE}: SGLang's loaders drop the stock tree's fused experts with no error, vLLM takes "
            f"them only where the checkpoint spells gate_proj/up_proj and drops w1/w3 names, and no sync into such a "
            f"family has run, so experts could keep serving launch weights while the rest syncs. Train it offline "
            f"(offline GRPO needs no server), or add the family (agent-docs/models/adding-a-model.md)."
        )


def _reject_unservable_model_types(config, backend: str, client_cls) -> None:
    unservable = sorted(client_cls.UNSERVABLE_MODEL_TYPES.keys() & config_model_types(config))
    if unservable:
        facts = "; ".join(
            f"{model_type}: {client_cls.UNSERVABLE_MODEL_TYPES[model_type]}" for model_type in unservable
        )
        raise ValueError(
            f"rollout_backend={backend!r} cannot serve model_type {unservable}, so the weight sync has "
            f"no served model to land in — {facts}. See {client_cls.__name__}.UNSERVABLE_MODEL_TYPES."
        )


def _reject_refused_families(families: list[tuple[str, type[EPMoELayerBase]]], client_cls) -> None:
    for where, cls in families:
        if cls.FAMILY.weight_sync_refusal is not None:
            raise ValueError(
                f"{cls.__name__} (at {where!r}) does not support weight sync, which feeds "
                f"{client_cls.ENGINE_RELEASE}'s model.load_weights directly: {cls.FAMILY.weight_sync_refusal}. "
                f"Online and environmental GRPO cannot train this model. Train it offline: offline GRPO needs no "
                f"server (agent-docs/training-methods/grpo/offline.md)."
            )


def reject_unsyncable_config(config, backend: str) -> None:
    """What ``config`` alone settles, for a script to raise before the load and the server preflights a refused family
    would pay for first. ``None`` for a config no rank could read, which the trainer's gate still sees as a model."""
    if config is None:
        return
    client_cls = resolve_weight_sync_client(backend)
    _reject_unservable_model_types(config, backend, client_cls)
    registry = f"model_type registry for {config.model_type}"
    _reject_refused_families([(registry, cls) for cls in ep_layer_classes_for_config(config)], client_cls)


def _reject_engine_fit(model: torch.nn.Module, backend: str, ep_layers: dict[str, EPMoELayerBase]) -> None:
    """Refuses an unservable family, one with a ``weight_sync_refusal``, a stock tree whose experts or spelling no engine
    layout covers (:func:`_reject_stock_tree`), and one missing a buffer the push must carry."""
    client_cls = resolve_weight_sync_client(backend)
    config = getattr(unwrap_framework_wrappers(model), "config", None)
    _reject_unservable_model_types(config, backend, client_cls)
    # Without live EP layers (the stock module tree) the registry's classes for the config carry the contract.
    if ep_layers:
        families = [(name, type(module)) for name, module in ep_layers.items()]
    else:
        families = [
            (f"model_type registry for {type(model).__name__}", cls) for cls in ep_layer_classes_for_config(config)
        ]
    _reject_refused_families(families, client_cls)
    if not ep_layers:
        _reject_stock_tree(config, sorted({cls.__name__ for _where, cls in families}), client_cls)
    for where, layer in ep_layers.items():
        for attr in _engine_loaded_buffer_attrs(layer, config):
            try:
                _engine_loaded_buffer(layer, attr)
            except AttributeError as missing:
                raise ValueError(
                    f"{type(layer).__name__} (at {where!r}) declares the buffer {attr!r} the weight sync must carry, "
                    f"but the layer holds none: the family moved or renamed it upstream, and {client_cls.ENGINE_RELEASE} "
                    f"would keep a value the trainer does not route with. Update {layer.FAMILY.model_types[0]}'s "
                    f"engine_loaded_buffers."
                ) from missing
    _warn_on_a_bias_the_engine_drops(ep_layers, client_cls, config)


def _warn_on_a_bias_the_engine_drops(ep_layers: dict[str, EPMoELayerBase], client_cls, config) -> None:
    """COLLECTIVE for a sharded slot. A checkpoint trained with bias balancing routes on its bias here while this engine
    never loads it, a systematic difference on every prompt. A warning, since ``routing_replay: rollout`` closes it."""
    if not client_cls.DROPPED_BALANCING_BIAS_MODEL_TYPES & set(config_model_types(config)):
        return
    carrying = 0
    for layer in ep_layers.values():
        slots = (resolve_balancing_slot(layer, attr) for attr in layer.FAMILY.balancing_slot.attrs)
        slot = next((slot for slot in slots if slot is not None), None)
        carrying += slot is not None and bool(materialize_dtensor(getattr(*slot)).any())
    if carrying and is_global_main_process():
        logger.warning(
            f"{carrying} MoE layers of this checkpoint hold a nonzero router bias, and {client_cls.ENGINE_RELEASE}'s "
            f"loader drops it: the trainer routes on the bias and the engine does not, a systematic gap on every "
            f"prompt rather than rounding noise. Set routing_replay: rollout, so the update trains through the "
            f"engine's selection."
        )


def _reject_balancing_state(model: torch.nn.Module) -> None:
    """Refuse enabled bias-update balancing state: the payload is parameters, plus the buffers a family declares at
    the value they loaded with, and no bias that moves between pushes."""
    # Enabled state, not the mode string, which the shipped scripts downgrade first; Zaya's native buffer is
    # neither adopted nor transient.
    balancing = sorted(
        {
            type(m).__name__
            for m in iter_balancing_routers(model)
            if getattr(m, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) or is_transient_balancing_router(m)
        }
    )
    if balancing:
        raise ValueError(
            f"router bias-update balancing is enabled on {', '.join(balancing)} while weight sync to an "
            f"external rollout engine is configured: the sync payload carries no bias that moves (an adopted "
            f"buffer or a transient side-buffer), so trainer routing drifts from the generator producing the "
            f"trajectories. Run weight-sync RL with "
            f"moe_balancing: none — the shipped GRPO scripts downgrade it automatically."
        )


def reject_unsyncable_model(model: torch.nn.Module, backend: str) -> None:
    """Refuses at construction; structural, so every rank raises alike."""
    ep_layers = dict(named_ep_layers(model))
    _reject_model_state(model, ep_layers)
    _reject_engine_fit(model, backend, ep_layers)
    _reject_balancing_state(model)


class _HubForwarder:
    """The engine skips an unknown name silently, hence the hub keys. A tensor a reverse ``WeightConverter`` claims is
    held until :meth:`flush`, as a many-to-one revert needs all its sources."""

    def __init__(
        self,
        client: WeightSyncSender,
        model: torch.nn.Module,
        ep_layers: dict[str, EPMoELayerBase],
        peft_prefix: str | None,
        guard: DeferredRankFailure,
    ):
        self._client = client
        self._guard = guard
        self._ep_layers = ep_layers
        self._peft_prefix = peft_prefix
        self._held: dict[str, torch.Tensor] = {}
        self._held_bytes = 0
        self._model = base_transformers_model(model)
        self._conversions = export_conversions(self._model)
        self._renamings, self._converters = reversed_export_transforms(self._conversions)

    def send(self, name: str, tensor: torch.Tensor, *, bounded: bool = True) -> None:
        """``bounded=False`` for a tensor the caller keeps alive and flushes itself, whose hold adds no bytes."""
        self._guard.run(partial(self._forward, name, tensor, bounded))

    def flush(self) -> None:
        self._guard.run(self._revert_held)

    def _forward(self, name: str, tensor: torch.Tensor, bounded: bool) -> None:
        for layer_name, layer in self._ep_layers.items():
            prefix = f"{layer_name}."
            if name.startswith(prefix):
                name = prefix + layer.FAMILY.to_hub_key(name[len(prefix) :])
                break
        if self._peft_prefix is not None:
            name = plain_peft_param_name(name, self._peft_prefix)
            if name is None:
                return
        if not self._conversions:
            self._client.update_named_param(name, tensor)
            return
        renamed, claimed_by = rename_source_key(name, self._renamings, self._converters, reverse=True)
        if claimed_by is None:
            self._client.update_named_param(renamed, tensor)
            return
        self._held[name] = tensor
        if not bounded:
            return
        self._held_bytes += tensor.nbytes
        if self._held_bytes > WEIGHT_SYNC_CHUNK_BYTES:
            raise RuntimeError(
                f"weight sync is holding {self._held_bytes / 2**30:.2f} GiB on the forwarding rank: "
                f"{len(self._held)} tensors up to {name!r} are claimed by a reverse WeightConverter of "
                f"{type(self._model).__name__} and cannot be sent until their sources are complete. The "
                f"held set lives on that one rank's GPU beside the gather, so it is bounded at "
                f"{WEIGHT_SYNC_CHUNK_BYTES / 2**30:.2f} GiB (HALO_WEIGHT_SYNC_CHUNK_MB) — a converter claiming "
                f"per-decoder-layer tensors needs a flush per layer, not one per gather phase."
            )

    def _revert_held(self) -> None:
        if not self._held:
            return
        held, self._held, self._held_bytes = self._held, {}, 0
        for hub_name, hub_tensor in revert_conversions(self._model, held, self._conversions).items():
            self._client.update_named_param(hub_name, hub_tensor)


def _send_ep_expert_weights(
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    guard: DeferredRankFailure,
    *,
    forwarding_rank: int = FORWARDING_RANK,
) -> None:
    """COLLECTIVE. Each layer is delivered to ``forwarding_rank`` alone, and layer k+1's gather is issued before layer k
    is forwarded (``HALO_WEIGHT_SYNC_GATHER_AHEAD`` layers ahead, 0 for strictly in turn), in one order on every rank.
    The retained assembly is the sync's largest rank-local allocation, so it runs under ``guard`` and retaining stops
    after a failure. The gather stays in step through a failed retained receive
    (:meth:`~halo.distributed.expert_parallel.moe_layer.gather.EPGatherMixin.gather_expert_tensors`)."""
    ahead = env_non_negative_int("HALO_WEIGHT_SYNC_GATHER_AHEAD", 1)
    layers = list(ep_layers.items())
    stream = collective_side_stream() if ahead else None
    issued: deque = deque()

    def issue(index: int) -> None:
        # Only the forwarding rank needs the assembled layer, and only while it can still send it.
        retain = forwarder is not None and guard.reason is None
        # The native expert LoRA is folded here; the PEFT adapters, never on experts, fold in the dense push.
        gather = partial(
            layers[index][1].start_gather_expert_state_dict,
            device="cuda",
            merge_lora=True,
            retain=retain,
            dst=forwarding_rank,
            stream=stream,
        )
        # Guarded only where it retains: elsewhere a raise is a group-wide failure, not this rank's own.
        issued.append(guard.run(gather) if retain else gather())

    for index in range(min(ahead, len(layers))):
        issue(index)
    for index, (layer_name, _) in enumerate(layers):
        if index + ahead < len(layers):
            issue(index + ahead)
        pending = issued.popleft()
        gathered = guard.run(pending.finish) if forwarder is not None and pending is not None else None
        del pending
        if gathered:
            for param_name, param_data in gathered.items():
                # Alive in ``gathered`` until the flush below, so a hub split's hold costs nothing here; Step-3.7's
                # fused bank alone is 5.6 GiB per layer at its release shape.
                forwarder.send(f"{layer_name}.{param_name}", param_data, bounded=False)
            # Per layer, so a hub split of these experts never outlives the next gather.
            forwarder.flush()
            del param_data  # the loop's last binding would hold one tensor through the next gather
        # Released before the next layer's gather finishes, so only the issued-ahead assemblies are alive beside it.
        del gathered


def _send_engine_loaded_buffers(ep_layers: dict[str, EPMoELayerBase], forwarder: _HubForwarder | None, config) -> None:
    """COLLECTIVE for a sharded buffer, so every rank walks them. Run before the dense push, whose flush ends the
    forwarder's holding."""
    for layer_name, layer in ep_layers.items():
        for attr in _engine_loaded_buffer_attrs(layer, config):
            data = materialize_dtensor(_engine_loaded_buffer(layer, attr))
            if forwarder is not None:
                forwarder.send(f"{layer_name}.{attr}", data)


def _send_dense_weights(
    model: torch.nn.Module,
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    folds: LoraFolds,
) -> None:
    """COLLECTIVE. One ``full_tensor()`` over FSDP2 DP and TP, the hand-sliced TP shards through their own drain;
    refuses a PEFT fold target the push did not send."""
    folded: set[int] = set()
    # Stream gather→send→drop (a dict = full dense copy/rank: OOM at 70B+); param order keeps collectives in step.
    for name, param in _dense_params(model, ep_layers):
        if id(param) in folds:
            folded.add(id(param))
        data = materialize_dtensor(lora_folded_data(param, folds))
        if forwarder is not None:
            # An unfolded param aliases the live weight; the client snapshots it, so no clone.
            forwarder.send(name, data)
        # Dropped before the next gather, so two full tensors are never alive at once.
        del data
    _reject_unpushed_folds(model, folds, folded)

    for name, full_tensor in iter_hand_sliced_full(model):
        if forwarder is not None:
            forwarder.send(name, full_tensor)
    if forwarder is not None:
        forwarder.flush()


def gather_and_send_weights(model: torch.nn.Module, sender: WeightSyncSender | None) -> None:
    """COLLECTIVE. A send can fail on the forwarding rank alone, so each runs under a deferred failure: a rank raising
    mid-loop would leave every peer blocked in the next gather until the watchdog fires. The caller closes the
    update."""
    # Read the shards the optimizer steps, not the unsharded copies a forward can leave registered.
    reshard_fsdp2_modules(model)
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    peft = is_peft_model(model)
    # Live EP layers, not has_ep_group: ep_size==1 is still EP-wrapped.
    ep_layers = dict(named_ep_layers(model))
    forwarder = (
        guard.run(partial(_HubForwarder, sender, model, ep_layers, model.prefix if peft else None, guard))
        if sender
        else None
    )
    # Out of place, not PEFT's merge/unmerge, whose bf16 unmerge misses the frozen base by a rounding step.
    folds = lora_fold_targets(model)
    _send_ep_expert_weights(ep_layers, forwarder, guard)
    _send_engine_loaded_buffers(ep_layers, forwarder, getattr(base_transformers_model(model), "config", None))
    _send_dense_weights(model, ep_layers, forwarder, folds)
    guard.reject()


def _finish_and_close(sender: WeightSyncSender) -> None:
    """Send the tail chunk and close the engine's update, aborting it if that raises before the close."""
    try:
        sender.finish_weight_update()
    except BaseException:
        sender.abort_weight_update()
        raise


def _reject_missing_client() -> None:
    raise RuntimeError(
        "The weight-sync push reached its forwarding rank with no engine client: the gather would run and "
        "send nothing, leaving the engine serving the old weights. The client must be formed before a push "
        "(the environmental trainer's _form_weight_sync_group; online GRPO's TRL vllm_client)."
    )


def sync_weights_to_client(
    model: torch.nn.Module, client: WeightSyncSender | None, is_main: bool, is_tp_main: bool
) -> None:
    """COLLECTIVE. Only the forwarding rank (global main, TP rank 0) sends; its missing ``client`` is refused on every
    rank at the close's verdict."""
    forwarding = is_main and is_tp_main
    sender = client if forwarding else None
    flush = DeferredRankFailure("weight-sync flush to the rollout engine")
    if forwarding and client is None:
        flush.run(_reject_missing_client)
    if sender is not None:
        sender.scope_co_load_groups(name for name, _ in model.named_modules())
    try:
        gather_and_send_weights(model, sender)
    except BaseException:
        # The push opened an update mid-gather, which only the forwarding rank can close.
        if sender is not None:
            sender.abort_weight_update()
        raise
    # After every gather, so no peer waits in a collective, but a peer must not roll out against a paused engine.
    if sender is not None:
        flush.run(partial(_finish_and_close, sender))
    flush.reject()


def sync_trainer_weights(trainer, client: WeightSyncSender | None) -> None:
    """COLLECTIVE. ``client`` is the caller's handle on the main process."""
    model = unwrap_framework_wrappers(trainer.model)
    config = trainer.parallelism_config
    is_main = trainer.accelerator.is_main_process
    is_tp_main = (trainer._get_tp_rank() == 0) if config.is_tp_mode else True

    log_memory = env_flag("HALO_WEIGHT_SYNC_MEM_LOG")
    if log_memory:
        log_cuda_memory("weight-sync pre")

    # Every rank waits for the push to land, and a raise must not skip the barrier the peers block in.
    with barrier_on_exit(f"Weight sync at step {trainer.state.global_step}"):
        sync_weights_to_client(model, client, is_main, is_tp_main)

    if log_memory:
        log_cuda_memory("weight-sync post")

    logger.debug(
        f"Synced distributed weights to the rollout engine at step {trainer.state.global_step} "
        f"(ep={config.has_ep_group}, tp={config.is_tp_mode}, "
        f"expert_tp={config.is_expert_tp_mode}, peft={is_peft_model(model)})"
    )
