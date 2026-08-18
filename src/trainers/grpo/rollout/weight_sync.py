"""Shared rollout-engine weight-sync internals for the online and environmental GRPO trainers.

Both push the trained policy to the rollout server over the vendored NCCL client: EP expert shards
first, then the FSDP2-DP / TP shards of every dense param. The gathers must run on **every** rank
(``full_tensor()`` and ``gather_expert_state_dict`` are collectives that hang if a rank skips them),
while only the forwarding rank (global-main, TP-rank 0 under TP) sends. PEFT/LoRA is folded into the
base and forwarded under base-model param names. The engine is a parameter: the expert layout and the
parallelism gate are read off the resolved client class (vLLM or SGLang).

Those sends sit between the gathers, so each runs under a :class:`DeferredRankFailure` and the verdict
is taken at a rank-uniform ``reject``; a forwarding rank raising mid-loop would otherwise leave every
peer blocked in the next layer's gather until the watchdog fires.

Every forwarded key is the hub spelling a gathered checkpoint would carry (:class:`_HubForwarder`),
because the engine loads by hub name and skips an unknown one without error.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import torch
from accelerate.utils import is_peft_model
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    rename_source_key,
    revert_weight_conversion,
)

from src.checkpoint.format import revert_conversions_for
from src.diagnostics.profiling import log_cuda_memory
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    config_model_types,
    ep_layer_classes_for_config,
    is_expert_weight_attr,
    to_hub_layer_key,
)
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.nccl.clients.base import WEIGHT_SYNC_CHUNK_BYTES, BaseWeightSyncClient, payload_bytes
from src.distributed.nccl.registry import resolve_weight_sync_client
from src.distributed.runtime import DeferredRankFailure, barrier_on_exit, materialize_dtensor
from src.distributed.tensor_parallel.state_dict import (
    iter_tp_sharded_non_dtensor_full,
    tp_sharded_non_dtensor_suffixes,
)
from src.env import env_flag
from src.models.moe_balancing import (
    NATIVE_BALANCING_BIAS_ADOPTED_ATTR,
    is_transient_balancing_router,
    iter_balancing_routers,
)
from src.models.patches.gpt_oss_sinks import SinksPolicy, neutralized_gpt_oss_sinks, stamped_sinks_policy
from src.models.structure import (
    merged_adapters,
    model_has_quantized_params,
    normalize_peft_param_name,
    unwrap_framework_wrappers,
    unwrap_model,
)
from src.trainers.mixins.ep_introspection import named_ep_layers

logger = logging.getLogger(__name__)

# What :class:`_HubForwarder` may hold back for a many-to-one reverse converter before refusing. The
# held tensors sit on the forwarding rank's GPU next to the gather that produced them, i.e. the
# rank-local allocation the streaming exists to avoid. The shipped claims (Step-3.7's vision q/k/v,
# one EP key) are megabytes; a converter claiming a per-decoder-layer tensor would hold the whole
# stack. Sized at one chunk, the same budget the client streams by.
_HELD_CONVERTER_BUDGET_BYTES = WEIGHT_SYNC_CHUNK_BYTES


def validate_weight_sync_support(model: torch.nn.Module) -> None:
    """Construction gate for the trainers that push weights to vLLM.

    Six failure classes are rejected here rather than at the first sync:

    - **Quantized bases (QLoRA)**: ``_send_dense_weights`` forwards raw ``named_parameters`` storage
      under base-weight names, so a bnb-quantized base ships packed non-floating-point tensors
      (``Params4bit`` uint8) the server rejects, and the per-sync LoRA merge/unmerge round-trip
      through 4-bit weights is lossy.
    - **GptOss with sinks removed** (the flash_attention_2 ``reset_sinks`` reset): the removed
      ``sinks`` slots leave ``named_parameters``, so nothing is pushed for them and the rollout engine
      keeps serving the pretrained sinks against a sink-free trainer, with no error at sync time.
    - **GptOss with trainable sinks** (``train_sinks``): an SFT-only policy. The frozen live sinks of
      ``reset_sinks: false`` are on-policy by construction; a sink that moves every step has no
      validated end-to-end sync into either rollout engine.
    - **Families whose layer class declares ``_supports_weight_sync = False``** (DeepSeek-V4,
      Inkling, Zaya, Cohere2 MoE, GLM-5 Next): the names this sync forwards go straight into vLLM's
      ``model.load_weights`` and cannot land; each class's ``_WEIGHT_SYNC_REFUSAL_REASON`` states the
      family's gap. Enforced through live EP instances when present, else through the registry off
      ``config.model_type``, since a wrapper-less run carries the same contract.
    - **Model types no pinned engine can serve** (``_WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES``): these
      spellings have no model class in the pinned engines, so the server cannot load the base model.
    - **Live bias-update balancing state**: the sync payload is parameters only, so an adopted native
      slot (a buffer) or a transient side-buffer is never pushed and trainer routing drifts from the
      generator. The shipped GRPO scripts downgrade ``moe_balancing`` to ``none`` before any bias
      state exists (``build_perf_callbacks``); this backstop catches hand-built drivers.
    """
    if model_has_quantized_params(model):
        raise ValueError(
            "QLoRA (quantized base weights) is not supported with vLLM weight sync: the sync forwards "
            "raw parameter storage under base-weight names, and bnb-packed non-floating-point tensors "
            "corrupt the served policy. Use plain LoRA (use_peft without load_in_4bit/load_in_8bit) "
            "or full fine-tuning."
        )
    # Refused rather than repaired (the second failure class above): every checkpoint writer re-emits
    # neutralized sinks, but this sync forwards named_parameters and has no such seam.
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
    model_types = config_model_types(model)
    # isinstance, not an attribute probe: a PEFT wrapper forwards ``__getattr__``, so a probe matches the wrapper.
    for where, cls in _sync_contract_classes(model):
        if not cls._supports_weight_sync:
            raise ValueError(
                f"{cls.__name__} (at {where!r}) does not support vLLM weight sync: the sync forwards "
                f"trainer parameter names straight into vLLM's model.load_weights, but "
                f"{cls._WEIGHT_SYNC_REFUSAL_REASON}. Online/environmental GRPO with vLLM weight sync "
                f"is unsupported for this model — see {cls.__name__}._supports_weight_sync."
            )
        unservable = model_types & set(cls._WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES)
        if unservable:
            raise ValueError(
                f"{cls.__name__} (at {where!r}) refuses weight sync for model_type "
                f"{sorted(unservable)}: no pinned rollout engine registers a model class for this "
                f"spelling, so the server cannot load the base model — see "
                f"{cls.__name__}._WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES for the engine facts."
            )
    # Enabled bias-update state, not the mode string: the shipped scripts downgrade the mode before any
    # state exists, so reaching here with an adopted slot or side-buffer means a hand-built driver
    # enabled balancing itself. These probes do not fire on Zaya's always-present native buffer (never
    # adopted, never transient), which its _supports_weight_sync=False refuses above.
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
            f"external rollout engine is configured: the sync payload carries parameters only, so the "
            f"balancing bias (an adopted buffer or a transient side-buffer) is never pushed and trainer "
            f"routing drifts from the generator producing the trajectories. Run weight-sync RL with "
            f"moe_balancing: none — the shipped GRPO scripts downgrade it automatically."
        )


def _sync_contract_classes(model: torch.nn.Module) -> list[tuple[str, type[EPMoELayerBase]]]:
    """(where, class) pairs carrying the family's weight-sync contract.

    Live EP instances when present, else the registry's classes for the model's config
    (:func:`ep_layer_classes_for_config`): ``ep_size == 1`` with ``use_grouped_gemm: false`` leaves the
    stock HF module tree, so a gate that only walked live modules would admit the families it exists
    to refuse.
    """
    instances = named_ep_layers(model)
    if instances:
        return [(name, type(module)) for name, module in instances.items()]
    config = getattr(unwrap_framework_wrappers(model), "config", None)
    return [(f"model_type registry for {type(model).__name__}", cls) for cls in ep_layer_classes_for_config(config)]


def _is_ep_expert_param(name: str, ep_layers: dict[str, EPMoELayerBase]) -> bool:
    """True if ``name`` is an expert weight of one of the EP layers (handled by the EP gather)."""
    return any(name.startswith(root + ".") and is_expert_weight_attr(name[len(root) + 1 :]) for root in ep_layers)


def _hub_param_name(name: str, ep_layers: dict[str, EPMoELayerBase]) -> str:
    """Rewrite one forwarded key from the live module spelling to the family's hub spelling.

    The same per-family :attr:`~EPMoELayerBase._EXPORT_KEY_RENAMES` rewrite
    :func:`~src.distributed.expert_parallel.expert_weights.gather_ep_layer_weights` applies to a
    gathered checkpoint, so vLLM receives what it would load from one. Identity outside EP layers and
    for every family whose two spellings agree.
    """
    for layer_name, layer in ep_layers.items():
        prefix = f"{layer_name}."
        if name.startswith(prefix):
            return prefix + to_hub_layer_key(name[len(prefix) :], type(layer))
    return name


def _hub_model(model: torch.nn.Module) -> torch.nn.Module:
    """The plain transformers model under every framework, toolkit and PEFT wrapper: the tree whose
    ``base_model_prefix`` and sub-model paths transformers' conversion mapping is scoped by."""
    base = unwrap_model(model)
    return unwrap_model(base.get_base_model()) if is_peft_model(base) else base


class _HubForwarder:
    """Forward gathered tensors to the engine client under the family's hub keys, the spelling the
    engine's loader reads. Built on the forwarding rank only.

    Live-tree names go through three rewrites in order: :attr:`~EPMoELayerBase._EXPORT_KEY_RENAMES`
    inside EP layers (Laguna); the PEFT base-name normalization (adapter-only tensors are dropped,
    their delta already merged); and, for a family declaring ``_EXPORTS_HUB_NAMESPACE`` (Step-3.7),
    transformers' save-side conversion revert. Renames are one-to-one and stream; a tensor a reverse
    ``WeightConverter`` claims is held until :meth:`flush`, since a many-to-one revert needs all of
    its sources together while the engine loads one tensor at a time.

    Every forward runs under the caller's ``guard``: this object is the only part of the sync that can
    fail on one rank alone (a host OOM pinning the snapshot, an HTTP/NCCL error from the client, a
    tensor the engine's key space rejects), and it does so between two group-wide gathers.
    """

    def __init__(
        self,
        client: Any,
        model: torch.nn.Module,
        ep_layers: dict[str, EPMoELayerBase],
        peft_prefix: str | None,
        guard: DeferredRankFailure,
    ):
        self._client = client
        self._guard = guard
        self._ep_layers = ep_layers
        self._peft_prefix = peft_prefix
        self._model: torch.nn.Module | None = None
        self._renamings: list[WeightRenaming] = []
        self._converters: list[WeightConverter] = []
        self._held: dict[str, torch.Tensor] = {}
        self._held_bytes = 0
        if not any(cls._EXPORTS_HUB_NAMESPACE for _where, cls in _sync_contract_classes(model)):
            return
        self._model = _hub_model(model)
        # The resolution every save-side revert uses, so the streamed renames and the held converts
        # come from the same reversed list the gathered save inverts.
        conversions = revert_conversions_for(self._model)
        for transform in (c.reverse_transform() for c in conversions[::-1]):
            (self._renamings if isinstance(transform, WeightRenaming) else self._converters).append(transform)

    def send(self, name: str, tensor: torch.Tensor) -> None:
        """Forward one live-tree tensor, deferring a failure to the sync's rank-uniform reject."""
        self._guard.run(partial(self._forward, name, tensor))

    def flush(self) -> None:
        """Forward what the reverse converters make of the held tensors, under the same guard."""
        self._guard.run(self._revert_held)

    def _forward(self, name: str, tensor: torch.Tensor) -> None:
        """Forward one tensor, or hold it for :meth:`flush` when a reverse converter claims it."""
        name = _hub_param_name(name, self._ep_layers)
        if self._peft_prefix is not None:
            name = normalize_peft_param_name(name, self._peft_prefix)
            if name is None:
                return
        if self._model is None:
            self._client.update_named_param(name, tensor)
            return
        renamed, claimed_by = rename_source_key(name, self._renamings, self._converters, reverse=True)
        if claimed_by is None:
            self._client.update_named_param(renamed, tensor)
            return
        self._held[name] = tensor
        self._held_bytes += payload_bytes(tensor)
        if self._held_bytes > _HELD_CONVERTER_BUDGET_BYTES:
            raise RuntimeError(
                f"weight sync is holding {self._held_bytes / 2**30:.2f} GiB on the forwarding rank: "
                f"{len(self._held)} tensors up to {name!r} are claimed by a reverse WeightConverter of "
                f"{type(self._model).__name__} and cannot be sent until their sources are complete. The "
                f"held set lives on that one rank's GPU beside the gather, so it is bounded at "
                f"{_HELD_CONVERTER_BUDGET_BYTES / 2**30:.2f} GiB — a converter claiming per-decoder-layer "
                f"tensors needs a flush per layer, not one per gather phase."
            )

    def _revert_held(self) -> None:
        """Run the reverse converters over the held tensors and forward what they produce."""
        if not self._held:
            return
        held, self._held, self._held_bytes = self._held, {}, 0
        for hub_name, hub_tensor in revert_weight_conversion(self._model, held).items():
            self._client.update_named_param(hub_name, hub_tensor)


def wants_fused_experts(expert_layout: str) -> bool:
    """Whether ``expert_layout`` is the checkpoint-fused one the engine loads.

    Both declared spellings are recognized and anything else raises: an inequality test would route an
    unknown third layout into one branch, and the gather it picks is rejected on arrival, after the
    engine is paused and partly written.
    """
    if expert_layout == BaseWeightSyncClient.FUSED_EXPERT_LAYOUT:
        return True
    if expert_layout == BaseWeightSyncClient.UNFUSED_EXPERT_LAYOUT:
        return False
    raise ValueError(
        f"unknown expert layout {expert_layout!r}: a weight-sync client declares "
        f"{BaseWeightSyncClient.UNFUSED_EXPERT_LAYOUT!r} (per-expert tensors) or "
        f"{BaseWeightSyncClient.FUSED_EXPERT_LAYOUT!r} (the checkpoint-fused pair), and the gather "
        f"must produce the one its receiver loads."
    )


def validate_backend_parallelism(backend: str, parallelism_config, model: torch.nn.Module) -> None:
    """Reject an engine/parallelism/model triple whose weight sync cannot work, at construction.

    Caught here rather than at the first sync: by then the run has loaded the model, collected a
    round of rollouts and paused the engine, and the failure lands mid-broadcast with the served
    weights already partly overwritten.
    """
    client_cls = resolve_weight_sync_client(backend)
    if parallelism_config.is_ep_mode and not client_cls.SUPPORTS_EXPERT_PARALLEL:
        # is_ep_mode is ep_size * expert_tp_size > 1, so pure ETP trips this too; name both sizes.
        raise ValueError(
            f"rollout_backend={backend!r} cannot be combined with expert distribution "
            f"(expert_parallel_size={parallelism_config.ep_size}, "
            f"expert_tensor_parallel_size={parallelism_config.expert_tp_size} — the sync is blocked by "
            f"their product, {parallelism_config.ep_group_size}, exceeding 1). The cross-container "
            f"weight-sync communicator needs NCCL's CUDA-IPC transports disabled, DeepEP needs them "
            f"enabled for symmetric memory, and both are process-global — NCCL caches them on first "
            f"read, so they cannot be scoped to one communicator. Use rollout_backend='vllm' for "
            f"expert-distributed runs (validated), or set both sizes to 1. "
            f"See agent-docs/infrastructure/rollout-servers.md."
        )
    # An engine that takes the per-expert layout has no fused contract to check.
    if not wants_fused_experts(client_cls.EXPERT_LAYOUT):
        return
    for where, cls in _sync_contract_classes(model):
        if not cls.implements_fused_expert_layout():
            raise ValueError(
                f"rollout_backend={backend!r} loads experts in the checkpoint-fused layout, which "
                f"{cls.__name__} (at {where!r}) does not implement — its "
                f"gather_fused_expert_state_dict is the base default. Sending the per-expert layout "
                f"instead is silently dropped or rejected on arrival, after the engine is paused and "
                f"partly written. Use rollout_backend='vllm', which takes the per-expert layout this "
                f"family already gathers."
            )


def expert_layout_for(trainer) -> str:
    """The receiving engine's expert layout, resolved identically on every rank.

    Read off the trainer's declared ``_rollout_backend`` rather than the client object: only the
    forwarding rank holds a client, and the gathers this selects are collective, so reading it from
    the instance would have the ranks build different layouts and hang. Online GRPO leaves the
    attribute ``None`` and keeps the base default, which is correct since it is vLLM-only.
    """
    backend = getattr(trainer, "_rollout_backend", None)
    return resolve_weight_sync_client(backend).EXPERT_LAYOUT if backend else BaseWeightSyncClient.EXPERT_LAYOUT


def _send_ep_expert_weights(
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    guard: DeferredRankFailure,
    expert_layout: str = BaseWeightSyncClient.EXPERT_LAYOUT,
) -> None:
    """Gather expert shards across the EP group and forward them. Collective on all ranks.

    ``expert_layout`` is the receiving engine's, not a property of the checkpoint. The gathers below
    are collective and must be decided identically on every rank, hence a threaded-in value rather
    than one read off ``forwarder``, which is ``None`` everywhere but the forwarding rank.

    The retained assembly is the sync's largest rank-local allocation (~28 GB for one 397B layer), so
    it runs under ``guard`` like the sends do: a retained gather finishes its collectives before it
    assembles, and an OOM there must reach the peers as a reason rather than drop this rank out of the
    next layer's gather. Retaining then stops, since there is nothing left to send.
    """
    # Resolved once, before the first gather: an unrecognized layout must refuse the sync outright,
    # not partway through the layers with the engine already paused.
    fused = wants_fused_experts(expert_layout)
    for layer_name, module in ep_layers.items():
        # Only the forwarding rank needs the assembled layer, and only while it can still send it.
        retain = forwarder is not None and guard.reason is None
        # PEFT's merge_adapter covers only the attention adapters, so the native expert-LoRA is folded here.
        # A family declaring no fused layout raises inside the base gather: an empty result is what
        # a non-retaining rank returns, so it cannot also carry the refusal.
        gather = partial(
            module.gather_fused_expert_state_dict if fused else module.gather_expert_state_dict,
            "cuda",
            merge_lora=True,
            retain=retain,
        )
        # Guarded only where it retains: a non-retaining rank runs the same collectives and keeps
        # nothing, so a raise there is a group-wide failure rather than this rank's own.
        gathered = guard.run(gather) if retain else gather()
        if forwarder is None or not gathered:
            continue
        for param_name, param_data in gathered.items():
            forwarder.send(f"{layer_name}.{param_name}", param_data)
        # Per layer, so a hub split of this layer's experts never outlives the next layer's gather.
        forwarder.flush()


def _send_dense_weights(
    model: torch.nn.Module,
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    peft: bool = False,
) -> None:
    """Gather non-expert (dense) params — one ``full_tensor()`` over FSDP2 DP and TP — and forward them.

    HF's ``tp_plan`` styles (dense) and the toolkit's attention-only TP (every MoE path) both place
    their shards as DTensors on the TP mesh, so ``materialize_dtensor`` returns each one full;
    only the hand-sliced params need their own gather.
    """
    # Hand-sliced TP shards (GptOss sinks) are not DTensors: shipping this rank's slice under the
    # full-tensor name would corrupt the weights, so skip here; the drain loop below sends them gathered.
    hand_sliced = tp_sharded_non_dtensor_suffixes(model)

    # Stream gather→send→drop (a dict = full dense copy/rank: OOM at 70B+); param order keeps collectives in step.
    for name, param in model.named_parameters():
        if _is_ep_expert_param(name, ep_layers):
            continue
        if hand_sliced and name.endswith(hand_sliced):
            continue
        raw = param.data
        data = materialize_dtensor(raw)
        # Sends buffer by reference and flush after PEFT unmerge, so a plain param must be cloned first.
        if peft and data is raw:
            data = data.clone()
        if forwarder is not None:
            forwarder.send(name, data)

    # Every rank drains this (collective); only the forwarding rank sends.
    for name, full_tensor in iter_tp_sharded_non_dtensor_full(model):
        if forwarder is not None:
            forwarder.send(name, full_tensor)
    if forwarder is not None:
        forwarder.flush()


def _flush_and_close(sender: Any) -> None:
    """Send the tail chunk and close the engine's update, aborting the update if that raises.

    ``reset_prefix_cache`` closes and resumes on its own success path, but a raise before the close
    (the D2H completion of the tail's host snapshots, or a chunk the engine rejects on arrival) would
    leave the engine quiesced behind an open reload with nothing else on this path to end it.
    """
    try:
        sender.reset_prefix_cache()
    except BaseException:
        sender.abort_weight_update()
        raise


def sync_weights_to_client(
    model: torch.nn.Module,
    client: Any | None,
    is_main: bool,
    is_tp_main: bool,
    expert_layout: str = BaseWeightSyncClient.EXPERT_LAYOUT,
) -> bool:
    """Gather the policy and push it to ``client``, then flush the buffered broadcast. Returns is-PEFT.

    Runs on **every** rank (the gathers are collective); only the forwarding rank (global-main, TP-rank 0
    under TP) sends.
    """
    # One forwarding-rank predicate for the push and the flush: two spellings that disagree would
    # leave the buffering rank never closing the update it opened.
    sender = client if (is_main and is_tp_main) else None
    try:
        peft = gather_and_send_weights(model, sender, expert_layout)
    except BaseException:
        # The push streams chunks into an update it opened mid-gather, so a raise past this point
        # would leave the engine quiesced behind an open reload, refusing every later sync and
        # queueing every rollout. Only the forwarding rank can close it.
        if sender is not None:
            sender.abort_weight_update()
        raise
    # The buffered broadcast lands after every gather, so a failure here blocks no peer inside a
    # collective, but a peer that continues past it drives its next rollout round against an engine
    # left paused mid-update. Same uniform verdict as the push, on the flush's own rank-local work.
    flush = DeferredRankFailure("weight-sync flush to the rollout engine")
    if sender is not None:
        flush.run(partial(_flush_and_close, sender))
    flush.reject()
    return peft


def gather_and_send_weights(
    model: torch.nn.Module,
    sender: Any | None,
    expert_layout: str = BaseWeightSyncClient.EXPERT_LAYOUT,
) -> bool:
    """Gather EP + dense/TP weights from ``model`` and forward to the engine via ``sender``.

    Runs on **every** rank (the gathers are collective); ``sender`` is the engine client on the
    forwarding rank and ``None`` elsewhere, so ``expert_layout`` is passed in rather than read off it.
    PEFT/LoRA is merged and forwarded under base-model names. The caller flushes afterwards with
    ``sender.reset_prefix_cache()``. Returns whether ``model`` is PEFT.
    """
    # FSDP2 leaves a forward's transient unsharded params registered while the optimizer steps the
    # shards, so the params a mid-training sync finds registered predate the last update: every
    # per-step push would ship a policy one optimizer step behind, and the merge below would fold the
    # adapter into a copy the next unshard discards. Same call the optimizer build and every
    # checkpoint writer make, for the same reason. Rank-uniform: a rank that skipped it would also
    # skip the DTensor gathers its peers enter.
    reshard_fsdp2_modules(model)
    # The forwarding rank's sends sit between the gathers below, and every one of them can fail on
    # that rank alone. Raising there would drop it out of the gather order its peers follow, so record
    # and carry on and let the reject decide.
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    with merged_adapters(model) as peft:
        # Live EP layers, not is_ep_mode: ep_size==1 is still EP-wrapped and the dense path ships an
        # unloadable layout.
        ep_layers = named_ep_layers(model)
        forwarder = (
            guard.run(partial(_HubForwarder, sender, model, ep_layers, model.prefix if peft else None, guard))
            if sender
            else None
        )
        _send_ep_expert_weights(ep_layers, forwarder, guard, expert_layout)
        _send_dense_weights(model, ep_layers, forwarder, peft)
    # Collective on every rank. Raises on all of them with the forwarding rank's cause, after the
    # adapters are unmerged, so a failed sync leaves the trainer's own weights untouched.
    guard.reject()
    return peft


def sync_trainer_weights(trainer, client: Any | None) -> bool:
    """Gather a distributed trainer's policy and push it to ``client``. Returns is-PEFT.

    Every rank must call this (all ranks join the gathers; only global-main forwards). ``client`` is
    the caller's own handle, the only difference between the online and env sync paths.
    """
    model = unwrap_framework_wrappers(trainer.model)
    config = trainer.parallelism_config
    is_main = trainer.accelerator.is_main_process
    is_tp_main = (trainer._get_tp_rank() == 0) if config.is_tp_mode else True

    # Read once, so the pre/post bracket cannot end up with only one half.
    log_memory = env_flag("HALO_WEIGHT_SYNC_MEM_LOG")
    if log_memory:
        log_cuda_memory("weight-sync pre")

    # Hold every rank until the forwarding rank's push lands: peers would otherwise drive rollouts
    # against a mid-update engine. Fenced because the push is main-rank-only, so a raise must not skip
    # the barrier its peers block in.
    with barrier_on_exit():
        peft = sync_weights_to_client(model, client, is_main, is_tp_main, expert_layout_for(trainer))

    if log_memory:
        log_cuda_memory("weight-sync post")

    logger.debug(
        f"Synced distributed weights to the rollout engine at step {trainer.state.global_step} "
        f"(ep={config.is_ep_mode}, tp={config.is_tp_mode}, "
        f"expert_tp={config.is_expert_tp_mode}, peft={peft})"
    )
    return peft
