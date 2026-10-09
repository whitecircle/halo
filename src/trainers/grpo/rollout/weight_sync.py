"""Shared rollout-engine weight-sync internals for the online and environmental GRPO trainers.

Both push the trained policy to the rollout server over the vendored NCCL client: EP expert shards
first, then the FSDP2-DP / TP shards of every dense param. The gathers must run on **every** rank
(``full_tensor()`` and ``gather_expert_state_dict`` are collectives that hang if a rank skips them),
while only the forwarding rank (global-main, TP-rank 0 under TP) sends. PEFT/LoRA is folded into each
base weight out of place as it is sent, and forwarded under base-model param names; the frozen base
is never written. Every family is gathered in its own hub checkpoint layout, which both engines'
loaders read; which families an engine serves at all is read off its client class at construction.

Those sends sit between the gathers, so each runs under a :class:`DeferredRankFailure` and the verdict
is taken at a rank-uniform ``reject``; a forwarding rank raising mid-loop would otherwise leave every
peer blocked in the next layer's gather until the watchdog fires.

Every forwarded key is the hub spelling a gathered checkpoint would carry (:class:`_HubForwarder`),
because the engine loads by hub name and skips an unknown one without error.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import partial
from typing import Any

import torch
from accelerate.utils import is_peft_model
from transformers.core_model_loading import rename_source_key

from src.checkpoint.format import revert_conversions
from src.diagnostics.profiling import log_cuda_memory
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    config_model_types,
    ep_layer_classes_for_config,
    is_expert_weight_attr,
    to_hub_layer_key,
)
from src.distributed.expert_parallel.hub_conversion import gathered_export_conversions, reversed_export_transforms
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.nccl.clients.base import WEIGHT_SYNC_CHUNK_BYTES, payload_bytes
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
    LoraFolds,
    base_transformers_model,
    lora_fold_targets,
    lora_folded_data,
    model_has_quantized_params,
    normalize_peft_param_name,
    unwrap_framework_wrappers,
)
from src.trainers.mixins.ep_introspection import named_ep_layers

logger = logging.getLogger(__name__)

# What :class:`_HubForwarder` may hold back for a many-to-one reverse converter before refusing. The
# held tensors sit on the forwarding rank's GPU next to the gather that produced them, i.e. the
# rank-local allocation the streaming exists to avoid. The shipped claims (Step-3.7's vision q/k/v,
# one EP key) are megabytes; a converter claiming a per-decoder-layer tensor would hold the whole
# stack. Sized at one chunk, the same budget the client streams by.
_HELD_CONVERTER_BUDGET_BYTES = WEIGHT_SYNC_CHUNK_BYTES


def validate_weight_sync_support(model: torch.nn.Module, backend: str) -> None:
    """Construction gate for the trainers that push weights to the ``backend`` rollout engine.

    Eight failure classes are rejected here rather than at the first sync:

    - **Quantized bases (QLoRA)**: ``_send_dense_weights`` forwards raw ``named_parameters`` storage
      under base-weight names, so a bnb-quantized base ships packed non-floating-point tensors
      (``Params4bit`` uint8) the server rejects.
    - **PEFT layers the sync cannot fold** (:func:`~src.models.structure.lora_fold_targets`): the
      delta of an ``nn.MultiheadAttention`` LoRA, trainable tokens or another tuner layer would never
      reach the pushed weights; nor would one on a tensor the dense push does not send (an EP layer's
      expert weights, whose gather folds only the native expert LoRA).
    - **GptOss with sinks removed** (the flash_attention_2 ``reset_sinks`` reset): the removed
      ``sinks`` slots leave ``named_parameters``, so nothing is pushed for them and the rollout engine
      keeps serving the pretrained sinks against a sink-free trainer, with no error at sync time.
    - **GptOss with trainable sinks** (``train_sinks``): an SFT-only policy. The frozen live sinks of
      ``reset_sinks: false`` are on-policy by construction; a sink that moves every step has no
      validated end-to-end sync into either rollout engine.
    - **Families whose layer class declares ``_supports_weight_sync = False``**: no end-to-end sync
      into either engine has been validated; each class's ``_WEIGHT_SYNC_REFUSAL_REASON`` states the
      family's gap. Enforced through live EP
      instances when present, else through the registry off ``config.model_type``, since a
      wrapper-less run carries the same contract.
    - **An EP family with no live EP wrapper** (``ep_size: 1`` with ``use_grouped_gemm: false``): the
      sync ships experts in the layout the family's ``gather_expert_state_dict`` emits, which only the
      EP wrapper provides; the plain HF module would stream a layout the engine cannot land.
    - **Model types the pinned ``backend`` cannot serve** (the client's ``UNSERVABLE_MODEL_TYPES``):
      no model class for the spelling, or a loader reading a layout no gather can emit, so the server
      has no base model for the stream to land in.
    - **Live bias-update balancing state**: the sync payload is parameters only, so an adopted native
      slot (a buffer) or a transient side-buffer is never pushed and trainer routing drifts from the
      generator. The shipped GRPO scripts downgrade ``moe_balancing`` to ``none`` before any bias
      state exists (``build_perf_callbacks``); this backstop catches hand-built drivers.
    """
    if model_has_quantized_params(model):
        raise ValueError(
            "QLoRA (quantized base weights) is not supported with rollout-engine weight sync: the sync forwards "
            "raw parameter storage under base-weight names, and bnb-packed non-floating-point tensors "
            "corrupt the served policy. Use plain LoRA (use_peft without load_in_4bit/load_in_8bit) "
            "or full fine-tuning."
        )
    ep_layers = named_ep_layers(model)
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
    client_cls = resolve_weight_sync_client(backend)
    unservable = sorted(config_model_types(model) & client_cls.UNSERVABLE_MODEL_TYPES.keys())
    if unservable:
        facts = "; ".join(
            f"{model_type}: {client_cls.UNSERVABLE_MODEL_TYPES[model_type]}" for model_type in unservable
        )
        raise ValueError(
            f"rollout_backend={backend!r} cannot serve model_type {unservable}, so the weight sync has "
            f"no served model to land in — {facts}. See {client_cls.__name__}.UNSERVABLE_MODEL_TYPES."
        )
    # isinstance, not an attribute probe: a PEFT wrapper forwards ``__getattr__``, so a probe matches the wrapper.
    families = _sync_contract_classes(model)
    for where, cls in families:
        if not cls._supports_weight_sync:
            raise ValueError(
                f"{cls.__name__} (at {where!r}) does not support weight sync: the sync feeds the "
                f"engine's model.load_weights directly, but "
                f"{cls._WEIGHT_SYNC_REFUSAL_REASON}. Online/environmental GRPO with weight sync "
                f"is unsupported for this model — see {cls.__name__}._supports_weight_sync."
            )
    # Every validated expert layout is a gather's: the wrapper's ``gather_expert_state_dict`` spells the
    # experts as the engine loads them. Without a wrapper the dense walk forwards the stock module
    # tree's fused expert tensors under module names — a layout no engine loader is validated against
    # and one the per-expert loaders skip before their "not found" warning.
    if families and not ep_layers:
        raise ValueError(
            f"{', '.join(sorted({cls.__name__ for _where, cls in families}))} resolves for model_type "
            f"{sorted(config_model_types(model))}, but the model carries no live EP wrapper "
            f"(expert_parallel_size 1 with use_grouped_gemm: false): weight sync ships experts in the layout "
            f"the family's gather_expert_state_dict emits, and without a wrapper it would forward the stock "
            f"module tree's fused expert tensors under module names, which the engine's loader drops with no "
            f"error — attention, norms and routers would sync while the experts keep serving launch weights. "
            f"Set use_grouped_gemm: true (the torchrun default), which installs the EP wrappers at "
            f"expert_parallel_size 1 too."
        )
    # Enabled bias-update state, not the mode string: the shipped scripts downgrade the mode before any
    # state exists, so reaching here with an adopted slot or side-buffer means a hand-built driver
    # enabled balancing itself. These probes do not fire on Zaya's always-present native buffer (never
    # adopted, never transient); both engines list the family as unservable above.
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


def _dense_params(
    model: torch.nn.Module, ep_layers: dict[str, EPMoELayerBase]
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    """The parameters :func:`_send_dense_weights` gathers and sends, in ``named_parameters`` order:
    all but the EP experts (the expert gather's) and the hand-sliced TP shards (sent gathered by the
    drain), which are not DTensors, so shipping this rank's slice under the full name would corrupt
    them."""
    hand_sliced = tp_sharded_non_dtensor_suffixes(model)
    for name, param in model.named_parameters():
        if _is_ep_expert_param(name, ep_layers) or (hand_sliced and name.endswith(hand_sliced)):
            continue
        yield name, param


def _reject_unpushed_folds(model: torch.nn.Module, folds: LoraFolds, pushed: set[int]) -> None:
    """Raise if a LoRA fold target is not among ``pushed``, the ids of the tensors the dense push sends.

    Only that push folds PEFT adapters, so the delta of any other target would never reach the engine.
    Structural, so every rank raises alike.
    """
    unpushed = folds.keys() - pushed
    if unpushed:
        names = sorted(name for name, param in model.named_parameters() if id(param) in unpushed)
        raise ValueError(
            f"PEFT LoRA adapts {names}, which the weight sync does not send through its dense push, the "
            f"only path that folds PEFT adapters: the served weights would miss their delta. EP expert "
            f"weights take native expert LoRA (list the expert projections in lora_target_modules)."
        )


def _hub_param_name(name: str, ep_layers: dict[str, EPMoELayerBase]) -> str:
    """Rewrite one forwarded key from the live module spelling to the family's hub spelling.

    The same per-family :attr:`~EPMoELayerBase._EXPORT_KEY_RENAMES` rewrite
    :func:`~src.distributed.expert_parallel.expert_weights.gather_ep_layer_weights` applies to a
    gathered checkpoint, so the engine receives what it would load from one. Identity outside EP layers and
    for every family whose two spellings agree.
    """
    for layer_name, layer in ep_layers.items():
        prefix = f"{layer_name}."
        if name.startswith(prefix):
            return prefix + to_hub_layer_key(name[len(prefix) :], type(layer))
    return name


class _HubForwarder:
    """Forward gathered tensors to the engine client under the family's hub keys, the spelling the
    engine's loader reads. Built on the forwarding rank only.

    Live-tree names go through three rewrites in order: :attr:`~EPMoELayerBase._EXPORT_KEY_RENAMES`
    inside EP layers (Laguna); the PEFT base-name normalization (adapter-only tensors are dropped,
    their delta already folded); and the revert the gathered save applies
    (:func:`~src.distributed.expert_parallel.hub_conversion.gathered_export_conversions`: what the load
    converted outside the per-expert merges, such as Step-3.7's namespace or a SigLIP tower's
    ``vision_model`` level). Renames are one-to-one and stream; a tensor a reverse ``WeightConverter``
    claims is held until :meth:`flush`, since a many-to-one revert needs all of its sources together
    while the engine loads one tensor at a time.

    Every forward runs under the caller's ``guard``: this object is the only part of the sync that can
    fail on one rank alone (a device OOM staging the snapshot, an HTTP/NCCL error from the client, a
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
        self._held: dict[str, torch.Tensor] = {}
        self._held_bytes = 0
        # The list the gathered save inverts, so the streamed renames and the held converts write
        # what a checkpoint of the same model carries.
        self._model = base_transformers_model(model)
        self._conversions = gathered_export_conversions(self._model)
        self._renamings, self._converters = reversed_export_transforms(self._conversions)

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
        if not self._conversions:
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
        for hub_name, hub_tensor in revert_conversions(self._model, held, self._conversions).items():
            self._client.update_named_param(hub_name, hub_tensor)


def _send_ep_expert_weights(
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    guard: DeferredRankFailure,
) -> None:
    """Gather expert shards across the EP group and forward them. Collective on all ranks.

    Every family is gathered in its own hub checkpoint layout, the one both engines' loaders read:
    fused pairs where the checkpoint stores them fused, per-expert tensors where it stores them per
    expert.

    The retained assembly is the sync's largest rank-local allocation (~28 GB for one 397B layer), so
    it runs under ``guard`` like the sends do: a retained gather finishes its collectives before it
    assembles, and an OOM there must reach the peers as a reason rather than drop this rank out of the
    next layer's gather. Retaining then stops, since there is nothing left to send.
    """
    for layer_name, module in ep_layers.items():
        # Only the forwarding rank needs the assembled layer, and only while it can still send it.
        retain = forwarder is not None and guard.reason is None
        # The dense push folds the PEFT adapters, none of which sit on expert weights, so the native
        # expert LoRA is folded here.
        gather = partial(module.gather_expert_state_dict, "cuda", merge_lora=True, retain=retain)
        # Guarded only where it retains: a non-retaining rank runs the same collectives and keeps
        # nothing, so a raise there is a group-wide failure rather than this rank's own.
        gathered = guard.run(gather) if retain else gather()
        if forwarder is not None and gathered:
            _forward_gathered_layer(forwarder, layer_name, gathered)
        # Released before the next layer's gather: the assembly is the sync's largest rank-local
        # allocation, and a binding kept across the loop would hold two of them.
        del gathered


def _forward_gathered_layer(forwarder: _HubForwarder, layer_name: str, gathered: dict[str, torch.Tensor]) -> None:
    """Forward one assembled EP layer, then flush the hub converters it may have fed.

    Per layer, so a hub split of this layer's experts never outlives the next layer's gather; a
    function of its own so the last tensor's binding dies with the layer.
    """
    for param_name, param_data in gathered.items():
        forwarder.send(f"{layer_name}.{param_name}", param_data)
    forwarder.flush()


def _send_dense_weights(
    model: torch.nn.Module,
    ep_layers: dict[str, EPMoELayerBase],
    forwarder: _HubForwarder | None,
    folds: LoraFolds,
) -> None:
    """Gather non-expert (dense) params — one ``full_tensor()`` over FSDP2 DP and TP — and forward them.

    HF's ``tp_plan`` styles (dense) and the toolkit's attention-only TP (every MoE path) both place
    their shards as DTensors on the TP mesh, so ``materialize_dtensor`` returns each one full;
    only the hand-sliced params need their own gather. Raises, on every rank, if a PEFT fold target
    was not among them (:func:`_reject_unpushed_folds`).
    """
    folded: set[int] = set()
    # Stream gather→send→drop (a dict = full dense copy/rank: OOM at 70B+); param order keeps collectives in step.
    for name, param in _dense_params(model, ep_layers):
        if id(param) in folds:
            folded.add(id(param))
        # Folded one tensor at a time, so the only extra memory is this param's temporaries. The fold
        # and the gather are collectives on every rank.
        data = materialize_dtensor(lora_folded_data(param, folds))
        if forwarder is not None:
            # A plain unfolded param aliases the live weight; the client buffers a snapshot, so it is
            # sent without a clone.
            forwarder.send(name, data)
        # Dropped before the next gather, so two full tensors are never alive at once.
        del data
    _reject_unpushed_folds(model, folds, folded)

    # Every rank drains this (collective); only the forwarding rank sends.
    for name, full_tensor in iter_tp_sharded_non_dtensor_full(model):
        if forwarder is not None:
            forwarder.send(name, full_tensor)
    if forwarder is not None:
        forwarder.flush()


def _flush_and_close(sender: Any) -> None:
    """Send the tail chunk and close the engine's update, aborting the update if that raises.

    ``reset_prefix_cache`` closes and resumes on its own success path, but a raise before the close
    (the completion of the tail's staged copies, or a chunk the engine rejects on arrival) would
    leave the engine quiesced behind an open reload with nothing else on this path to end it.
    """
    try:
        sender.reset_prefix_cache()
    except BaseException:
        sender.abort_weight_update()
        raise


def _refuse_missing_client() -> None:
    raise RuntimeError(
        "The weight-sync push reached its forwarding rank with no engine client: the gather would run and "
        "send nothing, leaving the engine serving the old weights. The client must be formed before a push "
        "(the environmental trainer's _form_weight_sync_group; online GRPO's TRL vllm_client)."
    )


def sync_weights_to_client(model: torch.nn.Module, client: Any | None, is_main: bool, is_tp_main: bool) -> None:
    """Gather the policy and push it to ``client``, then flush the buffered broadcast.

    Runs on **every** rank (the gathers are collective); only the forwarding rank (global-main, TP-rank 0
    under TP) sends, and it must hold a ``client``: without one the push would gather the whole policy
    and send none of it, leaving the engine on its old weights. That refusal is raised on every rank at
    the flush's verdict, after the gather the peers are in, never on the forwarding rank alone.
    """
    # One forwarding-rank predicate for the push and the flush: two spellings that disagree would
    # leave the buffering rank never closing the update it opened.
    forwarding = is_main and is_tp_main
    sender = client if forwarding else None
    flush = DeferredRankFailure("weight-sync flush to the rollout engine")
    if forwarding and client is None:
        flush.run(_refuse_missing_client)
    # The engine fuses a co-load group only where the pushed model declares every member, so the
    # client's groups are scoped to this module tree before the first chunk.
    if sender is not None:
        sender.scope_co_load_groups(name for name, _ in model.named_modules())
    try:
        gather_and_send_weights(model, sender)
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
    if sender is not None:
        flush.run(partial(_flush_and_close, sender))
    flush.reject()


def gather_and_send_weights(model: torch.nn.Module, sender: Any | None) -> None:
    """Gather EP + dense/TP weights from ``model`` and forward to the engine via ``sender``.

    Runs on **every** rank (the gathers are collective); ``sender`` is the engine client on the
    forwarding rank and ``None`` elsewhere. PEFT/LoRA is folded into each base weight out of place and
    forwarded under base-model names. The caller flushes afterwards with ``sender.reset_prefix_cache()``.
    """
    # FSDP2 leaves a forward's transient unsharded params registered while the optimizer steps the
    # shards, so the params a mid-training sync finds registered predate the last update: every
    # per-step push would ship a policy one optimizer step behind, folded from stale adapters. Same
    # call the optimizer build and every checkpoint writer make, for the same reason. Rank-uniform: a
    # rank that skipped it would also skip the DTensor gathers its peers enter.
    reshard_fsdp2_modules(model)
    # The forwarding rank's sends sit between the gathers below, and every one of them can fail on
    # that rank alone. Raising there would drop it out of the gather order its peers follow, so record
    # and carry on and let the reject decide.
    guard = DeferredRankFailure("weight-sync push to the rollout engine")
    peft = is_peft_model(model)
    # Live EP layers, not is_ep_mode: ep_size==1 is still EP-wrapped and the dense path ships an
    # unloadable layout.
    ep_layers = named_ep_layers(model)
    forwarder = (
        guard.run(partial(_HubForwarder, sender, model, ep_layers, model.prefix if peft else None, guard))
        if sender
        else None
    )
    # Out of place, not PEFT's merge/unmerge: in bf16 the unmerge misses the frozen base by a rounding
    # step, and the sync repeats every few steps for the whole run. Resolved before any tensor streams,
    # so a layer it cannot fold raises first; a target the push would not send is refused at
    # construction and checked again as the push ends.
    folds = lora_fold_targets(model)
    _send_ep_expert_weights(ep_layers, forwarder, guard)
    _send_dense_weights(model, ep_layers, forwarder, folds)
    # Collective on every rank. Raises on all of them with the forwarding rank's cause; the sync
    # writes none of the trainer's own weights, so a failed one leaves them untouched.
    guard.reject()


def sync_trainer_weights(trainer, client: Any | None) -> None:
    """Gather a distributed trainer's policy and push it to ``client``.

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
        sync_weights_to_client(model, client, is_main, is_tp_main)

    if log_memory:
        log_cuda_memory("weight-sync post")

    logger.debug(
        f"Synced distributed weights to the rollout engine at step {trainer.state.global_step} "
        f"(ep={config.is_ep_mode}, tp={config.is_tp_mode}, "
        f"expert_tp={config.is_expert_tp_mode}, peft={is_peft_model(model)})"
    )
