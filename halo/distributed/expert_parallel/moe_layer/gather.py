"""The collective half of an EP export: expert tensors gathered across the expert-TP and dispatch-EP groups, to every
rank or to one destination. Each rank calls every gather; ``retain=False`` joins every collective without assembling
the result."""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Callable, Mapping
from typing import ClassVar

import torch

from halo.distributed.expert_parallel.config import EPConfig
from halo.models.moe.expert_layout import ExpertLayout, to_hub_expert_tensors
from halo.models.moe.expert_lora_spec import expert_lora_delta
from halo.models.moe.family_specs import MoEFamilySpec
from halo.models.param_roles import persistent_buffers
from halo.runtime.collectives import all_gather_cat, gather_cat_to_rank, group_holds_rank
from halo.runtime.dtensor import full_cpu_tensor, materialize_dtensor
from halo.runtime.ranks import get_global_rank


class ExpertGather:
    """An expert export whose collectives are issued; :meth:`finish` hands it over in the hub layout on the caller's
    stream, ordered after this gather alone."""

    def __init__(
        self,
        layout: ExpertLayout,
        gathered: dict[str, torch.Tensor],
        *,
        device: torch.device | str,
        ready: torch.cuda.Event | None,
    ):
        self._layout = layout
        self._gathered = gathered
        self._device = device
        self._ready = ready

    def finish(self) -> dict:
        """The hub-layout tensors on ``device``; ``{}`` where nothing was retained."""
        gathered, self._gathered = self._gathered, {}
        if self._ready is not None:
            stream = torch.cuda.current_stream()
            stream.wait_event(self._ready)
            for tensor in gathered.values():
                tensor.record_stream(stream)
        return to_hub_expert_tensors(self._layout, gathered, device=self._device) if gathered else {}


class EPGatherMixin:
    """Expert export, mixed into :class:`EPMoELayerBase`, whose construction builds the state annotated here."""

    FAMILY: ClassVar[MoEFamilySpec]
    ep_config: EPConfig
    expert_tp_size: int
    expert_tp_group: object
    expert_lora_scaling: float
    has_expert_lora: bool
    _expert_lora_attrs: frozenset[str]
    _base_expert_named_params: Callable[[], list]
    expert_named_params: Callable[[], list]

    # The axis expert TP shards each runtime attribute along; ``down_proj_bias`` stays whole on expert-TP rank 0.
    EXPERT_TP_SHARD_DIMS: ClassVar[Mapping[str, int]] = {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 1,
        "gate_proj_bias": 1,
        "up_proj_bias": 1,
    }

    def gather_expert_tensors(
        self, local: Mapping[str, torch.Tensor], *, retain: bool = True, dst: int | None = None
    ) -> dict:
        """COLLECTIVE over the expert-TP and dispatch-EP groups. ``local`` (this rank's tensors by runtime attribute,
        matmul convention) with every expert and expert-TP shard, in ``local``'s order; ``{}`` without ``retain``.
        With ``dst`` (a global rank) only that rank receives, and a rank outside the EP group holding it joins
        nothing; within it the expert-TP shards still gather across their group.

        Only a retaining rank holds earlier attributes through a later receive, so it is the rank that runs out. It
        drops them and joins the collective anyway, raising past the last one, so its peers never pair with the wrong
        tensor. With nothing to drop the failure is the peers' own footprint and raises at once, which they wait out
        until the NCCL watchdog fires, as for any rank that fails to allocate its receive."""
        if dst is not None and not group_holds_rank(self.ep_config.process_group, dst):
            return {}
        gathered = {}
        failure: torch.cuda.OutOfMemoryError | None = None

        def in_step(collective: Callable[[bool], torch.Tensor | None]) -> torch.Tensor | None:
            nonlocal failure
            try:
                return collective(retain and failure is None)
            except torch.cuda.OutOfMemoryError as oom:
                if not gathered:
                    raise
                failure = failure or oom
                gathered.clear()
                return collective(False)

        group, ep_size = self.ep_config.dispatch_ep_group, self.ep_config.ep_size
        for attr, tensor in local.items():
            if attr in self.EXPERT_TP_SHARD_DIMS:
                dim = self.EXPERT_TP_SHARD_DIMS[attr]
                tensor = in_step(
                    lambda _, t=tensor, d=dim: all_gather_cat(t, d, self.expert_tp_group, self.expert_tp_size)
                )
            if dst is None:
                tensor = in_step(lambda _, t=tensor: all_gather_cat(t, 0, group, ep_size))
            elif ep_size > 1:
                if not group_holds_rank(group, dst):
                    continue
                tensor = in_step(
                    lambda receive, t=tensor: gather_cat_to_rank(t, group, ep_size, dst=dst, receive=receive)
                )
            elif get_global_rank() != dst:
                tensor = None
            if retain and failure is None and tensor is not None:
                gathered[attr] = tensor
        if failure is not None:
            raise failure
        return gathered

    def start_gather_expert_state_dict(
        self,
        *,
        device: torch.device | str = "cpu",
        merge_lora: bool = False,
        retain: bool = True,
        dst: int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> ExpertGather:
        """COLLECTIVE (see :meth:`gather_expert_tensors`). Issues this layer's gather and returns before it completes
        when ``stream`` is given, so the caller can forward the previous layer meanwhile; the collective order on
        every rank is the order of the calls. ``merge_lora`` folds each grouped-LoRA delta into its weight, so the RL
        weight sync ships the trained experts rather than the frozen base."""
        # Gated on built adapters, not the flag: the weight sync asks for the fold unconditionally.
        if merge_lora and self.has_expert_lora and self.expert_tp_size > 1:
            raise NotImplementedError(
                f"{type(self).__name__}: merge_lora is not implemented for the expert-TP gather "
                f"(expert_tensor_parallel_size={self.expert_tp_size}) — the export would carry "
                f"the FROZEN base experts. Gather with expert_tensor_parallel_size=1, or "
                f"implement the fold for the sharded layout."
            )
        layout = self.FAMILY.expert_layout
        if dst is not None and not group_holds_rank(self.ep_config.process_group, dst):
            return ExpertGather(layout, {}, device=device, ready=None)
        caller = torch.cuda.current_stream() if stream is not None else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            if stream is not None:
                stream.wait_stream(caller)
            local = {}
            for attr, param in self._base_expert_named_params():
                local[attr] = materialize_dtensor(param.data)
                if merge_lora and attr in self._expert_lora_attrs:
                    lora_a = materialize_dtensor(getattr(self, f"{attr}_lora_A").data)
                    lora_b = materialize_dtensor(getattr(self, f"{attr}_lora_B").data)
                    delta = expert_lora_delta(lora_a, lora_b, self.expert_lora_scaling)
                    local[attr] = (local[attr].float() + delta).to(local[attr].dtype)
            gathered = self.gather_expert_tensors(local, retain=retain, dst=dst)
            del local
            ready = None
            if stream is not None:
                ready = torch.cuda.Event()
                ready.record(stream)
        return ExpertGather(layout, gathered, device=device, ready=ready)

    def gather_expert_state_dict(
        self,
        *,
        device: torch.device | str = "cpu",
        merge_lora: bool = False,
        retain: bool = True,
        dst: int | None = None,
    ) -> dict:
        """COLLECTIVE (see :meth:`gather_expert_tensors`). This layer's experts in the family's hub layout on
        ``device``, on every rank that retains or only on ``dst``."""
        return self.start_gather_expert_state_dict(
            device=device, merge_lora=merge_lora, retain=retain, dst=dst
        ).finish()

    def is_expert_param(self, name: str) -> bool:
        """Whether ``name``, relative to this layer, is one of its expert tensors, native adapters included. Save and
        weight sync split by it."""
        return name.split(".")[0] in {attr for attr, _param in self.expert_named_params()}

    def gather_layer_state_dict(self, layer_name: str, *, merge_lora: bool = False, retain: bool = True) -> dict:
        """COLLECTIVE (see :meth:`gather_expert_tensors`, plus each FSDP2-sharded replicated tensor): the whole layer
        for a checkpoint in the hub spelling, gathered experts then persistent replicated params and buffers. No LoRA
        layer sits inside an EP layer, so replicated tensors are written as they are. Persistent only, since a
        non-persistent buffer is absent from the sharded save and would break merged-from-sharded == gathered."""
        experts = self.gather_expert_state_dict(merge_lora=merge_lora, retain=retain)
        gathered = {f"{layer_name}.{self.FAMILY.to_hub_key(key)}": tensor for key, tensor in experts.items()}
        del experts
        for name, tensor in itertools.chain(self.named_parameters(), persistent_buffers(self)):
            if self.is_expert_param(name):
                continue
            if retain:
                gathered.setdefault(f"{layer_name}.{self.FAMILY.to_hub_key(name)}", full_cpu_tensor(tensor.data))
            else:
                materialize_dtensor(tensor.data)  # the same collective, without the host copy
        return gathered
