"""The collective half of an EP export: expert tensors gathered across the expert-TP and dispatch-EP groups. Each rank
calls every gather; ``retain=False`` joins every collective without assembling the result."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from typing import ClassVar

import torch

from halo.distributed.expert_parallel.config import EPConfig
from halo.models.moe.expert_layout import to_hub_expert_tensors
from halo.models.moe.family_specs import MoEFamilySpec
from halo.models.param_roles import persistent_buffers
from halo.runtime.collectives import all_gather_cat
from halo.runtime.dtensor import full_cpu_tensor, materialize_dtensor


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

    def gather_expert_tensors(self, local: Mapping[str, torch.Tensor], *, retain: bool = True) -> dict:
        """COLLECTIVE over the expert-TP and dispatch-EP groups. ``local`` (this rank's tensors by runtime attribute,
        matmul convention) with every expert and expert-TP shard, in ``local``'s order; ``{}`` without ``retain``."""
        gathered = {}
        for attr, tensor in local.items():
            if attr in self.EXPERT_TP_SHARD_DIMS:
                tensor = all_gather_cat(
                    tensor, self.EXPERT_TP_SHARD_DIMS[attr], self.expert_tp_group, self.expert_tp_size
                )
            tensor = all_gather_cat(tensor, 0, self.ep_config.dispatch_ep_group, self.ep_config.ep_size)
            if retain:
                gathered[attr] = tensor
        return gathered

    def gather_expert_state_dict(
        self, *, device: torch.device | str = "cpu", merge_lora: bool = False, retain: bool = True
    ) -> dict:
        """COLLECTIVE (see :meth:`gather_expert_tensors`). This layer's experts in the family's hub layout on
        ``device``. ``merge_lora`` folds each grouped-LoRA delta into its weight, so the RL weight sync ships the
        trained experts rather than the frozen base."""
        # Gated on built adapters, not the flag: the weight sync asks for the fold unconditionally.
        if merge_lora and self.has_expert_lora and self.expert_tp_size > 1:
            raise NotImplementedError(
                f"{type(self).__name__}: merge_lora is not implemented for the expert-TP gather "
                f"(expert_tensor_parallel_size={self.expert_tp_size}) — the export would carry "
                f"the FROZEN base experts. Gather with expert_tensor_parallel_size=1, or "
                f"implement the fold for the sharded layout."
            )
        local = {}
        for attr, param in self._base_expert_named_params():
            local[attr] = materialize_dtensor(param.data)
            if merge_lora and attr in self._expert_lora_attrs:
                lora_a = materialize_dtensor(getattr(self, f"{attr}_lora_A").data)
                lora_b = materialize_dtensor(getattr(self, f"{attr}_lora_B").data)
                delta = self.expert_lora_scaling * torch.bmm(lora_a.float(), lora_b.float())
                local[attr] = (local[attr].float() + delta).to(local[attr].dtype)
        gathered = self.gather_expert_tensors(local, retain=retain)
        del local
        return to_hub_expert_tensors(self.FAMILY.expert_layout, gathered, device=device) if retain else {}

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
