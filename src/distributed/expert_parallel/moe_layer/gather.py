"""The collective half of an EP export: this rank's expert tensors gathered across the expert-TP and dispatch-EP
groups, then written in the family's :class:`~src.models.moe.expert_layout.ExpertLayout` under its hub keys.

Every gather is collective over those groups: each rank calls it, and ``retain=False`` joins every collective
without assembling the result, which only the writing rank needs.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from typing import ClassVar

import torch

from src.distributed.expert_parallel.config import EPConfig
from src.models.moe.expert_layout import to_hub_expert_tensors
from src.models.moe.family_specs import MoEFamilySpec
from src.models.param_roles import persistent_buffers
from src.runtime.collectives import all_gather_cat
from src.runtime.dtensor import full_cpu_tensor, materialize_dtensor


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
        """COLLECTIVE over the expert-TP and dispatch-EP groups. ``local`` (this rank's expert tensors by runtime
        attribute, matmul convention) with every expert and every expert-TP shard, in ``local``'s order; ``{}``
        without ``retain``."""
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
        ``device``, keyed relative to the layer before the export renames.

        ``merge_lora`` folds each adapted weight's grouped-LoRA delta ``scaling·(A@B)`` into it, on the full plain
        tensor, so the RL weight sync ships the trained experts rather than the frozen base.
        """
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

    def gather_layer_state_dict(self, layer_name: str, *, merge_lora: bool = False, retain: bool = True) -> dict:
        """COLLECTIVE (see :meth:`gather_expert_tensors`, plus each FSDP2-sharded replicated tensor). The whole layer
        for a checkpoint under ``layer_name``, in the hub spelling: the gathered experts, then every persistent
        replicated param and buffer (router, shared experts, balancing slots).

        No LoRA layer sits inside an EP layer (the trainer refuses one), so the replicated tensors are written as
        they are. Persistent buffers only: a non-persistent one (a rotary cache) is absent from the sharded save, and
        writing it here would break merged-from-sharded == gathered.
        """
        experts = self.gather_expert_state_dict(merge_lora=merge_lora, retain=retain)
        gathered = {f"{layer_name}.{self.FAMILY.to_hub_key(key)}": tensor for key, tensor in experts.items()}
        del experts
        expert_roots = {name.split(".")[0] for name, _ in self.expert_named_params()}
        for name, tensor in itertools.chain(self.named_parameters(), persistent_buffers(self)):
            if name.split(".")[0] in expert_roots:
                continue
            if retain:
                gathered.setdefault(f"{layer_name}.{self.FAMILY.to_hub_key(name)}", full_cpu_tensor(tensor.data))
            else:
                materialize_dtensor(tensor.data)  # the same collective, without the host copy
        return gathered
