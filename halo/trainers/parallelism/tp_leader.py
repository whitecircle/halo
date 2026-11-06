"""Broadcasts from the TP/ETP-group leader, so ranks sharing one sharded replica process identical inputs."""

import torch
import torch.distributed as dist

from halo.distributed.mesh import tp_group
from halo.runtime.collectives import nccl_safe_broadcast
from halo.runtime.ranks import get_global_rank
from halo.trainers.parallelism.ep_layers import require_ep_config


class TPLeaderBroadcastMixin:
    """Leader broadcasts for the generation paths; no-ops outside TP and ETP. Siblings run collectives inside
    ``model.forward``, so independently sampled rollouts would deadlock them."""

    def _broadcast_object_from_tp_leader(self, obj):
        group = self._get_tp_or_etp_process_group()
        if group is None:
            return obj
        container = [obj]
        dist.broadcast_object_list(container, src=self._get_tp_group_src_rank(), group=group)
        return container[0]

    def _broadcast_tensors_from_tp_leader(self, result: dict) -> dict:
        """In place from the leader; NCCL-unsupported dtypes (the int16 routing-replay masks) ride a uint8 bit view."""
        group = self._get_tp_or_etp_process_group()
        if group is None:
            return result
        src = self._get_tp_group_src_rank()
        for value in result.values():
            if isinstance(value, torch.Tensor):
                nccl_safe_broadcast(value, src=src, group=group)
        return result

    def _is_tp_follower(self) -> bool:
        return self._get_tp_or_etp_process_group() is not None and self._get_tp_group_src_rank() != get_global_rank()

    def _get_tp_or_etp_group_size(self) -> int:
        group = self._get_tp_or_etp_process_group()
        return 1 if group is None else dist.get_world_size(group)

    def _get_tp_or_etp_process_group(self) -> dist.ProcessGroup | None:
        if self.parallelism_config.is_tp_mode:
            return tp_group(self._device_mesh)
        if self.parallelism_config.is_expert_tp_mode:
            return require_ep_config(self._ep_config).expert_tp_group
        return None

    def _get_tp_group_src_rank(self) -> int:
        """Global rank of the group's ``tp_rank == 0``, the source of every leader broadcast."""
        if self.parallelism_config.is_tp_mode:
            return get_global_rank() - self._get_tp_rank()
        if self.parallelism_config.is_expert_tp_mode:
            group = require_ep_config(self._ep_config).expert_tp_group
            if group is not None:
                return dist.get_process_group_ranks(group)[0]
        return get_global_rank()
