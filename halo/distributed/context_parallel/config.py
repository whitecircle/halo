"""Ulysses CP's process groups: contiguous ``cp_size`` rank blocks inside each NVLink domain of a pipeline stage.
Domain-local only, since the all-to-all is bandwidth-heavy and DeepEP dispatch under EP+CP is domain-local."""

from __future__ import annotations

import logging

from halo.distributed.group_layout import NodeLocalLayout
from halo.runtime.collectives import build_groups
from halo.runtime.ranks import get_global_rank, get_global_world_size, has_process_group, is_global_main_process

logger = logging.getLogger(__name__)


class CPConfigError(ValueError):
    """Raised when Ulysses CP is misconfigured for the given model."""


class CPConfig:
    """COLLECTIVE over the default group: this rank's CP process group, ``cp_rank`` and ``cp_group_idx``. Without an
    initialized process group only ``cp_size == 1`` builds, with no group."""

    def __init__(self, cp_size: int, *, stage_world_size: int, stage_base_rank: int, nvlink_domain_size: int):
        self.cp_size = cp_size
        self.stage_base_rank = stage_base_rank

        if not has_process_group():
            if cp_size > 1:
                raise RuntimeError(
                    f"context_parallel_size={cp_size} needs an initialized torch.distributed "
                    f"process group: without one Ulysses has no all-to-all, so this process would "
                    f"attend over sequence chunk 0 alone. Launch with torchrun, or set "
                    f"context_parallel_size=1."
                )
            self.process_group, self.cp_rank, self.cp_group_idx = None, 0, 0
            logger.warning("Distributed not initialized - using single process mode")
            return

        global_rank = get_global_rank()
        layout = NodeLocalLayout(world_size=stage_world_size, group_size=cp_size, domain_size=nvlink_domain_size)
        self.cp_rank, self.cp_group_idx = layout.rank_and_group(global_rank - stage_base_rank)
        groups_per_stage = layout.num_groups
        rank_lists = [
            [base + rank for rank in layout.group_ranks(group_idx)]
            for base in range(0, get_global_world_size(), stage_world_size)
            for group_idx in range(groups_per_stage)
        ]
        mine = stage_base_rank // stage_world_size * groups_per_stage + self.cp_group_idx
        self.process_group = build_groups(rank_lists, mine=mine, what="CP group")
        self._cp_group_ranks = rank_lists[mine]

        # Membership and layout must agree, or the first CP all-to-all hangs.
        if global_rank not in self._cp_group_ranks:
            raise RuntimeError(
                f"CP group selection inconsistency: global_rank={global_rank} not in its "
                f"selected CP group {self._cp_group_ranks} (cp_group_idx={self.cp_group_idx}, "
                f"cp_rank={self.cp_rank}, stage_base_rank={stage_base_rank}, "
                f"nvlink_domain_size={nvlink_domain_size})."
            )

        if is_global_main_process():
            logger.info(
                f"CP process groups created: cp_size={cp_size}, {groups_per_stage} groups per stage (= data-parallel "
                f"size), {nvlink_domain_size} GPUs per NVLink domain; rank 0's group {self._cp_group_ranks}"
            )
        logger.debug(f"CP rank {global_rank}: group_idx={self.cp_group_idx}, cp_rank={self.cp_rank}")
