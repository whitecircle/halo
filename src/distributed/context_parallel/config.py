"""Context Parallelism (CP) configuration, process-group accessors, and the sequence-axis ops
(:func:`split_sequence_for_cp`, :func:`cp_boundary_shift`) every CP caller shares.

:class:`CPConfig` builds CP groups inside a single NVLink domain — the only shape
``ParallelismConfig`` admits (Ulysses all-to-all is bandwidth-heavy, and DeepEP dispatch under EP+CP
is domain-local), so DP size = ``world_size / cp_size`` counts whole domains' worth of groups.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from src.distributed.group_layout import (
    node_local_group_ranks,
    node_local_groups_per_domain,
    node_local_rank_and_group,
)
from src.distributed.runtime import (
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    get_nccl_timeout,
    is_global_main_process,
)

logger = logging.getLogger(__name__)


class CPConfig:
    """Context Parallelism configuration and process-group construction.

    Groups are NVLink-domain-local (see the module docstring); the DP rank the data loader shards by
    comes from ``ParallelismConfig.get_data_parallel_rank()``.
    """

    def __init__(
        self,
        cp_size: int,
        world_size: int | None = None,
        gpus_per_node: int | None = None,
        rank_offset: int = 0,
    ):
        """Args:
        world_size: width of ONE rank block (a pipeline stage; the whole job without PP);
            auto-detected from the global world size if None.
        gpus_per_node: NVLink-domain size; auto-detected from LOCAL_WORLD_SIZE if None.
        rank_offset: first global rank of this rank block (pipeline-stage base, 0 without PP). Layout
            math runs in block-local rank space; emitted membership is lifted back to global ranks.
        """
        self.cp_size = cp_size

        if world_size is None:
            world_size = get_global_world_size()
        self.world_size = world_size
        self.rank_offset = rank_offset

        # Same unit and same loud fallback as EPConfig: ``gpus_per_node`` is the NVLINK-DOMAIN size,
        # which production passes as ``ParallelismConfig.nvlink_domain_size``. Taking LOCAL_WORLD_SIZE
        # silently would cap node-local CP at one OS node on an NVL72 rack with nothing in the log.
        if gpus_per_node is None:
            gpus_per_node = get_local_world_size()
            logger.warning(
                "CPConfig built without an explicit domain size — assuming LOCAL_WORLD_SIZE "
                f"({gpus_per_node}). On a multi-node NVLink domain (NVL72) that is the wrong unit; "
                "pass ParallelismConfig.nvlink_domain_size."
            )
        if cp_size > gpus_per_node:
            raise ValueError(f"CP size ({cp_size}) cannot exceed the NVLink-domain size ({gpus_per_node})")
        if gpus_per_node % cp_size != 0:
            raise ValueError(f"CP size ({cp_size}) must divide the NVLink-domain size ({gpus_per_node})")
        self._create_cp_groups(gpus_per_node)

        # Sizes are reported, never stored: ``ParallelismConfig`` owns data_parallel_size (its
        # divisor also counts TP/ETP) and the domain count, and a second copy here would drift.
        if is_global_main_process():
            logger.info(f"CP Config: cp_size={self.cp_size}, world_size={world_size}")
            logger.info(f"  Data parallel size: {world_size // self.cp_size} (distinct batches)")
            logger.info(f"  NVLink domains={world_size // gpus_per_node}, GPUs/domain={gpus_per_node}")

    def _init_single_process_defaults(self) -> bool:
        """Set defaults for single-process mode. Returns True if distributed is not initialized.

        ``cp_size > 1`` cannot take this path: with no process group there is no all-to-all, so
        every rank would keep sequence chunk 0 and quietly train on ``1/cp_size`` of every sample.
        """
        if dist.is_initialized():
            return False
        if self.cp_size > 1:
            raise RuntimeError(
                f"CPConfig(cp_size={self.cp_size}) needs an initialized torch.distributed process "
                f"group: without one Ulysses has no all-to-all, so this process would attend over "
                f"sequence chunk 0 alone. Launch with torchrun, or set cp_size=1."
            )
        self.process_group = None
        self.cp_rank = 0
        self.cp_group_idx = 0
        logger.warning("Distributed not initialized - using single process mode")
        return True

    def _create_cp_groups(self, domain_size: int):
        """Create CP process groups over contiguous rank blocks inside each ``domain_size`` domain.

        Membership math comes from the shared ``group_layout`` source in block-local rank space,
        lifted to global ranks per block; every rank creates ALL groups in the same order.
        """
        if self._init_single_process_defaults():
            return

        global_rank = get_global_rank()

        num_domains = self.world_size // domain_size
        num_cp_groups = num_domains * node_local_groups_per_domain(domain_size, self.cp_size)
        self.cp_rank, self.cp_group_idx = node_local_rank_and_group(
            global_rank - self.rank_offset, domain_size, self.cp_size
        )

        self.process_group = None
        self._cp_group_ranks = None

        # ``dist.new_group`` is collective: under PP every rank must help create the OTHER stages' groups.
        for base in range(0, get_global_world_size(), self.world_size):
            for group_idx in range(num_cp_groups):
                cp_group_ranks = [base + r for r in node_local_group_ranks(group_idx, domain_size, self.cp_size)]
                group = dist.new_group(cp_group_ranks, timeout=get_nccl_timeout())
                if base == self.rank_offset and group_idx == self.cp_group_idx:
                    self.process_group = group
                    self._cp_group_ranks = cp_group_ranks

        # Fail fast if membership and layout disagree (else the first CP all-to-all hangs).
        if self._cp_group_ranks is None or global_rank not in self._cp_group_ranks:
            raise RuntimeError(
                f"CP group selection inconsistency: global_rank={global_rank} not in its "
                f"selected CP group {self._cp_group_ranks} (cp_group_idx={self.cp_group_idx}, "
                f"cp_rank={self.cp_rank}, rank_offset={self.rank_offset}, domain_size={domain_size})."
            )

        if is_global_main_process():
            logger.info("CP process groups created:")
            logger.info(f"  - CP group ranks (rank 0): {self._cp_group_ranks}")
            logger.info(f"  - Total CP groups (=DP size): {num_cp_groups}")
        logger.debug(f"CP rank {global_rank}: group_idx={self.cp_group_idx}, cp_rank={self.cp_rank}")


def cp_boundary_shift(
    logits: torch.Tensor,
    local_labels: torch.Tensor,
    boundary_labels: torch.Tensor | None,
    is_last_rank: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal shift of one CP rank's local ``(logits, labels)`` chunk → ``(shift_logits, shift_labels)``.

    The last rank (or a rank without a boundary label) uses the standard shift. A non-final rank's
    final logit predicts the NEXT chunk's first token, so the boundary pair ``(logits[:, -1:],
    boundary_labels)`` is appended instead of dropped — keeping every position's supervision and the
    per-rank shifted length equal to the chunk length.
    """
    if is_last_rank or boundary_labels is None:
        return logits[:, :-1, :], local_labels[:, 1:]
    # Every local logit keeps supervision, so the full logits tensor is passed as-is (no [B, chunk, V] copy).
    return logits, torch.cat([local_labels[:, 1:], boundary_labels], dim=1)


def split_sequence_for_cp(
    tensor: torch.Tensor,
    cp_config: CPConfig,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Split a tensor's sequence dimension for context parallelism."""
    if cp_config.cp_size == 1:
        return tensor

    seq_len = tensor.shape[seq_dim]
    if seq_len % cp_config.cp_size != 0:
        raise ValueError(f"Sequence length {seq_len} must be divisible by cp_size {cp_config.cp_size}")

    chunk_size = seq_len // cp_config.cp_size
    start = cp_config.cp_rank * chunk_size

    return tensor.narrow(seq_dim, start, chunk_size).contiguous()
