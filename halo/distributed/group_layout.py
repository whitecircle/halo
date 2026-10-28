"""Process-group rank layouts: which ranks form which EP/CP group, in block-local rank space."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class NodeLocalLayout:
    """Groups as contiguous ``group_size`` blocks tiling each ``domain_size``-wide NVLink domain, numbered
    domain-major (2 domains of 8, group_size=4)::

        group 0: [0, 1, 2, 3]     group 1: [4, 5, 6, 7]        # domain 0
        group 2: [8, 9, 10, 11]   group 3: [12, 13, 14, 15]    # domain 1
    """

    world_size: int
    group_size: int
    domain_size: int

    @property
    def num_groups(self) -> int:
        return self.world_size // self.domain_size * (self.domain_size // self.group_size)

    def rank_and_group(self, rank: int) -> tuple[int, int]:
        """``(rank_in_group, group_idx)`` of ``rank``."""
        domain_id, domain_local_rank = divmod(rank, self.domain_size)
        group_idx = domain_id * (self.domain_size // self.group_size) + domain_local_rank // self.group_size
        return domain_local_rank % self.group_size, group_idx

    def group_ranks(self, group_idx: int) -> list[int]:
        domain_id, local_group = divmod(group_idx, self.domain_size // self.group_size)
        base = domain_id * self.domain_size + local_group * self.group_size
        return [base + i for i in range(self.group_size)]

    def replica_ranks(self, rank_in_group: int) -> list[int]:
        """The ranks at ``rank_in_group`` in every group: an EP rank's expert replicas, a CP position."""
        return [
            domain * self.domain_size + local_group * self.group_size + rank_in_group
            for domain in range(self.world_size // self.domain_size)
            for local_group in range(self.domain_size // self.group_size)
        ]

    def etp_partition(self, ep_rank: int, ep_size: int, expert_tp_size: int) -> tuple[int, int]:
        """``(dispatch_ep_rank, expert_tp_rank)``: the EP group is ``expert_tp_size`` contiguous dispatch
        chunks of ``ep_size`` ranks, since the intranode P2P kernel needs contiguous IPC peers."""
        return ep_rank % ep_size, ep_rank // ep_size


@dataclass(frozen=True, kw_only=True)
class CrossNodeLayout:
    """The column-block layout of global-scope EP (``ep_scope="global"``); construction raises on a group that
    cannot tile it.

    DeepEP's intranode P2P kernel addresses peers by device offset, so each group takes one contiguous block per
    domain (group-rank = ``rdma_rank * members_per_domain + nvl_rank``). World=16, 2 domains of 8, group=8::

        EP group 0: [0, 1, 2, 3, 8, 9, 10, 11]      # block 0 of each domain
        EP group 1: [4, 5, 6, 7, 12, 13, 14, 15]    # block 1 of each domain
    """

    world_size: int
    group_size: int
    domain_size: int
    num_groups: int = field(init=False)
    members_per_domain: int = field(init=False)
    num_domains: int = field(init=False)

    def __post_init__(self) -> None:
        if self.domain_size <= 0 or self.world_size % self.domain_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be a positive multiple of nvlink_domain_size "
                f"({self.domain_size}) for cross-node EP."
            )
        num_domains = self.world_size // self.domain_size
        if self.group_size % num_domains != 0:
            raise ValueError(
                f"Cross-node EP group size ({self.group_size}) must be a multiple of the number of NVLink "
                f"domains ({num_domains}) so each group spans every domain with an equal share of members. "
                f"Use ep_group_size == world_size ({self.world_size}, a single global group) or NVLink-local EP "
                f"(ep_scope='node')."
            )
        members_per_domain = self.group_size // num_domains
        if self.domain_size % members_per_domain != 0:
            raise ValueError(
                f"Cross-node EP members-per-domain ({members_per_domain} = ep_group_size {self.group_size} / "
                f"num_domains {num_domains}) must divide the NVLink domain ({self.domain_size}) so each "
                f"group's per-domain members form a contiguous device block. Use ep_group_size == world_size "
                f"({self.world_size}) or NVLink-local EP (ep_scope='node')."
            )
        object.__setattr__(self, "num_groups", self.domain_size // members_per_domain)
        object.__setattr__(self, "members_per_domain", members_per_domain)
        object.__setattr__(self, "num_domains", num_domains)

    def rank_and_group(self, rank: int) -> tuple[int, int]:
        """``(ep_rank, group_idx)`` of ``rank``."""
        domain, pos = divmod(rank, self.domain_size)
        return domain * self.members_per_domain + pos % self.members_per_domain, pos // self.members_per_domain

    def group_ranks(self, group_idx: int) -> list[int]:
        """Group ``group_idx``'s ranks in group-rank order (domain-major)."""
        return [
            domain * self.domain_size + group_idx * self.members_per_domain + i
            for domain in range(self.num_domains)
            for i in range(self.members_per_domain)
        ]

    def replica_ranks(self, ep_rank: int) -> list[int]:
        """The ranks holding the same expert shard as ``ep_rank`` (its expert-grad replica group)."""
        domain, i = divmod(ep_rank, self.members_per_domain)
        return [domain * self.domain_size + g * self.members_per_domain + i for g in range(self.num_groups)]

    def etp_partition(self, ep_rank: int, ep_size: int, expert_tp_size: int) -> tuple[int, int]:
        """``(dispatch_ep_rank, expert_tp_rank)``: each domain's block is one ETP group, keeping its
        all-reduce on NVLink, and the dispatch group strides across domains."""
        return ep_rank // expert_tp_size, ep_rank % expert_tp_size


def ep_layout(
    *, world_size: int, group_size: int, domain_size: int, node_local: bool
) -> NodeLocalLayout | CrossNodeLayout:
    layout = NodeLocalLayout if node_local else CrossNodeLayout
    return layout(world_size=world_size, group_size=group_size, domain_size=domain_size)


def reject_node_local_ep_group(ep_group_size: int, domain_size: int, *, ep_size: int, expert_tp_size: int) -> None:
    """Reject a node-local EP group that does not fit in and tile its NVLink domain."""
    shape = f"ep_group_size = expert_parallel_size({ep_size}) * expert_tensor_parallel_size({expert_tp_size})."
    if ep_group_size > domain_size:
        raise ValueError(
            f"Node-local EP group size ({ep_group_size}) cannot exceed the NVLink domain "
            f"({domain_size}). {shape} Use ep_scope='global' for EP spanning NVLink domains (RDMA)."
        )
    if domain_size % ep_group_size != 0:
        raise ValueError(f"EP group size ({ep_group_size}) must divide the NVLink domain ({domain_size}). {shape}")


def reject_cross_node_ep_group(
    ep_group_size: int, scope_size: int, *, scope_desc: str, ep_size: int, expert_tp_size: int
) -> None:
    """Reject a global-scope EP group that does not fit and divide its scope (``scope_desc`` names it:
    the world, or one pipeline stage's world under PP)."""
    shape = f"ep_group_size = expert_parallel_size({ep_size}) * expert_tensor_parallel_size({expert_tp_size})."
    if ep_group_size > scope_size:
        raise ValueError(f"EP group size ({ep_group_size}) cannot exceed {scope_desc} ({scope_size}). {shape}")
    if scope_size % ep_group_size != 0:
        raise ValueError(f"EP group size ({ep_group_size}) must divide {scope_desc} ({scope_size}). {shape}")


def reject_cross_node_etp_shape(expert_tp_size: int, members_per_domain: int, ep_size: int, num_domains: int) -> None:
    """Reject a cross-node EP+ETP shape that is not exactly one ETP group per NVLink domain."""
    if expert_tp_size != members_per_domain or ep_size != num_domains:
        raise ValueError(
            f"Cross-node EP+ETP supports one ETP group per NVLink domain only: requires "
            f"expert_tensor_parallel_size ({expert_tp_size}) == EP members per domain "
            f"({members_per_domain}) and expert_parallel_size ({ep_size}) == domains "
            f"spanned ({num_domains}). Set expert_tensor_parallel_size={members_per_domain} with "
            f"expert_parallel_size={num_domains}, or use NVLink-local EP (ep_scope='node')."
        )
