"""Process-group rank layout — which global ranks form which EP/CP group.

``ParallelismConfig`` and ``EPConfig``/``CPConfig`` construction both derive their membership from
these functions. Dependency-free leaf, so the CP path uses it without importing DeepEP.

Node-local (``node_local_*``): each NVLink domain's ``domain_size`` GPUs split into contiguous
``group_size`` blocks, numbered domain-major. ``domain_size`` is the NVLink-domain size, not
``LOCAL_WORLD_SIZE`` (2 domains of 8, group_size=4)::

    group 0: [0, 1, 2, 3]     group 1: [4, 5, 6, 7]        # domain 0
    group 2: [8, 9, 10, 11]   group 3: [12, 13, 14, 15]    # domain 1

Cross-node (``cross_node_*``, ``ep_scope="global"``): "column-block" layout — DeepEP's intranode P2P
kernel addresses peers by device offset, so each group needs contiguous CUDA-IPC peers per domain
(group-rank = ``rdma_rank * members_per_domain + nvl_rank``). World=16, 2 domains of 8, group=8::

    EP group 0: [0, 1, 2, 3, 8, 9, 10, 11]      # block 0 of each domain
    EP group 1: [4, 5, 6, 7, 12, 13, 14, 15]    # block 1 of each domain
"""

from __future__ import annotations


def node_local_groups_per_domain(domain_size: int, group_size: int) -> int:
    """Number of contiguous ``group_size`` blocks that tile one NVLink domain."""
    return domain_size // group_size


def node_local_rank_and_group(global_rank: int, domain_size: int, group_size: int) -> tuple[int, int]:
    """Return ``(rank_in_group, group_idx)`` for ``global_rank`` under the node-local layout.

    ``rank_in_group`` is the rank's position within its group (0..group_size-1); ``group_idx``
    is the global index of the group, numbered domain-major.
    """
    domain_id = global_rank // domain_size
    domain_local_rank = global_rank % domain_size
    groups_per_domain = node_local_groups_per_domain(domain_size, group_size)
    rank_in_group = domain_local_rank % group_size
    group_idx = domain_id * groups_per_domain + domain_local_rank // group_size
    return rank_in_group, group_idx


def node_local_group_ranks(group_idx: int, domain_size: int, group_size: int) -> list[int]:
    """Global ranks of group ``group_idx`` — a contiguous block within one NVLink domain."""
    groups_per_domain = node_local_groups_per_domain(domain_size, group_size)
    domain_id = group_idx // groups_per_domain
    local_group = group_idx % groups_per_domain
    base = domain_id * domain_size + local_group * group_size
    return [base + i for i in range(group_size)]


def node_local_replica_ranks(rank_in_group: int, world_size: int, domain_size: int, group_size: int) -> list[int]:
    """Global ranks sharing ``rank_in_group`` across every group (EP expert replicas / CP positions)."""
    groups_per_domain = node_local_groups_per_domain(domain_size, group_size)
    num_domains = world_size // domain_size
    return [
        domain * domain_size + local_group * group_size + rank_in_group
        for domain in range(num_domains)
        for local_group in range(groups_per_domain)
    ]


def etp_dispatch_coords(ep_rank: int, ep_size: int, expert_tp_size: int, node_local: bool) -> tuple[int, int]:
    """``(dispatch_ep_rank, expert_tp_rank)`` for an EP-group rank under expert TP.

    Node-local: the EP group is laid out as ``expert_tp_size`` contiguous dispatch (sub-EP) chunks of
    ``ep_size`` ranks (DeepEP's intranode P2P kernel needs contiguous IPC peers), so
    ``dispatch = ep_rank % ep_size`` and ``expert_tp = ep_rank // ep_size``. Cross-node: each domain's
    contiguous block is one ETP group (keeps the ETP all-reduce on NVLink) and the dispatch group
    strides across domains, so ``expert_tp = ep_rank % expert_tp_size`` and
    ``dispatch = ep_rank // expert_tp_size``.

    ETP partners share ``dispatch_ep_rank`` and therefore a DP batch; ``EPConfig``'s group
    construction and ``ParallelismConfig.get_data_parallel_rank`` both use this formula.
    """
    if node_local:
        return ep_rank % ep_size, ep_rank // ep_size
    return ep_rank // expert_tp_size, ep_rank % expert_tp_size


def cross_node_layout(world_size: int, ep_group_size: int, nvlink_domain_size: int) -> tuple[int, int, int]:
    """Resolve and validate the column-block layout for cross-node (global-scope) EP.

    Returns ``(num_ep_groups, members_per_domain, num_domains)``.

    Raises:
        ValueError: if ``ep_group_size`` cannot tile the world as equal contiguous per-domain
            blocks (each group spanning every NVLink domain).
    """
    if nvlink_domain_size <= 0 or world_size % nvlink_domain_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be a positive multiple of nvlink_domain_size "
            f"({nvlink_domain_size}) for cross-node EP."
        )
    num_domains = world_size // nvlink_domain_size
    if ep_group_size % num_domains != 0:
        raise ValueError(
            f"Cross-node EP group size ({ep_group_size}) must be a multiple of the number of NVLink "
            f"domains ({num_domains}) so each group spans every domain with an equal share of members. "
            f"Use ep_group_size == world_size ({world_size}, a single global group) or NVLink-local EP "
            f"(ep_scope='node')."
        )
    members_per_domain = ep_group_size // num_domains
    if nvlink_domain_size % members_per_domain != 0:
        raise ValueError(
            f"Cross-node EP members-per-domain ({members_per_domain} = ep_group_size {ep_group_size} / "
            f"num_domains {num_domains}) must divide the NVLink domain ({nvlink_domain_size}) so each "
            f"group's per-domain members form a contiguous device block. Use ep_group_size == world_size "
            f"({world_size}) or NVLink-local EP (ep_scope='node')."
        )
    num_ep_groups = nvlink_domain_size // members_per_domain
    return num_ep_groups, members_per_domain, num_domains


def reject_node_local_ep_group(
    ep_group_size: int, domain_size: int, *, ep_size: int, expert_tp_size: int, remedy: str
) -> None:
    """Reject a node-local EP group that does not tile its NVLink domain.

    A node-local dispatch group is a contiguous block inside one domain, so it must both fit in the
    domain and tile it evenly. Checked twice: ``ParallelismConfig`` at config time and ``EPConfig`` at
    group construction; each caller appends its own ``remedy``.
    """
    shape = f"ep_group_size = ep_size({ep_size}) * expert_tp_size({expert_tp_size})."
    if ep_group_size > domain_size:
        raise ValueError(
            f"Node-local EP group size ({ep_group_size}) cannot exceed the NVLink domain "
            f"({domain_size}). {shape} {remedy}"
        )
    if domain_size % ep_group_size != 0:
        raise ValueError(f"EP group size ({ep_group_size}) must divide the NVLink domain ({domain_size}). {shape}")


def reject_cross_node_ep_group(
    ep_group_size: int, scope_size: int, *, scope_desc: str, ep_size: int, expert_tp_size: int
) -> None:
    """Reject a cross-node (global-scope) EP group that does not fit and divide its scope.

    ``scope_desc``/``scope_size`` name the scope in the caller's terms: the world for a plain job,
    one pipeline stage's world when ``pp_size > 1``.
    """
    shape = f"ep_group_size = ep_size({ep_size}) * expert_tp_size({expert_tp_size})."
    if ep_group_size > scope_size:
        raise ValueError(f"EP group size ({ep_group_size}) cannot exceed {scope_desc} ({scope_size}). {shape}")
    if scope_size % ep_group_size != 0:
        raise ValueError(f"EP group size ({ep_group_size}) must divide {scope_desc} ({scope_size}). {shape}")


def reject_cross_node_etp_shape(
    expert_tp_size: int, members_per_domain: int, ep_size: int, num_domains: int, *, remedy: str
) -> None:
    """Reject a cross-node EP+ETP shape that is not exactly one ETP group per NVLink domain.

    The expert-TP all-reduce must stay on NVLink, so each domain's contiguous block is one ETP group
    and the dispatch group strides across domains. ``ParallelismConfig`` checks this before the model
    loads; ``EPConfig`` re-checks it at group construction, where ``remedy`` can name the offending
    group.
    """
    if expert_tp_size != members_per_domain or ep_size != num_domains:
        raise ValueError(
            f"Cross-node EP+ETP supports one ETP group per NVLink domain only: requires "
            f"expert_tp_size ({expert_tp_size}) == EP members per domain ({members_per_domain}) "
            f"and ep_size ({ep_size}) == domains spanned ({num_domains}). {remedy}"
        )


def cross_node_group_ranks(group_idx: int, world_size: int, ep_group_size: int, nvlink_domain_size: int) -> list[int]:
    """Global ranks of cross-node EP group ``group_idx`` in group-rank order (domain-major)."""
    _, m, num_domains = cross_node_layout(world_size, ep_group_size, nvlink_domain_size)
    ranks = []
    for domain in range(num_domains):
        base = domain * nvlink_domain_size + group_idx * m
        ranks.extend(base + i for i in range(m))
    return ranks


def cross_node_rank_and_group(
    global_rank: int, world_size: int, ep_group_size: int, nvlink_domain_size: int
) -> tuple[int, int]:
    """Return ``(ep_rank, ep_group_idx)`` for ``global_rank`` under the column-block layout."""
    _, m, _ = cross_node_layout(world_size, ep_group_size, nvlink_domain_size)
    domain = global_rank // nvlink_domain_size
    pos = global_rank % nvlink_domain_size
    ep_group_idx = pos // m
    ep_rank = domain * m + (pos % m)
    return ep_rank, ep_group_idx


def cross_node_replica_ranks(ep_rank: int, world_size: int, ep_group_size: int, nvlink_domain_size: int) -> list[int]:
    """Global ranks holding the same expert shard as ``ep_rank`` (expert-grad replica group)."""
    num_ep_groups, m, _ = cross_node_layout(world_size, ep_group_size, nvlink_domain_size)
    domain = ep_rank // m
    i = ep_rank % m
    return [domain * nvlink_domain_size + g * m + i for g in range(num_ep_groups)]
