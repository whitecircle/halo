"""The sync topology choice and where each rollout server's workers sit in one weight-sync group: the trainer is rank 0
and each server's workers take a contiguous block of ranks, servers of one node adjacent."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple
from urllib.parse import urlparse

SyncTopology = Literal["per_server", "single_group"]


class GroupLayout(NamedTuple):
    """``order`` lists server indexes in rank order; ``rank_offsets`` is by server index; ``world_size`` counts the
    trainer."""

    order: tuple[int, ...]
    rank_offsets: tuple[int, ...]
    world_size: int


def resolve_group_layout(nodes: Sequence[str], engine_worlds: Sequence[int]) -> GroupLayout:
    """Servers sorted by node label, config order kept within a node, so each node's servers are one contiguous rank
    block: a ring over interleaved servers would cross that node's fabric links twice."""
    if len(nodes) != len(engine_worlds):
        raise ValueError(f"{len(nodes)} node labels for {len(engine_worlds)} servers")
    order = tuple(sorted(range(len(nodes)), key=lambda index: (nodes[index], index)))
    offsets = [0] * len(nodes)
    next_rank = 1
    for index in order:
        offsets[index] = next_rank
        next_rank += engine_worlds[index]
    return GroupLayout(order=order, rank_offsets=tuple(offsets), world_size=next_rank)


def reject_split_nodes(nodes: Sequence[str], addresses: Sequence[str], urls: Sequence[str]) -> None:
    """Refuse one address listed under two node labels: the sort would split that node's servers across the ring."""
    labels_at: dict[str, set[str]] = {}
    for node, address in zip(nodes, addresses, strict=True):
        labels_at.setdefault(address, set()).add(node)
    for address, labels in labels_at.items():
        if len(labels) > 1:
            servers = [url for url, found in zip(urls, addresses, strict=True) if found == address]
            raise ValueError(
                f"Rollout servers {servers} all resolve to {address} but carry node labels {sorted(labels)}, so the "
                f"group's rank order would split that node. Give them one 'node' label in rollout_server_configs."
            )


def resolve_group_host(servers: Sequence[Mapping[str, Any]]) -> str | None:
    """The one address a group of several servers advertises, ``None`` where no entry sets one. A group has one
    rendezvous, so a per-server port or a second address has nothing to bind."""
    if any("group_port" in server for server in servers):
        raise ValueError(
            "rollout_server_configs entries carry a group_port, but rollout_sync_topology: single_group forms one group "
            "on one trainer port. Set rollout_group_port once, or use rollout_sync_topology: per_server."
        )
    hosts = {server["group_host"] for server in servers if "group_host" in server}
    if len(hosts) > 1:
        raise ValueError(
            f"rollout_server_configs entries carry different group_host values {sorted(hosts)}, but "
            f"rollout_sync_topology: single_group advertises one address to every server. Use one group_host, or "
            f"rollout_sync_topology: per_server."
        )
    return next(iter(hosts), None)


def resolve_node_label(server: Mapping[str, Any]) -> str:
    """The entry's ``node`` label, else its URL host, which an IP and a DNS name of one node spell apart."""
    return server.get("node") or urlparse(server["url"]).hostname or server["url"]
