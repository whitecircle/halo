"""The sync topology choice and where each rollout server's workers sit in one weight-sync group: the trainer is rank 0
and each server's workers take a contiguous block of ranks, servers of one node adjacent."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple, get_args
from urllib.parse import urlparse

SyncTopology = Literal["per_server", "single_group"]
# ``auto`` resolves to one of them once the servers have been read.
SyncTopologySetting = Literal["auto", SyncTopology]


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


def reject_pipeline_stages(engine: str, urls: Sequence[str], stages: Sequence[int]) -> None:
    """Refuse a pipeline-parallel server: no rank formula reads its stage, so its workers would rank as
    ``rank_offset + tp_rank`` and collide."""
    for url, count in zip(urls, stages, strict=True):
        if count > 1:
            raise RuntimeError(
                f"{engine} server {url} runs --pp-size {count}: its workers rank as rank_offset + tp_rank, so the "
                f"stages collide in a group shared with other servers. Serve with --pp-size 1, or set "
                f"rollout_sync_topology: per_server."
            )


def reject_mixed_localities(urls: Sequence[str], is_local: Sequence[bool], group_host: str | None) -> None:
    """Refuse servers on this host beside servers on others behind one advertised address: it is loopback, which the
    remote ones cannot dial, or the external NIC, where firewalls drop the local ones' hairpin traffic."""
    if group_host is not None or all(is_local) or not any(is_local):
        return
    local = [url for url, found in zip(urls, is_local, strict=True) if found]
    raise ValueError(
        f"Rollout servers {local} run on this host and the rest on others, but one group advertises one address. Set "
        f"group_host on the rollout_server_configs entries to an address every server can dial, or use "
        f"rollout_sync_topology: per_server."
    )


def validate_sync_topology(topology: str, servers: Sequence[Mapping[str, Any]]) -> None:
    """Raise on a topology, or on server entries, that the sync cannot honour. One group has one rendezvous, so a
    per-server port or a second address has nothing to bind, and a node label orders only the servers of one group."""
    if topology not in get_args(SyncTopologySetting):
        raise ValueError(
            f"Unknown rollout_sync_topology {topology!r}; supported: {list(get_args(SyncTopologySetting))}"
        )
    if topology != "single_group":
        if any("node" in server for server in servers):
            raise ValueError(
                "rollout_server_configs entries set node, which only orders the servers of one group. Remove it, or "
                "set rollout_sync_topology: single_group."
            )
        return
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


def resolve_group_host(servers: Sequence[Mapping[str, Any]]) -> str | None:
    """The one address a validated single group advertises, ``None`` where no entry sets one."""
    return next((server["group_host"] for server in servers if "group_host" in server), None)


def resolve_node_label(server: Mapping[str, Any]) -> str:
    """The entry's ``node`` label, else its URL host, which an IP and a DNS name of one node spell apart."""
    return server.get("node") or urlparse(server["url"]).hostname or server["url"]
