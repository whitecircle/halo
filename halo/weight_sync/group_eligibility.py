"""Whether a run's rollout servers can share one weight-sync group, and the topology an ``auto`` setting resolves to:
one group needs servers that load what one broadcast carries the same way and that the group layout can rank."""

from collections.abc import Sequence
from typing import NamedTuple

from halo.weight_sync.clients.base import ServerFlags
from halo.weight_sync.group_layout import (
    SyncTopology,
    SyncTopologySetting,
    reject_mixed_localities,
    reject_pipeline_stages,
    reject_split_nodes,
)

# A launch flag the servers of one group must agree on: every one but the image check, the id each launch has alone,
# the pipeline stages, which the group's layout read refuses, and the engine facts that only get logged.
GROUP_UNIFORM_FLAGS = tuple(
    name for name in ServerFlags._fields if name not in ("toolkit_image", "instance_id", "pipeline_stages", "extras")
)


class ServerReading(NamedTuple):
    """What one server answers for the group: ``model`` and ``flags`` are ``None`` where it did not answer."""

    url: str
    node: str
    host: str
    is_local: bool
    model: str | None
    flags: ServerFlags | None
    pipeline_stages: int


def find_unread_server(
    urls: Sequence[str], models: Sequence[str | None], flags: Sequence[ServerFlags | None]
) -> str | None:
    """The first url whose server reported no served model or no launch flags, else ``None``."""
    return next(
        (url for url, model, found in zip(urls, models, flags, strict=True) if model is None or found is None), None
    )


def find_fleet_disagreement(
    urls: Sequence[str], models: Sequence[str | None], flags: Sequence[ServerFlags | None]
) -> tuple[str, dict[str, object]] | None:
    """The first attribute (the served model, then each :data:`GROUP_UNIFORM_FLAGS` entry) on which the servers that
    report it differ, with its value per url; ``None`` when they agree. A flag a server does not report is not
    compared."""
    attributes: dict[str, dict[str, object]] = {"served model": {}}
    for url, model, found in zip(urls, models, flags, strict=True):
        if model is not None:
            attributes["served model"][url] = model
        for name in GROUP_UNIFORM_FLAGS:
            if found is not None and (value := getattr(found, name)) is not None:
                attributes.setdefault(name, {})[url] = value
    return next(((what, found) for what, found in attributes.items() if len(set(found.values())) > 1), None)


def find_single_group_refusal(readings: Sequence[ServerReading], group_host: str | None) -> str | None:
    """Why the servers cannot share one group, or ``None``: one that did not answer, a fleet that disagrees, a
    pipeline-parallel server, a node split across labels, or local servers beside remote ones with no ``group_host``."""
    urls = [reading.url for reading in readings]
    models = [reading.model for reading in readings]
    flags = [reading.flags for reading in readings]
    if (unread := find_unread_server(urls, models, flags)) is not None:
        return f"{unread} did not report its served model and launch flags"
    if (disagreement := find_fleet_disagreement(urls, models, flags)) is not None:
        what, found = disagreement
        return f"the servers disagree on their {what} ({found})"
    try:
        reject_pipeline_stages("Rollout", urls, [reading.pipeline_stages for reading in readings])
        reject_split_nodes([reading.node for reading in readings], [reading.host for reading in readings], urls)
        reject_mixed_localities(urls, [reading.is_local for reading in readings], group_host)
    except (ValueError, RuntimeError) as refusal:
        return str(refusal)
    return None


def resolve_sync_topology(
    setting: SyncTopologySetting, readings: Sequence[ServerReading], group_host: str | None
) -> tuple[SyncTopology, str]:
    """``(topology, why)``. An explicit setting wins unread. ``auto`` is ``single_group`` for two or more servers that
    :func:`find_single_group_refusal` accepts, else ``per_server``."""
    if setting != "auto":
        return setting, f"rollout_sync_topology: {setting}"
    if len(readings) < 2:
        return "per_server", "one server has nothing to share a group with"
    if (refusal := find_single_group_refusal(readings, group_host)) is not None:
        return "per_server", refusal
    return "single_group", f"{len(readings)} servers report one served model and one set of weight-loading flags"
