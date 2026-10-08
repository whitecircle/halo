"""Where every rank reads a checkpoint from, agreed before any rank loads it.

A load names its source once, as a local directory or a Hub repo id at a revision, but each rank
resolves that name on its own node: the directory as that node mounts it, the snapshot in the Hub
cache its ``HF_HOME`` points at. :func:`resolve_model_source` fetches the snapshot main-rank-first,
then has the world agree on what every rank found before any of them reads a weight.
"""

from __future__ import annotations

import os
import socket
from typing import NamedTuple

from huggingface_hub import constants as hub_constants
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

from src.checkpoint.format import (
    HUB_SUBFOLDER_IGNORE_PATTERNS,
    SAFETENSORS_INDEX_FILE,
    resolve_checkpoint_weights,
)
from src.distributed.filesystem import fs_aware_main_first, store_join_recorded_failure
from src.distributed.runtime import (
    broadcast_from_rank0,
    fs_aware_load_rank,
    is_input_shared_filesystem,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.log import KEY_PREVIEW_COUNT

# What a rank that finds the source as a directory reports to the agreement, in place of a commit.
_LOCAL_DIRECTORY = "a local directory"


class _SourceView(NamedTuple):
    """One rank's resolution of the source."""

    # :data:`_LOCAL_DIRECTORY`, or the commit this rank's Hub cache holds.
    resolved: str
    # The snapshot directory this rank reads a Hub source from; None for a local directory.
    snapshot: str | None = None
    # The weight files that snapshot declares (:func:`_weight_files`).
    weight_files: tuple[str, ...] = ()


def resolve_model_source(
    model_name_or_path: str, revision: str | None, *, tag: str, whole_repo: bool = False
) -> str | None:
    """Fetch ``model_name_or_path`` main-rank-first and return the revision every rank loads it at.

    COLLECTIVE-EQUIVALENT — every rank calls it with the same ``tag``, equally often.

    The fetch takes the repo's top-level files (:data:`~src.checkpoint.format.HUB_SUBFOLDER_IGNORE_PATTERNS`),
    where a model's config, tokenizer, weights and remote code sit, so the weight dumps a repo also ships
    in subfolders stay on the Hub. ``whole_repo`` is for a loader that reads subfolders too (a
    sentence-transformers pipeline keeps its modules in them).

    The fetch is scoped like :func:`~src.distributed.filesystem.fs_aware_main_first`: global rank 0
    on a shared input filesystem, each node's local rank 0 on a per-node one. Every rank then
    resolves the source on its own node, and the verdicts are joined over the c10d store, which also
    absorbs the per-node fetch skew the next collective would otherwise wait out under the NCCL
    watchdog. A rank that cannot see the source fails the whole world naming it: a local checkpoint
    present on some nodes only, or a Hub snapshot fetched into a cache this rank does not read (a
    shared input declaration over node-local ``HF_HOME``).

    For a Hub repo the return is the commit every rank's cache resolved, agreed across ranks:
    per-node caches staged at different times, or a revision that moved between two nodes' fetches,
    would otherwise load different weights on different nodes with no error. Every rank's snapshot
    must then hold each weight file global rank 0's snapshot names
    (:func:`_reject_incomplete_snapshots`), so every rank reads the weights at that commit from its
    own cache, with no hub request. A local directory returns ``revision`` unchanged.
    """
    fetch = fs_aware_load_rank()
    failure: BaseException | None = None
    view: _SourceView | None = None
    with fs_aware_main_first(f"model_source/{tag}"):
        try:
            view = _resolve_on_this_rank(model_name_or_path, revision, fetch=fetch, whole_repo=whole_repo)
        except Exception as exc:  # joined below, so every rank raises with it
            failure = exc
    store_join_recorded_failure(f"model_source/{tag}", failure, f"Resolving the checkpoint {model_name_or_path!r}")
    reject_divergent_settings(
        {"resolved as": view.resolved},
        f"The checkpoint {model_name_or_path!r} each rank resolved",
        "Per-node Hub caches hold different commits of it (staged at different times, or its revision "
        "moved between two nodes' fetches), so the nodes would train different weights with no other "
        "error. Pin the revision to one commit, or refresh every node's cache to the same one.",
    )
    if view.resolved == _LOCAL_DIRECTORY:
        return revision
    _reject_incomplete_snapshots(model_name_or_path, view, fetched=fetch)
    return view.resolved


def _resolve_on_this_rank(
    model_name_or_path: str, revision: str | None, *, fetch: bool, whole_repo: bool
) -> _SourceView:
    """This rank's view of the source: a local directory, or the snapshot its Hub cache holds.

    ``fetch`` downloads whatever the cache lacks; without it the cache alone answers, so a rank that
    is not its scope's fetcher never contacts the hub here.
    """
    if os.path.isdir(model_name_or_path):
        return _SourceView(_LOCAL_DIRECTORY)
    try:
        snapshot = snapshot_download(
            model_name_or_path,
            revision=revision,
            local_files_only=not fetch,
            ignore_patterns=None if whole_repo else list(HUB_SUBFOLDER_IGNORE_PATTERNS),
        )
    except (HFValidationError, LocalEntryNotFoundError) as exc:
        raise FileNotFoundError(_unresolvable_reason(model_name_or_path, revision, exc if fetch else None)) from exc
    return _SourceView(os.path.basename(snapshot), snapshot, _weight_files(snapshot))


def _weight_files(snapshot: str) -> tuple[str, ...]:
    """The files a load reads the snapshot's weights from: the safetensors index and the shards it
    names, or the single weight file (:func:`~src.checkpoint.format.resolve_checkpoint_weights`)."""
    layout = resolve_checkpoint_weights(snapshot)
    index = (SAFETENSORS_INDEX_FILE,) if layout.index is not None else ()
    legacy = (layout.legacy_bin,) if layout.legacy_bin is not None else ()
    return (*index, *layout.shard_files, *legacy)


def _reject_incomplete_snapshots(model_name_or_path: str, view: _SourceView, *, fetched: bool) -> None:
    """Fail every rank when a rank's snapshot lacks a weight file global rank 0's snapshot names. COLLECTIVE.

    huggingface_hub's cache-only resolution returns any snapshot folder of the commit without
    checking its files, and the config and tokenizer reads ahead of the load leave one in every
    node's cache. Such a weightless snapshot resolves like a complete one, after which
    ``from_pretrained`` downloads the weights on every rank of that node and the lazy loaders fail on
    a missing shard.
    """
    required = broadcast_from_rank0(view.weight_files)
    missing = [name for name in required if not os.path.isfile(os.path.join(view.snapshot, name))]
    reason = None
    if missing:
        reason = (
            f"This rank's Hub cache ({hub_constants.HF_HUB_CACHE}) on {socket.gethostname()} holds the "
            f"snapshot of {model_name_or_path!r} at commit {view.resolved} without {len(missing)} of the "
            f"{len(required)} weight file(s) global rank 0's snapshot names "
            f"({', '.join(missing[:KEY_PREVIEW_COUNT])}). {_cache_remedy(fetched)}"
        )
    reject_across_ranks(reason, f"Resolving the checkpoint {model_name_or_path!r}", exc_type=FileNotFoundError)


def _unresolvable_reason(model_name_or_path: str, revision: str | None, fetch_error: Exception | None) -> str:
    """Why this rank holds no copy of the source: its own failed fetch, or the input-filesystem
    declaration that left it reading a cache nobody fetched into."""
    missing = (
        f"{model_name_or_path!r} is not a directory on {socket.gethostname()}, and this rank's Hub cache "
        f"({hub_constants.HF_HUB_CACHE}) holds no snapshot of it at revision {revision or 'main'}."
    )
    if fetch_error is not None:
        return f"{missing} This rank fetched for its scope, and the fetch failed: {type(fetch_error).__name__}: {fetch_error}"
    return f"{missing} {_cache_remedy(fetched=False)}"


def _cache_remedy(fetched: bool) -> str:
    """Who filled the cache this rank reads, and what makes that cache hold the source."""
    if fetched:
        return (
            "This rank fetched for its scope, and the hub left the snapshot incomplete: an offline fetch "
            "(HF_HUB_OFFLINE) or an unreachable hub serves whatever the cache holds. Fetch with the hub "
            "reachable, or stage the complete snapshot into this cache."
        )
    if is_input_shared_filesystem():
        return (
            "The input filesystem is declared shared (DIST_INPUT_SHARED_FILESYSTEM, else "
            "DIST_SHARED_FILESYSTEM), so global rank 0 alone fetched for every node: a local checkpoint "
            "must sit on a mount every node sees, and a Hub snapshot in an HF_HOME every node shares. "
            "Otherwise declare per-node input with DIST_INPUT_SHARED_FILESYSTEM=0, so each node's local "
            "rank 0 fetches its own copy."
        )
    return (
        "The input filesystem is declared per-node, so this node's local rank 0 fetched into this cache; "
        "a local checkpoint must exist on every node."
    )
