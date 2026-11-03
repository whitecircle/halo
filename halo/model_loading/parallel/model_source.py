"""Where every rank reads a checkpoint from, agreed before any rank loads it, and the config and modality verdict read
ahead of a load. Each rank resolves the source on its own node, so the world agrees on what every rank found."""

from __future__ import annotations

import logging
import os
import socket
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from huggingface_hub import constants as hub_constants
from huggingface_hub import is_offline_mode, snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from transformers import AutoConfig, PretrainedConfig

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE, index_declares_ep_shards
from halo.checkpoint.reading import (
    ep_sharded_checkpoint_reason,
    has_whole_model_weight_file,
    read_checkpoint_index,
    resolve_checkpoint_weights,
)
from halo.log import KEY_PREVIEW_COUNT, warn_once
from halo.models.modality import is_vlm_model, is_vlm_name
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import (
    rank0_verdict,
    rank_consensus,
    reject_across_ranks,
    reject_divergent_settings,
    store_join_recorded_failure,
)
from halo.runtime.filesystem import (
    fs_aware_main_first,
    hub_metadata_main_first,
    is_fs_load_rank,
    is_input_shared_filesystem,
)

if TYPE_CHECKING:
    # Annotation only: the entry scripts' argument layer sits beside this package.
    from halo.configs.script_arguments import ModelArguments

logger = logging.getLogger(__name__)

# A model load fetches the repo's top-level files only (fnmatch's ``*`` crosses ``/``).
HUB_SUBFOLDER_IGNORE_PATTERNS = ("*/*",)
# What a rank that finds the source as a directory reports to the agreement, in place of a commit.
_LOCAL_DIRECTORY = "a local directory"
# transformers' gate refusing a config whose class is remote code the caller did not trust.
_REMOTE_CODE_GATE = "resolve_trust_remote_code"

# Warn once per model path, not once per call: the probe runs from several call sites per rank.
_name_heuristic_warned: set[str] = set()


@dataclass(frozen=True)
class ModelSource:
    """A checkpoint as every rank loads it: ``name`` as the run names it, at the agreed ``revision``,
    read from ``local_dir`` on this rank (the directory, or this rank's verified Hub snapshot)."""

    name: str
    revision: str | None
    local_dir: str


class _SourceView(NamedTuple):
    """One rank's resolution of the source."""

    # :data:`_LOCAL_DIRECTORY`, or the commit this rank's Hub cache holds.
    resolved: str
    # The snapshot directory this rank reads a Hub source from; None for a local directory.
    snapshot: str | None = None
    weight_files: tuple[str, ...] = ()


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


def _unresolvable_reason(model_name_or_path: str, revision: str | None, fetch_error: Exception | None) -> str:
    """Why this rank holds no copy: its failed fetch, the input-filesystem declaration, or an offline hub."""
    missing = (
        f"{model_name_or_path!r} is not a directory on {socket.gethostname()}, and this rank's Hub cache "
        f"({hub_constants.HF_HUB_CACHE}) holds no snapshot of it at revision {revision or 'main'}."
    )
    if is_offline_mode():
        return (
            f"{missing} The hub is offline (HF_HUB_OFFLINE), so no rank fetched: a local checkpoint must "
            "exist at this path on this node, and a Hub snapshot must be staged complete into this rank's cache."
        )
    if fetch_error is not None:
        return f"{missing} This rank fetched for its scope, and the fetch failed: {type(fetch_error).__name__}: {fetch_error}"
    return f"{missing} {_cache_remedy(fetched=False)}"


def _weight_files(snapshot: str) -> tuple[str, ...]:
    """The files a load reads a snapshot's weights from: the index and its shards, or the single weight file."""
    layout = resolve_checkpoint_weights(snapshot)
    index = (SAFETENSORS_INDEX_FILE,) if layout.index is not None else ()
    legacy = (layout.legacy_bin,) if layout.legacy_bin is not None else ()
    return (*index, *layout.shard_files, *legacy)


def _resolve_on_this_rank(
    model_name_or_path: str, revision: str | None, *, fetch: bool, whole_repo: bool
) -> _SourceView:
    """A local directory or this rank's cached snapshot; only a ``fetch`` rank downloads. Offline it reads the cache
    like every rank, as a call that may fetch lists a pinned commit through the Hub and raises even over a complete
    snapshot."""
    if os.path.isdir(model_name_or_path):
        return _SourceView(_LOCAL_DIRECTORY)
    try:
        snapshot = snapshot_download(
            model_name_or_path,
            revision=revision,
            local_files_only=not fetch or is_offline_mode(),
            ignore_patterns=None if whole_repo else list(HUB_SUBFOLDER_IGNORE_PATTERNS),
        )
    except (HFValidationError, LocalEntryNotFoundError) as exc:
        raise FileNotFoundError(_unresolvable_reason(model_name_or_path, revision, exc if fetch else None)) from exc
    return _SourceView(os.path.basename(snapshot), snapshot, _weight_files(snapshot))


def _reject_incomplete_snapshots(model_name_or_path: str, view: _SourceView, *, fetched: bool) -> None:
    """COLLECTIVE: fail every rank when a snapshot lacks a weight file rank 0's names. huggingface_hub's cache-only
    resolution does not check files, and the config and tokenizer reads leave a weightless snapshot in every node's
    cache."""
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


def resolve_model_source(
    model_name_or_path: str, revision: str | None, *, tag: str, whole_repo: bool = False
) -> ModelSource:
    """COLLECTIVE-EQUIVALENT (every rank, same ``tag``, equally often): fetch main-rank-first and return the source
    every rank loads. A Hub repo resolves to one commit every rank's cache holds, as caches staged apart would train
    different weights with no error; a local directory keeps ``revision``. ``whole_repo`` fetches subfolders too."""
    fetch = is_fs_load_rank()
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
        return ModelSource(model_name_or_path, revision, os.path.abspath(model_name_or_path))
    _reject_incomplete_snapshots(model_name_or_path, view, fetched=fetch)
    return ModelSource(model_name_or_path, view.resolved, view.snapshot)


def reject_ep_sharded_checkpoint(source: ModelSource) -> None:
    """COLLECTIVE (rank 0's probe): refuse a per-rank EP-sharded checkpoint, whose keys would otherwise match no expert
    pattern and surface as never materialized."""
    if rank0_verdict(
        lambda: index_declares_ep_shards(read_checkpoint_index(source.local_dir, missing_ok=True)),
        f"EP checkpoint-format probe of {source.name!r}",
    ):
        raise ValueError(ep_sharded_checkpoint_reason(source.local_dir))


def resolve_lazy_checkpoint(
    source: ModelSource, *, lazy_requested: bool, layout_supported: Callable[[str], bool]
) -> bool:
    """COLLECTIVE: refuse an EP-sharded ``source``, then return whether every rank streams it lazily. Rank 0 decides,
    as the branches enter different collectives and a per-rank probe of a partially populated cache would split
    them."""
    reject_ep_sharded_checkpoint(source)
    return broadcast_from_rank0(
        lazy_requested
        and has_whole_model_weight_file(source.local_dir, safetensors_only=True)
        and layout_supported(source.local_dir)
    )


def fetch_checkpoint_config(model_config: ModelArguments, *, tag: str) -> PretrainedConfig:
    """The run's checkpoint config at its pinned revision, read main-rank-first under ``tag``
    (:func:`~halo.runtime.filesystem.hub_metadata_main_first`), ahead of the source agreement."""
    return hub_metadata_main_first(
        tag,
        lambda local_files_only: AutoConfig.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            trust_remote_code=model_config.trust_remote_code,
            local_files_only=local_files_only,
        ),
    )


def _probe_checkpoint_config(
    model_name_or_path: str, revision: str | None, trust_remote_code: bool, *, local_files_only: bool
):
    """The config for the modality verdict, ``None`` when the hub read fails; a remote-code config under
    ``trust_remote_code: false`` is refused. Only the hub read is caught, as a coordination failure must not become a
    heuristic."""
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # unreachable or missing config: fall back to the name heuristic below
        if any(frame.name == _REMOTE_CODE_GATE for frame in traceback.extract_tb(exc.__traceback__)):
            raise ValueError(
                f"'{model_name_or_path}' defines its architecture in remote code, which this run does not "
                f"trust (trust_remote_code: false), so the model load would refuse it. Set "
                f"trust_remote_code: true to run that code."
            ) from exc
        # The fallback can disagree with the config answer, so a rank that missed the hub says so.
        warn_once(
            logger,
            _name_heuristic_warned,
            model_name_or_path,
            f"Could not load a config for '{model_name_or_path}' ({type(exc).__name__}: {exc}); "
            "falling back to the model-name heuristic to decide text-vs-multimodal. A verdict that "
            "differs across ranks fails the probe on every rank.",
        )
        return None


def probe_checkpoint(
    model_name_or_path: str, revision: str | None = None, *, trust_remote_code: bool = False
) -> tuple[Any, bool]:
    """The checkpoint's config and its modality verdict, from one hub read; COLLECTIVE once the process group exists. A
    verdict split across ranks raises everywhere, as ranks would build different model classes and hang in different
    store phases."""
    config = hub_metadata_main_first(
        "vlm_probe",
        lambda local_files_only: _probe_checkpoint_config(
            model_name_or_path, revision, trust_remote_code, local_files_only=local_files_only
        ),
    )
    verdict = is_vlm_model(model_name_or_path, config) if config is not None else is_vlm_name(model_name_or_path)
    vlm = rank_consensus(verdict)
    if vlm.any_ok and not vlm.all_ok:
        source = "read its config" if config is not None else "could not read its config and fell back to its name"
        raise RuntimeError(
            f"'{model_name_or_path}' probes as multimodal on some ranks and text-only on others (this "
            f"rank {source}: {'multimodal' if verdict else 'text-only'}), so the ranks would build "
            f"different model classes. Make the same checkpoint readable on every node — a local "
            f"directory on a mount every node sees, or a Hub snapshot in every node's cache — and pin "
            f"model_revision so per-node caches cannot hold different commits."
        )
    return config, verdict
