"""The COLLECTIVE refusal of ranks that must hold the same rows of a dataset but loaded different ones."""

import hashlib

import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, LargeList, List

from halo.data.sources.dataset_cache import HALO_S3_DATASET_CACHE_DIR
from halo.log import KEY_PREVIEW_COUNT
from halo.runtime.collectives import combine_across_ranks
from halo.runtime.consensus import agree_across_ranks
from halo.runtime.ranks import get_global_world_size, get_local_world_size

# Rows per split hashed into the identity the holders of one replica compare: spread over the split,
# so a re-push that kept the row count still differs somewhere among them.
_IDENTITY_SAMPLE_ROWS = 16


def _stores_file_path(feature) -> bool:
    """Whether ``feature`` holds a media value (a storage struct carrying a ``path``) at any depth, decoded or
    not, which ``require_decoding`` cannot tell."""
    if isinstance(feature, dict):
        return any(_stores_file_path(sub) for sub in feature.values())
    if isinstance(feature, (list, tuple)):
        return _stores_file_path(feature[0])
    if isinstance(feature, (List, LargeList)):
        return _stores_file_path(feature.feature)
    storage = getattr(feature, "pa_type", None)
    return isinstance(storage, pa.StructType) and storage.get_field_index("path") >= 0


def _feed_values(hasher, array: pa.Array) -> None:
    """Feed ``array``'s logical values to ``hasher`` level by level (nulls, list lengths, leaves), so equal
    values feed equal bytes whatever the chunking or buffer layout of the copy they were read from."""
    if isinstance(array.type, pa.BaseExtensionType):
        array = array.storage
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    hasher.update(array.is_null().to_numpy(zero_copy_only=False).tobytes())
    kind = array.type
    if pa.types.is_struct(kind):
        for field in array.flatten():
            _feed_values(hasher, field)
    elif pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        hasher.update(array.value_lengths().fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64).tobytes())
        _feed_values(hasher, array.flatten())
    elif pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_boolean(kind):
        hasher.update(array.drop_null().to_numpy(zero_copy_only=False).tobytes())
    else:
        for value in array.drop_null():
            # Binary and string values hash their own buffer; the rare remaining types are small.
            payload = value.as_buffer() if hasattr(value, "as_buffer") else repr(value.as_py()).encode()
            hasher.update(len(payload).to_bytes(8, "little"))
            hasher.update(payload)


def _split_identity(split: Dataset) -> str:
    """Row count, schema and a digest of evenly spaced rows, what the holders of one replica compare. Media
    columns stay out, since their stored path can name a file under a node's own cache."""
    rows = len(split)
    columns = [name for name, feature in split.features.items() if not _stores_file_path(feature)]
    picks = (
        sorted({i * (rows - 1) // (_IDENTITY_SAMPLE_ROWS - 1) for i in range(_IDENTITY_SAMPLE_ROWS)})
        if rows and columns
        else []
    )
    hasher = hashlib.blake2b(str(split.features).encode(), digest_size=8)
    arrow_rows = split.with_format("arrow")
    for pick in picks:
        row = arrow_rows[pick]
        for name in columns:
            _feed_values(hasher, row.column(name).combine_chunks())
    return f"{rows} rows, content {hasher.hexdigest()}"


def _describe_divergence(gathered: list[tuple[int, dict[str, str]]], num_replicas: int, local_world: int) -> str:
    """The first replica whose holders loaded different rows, as the refusal names it; ``""`` when none did."""
    holders: dict[int, dict[str, list[int]]] = {}
    for rank, (rank_replica, rank_identity) in enumerate(gathered):
        holders.setdefault(rank_replica, {}).setdefault(repr(rank_identity), []).append(rank)
    divergent = [(held, versions) for held, versions in sorted(holders.items()) if len(versions) > 1]
    if not divergent:
        return ""
    held, versions = divergent[0]
    described = "; ".join(
        f"ranks {ranks[:KEY_PREVIEW_COUNT]}{'…' if len(ranks) > KEY_PREVIEW_COUNT else ''} "
        f"(node(s) {sorted({rank // local_world for rank in ranks})[:KEY_PREVIEW_COUNT]}) hold {version}"
        for version, ranks in sorted(versions.items(), key=lambda item: item[1][0])
    )
    scope = f" of data-parallel replica {held}" if num_replicas > 1 else ""
    return f"the ranks{scope} that must hold the same rows loaded different ones — {described}"


def reject_divergent_replicas(dataset: DatasetDict, path: str, *, replica: int, num_replicas: int) -> None:
    """COLLECTIVE: fail loud when ranks that must hold the same rows loaded different ones.

    A replicated load holds one copy of the rows on every rank (``num_replicas=1``), a sharded one one per
    data-parallel replica. Every rank reads its own copy of the source (a per-node S3 cache, a pre-staged
    directory, a cached hub revision), so one stale node trains other rows than its peers with nothing
    raised. The identity names every split, since the split set decides how many coordinated operations
    the consumers run. One all-reduce when every rank holds the same replica, else a rank-0 combine.
    """
    world = get_global_world_size()
    if world <= 1 or num_replicas >= world:
        return
    identity = {name: _split_identity(dataset[name]) for name in sorted(dataset)}
    if num_replicas == 1 and agree_across_ranks(identity).agreed:
        return
    local_world = get_local_world_size()
    divergence = combine_across_ranks(
        (replica, identity),
        lambda gathered: _describe_divergence(gathered, num_replicas, local_world),
        what="the dataset replica identities",
    )
    if divergence:
        raise ValueError(
            f"Dataset {path}: {divergence}. Each rank reads its own copy of the source, so a node serving a "
            f"stale one (S3 unreachable there with an old cache under {HALO_S3_DATASET_CACHE_DIR}, an older "
            f"pre-staged directory, a different cached hub revision) trains other rows than its peers. Re-sync "
            f"or clear the stale copy and relaunch."
        )
