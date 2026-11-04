"""Dataset loading from S3, the Hub or local disk into the train/test ``DatasetDict`` a trainer reads."""

import hashlib
import pickle

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from halo.data.pipeline.coordinated_map import run_load_rank_first
from halo.data.pipeline.map_cache_key import stamp_cache_key
from halo.data.preprocessed.metadata import is_preprocessed_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE
from halo.data.render_columns import drop_empty_conversation_rows, reject_missing_render_column
from halo.data.sources.corpus import build_corpus, placeholder_test_split, resolve_dataset_ratios, subset_dataset_dict
from halo.data.sources.paths import (
    DATA_FILE_BUILDERS,
    METADATA_FILE,
    TRAIN_TEST_SPLITS,
    data_file_builder,
    eval_split_name,
    hub_repo_id,
    parse_dataset_source,
    parse_hub_spec,
    reject_ambiguous_hub_source,
    reject_partial_splits,
    s3_object_key,
)
from halo.data.sources.replica_identity import reject_divergent_replicas
from halo.data.sources.s3_dataset_store import load_dataset_from_s3_uri, share_control_files
from halo.data.sources.sharded_dataset import ShardedDatasetLoader
from halo.log import info_logger
from halo.runtime.collectives import combine_across_ranks
from halo.runtime.consensus import (
    agree_probe_across_ranks,
    rank_consensus,
    reject_divergent_settings,
    store_join_recorded_failure,
)
from halo.runtime.filesystem import agree_input_probe_across_ranks, is_fs_load_rank, is_input_shared_filesystem
from halo.runtime.ranks import get_node_rank, is_global_main_process

# INFO: this module's lines are the run's only record of what data actually loaded.
logger = info_logger(__name__)

# Fixed so every rank derives the same train/test split independently of the caller's data seed.
_TRAIN_TEST_SPLIT_SEED = 42

_DEFAULT_DATA_SEED = 42

# Separates the ``conversation`` fallback from a declared column, the only one worth failing on.
_UNDECLARED = object()


def load_dataset_from_source(path: str) -> Dataset | DatasetDict:
    """S3 URI, Hub (``id:config`` suffix) or local path, auto-detected."""
    source_type, _bucket, key = parse_dataset_source(path)

    if source_type == "s3":
        if is_global_main_process():
            logger.info(f"Loading dataset from S3: {path}")
        return load_dataset_from_s3_uri(path, keep_in_memory=False)

    elif source_type == "hf_hub":
        reject_ambiguous_hub_source(path)
        # @split picks a split when there is no standard "train" (e.g. @train_sft).
        path, config_name, split_name = parse_hub_spec(path)
        if is_global_main_process():
            logger.info(
                f"Loading dataset from HuggingFace Hub: {path}"
                + (f" (config={config_name})" if config_name else "")
                + (f" (split={split_name})" if split_name else "")
            )
        try:
            return load_dataset(path, config_name, split=split_name, keep_in_memory=False)
        except (FileNotFoundError, OSError, ValueError) as e:
            raise FileNotFoundError(
                f"Could not load dataset '{path}' from HuggingFace Hub. "
                f"Make sure the dataset exists and you have access. Error: {e}"
            ) from e

    elif source_type == "local":
        builder = data_file_builder(key)
        if is_global_main_process():
            logger.info(f"Loading dataset from local {'file' if builder else 'path'}: {key}")
        try:
            if builder is not None:
                return load_dataset(builder, data_files=key, keep_in_memory=False)
            return load_from_disk(key, keep_in_memory=False)
        except (FileNotFoundError, OSError, ValueError) as e:
            raise FileNotFoundError(
                f"Could not load dataset from local path '{path}'. A path ending in "
                f"{'/'.join(DATA_FILE_BUILDERS)} is read as a data file; anything else must be a "
                f"directory written by ``save_to_disk``. Error: {e}"
            ) from e

    else:
        raise ValueError(f"Unknown dataset source type: {source_type} for path: {path}")


def load_dataset_split(source: str, *, split: str, config: str | None = None) -> Dataset:
    """One split: a Hub id read at ``config`` and ``split``, which replace a ``:config`` / ``@split`` suffix; any other
    source as :func:`load_dataset_from_source`, a bare Dataset or data file being its one split and a DatasetDict
    having to hold ``split``."""
    source_type, _bucket, path = parse_dataset_source(source)
    if source_type == "hf_hub":
        reject_ambiguous_hub_source(source)
        if hub_repo_id(source) != source:
            raise ValueError(
                f"Dataset {source!r} carries a ':config' or '@split' suffix; pass the bare 'org/name' and name the "
                f"config and split through their own arguments (--config / --split)."
            )
        return load_dataset(source, config, split=split)
    loaded = load_dataset_from_source(source)
    if isinstance(loaded, Dataset):
        return loaded
    if data_file_builder(path) is not None:
        return loaded["train"]
    if split not in loaded:
        raise ValueError(f"split {split!r} is not in {source}, which holds {list(loaded)}")
    return loaded[split]


def _warn_test_size_ignored(path: str, test_size: float | None, kind: str) -> None:
    """``test_size`` cannot re-split a sharded or pre-processed dataset, whose split was fixed at preparation."""
    if test_size is None:
        return
    if is_global_main_process():
        logger.warning(
            f"test_size={test_size} is ignored for the {kind} dataset at {path}: its train/test split "
            f"was fixed when the dataset was prepared. Re-prepare it with "
            f"scripts/before_training/prepare_dataset.py --test_size to change the split."
        )


def _load_whole_dataset(path: str, test_size: float | None, *, placeholder_test: bool) -> DatasetDict:
    dataset = load_dataset_from_source(path)

    if isinstance(dataset, Dataset):
        if test_size is not None:
            return dataset.train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
        if placeholder_test:
            return placeholder_test_split(dataset, path)
        return DatasetDict({"train": dataset})

    if "train" not in dataset:
        available = [str(s) for s in dataset]
        train_like = [s for s in available if s.startswith("train")]
        suggestion = (train_like or available)[0]
        raise ValueError(
            f"Dataset {path} has no 'train' split (available splits: {available}). "
            f"Append an '@split' selector to pick one, e.g. '{path}@{suggestion}'."
        )
    held_out = eval_split_name(dataset)
    if held_out not in (None, "test"):
        if is_global_main_process():
            logger.info(f"Dataset {path} has no test split; its {held_out!r} split is the test split")
        dataset["test"] = dataset.pop(held_out)
    if "test" not in dataset:
        if test_size is not None:
            return dataset["train"].train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
        if placeholder_test:
            return placeholder_test_split(dataset["train"], path)
    elif test_size is not None:
        combined = concatenate_datasets([dataset["train"], dataset["test"]])
        if is_global_main_process():
            logger.info(f"Re-split dataset {path} with test_size={test_size}")
        return combined.train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
    return dataset


def _share_within_input_scope(loaded: tuple[DatasetDict, bytes] | None, path: str) -> DatasetDict:
    """COLLECTIVE: the load rank's DatasetDict on every rank of its input scope. The pickle of memory-mapped tables
    holds paths under ``HF_DATASETS_CACHE``, whose sharing the startup probe verified, so peers open its files."""
    loading_rank = is_fs_load_rank()
    scope = None if is_input_shared_filesystem() else get_node_rank()
    by_scope = combine_across_ranks(
        (scope, loaded[1]) if loading_rank else None,
        lambda gathered: dict(entry for entry in gathered if entry is not None),
        what=f"the loads of dataset {path}",
    )
    dataset, failure = (loaded[0], None) if loading_rank else (None, None)
    if not loading_rank:
        try:
            dataset = pickle.loads(by_scope[scope])
        except BaseException as e:  # joined below, so every rank raises with it
            failure = e
    store_join_recorded_failure("shared_dataset_load", failure, f"Opening the load rank's copy of dataset {path}")
    return dataset


def _load_replicated(path: str, test_size: float | None, *, placeholder_test: bool) -> DatasetDict:
    """Loaded whole on every rank (the first fills the cache) and agreed. A Hub source loads on each scope's load rank
    alone, which hands its copy over: the others would resolve it online again, about ten requests each."""

    def load() -> DatasetDict:
        return _load_whole_dataset(path, test_size, placeholder_test=placeholder_test)

    def load_and_pickle() -> tuple[DatasetDict, bytes] | None:
        if not is_fs_load_rank():
            return None
        dataset = load()
        return dataset, pickle.dumps(dataset)

    operation_name = f"loading dataset {path}"
    if parse_dataset_source(path)[0] == "hf_hub":
        dataset = _share_within_input_scope(run_load_rank_first(load_and_pickle, operation_name), path)
    else:
        dataset = run_load_rank_first(load, operation_name)
    reject_divergent_replicas(dataset, path, replica=0, num_replicas=1)
    return dataset


def _load_sharded(
    path: str, data_parallel_rank: int, data_parallel_size: int
) -> tuple[DatasetDict, ShardedDatasetLoader]:
    """COLLECTIVE: joined on the c10d store, so a rank-local failure reaches every rank as the real cause and disjoint
    shards' downloads are absorbed under ``DIST_STORE_TIMEOUT_HOURS``, not the next collective's NCCL budget."""
    if is_global_main_process():
        logger.info(f"Loading sharded dataset from {path}")
        logger.info(f"  Data parallel rank: {data_parallel_rank}/{data_parallel_size}")
    loader = ShardedDatasetLoader(dataset_path=path, dp_rank=data_parallel_rank, dp_size=data_parallel_size)
    result, failure = None, None
    try:
        result = loader.load()
    except BaseException as e:  # a KeyboardInterrupt must reach the join too, or peers hang
        failure = e
    store_join_recorded_failure("sharded_dataset_load", failure, "Sharded dataset load")
    # Keyed by split, so a split whose index only some ranks found diverges here as well.
    reject_divergent_settings(
        loader.shard_index_digests(),
        f"The shard index of {loader.dataset_path}",
        "Every rank assigns its shards off its own read of each split's shard_index.json, so ranks "
        "reading different ones (a stale per-node copy or control-file mirror, a re-prepare racing "
        "the launch) load overlapping or missing shards.",
    )
    reject_divergent_replicas(result, path, replica=data_parallel_rank, num_replicas=data_parallel_size)
    return result, loader


def is_sharded_dataset_across_ranks(path) -> bool:
    """COLLECTIVE: the per-rank S3 probe only errs toward False (a creds flake), so the agreed MAX recovers it."""
    local = isinstance(path, str) and ShardedDatasetLoader.is_sharded_dataset(path)
    return agree_probe_across_ranks(local, path, "is_sharded_dataset")


def _load_dataset_from_path(
    path: str,
    test_size: float | None,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    *,
    placeholder_test: bool = True,
) -> tuple[DatasetDict, bool]:
    """``(dataset, sharded)``: this rank's shards, else the whole source with its test split (validation, a
    ``test_size`` re-split or a placeholder). ``placeholder_test=False`` leaves a train-only source train-only, for a
    corpus deciding its held-out split."""
    # No dp_size gate: a sharded layout has no root dataset_info.json, so a fall-through load crashes at dp==1.
    if is_sharded_dataset_across_ranks(path):
        _warn_test_size_ignored(path, test_size, kind="sharded")
        return _load_sharded(path, data_parallel_rank, data_parallel_size)[0], True
    return _load_replicated(path, test_size, placeholder_test=placeholder_test), False


def _content_signature(dataset: DatasetDict) -> str:
    """Content-derived until a coordinated map diverges it per rank, so a re-push with identical shapes still changes
    the forced cache keys."""
    return ",".join(f"{name}:{dataset[name]._fingerprint}" for name in sorted(dataset))


def _load_single(
    path: str,
    test_size: float | None,
    ratio: float,
    seed: int,
    data_parallel_rank: int,
    data_parallel_size: int,
    *,
    conversation_field: str,
    conversation_declared: bool,
    conversation_knob: str,
    tools_field: str | None,
) -> tuple[DatasetDict, bool, str]:
    """A checked, filtered and subset ``DatasetDict``, whether it loaded sharded, and its pre-filter signature."""
    ds, sharded = _load_dataset_from_path(path, test_size, data_parallel_rank, data_parallel_size)
    # Only a raw sharded load can return a partial DatasetDict (missing splits are skipped).
    reject_partial_splits(ds, path)
    content_sig = _content_signature(ds)
    drop_empty_conversation_rows(ds, path, conversation_field, required=conversation_declared, knob=conversation_knob)
    if tools_field:
        # An optional knob's typo never surfaces downstream: rows just render without tools.
        reject_missing_render_column(ds, path, "tools_field", tools_field)
    return subset_dataset_dict(ds, ratio, seed), sharded, content_sig


def _reject_empty_splits(ds: DatasetDict, sharded: bool) -> None:
    """COLLECTIVE when ``sharded``: an empty train slice on any rank raises on every rank, while the test split raises
    only when empty on all."""
    train_empty = len(ds["train"]) == 0
    if sharded:
        # Subsetting can floor one rank's slice to zero rows; peers must raise with it, not block in the test verdict.
        train_empty = not rank_consensus(not train_empty).all_ok
    if train_empty:
        raise ValueError("No training data after schema normalization. Please check your dataset.")
    if is_global_main_process():
        logger.info(f"Training data after schema normalization: {len(ds['train'])}")
        logger.info(f"Columns in training dataset: {ds['train'].column_names}")

    test_empty = len(ds["test"]) == 0
    if sharded:
        # A rank-local empty eval split is the trainer's equalization's to refuse; raising alone here strands peers.
        test_empty = rank_consensus(test_empty).all_ok
    if test_empty:
        raise ValueError("No test data after schema normalization. Please check your dataset.")
    if is_global_main_process():
        logger.info(f"Test data after schema normalization: {len(ds['test'])}")
        logger.info(f"Columns in test dataset: {ds['test'].column_names}")


def load_datasets(
    path: str | list,
    test_size: float | None,
    dataset_ratio: float | list | None,
    conversation_field: str | None = _UNDECLARED,
    seed: int = _DEFAULT_DATA_SEED,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    tools_field: str | None = None,
    conversation_knob: str = "conversation_field",
):
    """A declared ``conversation_field`` must exist; the refusal names ``conversation_knob``, the config field the
    caller read it from (``prompt_field`` on GRPO prompt scripts)."""
    dataset_ratio = resolve_dataset_ratios(path, dataset_ratio)
    conversation_declared = conversation_field is not _UNDECLARED
    if not conversation_declared:
        conversation_field = "conversation"
    render = {
        "conversation_field": conversation_field,
        "conversation_declared": conversation_declared,
        "conversation_knob": conversation_knob,
        "tools_field": tools_field,
    }
    # No outer barrier: _load_dataset_from_path coordinates the downloads and the rest is rank-deterministic.
    if isinstance(path, list):
        # Each entry loads fully: the list is sharded once as a whole, so a DP slice of a sharded entry loses data.
        loaded = [_load_dataset_from_path(entry, test_size, placeholder_test=False) for entry in path]
        content_sig = ",".join(_content_signature(entry_ds) for entry_ds, _ in loaded)
        ds, sharded = build_corpus(path, loaded, dataset_ratio, seed, **render), False
    else:
        ds, sharded, content_sig = _load_single(
            path, test_size, dataset_ratio, seed, data_parallel_rank, data_parallel_size, **render
        )
    _reject_empty_splits(ds, sharded)

    # HF's own fingerprints diverge between writer and loader ranks, breaking downstream cache keys.
    path_repr = ",".join(path) if isinstance(path, list) else str(path)
    ratio_repr = ",".join([str(r) for r in dataset_ratio]) if isinstance(dataset_ratio, list) else str(dataset_ratio)
    meta = (
        f"{path_repr}|{test_size}|{ratio_repr}|{seed}|{len(ds['train'])}|{len(ds['test'])}"
        f"|{conversation_field}|{content_sig}"
    )
    # Equal-length shards would otherwise stamp identical keys and load rank 0's mapped shard.
    if sharded:
        meta += f"|dp{data_parallel_rank}/{data_parallel_size}"
    base_fp = hashlib.md5(meta.encode()).hexdigest()
    ds["train"]._fingerprint = f"{base_fp}-train"
    ds["test"]._fingerprint = f"{base_fp}-test"
    stamp_cache_key(ds["train"], f"loaded_{base_fp}_train")
    stamp_cache_key(ds["test"], f"loaded_{base_fp}_test")

    return ds


def share_s3_control_files(path: str | list) -> None:
    """COLLECTIVE: read each S3 source's control files once for the launch; later probes read none on other ranks."""
    for entry in path if isinstance(path, list) else [path]:
        source_type, bucket, key = parse_dataset_source(entry)
        if source_type == "s3":
            control_keys = [
                s3_object_key(key, METADATA_FILE),
                *(s3_object_key(key, split, SHARD_INDEX_FILE) for split in TRAIN_TEST_SPLITS),
            ]
            share_control_files(bucket, control_keys)


def is_presharded_dataset_load_across_ranks(path, data_parallel_size: int) -> bool:
    """COLLECTIVE: whether ``path`` gives each of several DP ranks a disjoint slice, which trainers pass as
    ``dataset_presharded`` so the DataLoader does not re-shard."""
    return data_parallel_size > 1 and isinstance(path, str) and is_sharded_dataset_across_ranks(path)


def load_preprocessed_dataset(
    path: str,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
) -> DatasetDict:
    """Pass the DATA-PARALLEL rank/size: CP/TP-group siblings share data, EP ranks (orthogonal to DP) do not."""
    if parse_dataset_source(path)[0] == "hf_hub":
        reject_ambiguous_hub_source(path)
        # Refused on every rank alike. ``load_dataset`` infers the Json builder from the saved tree's ``state.json``
        # and would return those as rows.
        raise ValueError(
            f"Pre-processed dataset {path} is on the Hub, which training does not read in place. "
            f"Download it (hf download {hub_repo_id(path)} --repo-type dataset --local-dir <dir>) "
            f"and point dataset at <dir>."
        )

    if is_sharded_dataset_across_ranks(path):
        ds, loader = _load_sharded(path, data_parallel_rank, data_parallel_size)
        # The shard index gives every rank the same global verdict; per-rank emptiness is the trainer's.
        totals = {split: loader.total_examples(split) for split in TRAIN_TEST_SPLITS if split in ds}
    else:
        ds = _load_replicated(path, test_size=None, placeholder_test=True)
        # Non-sharded data is replica-identical, so a local empty split is globally empty.
        totals = {split: len(ds[split]) for split in TRAIN_TEST_SPLITS if split in ds}

    # The sharded loader skips missing splits; without this the consumer dies on ds["test"] later.
    reject_partial_splits(ds, path)
    empty = sorted(split for split, total in totals.items() if total == 0)
    if empty:
        raise ValueError(
            f"No data in the {empty} split(s) of the pre-processed dataset at {path}: they exist but "
            f"hold zero examples. Re-preprocess with non-empty splits (check test_size and the "
            f"preprocessing filters)."
        )
    return ds


def load_datasets_auto(
    path: str | list,
    test_size: float | None,
    dataset_ratio: float | list | None,
    conversation_field: str | None = _UNDECLARED,
    seed: int = _DEFAULT_DATA_SEED,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    tools_field: str | None = None,
) -> tuple[DatasetDict, bool]:
    """``(DatasetDict, is_preprocessed)``. Pass the DATA-PARALLEL rank/size, so CP/TP-group siblings share shards."""
    # Cross-rank agreed: a transient S3 error must not split ranks onto the raw and preprocessed paths.
    preprocessed = isinstance(path, str) and agree_input_probe_across_ranks(
        lambda: is_preprocessed_dataset(path), path, "is_preprocessed_dataset"
    )
    if preprocessed:
        if is_global_main_process():
            logger.info(f"Detected pre-processed dataset at {path}")
        _warn_test_size_ignored(path, test_size, kind="pre-processed")
        if resolve_dataset_ratios(path, dataset_ratio) != 1:
            raise ValueError(
                f"dataset_ratio={dataset_ratio} cannot apply to the pre-processed dataset at {path}: its rows were "
                f"fixed when the dataset was prepared, and the run would train all of them. Unset dataset_ratio, or "
                f"subset the source before preparing it."
            )
        return load_preprocessed_dataset(path, data_parallel_rank, data_parallel_size), True

    ds = load_datasets(
        path=path,
        test_size=test_size,
        dataset_ratio=dataset_ratio,
        conversation_field=conversation_field,
        seed=seed,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        tools_field=tools_field,
    )
    return ds, False
