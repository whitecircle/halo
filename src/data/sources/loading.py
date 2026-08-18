"""Dataset loading from S3, HuggingFace Hub, or local disk, with split/subset/sharding helpers."""

import hashlib

import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from src.data.pipeline.preprocessed_metadata import is_preprocessed_dataset
from src.data.pipeline.processing import coordinated_filter, missing_render_column_splits, require_render_column
from src.data.probe_consensus import agree_probe_across_ranks
from src.data.sources.paths import DATA_FILE_BUILDERS, parse_dataset_source, parse_hub_spec
from src.data.sources.s3_client import load_dataset_from_s3_uri
from src.data.sources.sharded_dataset import ShardedDatasetLoader
from src.data.vlm import VLM_IMAGE_COLUMNS, carried_image_columns
from src.distributed.filesystem import fs_aware_main_first, store_join_recorded_failure
from src.distributed.runtime import current_device, get_global_world_size

# INFO opt-in (the convention vlm_setup and the training scripts use): this module's INFO lines are
# the run's only record of what data actually loaded (columns kept, split sizes, source dispatch).
logger = get_logger(__name__, log_level="INFO")

_PLACEHOLDER_TEST_ROWS = 100

# Fixed so every rank derives the same train/test split independently of the caller's data seed.
_TRAIN_TEST_SPLIT_SEED = 42

# Default for the caller-facing ``seed`` (ratio-subsetting), shared so the entry points and the
# subset helpers cannot drift onto different defaults.
_DEFAULT_DATA_SEED = 42

# ``load_datasets``' conversation-column fallback, and the sentinel separating it from a column the
# caller declared. Only a declared column is a render contract worth failing on: the
# preference/reward/classification scripts pass no field and carry no such column.
_FALLBACK_CONVERSATION_FIELD = "conversation"
_UNDECLARED = object()

# The image-column spellings a vision route hard-codes: TRL's preference trainers read them off the
# first sample to pick the vision branch, and list only these two among their signature columns. A
# run that declares its images elsewhere is aliased onto the first (:func:`alias_images_column`).
_VISION_ROUTE_COLUMN = "images"
_VISION_ROUTE_COLUMNS = (_VISION_ROUTE_COLUMN, "image")
# The tools column TRL's RewardTrainer hands to the chat template (:func:`alias_tools_column`).
_TOOLS_ROUTE_COLUMN = "tools"

# Columns whose loss changes what a run trains on: the row fields of the supported dataset formats
# and the tokenized triple a prepared dataset carries. The run's own declared render columns are
# threaded in by the caller and pinned through the concatenation rather than only warned about.
_ESSENTIAL_COLUMNS = (
    _FALLBACK_CONVERSATION_FIELD,
    "prompt",
    "target",
    "chosen",
    "rejected",
    "completion",
    "completions",
    "rewards",
    "label",
    "input_ids",
    "attention_mask",
    "labels",
    *VLM_IMAGE_COLUMNS,
)

__all__ = [
    "alias_images_column",
    "alias_tools_column",
    "is_presharded_dataset_load",
    "is_sharded_dataset_coordinated",
    "load_dataset_from_source",
    "load_datasets",
    "load_datasets_auto",
    "load_preprocessed_dataset",
    "reject_image_columns",
]


def load_dataset_from_source(path: str) -> Dataset | DatasetDict:
    """Load a dataset from S3 URI, HuggingFace Hub (supports ``id:config`` suffix), or local path (auto-detected)."""
    source_type, _bucket, _key = parse_dataset_source(path)

    if source_type == "s3":
        logger.info(f"Loading dataset from S3: {path}")
        return load_dataset_from_s3_uri(path, keep_in_memory=False)

    elif source_type == "hf_hub":
        # @split picks a split when there is no standard "train" (e.g. @train_sft).
        path, config_name, split_name = parse_hub_spec(path)
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
        builder = next((b for ext, b in DATA_FILE_BUILDERS.items() if path.endswith(ext)), None)
        logger.info(f"Loading dataset from local {'file' if builder else 'path'}: {path}")
        try:
            if builder is not None:
                return load_dataset(builder, data_files=path, keep_in_memory=False)
            return load_from_disk(path, keep_in_memory=False)
        except (FileNotFoundError, OSError, ValueError) as e:
            raise FileNotFoundError(
                f"Could not load dataset from local path '{path}'. A path ending in "
                f"{'/'.join(DATA_FILE_BUILDERS)} is read as a data file; anything else must be a "
                f"directory written by ``save_to_disk``. Error: {e}"
            ) from e

    else:
        raise ValueError(f"Unknown dataset source type: {source_type} for path: {path}")


def _require_train_test_splits(ds: DatasetDict, path: str) -> None:
    """Raise on a partial DatasetDict: sharded/preprocessed loads skip missing splits, and the
    trainers index ``ds["test"]`` unconditionally, giving a ``KeyError`` far from the cause."""
    missing = [split for split in ("train", "test") if split not in ds]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing the {missing} split(s). Trainers require both 'train' and "
            f"'test' — re-prepare the dataset with a test split "
            f"(scripts/before_training/prepare_dataset.py writes one when --test-size is set)."
        )


def _warn_test_size_ignored(path: str, test_size: float | None, kind: str) -> None:
    """Warn that ``test_size`` cannot re-split an already-prepared dataset.

    It re-splits only a dataset loaded whole; a sharded or pre-processed one carries the split that
    was decided at preparation time, so a ``test_size`` in the training YAML does nothing. Silent
    when unset, so a normal run does not warn.
    """
    if test_size is None:
        return
    logger.warning(
        f"test_size={test_size} is ignored for the {kind} dataset at {path}: its train/test split "
        f"was fixed when the dataset was prepared. Re-prepare it with "
        f"scripts/before_training/prepare_dataset.py --test-size to change the split."
    )


def _placeholder_test_split(train_ds: Dataset, path: str) -> DatasetDict:
    """Wrap a train-only dataset with a placeholder test split (first ``_PLACEHOLDER_TEST_ROWS`` rows),
    so downstream eval always has a non-empty test split. Used when a source has no test split.
    """
    logger.warning(f"Dataset {path} has no test split, using first {_PLACEHOLDER_TEST_ROWS} samples as test")
    return DatasetDict({"train": train_ds, "test": train_ds.select(range(min(_PLACEHOLDER_TEST_ROWS, len(train_ds))))})


def _sharded_load_with_join(loader: ShardedDatasetLoader) -> DatasetDict:
    """This rank's shard load, joined across the world on the c10d store.

    A rank-local failure (a cold per-shard cache during an S3 outage, ENOSPC on one volume) reaches
    every rank as the real cause instead of blocking the peers, and the widely varying first-run
    download times of disjoint shards are absorbed under ``DIST_STORE_TIMEOUT_HOURS`` rather than the
    next collective's NCCL budget. Collective-equivalent: the sharded verdict is agreed beforehand.
    """
    result, failure = None, None
    try:
        result = loader.load()
    except BaseException as e:  # a KeyboardInterrupt must reach the join too, or peers hang
        failure = e
    store_join_recorded_failure("sharded_dataset_load", failure, "Sharded dataset load")
    _reject_divergent_split_presence(result, loader.dataset_path)
    return result


def _reject_divergent_split_presence(dataset: DatasetDict, path: str) -> None:
    """Raise when the set of splits a load produced differs across ranks.

    Which splits a source yields drives how many coordinated dataset operations its consumers run, so
    a rank that reads a different set desynchronizes the store phases its peers enter.
    ``ShardedDatasetLoader.load`` skips a split whose index it cannot find, and a half-populated cache
    directory does the same to a plain load. Verdict and refusal are both world-agreed.
    """
    if get_global_world_size() <= 1:
        return
    splits = ("train", "test")
    present = torch.tensor([1 if name in dataset else 0 for name in splits], device=current_device())
    anywhere, everywhere = present.clone(), present.clone()
    dist.all_reduce(anywhere, op=dist.ReduceOp.MAX)
    dist.all_reduce(everywhere, op=dist.ReduceOp.MIN)
    diverged = [name for name, hi, lo in zip(splits, anywhere.tolist(), everywhere.tolist(), strict=True) if hi != lo]
    if diverged:
        raise ValueError(
            f"Dataset {path} loaded split(s) {diverged} on some ranks but not others — the per-rank "
            f"read disagreed (a transient S3 credential/throttling fault, or a half-populated cache "
            f"directory, is the usual cause). Ranks would then run different numbers of coordinated "
            f"dataset operations. Re-launch, or pre-cache the dataset so every rank reads the same source."
        )


def _load_dataset_from_path(
    path: str,
    test_size: float | None,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    *,
    placeholder_test: bool = True,
) -> tuple[DatasetDict, bool]:
    """Load a dataset and ensure train/test splits; returns ``(dataset, sharded)``.

    For sharded datasets each rank loads only its assigned shards when data_parallel_size > 1
    (``sharded=True``, each DP rank then holding a rank-specific slice). Without a test split:
    re-split train by test_size, else use the first 100 train rows as test.
    ``placeholder_test=False`` returns a train-only source as a train-only ``DatasetDict`` instead.
    """
    # Cross-rank-agreed probe: a per-rank S3-creds fault must not split ranks onto different data
    # paths. No dp_size gate: a sharded layout has no root dataset_info.json, so a fall-through load
    # fails at dp==1.
    if is_sharded_dataset_coordinated(path):
        logger.info(f"Loading sharded dataset from {path}")
        logger.info(f"  Data parallel rank: {data_parallel_rank}/{data_parallel_size}")
        # Every sharded load passes through here, single path and list entry alike.
        _warn_test_size_ignored(path, test_size, kind="sharded")
        loader = ShardedDatasetLoader(
            dataset_path=path,
            global_rank=data_parallel_rank,
            world_size=data_parallel_size,
        )
        return _sharded_load_with_join(loader), True

    with fs_aware_main_first("dataset_load"):
        dataset = load_dataset_from_source(path)

        if isinstance(dataset, Dataset):
            if test_size is not None:
                dataset = dataset.train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
            elif placeholder_test:
                dataset = _placeholder_test_split(dataset, path)
            else:
                dataset = DatasetDict({"train": dataset})
        elif isinstance(dataset, DatasetDict):
            if "train" not in dataset:
                available = [str(s) for s in dataset]
                train_like = [s for s in available if s.startswith("train")]
                suggestion = (train_like or available)[0]
                raise ValueError(
                    f"Dataset {path} has no 'train' split (available splits: {available}). "
                    f"Append an '@split' selector to pick one, e.g. '{path}@{suggestion}'."
                )
            if "test" not in dataset:
                if test_size is not None:
                    dataset = dataset["train"].train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
                elif placeholder_test:
                    dataset = _placeholder_test_split(dataset["train"], path)
            elif test_size is not None:
                combined = concatenate_datasets([dataset["train"], dataset["test"]])
                dataset = combined.train_test_split(test_size, seed=_TRAIN_TEST_SPLIT_SEED)
                logger.info(f"Re-split dataset {path} with test_size={test_size}")

    return dataset, False


def _content_signature(dataset: DatasetDict) -> str:
    """Content-derived, process-stable signature of a freshly loaded DatasetDict.

    The raw per-split ``_fingerprint``, read before any coordinated map/filter, whose writer and
    loader fingerprints diverge. At load time it is content-derived and stable within a filesystem
    domain, so a re-push with identical shapes still changes it and the forced cache keys built from
    it cannot serve stale mapped rows.
    """
    return ",".join(f"{name}:{getattr(dataset[name], '_fingerprint', None) or 'nofp'}" for name in sorted(dataset))


def _get_subset_from_dataset(dataset: Dataset, dataset_ratio: float | None, seed: int = _DEFAULT_DATA_SEED) -> Dataset:
    """Subset the dataset to dataset_ratio (0.0-1.0). Seeded for cross-rank reproducibility."""
    if dataset_ratio is None or abs(dataset_ratio - 1) <= 0.001:
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), int(dataset_ratio * len(dataset)), replace=False)
    dataset = dataset.select(indices)
    return dataset


def _get_subset_from_dataset_dict(
    dataset: DatasetDict, dataset_ratio: float | None, seed: int = _DEFAULT_DATA_SEED
) -> DatasetDict:
    """Subset both train and test to dataset_ratio (test uses seed+1 to avoid overlap).

    A corpus entry that shipped no test split stays train-only; the ratio applies to what is there.
    """
    dataset["train"] = _get_subset_from_dataset(dataset["train"], dataset_ratio, seed)
    if "test" in dataset:
        dataset["test"] = _get_subset_from_dataset(
            dataset["test"], dataset_ratio, seed + 1
        )  # Different seed for test to avoid overlap
    return dataset


def _filter_empty_conversations(dataset: Dataset, conversation_field: str, split_name: str = "") -> Dataset:
    """Filter out rows where the conversation field is None or empty.

    ``split_name`` is threaded into the cache desc so train/test caches do not collide. No-op when
    the field is absent, since filtering on a non-existent column would drop every row.
    """
    if conversation_field not in dataset.column_names:
        return dataset

    def has_valid_conversation(example):
        return (
            conversation_field in example
            and example[conversation_field] is not None
            and len(example[conversation_field]) > 0
        )

    # num_proc=None avoids spawning processes during the distributed loading phase.
    desc = f"filtering empty {conversation_field}"
    if split_name:
        desc = f"{desc} ({split_name})"

    return coordinated_filter(
        dataset,
        has_valid_conversation,
        desc=desc,
        num_proc=None,
    )


def reject_image_columns(dataset, method: str) -> None:
    """Raise when a method with no vision path is handed a dataset carrying images.

    Every such method prunes its dataset to the columns it names, so an image column would be dropped
    with no diagnostic and the run would train on text alone at full cost. Read off the shared
    :data:`~src.data.vlm.VLM_IMAGE_COLUMNS` spellings.
    """
    carried = sorted(carried_image_columns(dataset))
    if not carried:
        return
    raise ValueError(
        f"{method} is text-only, but the dataset carries the image column(s) {carried}. They would "
        f"be pruned before the first batch and the run would train on the text alone. Drop the "
        f"column(s), or train this data with a method that has a vision path (SFT, distillation, "
        f"DPO, SMPO)."
    )


def _alias_render_column(
    dataset: DatasetDict, knob: str, field: str | None, target: str, accepted: tuple[str, ...], path: str
) -> DatasetDict:
    """Rename the column ``knob`` declares onto ``target``, the spelling a trainer hard-codes.

    The column is validated first so a mistyped name raises at the rename, ahead of any training
    step (ahead of the model load on the scripts that prepare data first); a declared name
    already among ``accepted`` needs no rename; a split that carries ``target`` beside the declared
    column is refused rather than overwritten.
    """
    if not field:
        return dataset
    require_render_column(dataset, path, knob, field)
    if field in accepted:
        return dataset
    occupied = sorted(name for name, split in dataset.items() if target in split.column_names)
    if occupied:
        raise ValueError(
            f"{knob}='{field}' has to be renamed to '{target}' (the column the trainer reads), but "
            f"split(s) {occupied} already carry one. Point {knob} at '{target}', or drop the other column."
        )
    return DatasetDict(
        {
            name: split.rename_column(field, target) if field in split.column_names else split
            for name, split in dataset.items()
        }
    )


def alias_images_column(dataset: DatasetDict, images_field: str | None, path: str) -> DatasetDict:
    """Rename a declared image column to ``images``, the spelling the vision routes hard-code.

    TRL's preference trainers decide their vision route by probing a sample for an ``image`` /
    ``images`` key, and their signature columns name only those two, so images stored under any
    other name are pruned to text while :func:`~src.data.vlm.is_vlm_run`, which reads
    ``images_field``, calls the same run multimodal. Aliasing ahead of the dispatch keeps the two
    verdicts identical.
    """
    return _alias_render_column(
        dataset, "images_field", images_field, _VISION_ROUTE_COLUMN, _VISION_ROUTE_COLUMNS, path
    )


def alias_tools_column(dataset: DatasetDict, tools_field: str | None, path: str) -> DatasetDict:
    """Rename a declared tools column to ``tools``, the only column TRL's ``RewardTrainer`` renders.

    The reward trainer chat-templates ``chosen``/``rejected`` itself and hands the row's ``tools``
    (a tool-schema list or its JSON string) to the template; no other column reaches it, so tools
    stored under another name would render without tools and without a diagnostic.
    """
    return _alias_render_column(dataset, "tools_field", tools_field, _TOOLS_ROUTE_COLUMN, (_TOOLS_ROUTE_COLUMN,), path)


def _require_tools_field_somewhere(paths: list, datasets: list, tools_field: str) -> None:
    """Hold ``tools_field`` to the dataset list rather than to each entry.

    A mixed corpus is a legitimate shape (a tool-use dataset concatenated with plain chat), and a
    source without the column renders its rows without tools. Only a knob no source can honour is
    treated as a typo and raises.
    """
    without = [
        entry_path
        for entry_path, ds_dict in zip(paths, datasets, strict=True)
        if missing_render_column_splits(ds_dict, tools_field)
    ]
    if not without:
        return
    if len(without) == len(paths):
        raise ValueError(
            f"tools_field='{tools_field}' names a column NONE of the {len(paths)} configured "
            f"dataset(s) carry ({without}) — every row would render without tools. Point tools_field "
            f"at an existing column in the YAML, or drop it."
        )
    logger.warning(
        f"tools_field='{tools_field}' is absent from {without}; rows from those source(s) render "
        f"without tools. Intentional for a mixed corpus — check the column name if it is not."
    )


def _apply_conversation_field(
    dataset: DatasetDict, path: str, conversation_field: str | None, *, required: bool
) -> None:
    """Drop rows with an empty conversation from whichever of train/test ``dataset`` carries, in place.

    ``required`` (the caller declared its render column, see :data:`_UNDECLARED`) additionally makes
    the column's absence fatal instead of the no-op filtering below.
    """
    if conversation_field is None:
        return
    if required:
        require_render_column(dataset, path, "conversation_field", conversation_field)
    for split in ("train", "test"):
        if split in dataset:
            dataset[split] = _filter_empty_conversations(dataset[split], conversation_field, split_name=split)


def _find_common_columns(datasets):
    """Find columns that exist in all datasets with compatible types."""
    if not datasets:
        return []

    common_columns = set(datasets[0].column_names)

    feature_types = {col: str(datasets[0].features[col]) for col in common_columns}

    for ds in datasets[1:]:
        for col in list(common_columns):
            if col not in ds.column_names:
                common_columns.remove(col)
                continue

            if col in feature_types and str(ds.features[col]) != feature_types[col]:
                common_columns.remove(col)
                logger.warning(
                    f"Removing column '{col}' due to schema mismatch: {feature_types[col]} vs {str(ds.features[col])}"
                )

    return list(common_columns)


def _pin_declared_columns(datasets: list[Dataset], declared_columns) -> list[Dataset]:
    """Union-fill each declared render column into the entries that do not carry it.

    A mixed corpus is a legitimate shape, but the schema intersection below would drop the column
    from the whole concatenation. A null fill is what a source without the column means, and it takes
    the carrying entry's feature type so it survives the intersection.
    """
    donor_features = {}
    for column in declared_columns:
        donor = next((ds for ds in datasets if column in ds.column_names), None)
        if donor is not None:
            donor_features[column] = donor.features[column]

    filled = []
    for dataset in datasets:
        for column, feature in donor_features.items():
            if column not in dataset.column_names:
                dataset = dataset.add_column(column, [None] * len(dataset), feature=feature)
        filled.append(dataset)
    return filled


def _normalize_dataset_schema(datasets, declared_columns=()):
    """Find common columns across all datasets and retain only those.

    ``declared_columns`` are the run's declared render columns (``conversation_field``,
    ``tools_field``): pinned through the intersection (:func:`_pin_declared_columns`) and named in
    the drop warning below if a schema mismatch loses one anyway.
    """
    if not datasets:
        return []

    datasets = _pin_declared_columns(datasets, declared_columns)
    common_columns = _find_common_columns(datasets)

    if not common_columns:
        raise ValueError("No common columns found across datasets! Cannot concatenate.")

    # Warn once, and only about essential columns some dataset actually had: a column no dataset
    # carries was never lost, and warning about it on every multi-dataset run would be noise.
    essential_columns = (*_ESSENTIAL_COLUMNS, *declared_columns)
    present_somewhere = set().union(*(ds.column_names for ds in datasets))
    lost_essentials = sorted(
        {col for col in essential_columns if col in present_somewhere and col not in common_columns}
    )
    if lost_essentials:
        logger.warning(
            f"Essential column(s) {lost_essentials} are dropped from the concatenation: present in "
            f"some datasets but absent from others or schema-mismatched across them."
        )

    removed_columns = set(datasets[0].column_names) - set(common_columns)
    if removed_columns:
        logger.info(f"Removing columns with incompatible schemas: {', '.join(removed_columns)}")

    return [ds.select_columns(common_columns) for ds in datasets]


def load_datasets(
    path: str | list,
    test_size: float | None,
    dataset_ratio: float | list | None,
    conversation_field: str | None = _UNDECLARED,
    seed: int = _DEFAULT_DATA_SEED,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    tools_field: str | None = None,
):
    # No outer barrier: _load_dataset_from_path coordinates the downloads and the rest is
    # rank-deterministic.
    if dataset_ratio is None:
        # Unset means the whole dataset, which is the normal case; warning here would fire on almost
        # every run to restate a default.
        dataset_ratio = [1] * len(path) if isinstance(path, list) else 1
    if isinstance(path, list) and isinstance(dataset_ratio, (int, float)):
        dataset_ratio = [float(dataset_ratio)] * len(path)
    if isinstance(path, list) and not isinstance(dataset_ratio, list):
        raise ValueError(
            "You should pass dataset_ratio for all of your datasets (list of floats), "
            "or a single float to apply to each dataset."
        )
    if not isinstance(path, list) and isinstance(dataset_ratio, list):
        raise ValueError("You should pass datasets for all of your dataset ratios.")
    if isinstance(path, list) and isinstance(dataset_ratio, list) and len(path) != len(dataset_ratio):
        raise ValueError(
            f"You have set {len(path)} datasets and {len(dataset_ratio)} dataset ratios, but it should be equal."
        )
    declared_conversation_field = conversation_field is not _UNDECLARED
    if not declared_conversation_field:
        conversation_field = _FALLBACK_CONVERSATION_FIELD

    # sharded=True (single-path branch only): DP ranks hold disjoint slices, so the forced cache keys
    # below must carry the DP identity.
    sharded = False
    if isinstance(path, list):
        # Load each entry in full: a list is replicated and accelerate shards the concatenation once.
        # Passing DP rank/size here would double-shard a sharded entry (1/dp of its own 1/dp),
        # dropping data with no diagnostic.
        loaded = [_load_dataset_from_path(d, test_size, placeholder_test=False) for d in path]
        all_datasets = [entry_ds for entry_ds, _ in loaded]
        content_sig = ",".join(_content_signature(d) for d in all_datasets)

        # Per entry, so a partial split or a missing render column names the offending dataset rather
        # than the whole list. A sharded entry missing a split is a per-rank shard-index fault rather
        # than a source that ships only train, so it raises; a plain train-only pool is handled below.
        for entry_path, (ds_dict, entry_sharded) in zip(path, loaded, strict=True):
            if entry_sharded:
                _require_train_test_splits(ds_dict, entry_path)
            else:
                # An entry's split set decides how many coordinated operations run over it below, so
                # a per-rank read that disagrees would desynchronize the store phases.
                _reject_divergent_split_presence(ds_dict, entry_path)
            _apply_conversation_field(ds_dict, entry_path, conversation_field, required=declared_conversation_field)
        if tools_field:
            _require_tools_field_somewhere(path, all_datasets, tools_field)

        # Declared render columns are contracts rather than candidates for the schema intersection:
        # pinned through the concatenation below, then re-checked on its result.
        declared_render_columns = {
            knob: column
            for knob, column in (
                ("conversation_field", conversation_field if declared_conversation_field else None),
                ("tools_field", tools_field),
            )
            if column
        }

        corpus_path = ", ".join(path)
        truncated_datasets = [
            _get_subset_from_dataset_dict(d, ratio, seed + i)  # Different seed for each dataset
            for i, (d, ratio) in enumerate(zip(all_datasets, dataset_ratio, strict=True))
        ]
        ds = DatasetDict()

        # An entry that ships no test split contributes training rows only, rather than 100 of its
        # own train rows into a sibling's real held-out split.
        train_datasets = [d["train"] for d in truncated_datasets]
        test_datasets = [d["test"] for d in truncated_datasets if "test" in d]
        train_only_entries = [p for p, d in zip(path, truncated_datasets, strict=True) if "test" not in d]
        if train_only_entries and test_datasets:
            logger.warning(
                f"Dataset(s) {train_only_entries} ship no test split: they contribute training rows "
                f"only, and the corpus test split comes from the entries that ship one. Set test_size "
                f"to carve a held-out split from every entry instead."
            )

        logger.info("Normalizing dataset schemas before concatenation...")

        # One column intersection over everything that gets concatenated: only some entries may
        # contribute test rows, and a per-side intersection would render the two splits under
        # different schemas, which the shared row processor downstream cannot map.
        num_train = len(train_datasets)
        normalized = _normalize_dataset_schema(train_datasets + test_datasets, declared_render_columns.values())
        train_datasets, test_datasets = normalized[:num_train], normalized[num_train:]

        logger.info(f"Concatenating {len(train_datasets)} training datasets")
        ds["train"] = concatenate_datasets(train_datasets)

        if test_datasets:
            logger.info(f"Concatenating {len(test_datasets)} test datasets")
            ds["test"] = concatenate_datasets(test_datasets)
        else:
            # No entry shipped one, so there is no held-out split to protect: the same fallback the
            # single-path load takes, over the whole corpus.
            ds["test"] = _placeholder_test_split(ds["train"], corpus_path)["test"]

        # The pin has to hold on the result: a declared column the intersection dropped anyway (a
        # schema mismatch across the entries, which no null fill can bridge) would render the whole
        # corpus without it, and for tools_field nothing downstream would report that.
        for knob, column in declared_render_columns.items():
            require_render_column(ds, corpus_path, knob, column)

    else:
        ds, sharded = _load_dataset_from_path(path, test_size, data_parallel_rank, data_parallel_size)
        # Only a raw sharded load can return a partial DatasetDict (missing splits are skipped).
        _require_train_test_splits(ds, path)
        content_sig = _content_signature(ds)
        _apply_conversation_field(ds, path, conversation_field, required=declared_conversation_field)
        if tools_field:
            # Optional knob, so a typo does not surface downstream: rows render without tools.
            require_render_column(ds, path, "tools_field", tools_field)
        ds = _get_subset_from_dataset_dict(ds, dataset_ratio, seed)

    if len(ds["train"]) == 0:
        raise ValueError("No training data after schema normalization. Please check your dataset.")
    else:
        logger.info(f"Training data after schema normalization: {len(ds['train'])}")
        logger.info(f"Columns in training dataset: {ds['train'].column_names}")

    if len(ds["test"]) == 0:
        raise ValueError("No test data after schema normalization. Please check your dataset.")
    else:
        logger.info(f"Test data after schema normalization: {len(ds['test'])}")
        logger.info(f"Columns in test dataset: {ds['test'].column_names}")

    # HF's own fingerprints diverge between writer and loader ranks, breaking downstream cache keys.
    # Unguarded: a failure here would leave the ranks disagreeing on every downstream cache key, and
    # string formatting and attribute stamping have no expected failure mode.
    path_repr = ",".join(path) if isinstance(path, list) else str(path)
    ratio_repr = ",".join([str(r) for r in dataset_ratio]) if isinstance(dataset_ratio, list) else str(dataset_ratio)
    meta = (
        f"{path_repr}|{test_size}|{ratio_repr}|{seed}|{len(ds['train'])}|{len(ds['test'])}"
        f"|{conversation_field}|{content_sig}"
    )
    # Without the DP identity, equal-length shards stamp identical keys and every downstream
    # coordinated map/filter cache collides, so non-writer ranks would load rank 0's mapped shard.
    if sharded:
        meta += f"|dp{data_parallel_rank}/{data_parallel_size}"
    base_fp = hashlib.md5(meta.encode()).hexdigest()
    ds["train"]._fingerprint = f"{base_fp}-train"
    ds["test"]._fingerprint = f"{base_fp}-test"
    ds["train"]._toolkit_cache_key = f"loaded_{base_fp}_train"
    ds["test"]._toolkit_cache_key = f"loaded_{base_fp}_test"

    return ds


def is_sharded_dataset_coordinated(path) -> bool:
    """``ShardedDatasetLoader.is_sharded_dataset`` agreed across all ranks.

    The per-rank S3 probe can transiently disagree (a creds error returns False on some ranks),
    splitting ranks onto sharded vs full paths and hanging NCCL. The probe only errs toward False, so
    the agreed MAX restores the sharded verdict on every rank.
    """
    local = isinstance(path, str) and ShardedDatasetLoader.is_sharded_dataset(path)
    return agree_probe_across_ranks(local, path, "is_sharded_dataset")


def is_presharded_dataset_load(path, data_parallel_size: int) -> bool:
    """Whether loading ``path`` shards data per data-parallel rank.

    True only for a sharded dataset with >1 DP rank (each holds a disjoint slice). Trainers pass this
    as ``dataset_presharded`` so the DataLoader does not re-shard already-split data. Uses the
    cross-rank-agreed probe so all ranks decide identically.
    """
    return data_parallel_size > 1 and isinstance(path, str) and is_sharded_dataset_coordinated(path)


def load_preprocessed_dataset(
    path: str,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
) -> DatasetDict:
    """Load a pre-processed dataset; sharded datasets load only this rank's shards.

    Pass the data-parallel rank/size, not global rank/world_size: CP/TP-group siblings must share
    data, while EP ranks (orthogonal to DP) get disjoint shards.
    """
    if is_sharded_dataset_coordinated(path):
        logger.info(f"Loading sharded pre-processed dataset from {path}")
        logger.info(f"  Data parallel rank: {data_parallel_rank}/{data_parallel_size}")
        loader = ShardedDatasetLoader(
            dataset_path=path,
            global_rank=data_parallel_rank,
            world_size=data_parallel_size,
        )
        ds = _sharded_load_with_join(loader)
        # Global emptiness is read from the shard index, giving the same verdict on every rank, so no
        # collective is needed. Per-rank emptiness is caught later by the trainer's pre-sharded eval
        # equalization.
        totals = {split: loader.get_total_examples(split) for split in ("train", "test") if split in ds}
    else:
        logger.info(f"Loading non-sharded pre-processed dataset from {path}")
        ds, _ = _load_dataset_from_path(path, test_size=None)
        # Non-sharded data is replica-identical, so a local empty split is globally empty.
        totals = {split: len(ds[split]) for split in ("train", "test") if split in ds}

    # The sharded loader skips missing splits; without this the consumer fails on ds["test"] later.
    _require_train_test_splits(ds, path)
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
    """Load SFT datasets, auto-detecting pre-processed (load directly) vs raw (tokenize at train).

    Detection works only for single-path SFT datasets. For sharded datasets pass data-parallel
    rank/size so CP/TP-group siblings share shards. Returns (DatasetDict, is_preprocessed).
    """
    # Cross-rank agreed, same split-brain hazard as the sharded probe: a transient S3 error would
    # split ranks onto the raw vs preprocessed path and hang NCCL.
    preprocessed = isinstance(path, str) and agree_probe_across_ranks(
        is_preprocessed_dataset(path), path, "is_preprocessed_dataset"
    )
    if preprocessed:
        logger.info(f"Detected pre-processed dataset at {path}")
        # test_size is applied only here on this branch: ``load_preprocessed_dataset`` never takes it.
        _warn_test_size_ignored(path, test_size, kind="pre-processed")
        ds = load_preprocessed_dataset(path, data_parallel_rank, data_parallel_size)
        return ds, True

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
