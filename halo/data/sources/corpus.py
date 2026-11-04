"""Multi-dataset corpora: per-entry ratios and subsets, one schema over every entry, and train/test concatenation."""

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets

from halo.data.render_columns import (
    drop_empty_conversation_rows,
    missing_render_column_splits,
    reject_missing_render_column,
)
from halo.data.sources.paths import reject_partial_splits
from halo.data.vlm_detection import VLM_IMAGE_COLUMNS
from halo.log import info_logger
from halo.runtime.ranks import is_global_main_process

logger = info_logger(__name__)

_PLACEHOLDER_TEST_ROWS = 100

# Columns whose silent loss changes what a run trains on: the supported row formats' fields and a prepared row.
_ESSENTIAL_COLUMNS = (
    "conversation",
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


def resolve_dataset_ratios(path: str | list, dataset_ratio: float | list | None) -> float | list:
    """One ratio per entry of a list ``path`` (a single float applies to each), ``1`` when unset."""
    if dataset_ratio is None:
        return [1] * len(path) if isinstance(path, list) else 1
    if isinstance(path, list) and isinstance(dataset_ratio, (int, float)):
        dataset_ratio = [float(dataset_ratio)] * len(path)
    if isinstance(path, list) and not isinstance(dataset_ratio, list):
        raise ValueError(
            "You should pass dataset_ratio for all of your datasets (list of floats), "
            "or a single float to apply to each dataset."
        )
    if not isinstance(path, list) and isinstance(dataset_ratio, list):
        raise ValueError("You should pass datasets for all of your dataset ratios.")
    if isinstance(path, list) and len(path) != len(dataset_ratio):
        raise ValueError(
            f"You have set {len(path)} datasets and {len(dataset_ratio)} dataset ratios, but it should be equal."
        )
    return dataset_ratio


def placeholder_test_split(train_ds: Dataset, path: str) -> DatasetDict:
    """For a source shipping no test split: its first ``_PLACEHOLDER_TEST_ROWS`` rows."""
    if is_global_main_process():
        logger.warning(f"Dataset {path} has no test split, using first {_PLACEHOLDER_TEST_ROWS} samples as test")
    return DatasetDict({"train": train_ds, "test": train_ds.select(range(min(_PLACEHOLDER_TEST_ROWS, len(train_ds))))})


def _subset_split(dataset: Dataset, dataset_ratio: float | None, seed: int) -> Dataset:
    """Seeded for cross-rank reproducibility."""
    if dataset_ratio is None or abs(dataset_ratio - 1) <= 0.001:
        return dataset

    indices = np.random.default_rng(seed).choice(len(dataset), int(dataset_ratio * len(dataset)), replace=False)
    return dataset.select(indices)


def subset_dataset_dict(dataset: DatasetDict, dataset_ratio: float | None, seed: int) -> DatasetDict:
    """Test draws with ``seed + 1``."""
    dataset["train"] = _subset_split(dataset["train"], dataset_ratio, seed)
    if "test" in dataset:
        dataset["test"] = _subset_split(dataset["test"], dataset_ratio, seed + 1)
    return dataset


def _reject_tools_field_nowhere(paths: list, datasets: list, tools_field: str) -> None:
    """Refuse a ``tools_field`` no entry carries (a typo); warn about entries without it, as in a mixed corpus."""
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
    if is_global_main_process():
        logger.warning(
            f"tools_field='{tools_field}' is absent from {without}; rows from those source(s) render "
            f"without tools. Intentional for a mixed corpus — check the column name if it is not."
        )


def _find_common_columns(datasets):
    """Columns every dataset carries with one feature type, in the first's order. A list: the order becomes the
    concatenation's column order, which maps keying on ``column_names`` read; a set's is hash-randomized per rank."""
    common_columns = list(datasets[0].column_names)
    feature_types = {col: str(datasets[0].features[col]) for col in common_columns}
    for ds in datasets[1:]:
        for col in list(common_columns):
            if col not in ds.column_names:
                common_columns.remove(col)
                continue
            if str(ds.features[col]) != feature_types[col]:
                common_columns.remove(col)
                if is_global_main_process():
                    logger.warning(
                        f"Removing column '{col}' due to schema mismatch: {feature_types[col]} vs {str(ds.features[col])}"
                    )
    return common_columns


def _pin_declared_columns(datasets: list[Dataset], declared_columns) -> list[Dataset]:
    """Null-fill declared render columns into entries lacking them, at the carrying entry's type, so the schema
    intersection keeps them."""
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
    if not datasets:
        return []
    datasets = _pin_declared_columns(datasets, declared_columns)
    common_columns = _find_common_columns(datasets)
    if not common_columns:
        raise ValueError("No common columns found across datasets! Cannot concatenate.")

    # Only essential columns some dataset actually had: a column no dataset carries was never lost.
    essential_columns = (*_ESSENTIAL_COLUMNS, *declared_columns)
    present_somewhere = set().union(*(ds.column_names for ds in datasets))
    lost_essentials = sorted(
        {col for col in essential_columns if col in present_somewhere and col not in common_columns}
    )
    if lost_essentials and is_global_main_process():
        logger.warning(
            f"Essential column(s) {lost_essentials} are dropped from the concatenation: present in "
            f"some datasets but absent from others or schema-mismatched across them."
        )

    removed_columns = set(datasets[0].column_names) - set(common_columns)
    if removed_columns and is_global_main_process():
        logger.info(f"Removing columns with incompatible schemas: {', '.join(removed_columns)}")

    return [ds.select_columns(common_columns) for ds in datasets]


def _concatenate_entries(paths: list[str], subsets: list[DatasetDict], declared_columns) -> DatasetDict:
    """The test split comes only from entries shipping one, else a placeholder."""
    train_datasets = [d["train"] for d in subsets]
    test_datasets = [d["test"] for d in subsets if "test" in d]
    train_only_entries = [p for p, d in zip(paths, subsets, strict=True) if "test" not in d]
    if train_only_entries and test_datasets and is_global_main_process():
        logger.warning(
            f"Dataset(s) {train_only_entries} ship no test split: they contribute training rows "
            f"only, and the corpus test split comes from the entries that ship one. Set test_size "
            f"to carve a held-out split from every entry instead."
        )

    if is_global_main_process():
        logger.info("Normalizing dataset schemas before concatenation...")
    # One intersection over both sides: the shared row processor cannot map two split schemas.
    num_train = len(train_datasets)
    normalized = _normalize_dataset_schema(train_datasets + test_datasets, declared_columns)
    train_datasets, test_datasets = normalized[:num_train], normalized[num_train:]

    if is_global_main_process():
        logger.info(f"Concatenating {len(train_datasets)} training datasets")
    ds = DatasetDict({"train": concatenate_datasets(train_datasets)})
    if test_datasets:
        if is_global_main_process():
            logger.info(f"Concatenating {len(test_datasets)} test datasets")
        ds["test"] = concatenate_datasets(test_datasets)
    else:
        ds["test"] = placeholder_test_split(ds["train"], ", ".join(paths))["test"]
    return ds


def build_corpus(
    paths: list[str],
    loaded: list[tuple[DatasetDict, bool]],
    ratios: list[float],
    seed: int,
    *,
    conversation_field: str,
    conversation_declared: bool,
    conversation_knob: str,
    tools_field: str | None,
) -> DatasetDict:
    """One train/test ``DatasetDict``. Each entry is checked and filtered on its own, so a refusal names the offending
    dataset; declared render columns are pinned through the concatenation, then re-checked."""
    # A SHARDED entry missing a split is a per-rank shard-index fault, not a source that ships only train.
    for entry_path, (entry_ds, entry_sharded) in zip(paths, loaded, strict=True):
        if entry_sharded:
            reject_partial_splits(entry_ds, entry_path)
        drop_empty_conversation_rows(
            entry_ds, entry_path, conversation_field, required=conversation_declared, knob=conversation_knob
        )
    datasets = [entry_ds for entry_ds, _ in loaded]
    if tools_field:
        _reject_tools_field_nowhere(paths, datasets, tools_field)

    declared_render_columns = {
        knob: column
        for knob, column in (
            (conversation_knob, conversation_field if conversation_declared else None),
            ("tools_field", tools_field),
        )
        if column
    }
    subsets = [
        subset_dataset_dict(entry_ds, ratio, seed + i)
        for i, (entry_ds, ratio) in enumerate(zip(datasets, ratios, strict=True))
    ]
    ds = _concatenate_entries(paths, subsets, declared_render_columns.values())
    # A schema mismatch no null fill bridges drops a declared column anyway; tools_field would say nothing.
    for knob, column in declared_render_columns.items():
        reject_missing_render_column(ds, ", ".join(paths), knob, column)
    return ds
