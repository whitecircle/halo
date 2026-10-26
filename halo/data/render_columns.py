"""The columns a run declares for rendering."""

from datasets import Dataset, DatasetDict

from halo.data.pipeline.coordinated_map import coordinated_filter
from halo.data.pipeline.map_cache_key import carry_cache_key
from halo.data.sources.paths import TRAIN_TEST_SPLITS, eval_split_name
from halo.data.vlm_routing import VLM_RAW_IMAGE_COLUMNS


def _rendered_splits(dataset: DatasetDict) -> list[str]:
    """The splits of ``dataset`` a loader renders: ``train`` and its held-out split (:func:`eval_split_name`),
    a lone ``validation`` split included."""
    return [split for split in ("train", eval_split_name(dataset)) if split in dataset]


def missing_render_column_splits(dataset: DatasetDict, column: str) -> list[str]:
    """The rendered splits that do not carry ``column``; an extra split a source carries is never rendered, so its
    schema is no contract."""
    return sorted(split for split in _rendered_splits(dataset) if column not in dataset[split].column_names)


def reject_missing_render_column(dataset: DatasetDict, path: str, knob: str, column: str) -> None:
    """Refuse a declared render column a raw dataset lacks, naming ``knob`` (a config field or a CLI flag): else a
    typo surfaces as a map worker's bare ``KeyError`` after the model load, or, for ``tools_field``, not at all."""
    missing = missing_render_column_splits(dataset, column)
    if not missing:
        return
    available = sorted(set().union(*(dataset[split].column_names for split in _rendered_splits(dataset))))
    raise ValueError(
        f"{knob}='{column}' names a column the dataset {path} does not carry (missing from split(s) "
        f"{missing}; available columns: {available}). Point {knob} at an existing column."
    )


def filter_empty_conversations(dataset: Dataset, conversation_field: str, split_name: str) -> Dataset:
    """Drop the rows whose ``conversation_field`` is None or empty; a no-op when the field is absent.
    ``split_name`` keeps the train and test filter caches apart."""
    if conversation_field not in dataset.column_names:
        return dataset

    def has_valid_conversation(example):
        return example[conversation_field] is not None and len(example[conversation_field]) > 0

    # num_proc=None avoids spawning processes during the distributed loading phase.
    return coordinated_filter(
        dataset,
        has_valid_conversation,
        desc=f"filtering empty {conversation_field} ({split_name})",
        num_proc=None,
    )


def _alias_render_column(
    dataset: DatasetDict, knob: str, field: str | None, target: str, accepted: tuple[str, ...], path: str
) -> DatasetDict:
    """Rename the column ``knob`` declares onto ``target``, the spelling a trainer hard-codes, after checking it
    exists; a name already ``accepted`` stays, and a split carrying ``target`` beside it is refused."""
    if not field:
        return dataset
    reject_missing_render_column(dataset, path, knob, field)
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
            name: carry_cache_key(split, split.rename_column(field, target), f"rename:{field}->{target}")
            if field in split.column_names
            else split
            for name, split in dataset.items()
        }
    )


def alias_images_column(dataset: DatasetDict, images_field: str | None, path: str) -> DatasetDict:
    """Rename a declared image column to ``images``: TRL's preference trainers route on an ``image``/``images``
    key and prune every other column, while :func:`~halo.data.vlm_routing.is_vlm_run` reads ``images_field``."""
    return _alias_render_column(
        dataset, "images_field", images_field, VLM_RAW_IMAGE_COLUMNS[0], VLM_RAW_IMAGE_COLUMNS, path
    )


def alias_tools_column(dataset: DatasetDict, tools_field: str | None, path: str) -> DatasetDict:
    """Rename a declared tools column to ``tools``, the one column TRL's ``RewardTrainer`` hands its chat
    template, so tools stored elsewhere do not render toolless in silence."""
    return _alias_render_column(dataset, "tools_field", tools_field, "tools", ("tools",), path)


def apply_conversation_field(
    dataset: DatasetDict, path: str, conversation_field: str | None, *, required: bool, knob: str
) -> None:
    """Drop rows with an empty conversation from whichever of train/test ``dataset`` carries, in place.

    ``required`` (the caller declared the column) makes its absence fatal instead of a no-op, naming
    ``knob``, the config field the script reads the column name from.
    """
    if conversation_field is None:
        return
    if required:
        reject_missing_render_column(dataset, path, knob, conversation_field)
    for split in TRAIN_TEST_SPLITS:
        if split in dataset:
            dataset[split] = filter_empty_conversations(dataset[split], conversation_field, split_name=split)
