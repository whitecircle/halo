"""TRL's own dataset preparation, run once per filesystem scope and shared with every other rank.

TRL's DPO, KTO and reward trainers tokenize in ``_prepare_dataset`` under
``PartialState().main_process_first()``: global rank 0 maps the whole corpus while every other rank
sits in an NCCL barrier bounded by the watchdog, whatever the filesystem layout, and the maps then key
HF's fingerprint of a closure over the trainer — unhashable, so random — so each rank writes its own
full tokenized copy into ``HF_DATASETS_CACHE`` on every run.
"""

from __future__ import annotations

import contextlib
import json

from accelerate import PartialState
from datasets import Dataset

from src.data.pipeline.processing import coordinated_dataset_transform

# TrainingArguments fields naming the process or the launch, not the data: ``local_rank`` differs per
# rank and the rest per launch, so keying them would publish one prepared copy per rank or per run.
_RUN_IDENTITY_ARGS = frozenset({"local_rank", "output_dir", "run_name", "resume_from_checkpoint"})

_MAIN_FIRST_METHODS = ("main_process_first", "local_main_process_first")


@contextlib.contextmanager
def _accelerate_main_first_disabled():
    """Make accelerate's main-first blocks plain blocks while TRL's preparation runs.

    Their barriers would pair with nothing: :func:`coordinated_dataset_transform` runs the preparation
    on the load rank alone while its peers wait on the store, and orders the ranks itself.
    """
    saved = {name: PartialState.__dict__[name] for name in _MAIN_FIRST_METHODS}
    for name in _MAIN_FIRST_METHODS:
        setattr(PartialState, name, lambda self: contextlib.nullcontext())
    try:
        yield
    finally:
        for name, method in saved.items():
            setattr(PartialState, name, method)


def prepared_data_settings(args) -> dict:
    """The training arguments a prepared dataset is keyed on: all of them but :data:`_RUN_IDENTITY_ARGS`.

    Over-keyed on purpose: TRL's preparation reads whichever fields its version reads, and a field the
    key missed would reuse rows prepared under another value.
    """
    return {name: value for name, value in json.loads(args.to_json_string()).items() if name not in _RUN_IDENTITY_ARGS}


class CoordinatedTRLDatasetPrepMixin:
    """Route TRL's ``_prepare_dataset`` through :func:`coordinated_dataset_transform`.

    Mixed in directly ahead of the TRL trainer class. The load rank of each filesystem scope prepares
    and publishes the dataset, the rest load it, and a relaunch with the same tokenizer, rows and
    settings reuses it. An ``IterableDataset`` keeps TRL's own path: its maps are lazy, so its
    main-first barriers hold nobody for long.
    """

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        prepare = super()._prepare_dataset
        if not isinstance(dataset, Dataset):
            return prepare(dataset, processing_class, args, dataset_name)

        def run() -> Dataset:
            with _accelerate_main_first_disabled():
                return prepare(dataset, processing_class, args, dataset_name)

        return coordinated_dataset_transform(
            dataset,
            run,
            f"{type(self).__name__} {dataset_name} dataset preparation",
            cache_key_extras={"processing_class": processing_class, "args": prepared_data_settings(args)},
        )
