"""TRL's dataset preparation, run once per filesystem scope. TRL's DPO, KTO and reward trainers tokenize under
``main_process_first()``, a watchdog-bounded barrier, and key HF's fingerprint off an unhashable closure, so each rank
writes its own full tokenized copy on every run."""

from __future__ import annotations

import contextlib
import json

from accelerate import PartialState
from datasets import Dataset

from halo.data.pipeline.coordinated_map import coordinated_dataset_transform

# Process or launch identity, which would publish one prepared copy per rank or per run. Every other field is keyed
# because TRL reads whichever its version reads.
_RUN_IDENTITY_ARGS = frozenset({"local_rank", "output_dir", "run_name", "resume_from_checkpoint"})

_MAIN_FIRST_METHODS = ("main_process_first", "local_main_process_first")


@contextlib.contextmanager
def _accelerate_main_first_disabled():
    """Make accelerate's main-first blocks plain while TRL's preparation runs. :func:`coordinated_dataset_transform`
    prepares on the load rank alone, so their barriers would pair with nothing."""
    saved = {name: PartialState.__dict__[name] for name in _MAIN_FIRST_METHODS}
    for name in _MAIN_FIRST_METHODS:
        setattr(PartialState, name, lambda self: contextlib.nullcontext())
    try:
        yield
    finally:
        for name, method in saved.items():
            setattr(PartialState, name, method)


class TRLDatasetPrepMixin:
    """Mix in directly ahead of the TRL trainer class. An ``IterableDataset`` keeps TRL's own path, since its lazy maps
    hold nobody at the barriers for long."""

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
            cache_key_extras={
                "processing_class": processing_class,
                "args": {
                    name: value
                    for name, value in json.loads(args.to_json_string()).items()
                    if name not in _RUN_IDENTITY_ARGS
                },
            },
        )
