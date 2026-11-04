#!/usr/bin/env python
"""Embedding training refuses a sentence-transformers train loader whose ranks would run unequal steps, which hangs.
That loader turns ``even_batches`` off; the toolkit's DP-sharded loader and a single process are unaffected."""

import ast
import inspect
import textwrap

import pytest
from accelerate.data_loader import BatchSamplerShard, prepare_data_loader
from datasets import Dataset, DatasetDict
from sentence_transformers.base.sampler import DefaultBatchSampler
from torch.utils.data import DataLoader, SequentialSampler

from halo.configs.embedding import EmbeddingConfig
from halo.trainers.embedding import embedding_rules
from halo.trainers.embedding.embedding_rules import reject_uneven_sentence_transformers_batches
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.repo import REPO_ROOT

_ROWS = Dataset.from_dict({"anchor": ["a", "b", "c"], "positive": ["x", "y", "z"]})
_SCRIPT = REPO_ROOT / "scripts" / "training" / "embedding.py"


def _check(
    *,
    drop_last: bool = False,
    world_size: int = 2,
    toolkit_loader: bool = False,
    tmp_path,
    monkeypatch,
    train_dataset=_ROWS,
    dispatch_batches: bool | None = None,
    split_batches: bool = False,
) -> None:
    monkeypatch.setattr(embedding_rules, "launcher_global_world_size", lambda: world_size)
    args = EmbeddingConfig(
        output_dir=str(tmp_path),
        dataloader_drop_last=drop_last,
        bf16=False,
        use_cpu=True,
        accelerator_config={"dispatch_batches": dispatch_batches, "split_batches": split_batches},
    )
    reject_uneven_sentence_transformers_batches(args, train_dataset, toolkit_loader=toolkit_loader)


def _sharded_iterable(n_shards: int):
    return Dataset.from_dict({"anchor": [str(i) for i in range(13)]}).to_iterable_dataset(num_shards=n_shards)


def test_the_kept_remainder_gives_ranks_unequal_batch_counts():
    """On the sampler and shard the sentence-transformers loader builds."""

    def lengths(drop_last: bool) -> list[int]:
        sampler = DefaultBatchSampler(SequentialSampler(range(10)), batch_size=2, drop_last=drop_last)
        return [
            len(BatchSamplerShard(sampler, num_processes=2, process_index=rank, even_batches=False)) for rank in (0, 1)
        ]

    assert lengths(drop_last=False) == [3, 2]
    assert lengths(drop_last=True) == [2, 2]


@pytest.mark.parametrize("drop_last", [False, True])
def test_a_shard_split_iterable_gives_ranks_unequal_batch_counts(drop_last):
    """With dispatch off, 3 file shards over 2 ranks are split by shard, and ``drop_last`` does not even them out."""
    counts = [
        sum(
            1
            for _ in prepare_data_loader(
                DataLoader(_sharded_iterable(3), batch_size=2, drop_last=drop_last),
                num_processes=2,
                process_index=rank,
                dispatch_batches=False,
                even_batches=False,
                put_on_device=False,
            )
        )
        for rank in (0, 1)
    ]
    assert counts[0] != counts[1], counts


@pytest.mark.parametrize("train_dataset", [_ROWS, DatasetDict(first=_ROWS, second=_ROWS)], ids=["dataset", "dict"])
def test_a_kept_remainder_is_refused_on_the_sentence_transformers_loader(train_dataset, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Set dataloader_drop_last: true"):
        _check(tmp_path=tmp_path, monkeypatch=monkeypatch, train_dataset=train_dataset)


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "n_shards, dispatch_batches", [(3, False), (1, None), (1, True)], ids=["shard-split", "default", "dispatched"]
)
def test_an_iterable_the_loader_cannot_split_evenly_is_refused(
    n_shards, dispatch_batches, drop_last, tmp_path, monkeypatch
):
    """Split by shard, the ranks' row counts differ.
    Dispatched (accelerate's default for an iterable), the collator's string modality fields cannot concatenate."""
    with pytest.raises(
        ValueError,
        match="Set dispatch_batches: false, and either reshard the dataset to at most 2 shards or set split_batches: true",
    ):
        _check(
            drop_last=drop_last,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            train_dataset=_sharded_iterable(n_shards),
            dispatch_batches=dispatch_batches,
        )


@pytest.mark.parametrize(
    "n_shards, split_batches", [(2, False), (1, False), (3, True)], ids=["2-shards", "1-shard", "split-batches"]
)
def test_an_evenly_sharded_iterable_passes(n_shards, split_batches, tmp_path, monkeypatch):
    """With at most one shard per process or split batches, ``IterableDatasetShard`` gives every rank equal batches."""
    _check(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        train_dataset=_sharded_iterable(n_shards),
        dispatch_batches=False,
        split_batches=split_batches,
    )


def test_a_single_process_keeps_its_remainder(tmp_path, monkeypatch):
    _check(world_size=1, tmp_path=tmp_path, monkeypatch=monkeypatch)
    _check(
        world_size=1,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        train_dataset=_sharded_iterable(3),
        dispatch_batches=False,
    )


def test_the_toolkit_loader_keeps_its_remainder(tmp_path, monkeypatch):
    _check(toolkit_loader=True, tmp_path=tmp_path, monkeypatch=monkeypatch)


def test_drop_last_passes_on_the_sentence_transformers_loader(tmp_path, monkeypatch):
    _check(drop_last=True, tmp_path=tmp_path, monkeypatch=monkeypatch)


def _called_names(source: str) -> set[str]:
    tree = ast.parse(textwrap.dedent(source))
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


def test_the_gate_runs_in_the_script_and_the_trainers_init():
    """The script refuses before the model loads; the trainer refuses a hand-built run the same way."""
    assert "reject_uneven_sentence_transformers_batches" in _called_names(_SCRIPT.read_text())
    assert "reject_uneven_sentence_transformers_batches" in _called_names(inspect.getsource(EmbeddingTrainer.__init__))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
