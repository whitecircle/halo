#!/usr/bin/env python
"""Embedding training refuses a sentence-transformers train loader whose ranks would run unequal steps, which hangs.
That loader turns ``even_batches`` off; the toolkit's DP-sharded loader and a single process are unaffected."""

import ast
import inspect
import textwrap

import pytest
from datasets import Dataset, DatasetDict

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


@pytest.mark.parametrize("train_dataset", [_ROWS, DatasetDict(first=_ROWS, second=_ROWS)], ids=["dataset", "dict"])
def test_a_kept_remainder_is_refused_on_the_sentence_transformers_loader(train_dataset, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Set dataloader_drop_last: true"):
        _check(tmp_path=tmp_path, monkeypatch=monkeypatch, train_dataset=train_dataset)


@pytest.mark.parametrize(
    "n_shards, dispatch_batches, drop_last",
    [
        pytest.param(3, False, False, id="shard-split-False"),
        pytest.param(1, None, False, id="default-False"),
        pytest.param(1, True, False, id="dispatched-False"),
        pytest.param(1, None, True, id="default-True"),
    ],
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
