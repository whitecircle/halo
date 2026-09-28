#!/usr/bin/env python
"""The embedding trainer refuses a batch sampler its toolkit loader would drop.

TP/ETP runs and pre-sharded datasets batch through the mixin's DP-sharded loader, which builds plain
batches and never reads ``batch_sampler``, so ``no_duplicates`` there would run with in-batch
duplicates (false negatives for in-batch-negative losses) and no message. Plain DP and pure EP keep
sentence-transformers' own loader, which applies it.

Run: pytest tests/cpu/trainers/test_embedding_batch_sampler_gate.py
"""

import ast
import inspect
import textwrap

import pytest

from src.configs.embedding_config import EmbeddingConfig
from src.trainers.embedding.trainer import EmbeddingTrainer


def _check(batch_sampler: str, *, toolkit_loader: bool, tmp_path) -> None:
    host = object.__new__(EmbeddingTrainer)
    host._needs_custom_dataloader = lambda: toolkit_loader
    args = EmbeddingConfig(output_dir=str(tmp_path), batch_sampler=batch_sampler, bf16=False, use_cpu=True)
    host._reject_batch_sampler_on_toolkit_loader(args)


@pytest.mark.parametrize("batch_sampler", ["no_duplicates", "no_duplicates_hashed", "group_by_label"])
def test_a_non_plain_sampler_is_refused_on_the_toolkit_loader(batch_sampler, tmp_path):
    with pytest.raises(ValueError, match=f"batch_sampler: {batch_sampler} is not applied"):
        _check(batch_sampler, toolkit_loader=True, tmp_path=tmp_path)


def test_plain_batches_pass_on_the_toolkit_loader(tmp_path):
    _check("batch_sampler", toolkit_loader=True, tmp_path=tmp_path)


def test_the_sentence_transformers_loader_keeps_its_samplers(tmp_path):
    _check("no_duplicates", toolkit_loader=False, tmp_path=tmp_path)


def test_the_gate_runs_in_the_trainers_init():
    source = textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__))
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_reject_batch_sampler_on_toolkit_loader" in called


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
