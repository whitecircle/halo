#!/usr/bin/env python
"""The embedding trainer refuses a batch sampler its toolkit loader would drop.
The DP-sharded loader ignores ``batch_sampler``, so ``no_duplicates`` would silently run with in-batch duplicates."""

import ast
import inspect
import textwrap

import pytest

from halo.configs.embedding import EmbeddingConfig
from halo.trainers.embedding.embedding_rules import reject_batch_sampler_on_toolkit_loader
from halo.trainers.embedding.trainer import EmbeddingTrainer


def _check(batch_sampler: str, *, toolkit_loader: bool, tmp_path) -> None:
    args = EmbeddingConfig(output_dir=str(tmp_path), batch_sampler=batch_sampler, bf16=False, use_cpu=True)
    reject_batch_sampler_on_toolkit_loader(args, toolkit_loader=toolkit_loader)


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
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reject_batch_sampler_on_toolkit_loader" in called


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
