#!/usr/bin/env python
"""The embedding trainer refuses a ``max_length`` the entry script has not resolved.

sentence-transformers' training collator truncates every text column to ``args.max_length``; a
non-positive value stands for the backbone's context window only after ``resolve_embedding_max_length``
runs, so a hand-built trainer handed one would train on texts cut to their special tokens.

Run: pytest tests/cpu/trainers/test_embedding_max_length_gate.py
"""

import ast
import inspect
import textwrap

import pytest

from src.configs.embedding import EmbeddingConfig
from src.trainers.embedding.trainer import EmbeddingTrainer, reject_unresolved_max_length


def _args(max_length, tmp_path) -> EmbeddingConfig:
    return EmbeddingConfig(output_dir=str(tmp_path), max_length=max_length, bf16=False, use_cpu=True)


@pytest.mark.parametrize("max_length", [0, -1])
def test_an_unresolved_length_is_refused(max_length, tmp_path):
    with pytest.raises(ValueError, match=f"max_length={max_length} reached the embedding trainer unresolved"):
        reject_unresolved_max_length(_args(max_length, tmp_path))


@pytest.mark.parametrize("max_length", [None, 16])
def test_a_resolved_or_absent_length_passes(max_length, tmp_path):
    reject_unresolved_max_length(_args(max_length, tmp_path))


def test_the_gate_runs_in_the_trainers_init():
    source = textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__))
    called = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reject_unresolved_max_length" in called


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
