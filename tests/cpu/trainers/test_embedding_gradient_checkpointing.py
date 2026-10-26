#!/usr/bin/env python
"""A SentenceTransformer must take the trainer's gradient-checkpointing call exactly as ``PreTrainedModel`` does.

transformers' trainer enables checkpointing through ``gradient_checkpointing_enable(
gradient_checkpointing_kwargs=..., every_n_layers=...)``; sentence-transformers' own override takes
the kwargs dict only, so without the toolkit patch every embedding run with checkpointing on dies at
that call, and one that swallowed the keyword would checkpoint every layer at the memory profile of a
run that asked for fewer. The embedding trainer installs the patch at construction.

Run: ``pytest -m cpu tests/cpu/trainers/test_embedding_gradient_checkpointing.py``
"""

import ast
import inspect
import textwrap

import pytest
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

from halo.trainers.embedding.preloaded_transformer import patch_sentence_transformer_gradient_checkpointing
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.models import TINY_QWEN3_CONFIG


class _Encoder(nn.Module):
    """The shape of sentence-transformers' ``Transformer`` module: a transformers model held as ``auto_model``."""

    def __init__(self):
        super().__init__()
        self.auto_model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))


def _layer_flags(model) -> list[bool]:
    return [m.gradient_checkpointing for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]


def test_every_n_layers_reaches_the_transformers_model():
    patch_sentence_transformer_gradient_checkpointing()
    model = SentenceTransformer(modules=[_Encoder(), Normalize()])

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}, every_n_layers=2)
    flags = _layer_flags(model)
    assert len(flags) >= 2 and flags == [index % 2 == 0 for index in range(len(flags))], flags
    assert model.transformers_model.is_gradient_checkpointing

    model.gradient_checkpointing_enable()
    assert all(_layer_flags(model))

    model.gradient_checkpointing_disable()
    assert not any(_layer_flags(model))


def test_a_pipeline_without_a_transformers_model_is_refused():
    patch_sentence_transformer_gradient_checkpointing()
    with pytest.raises(ValueError, match="holds no transformers model"):
        SentenceTransformer(modules=[Normalize()]).gradient_checkpointing_enable(every_n_layers=1)


def test_the_trainer_installs_the_patch_at_construction():
    constructor = ast.parse(textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__)))
    called = {
        node.func.id
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "patch_sentence_transformer_gradient_checkpointing" in called


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
