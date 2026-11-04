#!/usr/bin/env python
"""Every wrapper the trainer enables checkpointing on takes transformers' full keyword set, or every such run fails at
that call. A pipeline stage builds its own checkpoint function and refuses ``offload``."""

import pytest
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.trainers.embedding.preloaded_transformer import patch_sentence_transformer_gradient_checkpointing
from tests.common.models import TINY_QWEN3_CONFIG

# The call transformers' Trainer.train makes.
TRAINER_CALL = {"gradient_checkpointing_kwargs": None, "every_n_layers": 1, "offload": True}


def _qwen3() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))


class _Encoder(nn.Module):
    """sentence-transformers' ``Transformer`` module: a transformers model held as ``auto_model``."""

    def __init__(self):
        super().__init__()
        self.auto_model = _qwen3()


def _assert_offloading(inner) -> None:
    assert inner.is_gradient_checkpointing
    installed = {layer._gradient_checkpointing_func.func for layer in inner.model.layers}
    assert len(installed) == 1 and checkpoint not in installed, "offload did not reach the transformers model"


def test_the_cp_model_hands_offload_to_the_transformers_model():
    inner = _qwen3()
    # The Ulysses validator reads a flash label; no attention kernel runs here.
    inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    wrapper.gradient_checkpointing_enable(**TRAINER_CALL)
    _assert_offloading(inner)


def test_a_sentence_transformer_hands_offload_to_the_transformers_model():
    patch_sentence_transformer_gradient_checkpointing()
    model = SentenceTransformer(modules=[_Encoder(), Normalize()])
    model.gradient_checkpointing_enable(**TRAINER_CALL)
    _assert_offloading(model.transformers_model)


def test_a_pipeline_stage_refuses_offload():
    stage = build_pipeline_stage(_qwen3(), 0, 2)
    with pytest.raises(ValueError, match="offload"):
        stage.gradient_checkpointing_enable(**TRAINER_CALL)
    assert not stage.is_gradient_checkpointing


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
