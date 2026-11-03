#!/usr/bin/env python
"""Four ``UlyssesModel`` / ``CPConfig`` seams that fail silently otherwise."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.unwrapping import CP_ATTENTION_INNER_ATTR, strip_cp_attention_level

_CP_MOD = "halo.distributed.context_parallel.config"

VOCAB = 64
SEQ_LEN = 8
BATCH = 2
NUM_LAYERS = 3


class _RecordingLogitsOnly(torch.nn.Module):
    """CP's patched attention calls ``flash_attn_func`` directly, which has no CPU kernel; the real validation and
    patching still ran."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Embedding(VOCAB, VOCAB)
        self.seen: dict = {}

    def forward(self, input_ids=None, **kwargs):
        self.seen = {"input_ids": input_ids, **kwargs}
        return CausalLMOutputWithPast(logits=self.proj(input_ids).float())


def _cp_wrapped_model():
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    # The validator requires a flash label, which transformers refuses to instantiate on CPU; no attention kernel is
    # ever called.
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    return UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1)), config


def _split_two_ways(wrapper):
    """Without a 2-rank job, the split still really narrows, away from position 0."""
    wrapper.cp_config.cp_size, wrapper.cp_config.cp_rank = 2, 1
    return SEQ_LEN // 2


@pytest.mark.parametrize("explicit_positions", [False, True], ids=["arange", "given"])
def test_the_model_gets_this_ranks_chunk_of_the_positions(tmp_path, explicit_positions):
    """Each attention layer applies RoPE to its own chunk, so positions must be the sequence's, not restarting at 0."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, config = _cp_wrapped_model()
        inner = _RecordingLogitsOnly(config)
        wrapper.model = inner
        chunk = _split_two_ways(wrapper)

        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        # Per-document resets, so only the real tensor reproduces them.
        packed = torch.tensor([[0, 1, 2, 3, 5, 6, 7, 8]], dtype=torch.long).expand(BATCH, SEQ_LEN).contiguous()
        expected = packed.clone() if explicit_positions else torch.arange(SEQ_LEN).expand(BATCH, SEQ_LEN)

        wrapper(input_ids=ids, labels=ids, position_ids=packed if explicit_positions else None)

        assert inner.seen["input_ids"].shape == (BATCH, chunk), "the model must still see its chunk"
        torch.testing.assert_close(inner.seen["position_ids"], expected[:, chunk:], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_strip_cp_attention_level_inverts_the_level_the_wrappers_insert(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, config = _cp_wrapped_model()
        patched = [name for name, _ in wrapper.model.named_parameters() if ".self_attn." in name]
        hub_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        hub = [name for name, _ in hub_model.named_parameters() if ".self_attn." in name]

        assert patched and all(f".self_attn.{CP_ATTENTION_INNER_ATTR}." in name for name in patched)
        assert sorted(strip_cp_attention_level(name) for name in patched) == sorted(hub)
    finally:
        dist.destroy_process_group()


def test_parameters_yield_each_tensor_once_under_its_hub_name(tmp_path):
    """An optimizer built from ``parameters()`` steps each tensor once; the trainer keys state by hub names."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, _ = _cp_wrapped_model()
        params = list(wrapper.parameters())
        names = [name for name, _ in wrapper.named_parameters()]

        assert len({id(param) for param in params}) == len(params) == len(list(wrapper.model.parameters()))
        assert [id(param) for _, param in wrapper.named_parameters()] == [id(param) for param in params]
        assert not any(CP_ATTENTION_INNER_ATTR in name for name in names)
    finally:
        dist.destroy_process_group()


def test_inputs_embeds_only_and_multimodal_calls_report_different_problems(tmp_path):
    """One branch for both would blame an ``inputs_embeds``-only call on pixel features."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, config = _cp_wrapped_model()
        wrapper.model = _RecordingLogitsOnly(config)
        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)

        with pytest.raises(ValueError, match="needs input_ids"):
            wrapper(input_ids=None, inputs_embeds=torch.zeros(BATCH, SEQ_LEN, 32))
        with pytest.raises(ValueError, match="multimodal features"):
            wrapper(input_ids=ids, labels=ids, pixel_values=torch.zeros(BATCH, 3, 4, 4))
    finally:
        dist.destroy_process_group()


def test_cp_size_above_one_refuses_to_run_without_a_process_group():
    """Without a group, this process would attend over chunk 0 alone, training on ``1/cp_size`` of each sample."""
    with (
        patch(f"{_CP_MOD}.has_process_group", return_value=False),
        pytest.raises(RuntimeError, match="initialized torch.distributed"),
    ):
        CPConfig(2, stage_world_size=8, stage_base_rank=0, nvlink_domain_size=8)


def test_cp_size_one_still_builds_without_a_process_group():
    """Anti-vacuity: the single-process path must survive."""
    with patch(f"{_CP_MOD}.has_process_group", return_value=False):
        config = CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1)
    assert config.process_group is None and config.cp_rank == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
