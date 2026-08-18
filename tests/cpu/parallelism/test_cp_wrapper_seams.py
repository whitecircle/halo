#!/usr/bin/env python
"""CPU tests for three ``UlyssesCPModelWrapper`` / ``CPConfig`` seams that fail silently otherwise.

* **Global positions are published, not gathered per layer.** The legacy attention path applies
  RoPE AFTER the all-to-all, where Q/K span the whole sequence, so a family hook needing positions
  (Mistral4's llama-4 scale) needs the FULL ``position_ids`` — which the wrapper holds, before the
  split. Rebuilding them inside each attention layer costs one ``dist.all_gather`` per layer per
  forward for a value the wrapper already holds.
* **The input guard names the actual problem.** An ``inputs_embeds``-only call must not be reported
  as "got multimodal features" — that branch is keyed on ``pixel_values``, not on ``input_ids``.
* **``cp_size > 1`` without a process group is refused.** Degrading to "every rank keeps chunk 0"
  quietly trains on ``1/cp_size`` of every sample behind a warning.

Run: ``python tests/cpu/parallelism/test_cp_wrapper_seams.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper

_CP_MOD = "src.distributed.context_parallel.config"

VOCAB = 64
SEQ_LEN = 8
BATCH = 2
NUM_LAYERS = 3


class _RecordingLogitsOnly(torch.nn.Module):
    """Stands in for the wrapped transformer, which cannot run in the CPU tier.

    CP's patched attention calls ``flash_attn_func`` directly and flash-attn has no CPU kernel. Only
    the wrapper's own ``forward`` is under test here; the real validation and patching still ran in
    :func:`_cp_wrapped_model`, and the layers they produced are what the publish is asserted on.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Embedding(VOCAB, VOCAB)
        self.seen: dict = {}

    def forward(self, input_ids=None, **kwargs):
        self.seen = {"input_ids": input_ids, **kwargs}
        return CausalLMOutputWithPast(logits=self.proj(input_ids).float())


def _cp_wrapped_model():
    """A real ``UlyssesCPModelWrapper`` over a tiny multi-layer Qwen3, patched for real."""
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
    # `validate_model_for_ulysses` requires a flash implementation, which transformers refuses to
    # instantiate on a CPU host. Build the tree eager and declare the label the validator reads:
    # no attention kernel is ever called.
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    return UlyssesCPModelWrapper(model, CPConfig(cp_size=1, world_size=1, gpus_per_node=1)), config


def _split_two_ways(wrapper):
    """Drive the wrapper as CP rank 0 of 2 without a 2-rank job: the split then really narrows, so
    'published the FULL positions' and 'published the local chunk' stop being the same tensor."""
    wrapper.cp_size = 2
    wrapper.cp_config.cp_size = 2
    return SEQ_LEN // 2


def test_every_patched_layer_is_collected_for_the_publish(tmp_path):
    """The cached list IS the patched set — a layer missing from it never gets the positions."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, _ = _cp_wrapped_model()
        patched = {id(m) for m in wrapper.model.modules() if isinstance(m, UlyssesAttentionBase)}

        assert len(patched) == NUM_LAYERS
        assert {id(layer) for layer in wrapper._attention_layers} == patched
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("explicit_positions", [False, True], ids=["arange", "given"])
def test_wrapper_publishes_the_full_positions_while_the_model_gets_the_chunk(tmp_path, explicit_positions):
    """The published tensor must span the whole sequence even though the model call is narrowed —
    the exact distinction the deleted per-layer all-gather existed to recover."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper, config = _cp_wrapped_model()
        layers = list(wrapper._attention_layers)
        inner = _RecordingLogitsOnly(config)
        wrapper.model = inner
        chunk = _split_two_ways(wrapper)

        ids = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
        # Packed-style positions: per-document resets, so only the real tensor reproduces them.
        packed = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long).expand(BATCH, SEQ_LEN).contiguous()
        expected = packed.clone() if explicit_positions else torch.arange(SEQ_LEN).expand(BATCH, SEQ_LEN)

        wrapper(input_ids=ids, labels=ids, position_ids=packed if explicit_positions else None)

        assert inner.seen["input_ids"].shape == (BATCH, chunk), "the model must still see its chunk"
        assert inner.seen["position_ids"].shape == (BATCH, chunk)
        for layer in layers:
            assert layer.global_position_ids is not None, "a patched layer got no global positions"
            assert layer.global_position_ids.shape == (BATCH, SEQ_LEN)
            torch.testing.assert_close(layer.global_position_ids, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_inputs_embeds_only_and_multimodal_calls_report_different_problems(tmp_path):
    """One branch for both meant an ``inputs_embeds``-only call was blamed on pixel features."""
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
    """No group means no all-to-all, so this process would attend over chunk 0 alone."""
    no_dist = types.SimpleNamespace(is_initialized=lambda: False)
    with patch(f"{_CP_MOD}.dist", no_dist), pytest.raises(RuntimeError, match="initialized torch.distributed"):
        CPConfig(cp_size=2, world_size=8, gpus_per_node=8)


def test_cp_size_one_still_builds_without_a_process_group():
    """Anti-vacuity: the single-process path is exactly the one that must survive."""
    no_dist = types.SimpleNamespace(is_initialized=lambda: False)
    with patch(f"{_CP_MOD}.dist", no_dist):
        config = CPConfig(cp_size=1, world_size=1, gpus_per_node=1)
    assert config.process_group is None and config.cp_rank == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
