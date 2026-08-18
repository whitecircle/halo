#!/usr/bin/env python
"""CPU tests for the streamed gathered save (FSDP2 / CP / TP).

``stream_gathered_checkpoint`` writes a gathered checkpoint chunk by chunk instead of buffering the
whole state dict, so the save rank's host RAM peaks at one decoder layer plus one pending shard
rather than the whole model (800 GB at Qwen3.5-397B-A17B). The artifact must stay indistinguishable
from the buffered writer's: the same keys, the same values, the same save-dtype and hub expert
layout, the same tie-consistency verdict. Only the part BOUNDARIES differ, which no reader sees.

    python tests/cpu/checkpoint/test_streamed_gathered_save.py
"""

import os

import pytest
import torch
from accelerate import PartialState
from transformers import CONFIG_MAPPING, AutoModelForCausalLM

PartialState()  # save_model_config logs through accelerate's logger

from safetensors import safe_open

import src.checkpoint.shard_writer as shard_writer_mod
import src.distributed.checkpoint.write as checkpoint_write_mod
import src.distributed.tensor_parallel.checkpoint as tp_checkpoint_mod
from src.checkpoint.format import load_full_state_dict, write_gathered_checkpoint
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.checkpoint.write import (
    chunked_saveable_tensors,
    conversion_chunk_key,
    stream_gathered_checkpoint,
)
from src.distributed.tensor_parallel.checkpoint import save_tp_model
from tests.common.checkpoint_io import weight_files

SHARD_SIZE = "64KB"


def _keys_in(shard_path) -> list[str]:
    """The keys one shard FILE holds — the index would hide a key claimed twice."""
    with safe_open(str(shard_path), framework="pt") as handle:
        return list(handle.keys())


def _tiny_moe():
    """A fused-expert MoE: transformers loads ``experts.gate_up_proj`` and the save must revert it to
    the per-expert hub layout, which is the one thing a per-chunk conversion could get wrong."""
    config = CONFIG_MAPPING["qwen3_moe"]()
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.num_hidden_layers = 3
    config.intermediate_size = 64
    config.moe_intermediate_size = 24
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.vocab_size = 128
    config.tie_word_embeddings = False
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config)


def _stream(model, output_dir, *, is_save_rank=True, max_shard_size=SHARD_SIZE):
    stream_gathered_checkpoint(
        model,
        chunked_saveable_tensors(model, retain=is_save_rank),
        output_dir,
        is_save_rank=is_save_rank,
        max_shard_size=max_shard_size,
    )


def test_chunks_partition_the_state_dict_by_layer():
    """Every key reaches exactly one chunk, and a decoder layer is never split across two — a split
    layer would hand the reverse conversion half of a fusion's sources."""
    model = _tiny_moe()
    chunks = list(chunked_saveable_tensors(model, retain=True))
    seen = [key for chunk in chunks for key in chunk]
    assert len(seen) == len(set(seen)), "a key reached two chunks"
    assert set(seen) == {name for name, _ in model.named_parameters()}
    assert len(chunks) == model.config.num_hidden_layers + 1, "one chunk per layer plus the rest"
    for chunk in chunks:
        assert len({conversion_chunk_key(key) for key in chunk}) == 1, sorted(chunk)


def test_non_writer_ranks_write_nothing(tmp_path):
    """``retain=False`` joins every gather and keeps nothing, so a non-writer must leave no file."""
    _stream(_tiny_moe(), str(tmp_path), is_save_rank=False)
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".safetensors")]


def test_streamed_artifact_matches_the_buffered_writer(tmp_path):
    """Same keys, same values, same hub expert layout — the file split is the only difference."""
    model = _tiny_moe()
    streamed, buffered = str(tmp_path / "streamed"), str(tmp_path / "buffered")
    os.makedirs(streamed)
    os.makedirs(buffered)

    _stream(model, streamed)
    write_gathered_checkpoint(model, {k: v.clone() for k, v in model.state_dict().items()}, buffered, SHARD_SIZE)

    streamed_state = load_full_state_dict(streamed)
    buffered_state = load_full_state_dict(buffered)
    assert set(streamed_state) == set(buffered_state)
    for key, tensor in streamed_state.items():
        assert torch.equal(tensor, buffered_state[key]), key
    fused = [key for key in streamed_state if key.endswith(("experts.gate_up_proj", "experts.down_proj"))]
    assert not fused, f"the per-chunk revert left module-fused expert keys: {fused}"
    assert any(".experts.0." in key for key in streamed_state), "premise: the hub layout is per-expert"


def test_writer_never_holds_more_than_one_shard(tmp_path):
    """The whole point: the pending batch is flushed before it can exceed ``max_shard_size``, so the
    writer's live tensors are bounded by one shard however large the model is."""
    peak_bytes, peak_count = [], []

    class _Recording(StageShardWriter):
        def add(self, key, tensor):
            super().add(key, tensor)
            peak_bytes.append(sum(t.numel() * t.element_size() for t in self._pending.values()))
            peak_count.append(len(self._pending))

    max_bytes = shard_writer_mod.parse_size_to_int(SHARD_SIZE)
    model = _tiny_moe()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(checkpoint_write_mod, "StageShardWriter", _Recording)
        _stream(model, str(tmp_path))

    total_keys = len(load_full_state_dict(str(tmp_path)))
    largest = max(p.numel() * p.element_size() for p in model.parameters())
    assert max(peak_bytes) <= max_bytes + largest, f"{max(peak_bytes)} > one shard ({max_bytes}) + one tensor"
    assert max(peak_count) < total_keys, "the writer held every key at once — nothing streamed"
    assert len([n for n in os.listdir(tmp_path) if n.endswith(".safetensors")]) > 1, "premise: multi-part"


def test_a_tp_hand_sliced_param_is_written_once_at_its_gathered_width(tmp_path, monkeypatch):
    """The TP save's trailing gather owns the hand-sliced params — the parameter walk must not also
    emit them.

    Those params (GptOss sinks) are not DTensors, so ``full_tensor()`` cannot reconstruct them and
    the walk yields this rank's ``[heads/tp]`` slice. A buffered writer overwrote it with the
    gathered tensor; the streamed one has already FLUSHED it, so the checkpoint would carry the key
    twice at two shapes — readable through the index, wrong to every tool that walks the shard files
    (``unfuse_moe_experts``, ``merge_models``, ``quantize_to_lowp``) — with ``total_size`` counting
    both.
    """
    model = _tiny_moe()
    attention = model.model.layers[0].self_attn
    attention.sinks = torch.nn.Parameter(torch.zeros(2))  # this rank's slice of 4 heads
    model._tp_sharded_non_dtensor = [("self_attn.sinks", 0)]
    key = "model.layers.0.self_attn.sinks"

    def _gathered(_model, state_dict, retain=True):
        """The mesh all-gather, stubbed to its RESULT: the full-width tensor on the save rank."""
        if retain:
            state_dict[key] = torch.ones(4)

    monkeypatch.setattr(tp_checkpoint_mod, "gather_tp_sharded_non_dtensor_params", _gathered)

    save_tp_model(model, str(tmp_path))

    written = [name for part in weight_files(str(tmp_path)) for name in _keys_in(tmp_path / part)]
    assert written.count(key) == 1, f"{key} was written {written.count(key)} times across the shard files"
    assert load_full_state_dict(str(tmp_path))[key].shape == (4,), "the checkpoint kept the TP slice"


def test_streamed_save_untangles_diverged_tied_embeddings(tmp_path):
    """The tie reconcile compares two keys that a streamed save meets in the same chunk but never
    holds beside the rest of the model; without it the config would claim a tie the weights broke."""
    model = _tiny_moe()
    model.config.tie_word_embeddings = True
    model.lm_head.weight.data.add_(1.0)
    _stream(model, str(tmp_path))
    assert model.config.tie_word_embeddings is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
