#!/usr/bin/env python
"""CPU tests for merge_ep_shards.py post-processing logic.

Verifies that EP-internal weight formats are correctly converted to
HuggingFace checkpoint format for all supported MoE architectures. These
transforms are the contract between ``_save_ep_sharded`` and
``from_pretrained`` — a wrong transpose or stale key here produces a
checkpoint that silently loads garbage (or fails to load at all), so the
tests assert exact shapes AND values, not just key presence.
"""

import json
import os
import sys
import tempfile
from collections import defaultdict
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

import scripts.after_training.merge_ep_shards as merge_mod
from scripts.after_training.merge_ep_shards import (
    _EP_EXPERT_PATTERN,
    _EP_EXPERT_SUFFIXES,
    _group_expert_weights,
    merge_ep_shards,
)
from src.checkpoint.format import EP_SHARD_KEY_RE
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import expert_weight_roots, resolve_ep_merge_layer_class

# The transform lives ON the EP layer class (inverse of that class's own gather).
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.glm5_next import EPGlm5NextMoELayer
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer

# The whole-dict oracle the streamed merge is pinned against (test-only; see its module docstring).
from tests.common.ep_merge_oracle import post_process_merged_weights

_transform_gptoss = EPGptOssMoELayer.merge_shards_to_hf
_transform_qwen3 = EPQwen3MoELayer.merge_shards_to_hf
_transform_individual = EPBailingMoELayer.merge_shards_to_hf
_transform_fused_glu_experts = EPQwen3_5MoELayer.merge_shards_to_hf

E, H, M = 8, 64, 32  # num_experts, hidden_size, intermediate_size
PREFIX = "model.layers.0"
# The prefix passed to a transform includes the MoE container: `mlp`, or `feed_forward` for LFM2.
PREFIX_MLP = f"{PREFIX}.mlp"
PREFIX_FF = f"{PREFIX}.feed_forward"


def test_transform_gptoss_no_gmm():
    """GptOss without grouped GEMM: rename only, weights pass through verbatim."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    result = _transform_gptoss(PREFIX_MLP, {"gate_up_proj": gate_up, "down_proj": down})

    assert set(result) == {
        f"{PREFIX}.mlp.experts.gate_up_proj",
        f"{PREFIX}.mlp.experts.down_proj",
    }
    assert torch.equal(result[f"{PREFIX}.mlp.experts.gate_up_proj"], gate_up)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj"], down)


def test_transform_gptoss_gmm_interleaves_gate_up():
    """GptOss grouped GEMM stores de-interleaved gate/up — merge re-interleaves them.

    Regression: gate must land on even channels and up on odd channels of the
    fused [E, H, 2M] tensor; a swap silently corrupts the SwiGLU.
    """
    gate = torch.randn(E, H, M)
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)
    result = _transform_gptoss(PREFIX_MLP, {"gate_proj_gmm": gate, "up_proj_gmm": up, "down_proj": down})

    gate_up = result[f"{PREFIX}.mlp.experts.gate_up_proj"]
    assert gate_up.shape == (E, H, 2 * M)
    assert torch.equal(gate_up[:, :, ::2], gate)
    assert torch.equal(gate_up[:, :, 1::2], up)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj"], down)


def test_transform_gptoss_gmm_with_bias():
    """GptOss grouped GEMM biases re-interleave the same way as the weights."""
    gate, up, down = torch.randn(E, H, M), torch.randn(E, H, M), torch.randn(E, M, H)
    gate_bias, up_bias, down_bias = torch.randn(E, M), torch.randn(E, M), torch.randn(E, H)
    result = _transform_gptoss(
        PREFIX_MLP,
        {
            "gate_proj_gmm": gate,
            "up_proj_gmm": up,
            "down_proj": down,
            "gate_proj_gmm_bias": gate_bias,
            "up_proj_gmm_bias": up_bias,
            "down_proj_bias": down_bias,
        },
    )

    bias = result[f"{PREFIX}.mlp.experts.gate_up_proj_bias"]
    assert bias.shape == (E, 2 * M)
    assert torch.equal(bias[:, ::2], gate_bias)
    assert torch.equal(bias[:, 1::2], up_bias)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj_bias"], down_bias)


def test_transform_gptoss_no_gmm_with_bias():
    """Non-gmm GptOss biases pass through under the experts prefix unchanged."""
    gate_up_bias = torch.randn(E, 2 * M)
    down_bias = torch.randn(E, H)
    result = _transform_gptoss(
        PREFIX_MLP,
        {
            "gate_up_proj": torch.randn(E, H, 2 * M),
            "down_proj": torch.randn(E, M, H),
            "gate_up_proj_bias": gate_up_bias,
            "down_proj_bias": down_bias,
        },
    )
    assert torch.equal(result[f"{PREFIX}.mlp.experts.gate_up_proj_bias"], gate_up_bias)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj_bias"], down_bias)


def test_transform_qwen3_unfuses_to_individual_experts():
    """Qwen3 MoE: matmul-convention 3D tensors → per-expert nn.Linear weights.

    EP keeps gate_proj/up_proj/down_proj fused [E, H, M]; the HF Qwen3 checkpoint
    wants one tensor per expert in nn.Linear convention. Verify the transpose
    direction per expert (a wrong transpose loads but produces garbage logits).
    """
    gate = torch.randn(E, H, M)
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)
    result = _transform_qwen3(PREFIX_MLP, {"gate_proj": gate, "up_proj": up, "down_proj": down})

    assert len(result) == 3 * E
    for i in range(E):
        g = result[f"{PREFIX}.mlp.experts.{i}.gate_proj.weight"]
        u = result[f"{PREFIX}.mlp.experts.{i}.up_proj.weight"]
        d = result[f"{PREFIX}.mlp.experts.{i}.down_proj.weight"]
        assert g.shape == (M, H) and u.shape == (M, H) and d.shape == (H, M)
        assert torch.equal(g, gate[i].transpose(0, 1))
        assert torch.equal(u, up[i].transpose(0, 1))
        assert torch.equal(d, down[i].transpose(0, 1))


def test_transform_fused_glu_experts():
    """Qwen3.5/DeepSeek-V4: transpose fused GLU matmul→nn.Linear under the experts prefix."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    result = _transform_fused_glu_experts(PREFIX_MLP, {"gate_up_proj": gate_up, "down_proj": down})

    gate_up_out = result[f"{PREFIX}.mlp.experts.gate_up_proj"]
    down_out = result[f"{PREFIX}.mlp.experts.down_proj"]
    assert gate_up_out.shape == (E, 2 * M, H)
    assert down_out.shape == (E, H, M)
    assert torch.equal(gate_up_out, gate_up.transpose(1, 2))
    assert torch.equal(down_out, down.transpose(1, 2))


def test_transform_fused_glu_feed_forward_container():
    """The prefix carries the MoE container (``feed_forward`` for LFM2-style blocks, ``mlp``
    elsewhere): the transform re-emits the experts under exactly that block."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    result = _transform_fused_glu_experts(PREFIX_FF, {"gate_up_proj": gate_up, "down_proj": down})

    assert set(result) == {
        f"{PREFIX}.feed_forward.experts.gate_up_proj",
        f"{PREFIX}.feed_forward.experts.down_proj",
    }
    assert torch.equal(result[f"{PREFIX}.feed_forward.experts.gate_up_proj"], gate_up.transpose(1, 2))
    assert torch.equal(result[f"{PREFIX}.feed_forward.experts.down_proj"], down.transpose(1, 2))


def test_transform_fused_glu_biases_pass_through():
    """Fused-GLU biases are 1-D per-expert and pass through without transpose."""
    gate_up_bias = torch.randn(E, 2 * M)
    down_bias = torch.randn(E, H)
    result = _transform_fused_glu_experts(
        PREFIX_MLP,
        {
            "gate_up_proj": torch.randn(E, H, 2 * M),
            "down_proj": torch.randn(E, M, H),
            "gate_up_proj_bias": gate_up_bias,
            "down_proj_bias": down_bias,
        },
    )
    assert torch.equal(result[f"{PREFIX}.mlp.experts.gate_up_proj_bias"], gate_up_bias)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj_bias"], down_bias)


def test_transform_individual():
    """Bailing MoE: 3D fused → individual per-expert keys (transposed)."""
    gate = torch.randn(E, H, M)
    up = torch.randn(E, H, M)
    down = torch.randn(E, M, H)
    result = _transform_individual(PREFIX_MLP, {"gate_proj": gate, "up_proj": up, "down_proj": down})

    assert len(result) == 3 * E
    for i in range(E):
        g = result[f"{PREFIX}.mlp.experts.{i}.gate_proj.weight"]
        u = result[f"{PREFIX}.mlp.experts.{i}.up_proj.weight"]
        d = result[f"{PREFIX}.mlp.experts.{i}.down_proj.weight"]
        assert g.shape == (M, H) and u.shape == (M, H) and d.shape == (H, M)
        assert torch.equal(g, gate[i].transpose(0, 1))
        assert torch.equal(u, up[i].transpose(0, 1))
        assert torch.equal(d, down[i].transpose(0, 1))


def test_group_expert_weights():
    """Expert weights group by layer; router/attention/shared-expert stay non-expert."""
    weights = {
        f"{PREFIX}.mlp.gate_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.up_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),  # router
        f"{PREFIX}.self_attn.q_proj.weight": torch.randn(H, H),
        f"{PREFIX}.mlp.shared_expert.gate_proj.weight": torch.randn(M, H),
    }

    expert_groups, non_expert = _group_expert_weights(weights)

    assert set(expert_groups[PREFIX_MLP].keys()) == {"gate_proj", "up_proj", "down_proj"}
    assert f"{PREFIX}.mlp.gate.weight" in non_expert
    assert f"{PREFIX}.self_attn.q_proj.weight" in non_expert
    # Trailing `.weight` keeps the shared expert off the bare ".mlp.gate_proj" expert suffix.
    assert f"{PREFIX}.mlp.shared_expert.gate_proj.weight" in non_expert


def test_group_expert_weights_multi_layer():
    """Grouping keys each MoE layer independently."""
    weights = {}
    for layer in range(3):
        p = f"model.layers.{layer}"
        weights[f"{p}.mlp.gate_up_proj"] = torch.randn(E, H, 2 * M)
        weights[f"{p}.mlp.down_proj"] = torch.randn(E, M, H)
    expert_groups, non_expert = _group_expert_weights(weights)
    assert set(expert_groups) == {f"model.layers.{i}.mlp" for i in range(3)}
    assert non_expert == {}


def test_group_expert_weights_feed_forward_container():
    """LFM2 expert keys live under ``.feed_forward.`` and must group (regression: a pattern
    hard-requiring ``.mlp.`` classifies LFM2 experts as non-expert and passes them through
    unmerged)."""
    weights = {
        f"{PREFIX}.feed_forward.gate_up_proj": torch.randn(E, H, 2 * M),
        f"{PREFIX}.feed_forward.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.feed_forward.gate.weight": torch.randn(E, H),
    }
    expert_groups, non_expert = _group_expert_weights(weights)
    assert set(expert_groups) == {PREFIX_FF}
    assert set(expert_groups[PREFIX_FF].keys()) == {"gate_up_proj", "down_proj"}
    assert f"{PREFIX}.feed_forward.gate.weight" in non_expert


def test_post_process_qwen3():
    """End-to-end Qwen3 post-processing: individual experts + router preserved."""
    weights = {
        f"{PREFIX}.mlp.gate_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.up_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),  # router
    }
    result = post_process_merged_weights(weights, "qwen3_moe", verbose=False)

    expert_keys = [k for k in result if ".experts." in k]
    assert len(expert_keys) == 3 * E
    assert result[f"{PREFIX}.mlp.experts.0.gate_proj.weight"].shape == (M, H)
    assert result[f"{PREFIX}.mlp.experts.0.down_proj.weight"].shape == (H, M)
    assert f"{PREFIX}.mlp.gate.weight" in result


def test_post_process_gptoss_gmm():
    """End-to-end GptOss grouped-GEMM post-processing keeps HF [E, H, 2M] layout."""
    weights = {
        f"{PREFIX}.mlp.gate_proj_gmm": torch.randn(E, H, M),
        f"{PREFIX}.mlp.up_proj_gmm": torch.randn(E, H, M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.mlp.router.weight": torch.randn(E, H),
    }
    result = post_process_merged_weights(weights, "gpt_oss", verbose=False)

    assert result[f"{PREFIX}.mlp.experts.gate_up_proj"].shape == (E, H, 2 * M)
    assert result[f"{PREFIX}.mlp.experts.down_proj"].shape == (E, M, H)
    assert f"{PREFIX}.mlp.router.weight" in result


def test_post_process_glm4():
    """GLM4 MoE Lite: fused runtime shards → per-expert hub keys
    ``mlp.experts.{i}.{gate,up,down}_proj.weight`` — the layout the gathered save emits and vLLM's
    per-expert loader requires; a fused ``experts.gate_up_proj`` output loads in transformers but
    cannot be served/synced. Values must equal the exact [gate; up] half split; shared expert +
    router pass through."""
    gate_up = torch.randn(E, H, 2 * M)  # matmul convention, halves [gate; up] on dim 2
    down = torch.randn(E, M, H)
    weights = {
        f"{PREFIX}.mlp.gate_up_proj": gate_up,
        f"{PREFIX}.mlp.down_proj": down,
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),
        f"{PREFIX}.mlp.shared_experts.gate_proj.weight": torch.randn(M, H),
    }
    result = post_process_merged_weights(weights, "glm4_moe_lite", verbose=False)

    expert_keys = {k for k in result if f"{PREFIX}.mlp.experts." in k}
    assert expert_keys == {
        f"{PREFIX}.mlp.experts.{i}.{proj}.weight" for i in range(E) for proj in ("gate_proj", "up_proj", "down_proj")
    }
    flinear = gate_up.transpose(1, 2)  # [E, 2M, H], halves [gate; up] on dim 1
    for i in range(E):
        assert torch.equal(result[f"{PREFIX}.mlp.experts.{i}.gate_proj.weight"], flinear[i, :M])
        assert torch.equal(result[f"{PREFIX}.mlp.experts.{i}.up_proj.weight"], flinear[i, M:])
        assert torch.equal(result[f"{PREFIX}.mlp.experts.{i}.down_proj.weight"], down[i].transpose(0, 1))
    assert not any("routed_experts" in k for k in result), "GLM4 must not emit the stale routed_experts prefix"
    assert f"{PREFIX}.mlp.gate.weight" in result
    assert f"{PREFIX}.mlp.shared_experts.gate_proj.weight" in result
    assert f"{PREFIX}.mlp.gate_up_proj" not in result


def test_post_process_lfm2_feed_forward_container():
    """LFM2 MoE: experts live under ``feed_forward`` (regression: the merge once required ``.mlp.``
    so LFM2 experts passed through unmerged → dropped on reload) and the hub layout is per-expert
    Llama-style names ``experts.{i}.w{1,3,2}.weight`` (w1 = gate, w3 = up, w2 = down) — the split
    order must match the class-declared _PER_EXPERT_UNFUSED_KEYS exactly (a swap has identical
    shapes and corrupts the SwiGLU silently)."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    weights = {
        f"{PREFIX}.feed_forward.gate_up_proj": gate_up,
        f"{PREFIX}.feed_forward.down_proj": down,
        f"{PREFIX}.feed_forward.gate.weight": torch.randn(E, H),
    }
    result = post_process_merged_weights(weights, "lfm2_moe", verbose=False)

    expert_keys = {k for k in result if f"{PREFIX}.feed_forward.experts." in k}
    assert expert_keys == {f"{PREFIX}.feed_forward.experts.{i}.w{n}.weight" for i in range(E) for n in (1, 2, 3)}
    flinear = gate_up.transpose(1, 2)
    for i in range(E):
        assert torch.equal(result[f"{PREFIX}.feed_forward.experts.{i}.w1.weight"], flinear[i, :M])  # gate
        assert torch.equal(result[f"{PREFIX}.feed_forward.experts.{i}.w3.weight"], flinear[i, M:])  # up
        assert torch.equal(result[f"{PREFIX}.feed_forward.experts.{i}.w2.weight"], down[i].transpose(0, 1))
    assert f"{PREFIX}.feed_forward.gate.weight" in result
    assert f"{PREFIX}.feed_forward.gate_up_proj" not in result


def test_post_process_glm4_unexpected_expert_param_raises():
    """A shard layout the per-expert split does not cover (e.g. a stray bias) must fail loud —
    silently dropping it would corrupt the merged checkpoint."""
    weights = {
        f"{PREFIX}.mlp.gate_up_proj": torch.randn(E, H, 2 * M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.mlp.gate_up_proj_bias": torch.randn(E, 2 * M),
    }
    with pytest.raises(ValueError, match="unexpected expert params"):
        post_process_merged_weights(weights, "glm4_moe_lite", verbose=False)


def test_post_process_deepseek_v4():
    """DeepSeek-V4: fused GLU under ``mlp.experts`` (contiguous [gate; up] halves, Qwen3.5-style
    transpose back to nn.Linear convention); router gate + shared expert pass through, and the
    transpose round-trips values exactly."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    weights = {
        f"{PREFIX}.mlp.gate_up_proj": gate_up,
        f"{PREFIX}.mlp.down_proj": down,
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),
        f"{PREFIX}.mlp.shared_experts.gate_proj.weight": torch.randn(M, H),
    }
    result = post_process_merged_weights(weights, "deepseek_v4", verbose=False)

    assert result[f"{PREFIX}.mlp.experts.gate_up_proj"].shape == (E, 2 * M, H)
    assert result[f"{PREFIX}.mlp.experts.down_proj"].shape == (E, H, M)
    assert torch.equal(result[f"{PREFIX}.mlp.experts.gate_up_proj"], gate_up.transpose(1, 2))
    assert torch.equal(result[f"{PREFIX}.mlp.experts.down_proj"], down.transpose(1, 2))
    assert f"{PREFIX}.mlp.gate.weight" in result
    assert f"{PREFIX}.mlp.shared_experts.gate_proj.weight" in result
    assert f"{PREFIX}.mlp.gate_up_proj" not in result


def test_post_process_bailing():
    """Bailing MoE: individual experts, 3 keys per expert."""
    weights = {
        f"{PREFIX}.mlp.gate_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.up_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
    }
    result = post_process_merged_weights(weights, "bailing_moe", verbose=False)

    assert len([k for k in result if "experts." in k]) == 3 * E
    assert result[f"{PREFIX}.mlp.experts.0.gate_proj.weight"].shape == (M, H)
    assert result[f"{PREFIX}.mlp.experts.0.down_proj.weight"].shape == (H, M)


def test_post_process_no_expert_weights_is_noop():
    """A dense (non-MoE) state dict has no EP expert keys and is returned as-is."""
    weights = {
        f"{PREFIX}.self_attn.q_proj.weight": torch.randn(H, H),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),
    }
    result = post_process_merged_weights(weights, "unknown_model", verbose=False)
    assert result is weights


def test_post_process_unknown_type_with_experts_raises():
    """An unmapped model_type WITH expert weights must fail loudly, never emit an
    EP-internal checkpoint that from_pretrained() cannot load (regression guard)."""
    weights = {
        f"{PREFIX}.mlp.gate_up_proj": torch.randn(E, H, 2 * M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
    }
    with pytest.raises(ValueError, match="no HF-layout transform"):
        post_process_merged_weights(weights, "mistral4_moe", verbose=False)


def test_model_type_resolves_to_the_transform_of_its_layer_class():
    """The merge picks a transform by resolving model_type -> EP layer class -> that class's
    ``merge_shards_to_hf``, so every spelling a class claims is mergeable — including Laguna, whose
    own wrapper inherits GLM-4's transform unchanged."""
    assert resolve_ep_merge_layer_class("glm4_moe_lite") is EPGlm4MoELayer
    assert resolve_ep_merge_layer_class("laguna") is EPLagunaMoELayer
    assert EPLagunaMoELayer.merge_shards_to_hf.__func__ is EPGlm4MoELayer.merge_shards_to_hf.__func__
    assert resolve_ep_merge_layer_class("bailing_moe_linear") is resolve_ep_merge_layer_class("bailing_moe")
    # No spelling heuristic: an unclaimed model_type is rejected, never guessed onto a layout.
    assert resolve_ep_merge_layer_class("my_glm4_moe_variant") is None


def _write_ep_checkpoint(input_dir, model_type, shards, ep_size):
    """Write per-rank shard files + index for a simulated EP-sharded checkpoint."""
    with open(os.path.join(input_dir, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "num_local_experts": E}, f)

    weight_map = {}
    for rank, shard in enumerate(shards):
        fname = f"model-{rank:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(input_dir, fname))
        for k in shard:
            weight_map[k] = fname

    index = {"metadata": {"ep_size": ep_size, "format": "ep_sharded"}, "weight_map": weight_map}
    with open(os.path.join(input_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def _load_merged(output_dir):
    merged = {}
    for f in os.listdir(output_dir):
        if f.endswith(".safetensors"):
            merged.update(load_file(os.path.join(output_dir, f)))
    return merged


def test_full_merge_qwen3_round_trip_values():
    """Full Qwen3 merge: shards concatenate along expert dim and the per-expert
    output exactly equals the transposed original slice (the load-time contract)."""
    ep_size = 2
    per = E // ep_size
    full_gate = torch.randn(E, H, M)
    full_up = torch.randn(E, H, M)
    full_down = torch.randn(E, M, H)
    router = torch.randn(E, H)

    shard0 = {
        f"{PREFIX}.mlp.gate_proj.shard_0": full_gate[:per].clone(),
        f"{PREFIX}.mlp.up_proj.shard_0": full_up[:per].clone(),
        f"{PREFIX}.mlp.down_proj.shard_0": full_down[:per].clone(),
        f"{PREFIX}.mlp.gate.weight": router,  # router only on rank 0
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_proj.shard_1": full_gate[per:].clone(),
        f"{PREFIX}.mlp.up_proj.shard_1": full_up[per:].clone(),
        f"{PREFIX}.mlp.down_proj.shard_1": full_down[per:].clone(),
    }

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "qwen3_moe", [shard0, shard1], ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        merged = _load_merged(output_dir)

        assert torch.equal(merged[f"{PREFIX}.mlp.gate.weight"], router)
        for i in range(E):
            g = merged[f"{PREFIX}.mlp.experts.{i}.gate_proj.weight"]
            assert g.shape == (M, H)
            assert torch.equal(g, full_gate[i].transpose(0, 1))
        assert not any(".shard_" in k for k in merged)
        assert f"{PREFIX}.mlp.gate_proj" not in merged


def test_full_merge_gptoss_round_trip_values():
    """Full GptOss (non-gmm) merge: HF [E, H, 2M] layout, exact values."""
    ep_size = 2
    per = E // ep_size
    full_gate_up = torch.randn(E, H, 2 * M)
    full_down = torch.randn(E, M, H)

    shard0 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_0": full_gate_up[:per].clone(),
        f"{PREFIX}.mlp.down_proj.shard_0": full_down[:per].clone(),
        f"{PREFIX}.mlp.router.weight": torch.randn(E, H),
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_1": full_gate_up[per:].clone(),
        f"{PREFIX}.mlp.down_proj.shard_1": full_down[per:].clone(),
    }

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", [shard0, shard1], ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        merged = _load_merged(output_dir)

        gu = merged[f"{PREFIX}.mlp.experts.gate_up_proj"]
        assert gu.shape == (E, H, 2 * M)
        assert torch.equal(gu, full_gate_up)
        assert torch.equal(merged[f"{PREFIX}.mlp.experts.down_proj"], full_down)


def test_full_merge_re_emits_every_tensor_at_its_stored_dtype():
    """The sharded writer already applied ``save_dtype_caster`` (bf16 everywhere except the
    module-tree keep-set: norms, balancing tensors, the family's fp32 pins), so the merge must
    re-emit each tensor exactly as stored. A second, name-only cast has no model tree to derive the
    pins from and folded them to bf16 — GLM-5 Next's ``A_log``/``dt_bias``, Inkling's short
    convolutions, DeepSeek-V4's ``attn_hc`` — breaking merged-from-sharded == gathered on the very
    tensors pinned because bf16 breaks their arithmetic."""
    ep_size = 2
    per = E // ep_size
    a_log = torch.randn(4, dtype=torch.float32)  # an fp32 pin that is neither a norm nor balancing state
    shard0 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_0": torch.randn(per, H, 2 * M, dtype=torch.bfloat16),
        f"{PREFIX}.mlp.down_proj.shard_0": torch.randn(per, M, H, dtype=torch.bfloat16),
        f"{PREFIX}.mlp.router.weight": torch.randn(E, H, dtype=torch.bfloat16),
        f"{PREFIX}.linear_attn.A_log": a_log,
        f"{PREFIX}.input_layernorm.weight": torch.randn(H, dtype=torch.float32),
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_1": torch.randn(per, H, 2 * M, dtype=torch.bfloat16),
        f"{PREFIX}.mlp.down_proj.shard_1": torch.randn(per, M, H, dtype=torch.bfloat16),
    }
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", [shard0, shard1], ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        merged = _load_merged(output_dir)
    assert merged[f"{PREFIX}.mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert merged[f"{PREFIX}.mlp.router.weight"].dtype == torch.bfloat16
    assert merged[f"{PREFIX}.input_layernorm.weight"].dtype == torch.float32
    assert merged[f"{PREFIX}.linear_attn.A_log"].dtype == torch.float32, "an fp32 pin was folded to bf16"
    assert torch.equal(merged[f"{PREFIX}.linear_attn.A_log"], a_log)


def test_merge_copies_auxiliary_files():
    """Tokenizer/config files are copied to the merged dir; input shards are not."""
    ep_size = 2
    per = E // ep_size
    shard0 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_0": torch.randn(per, H, 2 * M),
        f"{PREFIX}.mlp.down_proj.shard_0": torch.randn(per, M, H),
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_1": torch.randn(per, H, 2 * M),
        f"{PREFIX}.mlp.down_proj.shard_1": torch.randn(per, M, H),
    }
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", [shard0, shard1], ep_size)
        with open(os.path.join(input_dir, "tokenizer.json"), "w") as f:
            json.dump({"type": "test"}, f)

        merge_ep_shards(input_dir, output_dir, verbose=False)

        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "tokenizer.json"))
        # Output is HF-style model-*.safetensors; the input shard names are rewritten, not copied.
        assert not any(".shard_" in k for k in _load_merged(output_dir))


def _gathered_reference(layer_cls, fused_flinear):
    """What the GATHERED save of the same weights emits: the family's ``gather_expert_state_dict``
    with the base fused gather stubbed to the F.linear-convention full tensors (no GPU/EP state
    needed). This is the invariant target for the sharded merge — vLLM's per-expert loaders and the
    hub conversion consume exactly this layout."""
    instance = object.__new__(layer_cls)
    with patch.object(EPMoELayerBase, "_gather_fused_expert_state_dict", return_value=dict(fused_flinear)):
        return layer_cls.gather_expert_state_dict(instance, device="cpu")


def _roundtrip_vs_gathered(layer_cls, model_type, container):
    """Write a tiny 2-rank fused sharded checkpoint, merge it, and assert the expert output is
    key-and-tensor identical to the gathered save of the same weights. bf16 inputs, as the sharded
    writer stores them."""
    ep_size = 2
    per = E // ep_size
    gate_up = torch.randn(E, H, 2 * M, dtype=torch.bfloat16)  # matmul convention runtime layout
    down = torch.randn(E, M, H, dtype=torch.bfloat16)
    router = torch.randn(E, H, dtype=torch.bfloat16)

    shard0 = {
        f"{PREFIX}.{container}.gate_up_proj.shard_0": gate_up[:per].clone(),
        f"{PREFIX}.{container}.down_proj.shard_0": down[:per].clone(),
        f"{PREFIX}.{container}.gate.weight": router,  # router only on rank 0
    }
    shard1 = {
        f"{PREFIX}.{container}.gate_up_proj.shard_1": gate_up[per:].clone(),
        f"{PREFIX}.{container}.down_proj.shard_1": down[per:].clone(),
    }

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, model_type, [shard0, shard1], ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        merged = _load_merged(output_dir)

    reference = {
        f"{PREFIX}.{container}.{key}": tensor
        for key, tensor in _gathered_reference(
            layer_cls,
            {
                "experts.gate_up_proj": gate_up.transpose(1, 2).contiguous(),
                "experts.down_proj": down.transpose(1, 2).contiguous(),
            },
        ).items()
    }
    expert_keys = set(merged) - {f"{PREFIX}.{container}.gate.weight"}
    assert expert_keys == set(reference), (
        f"merged-from-sharded expert keys diverge from the gathered save:\n"
        f"  merged-only:   {sorted(expert_keys - set(reference))[:6]}\n"
        f"  gathered-only: {sorted(set(reference) - expert_keys)[:6]}"
    )
    for key, tensor in reference.items():
        assert torch.equal(merged[key], tensor), f"{key} diverges from the gathered save"
    # The key-set compare above is what rejects leftover runtime keys: for Gemma4 the HF key IS
    # ``<prefix>.gate_up_proj``, so a "no runtime key" check would be vacuous.
    assert torch.equal(merged[f"{PREFIX}.{container}.gate.weight"], router)
    assert not any(".shard_" in k for k in merged)


@pytest.mark.parametrize(
    ("layer_cls", "model_type", "container"),
    [
        # Per-expert hub layouts, split by the class-declared _PER_EXPERT_UNFUSED_KEYS.
        (EPGlm4MoELayer, "glm4_moe_lite", "mlp"),
        (EPLagunaMoELayer, "laguna", "mlp"),
        (EPLfm2MoELayer, "lfm2_moe", "feed_forward"),
        # Fused hub layouts: the base merge is the exact inverse of the base gather.
        (EPCohere2MoELayer, "cohere2_moe", "mlp"),
        (EPQwen3_5MoELayer, "qwen3_5_moe", "mlp"),
        (EPGlm5NextMoELayer, "glm5_next", "mlp"),
        (EPDeepseekV4MoELayer, "deepseek_v4", "mlp"),
        (EPMistral4MoELayer, "mistral4", "mlp"),
        (EPZayaMoELayer, "zaya", "mlp"),
        # Overriding families: the merge override must track the gather override key for key.
        (EPGemma4MoELayer, "gemma4", "mlp.experts"),
    ],
)
def test_full_merge_matches_gathered_save(layer_cls, model_type, container):
    """merged-from-sharded == gathered, key and tensor, for every family with fused expert storage.

    This is the invariant that lets the transform live on the class: a family is merge-supported
    because its ``merge_shards_to_hf`` is the inverse of its own ``gather_expert_state_dict``, and
    this asserts exactly that end to end through the real merge script."""
    _roundtrip_vs_gathered(layer_cls, model_type, container)


def test_merge_renames_module_spellings_to_hub_spellings():
    """Laguna is the roster's rename-bearing family (``_EXPORT_KEY_RENAMES``): the merge must
    re-spell ALL output keys — non-expert included — to the hub layout, exactly as the gathered
    save does. A dropped rename ships ``gate.e_score_correction_bias`` / ``shared_experts.*``,
    which from_pretrained treats as unexpected+missing: the served model silently routes on the
    pretrained bias with a random shared expert. The balancing tensor keeps its stored fp32 (a
    bf16 round trip quantizes away the ~1e-3 sign steps)."""
    ep_size = 2
    per = E // ep_size
    balancing = torch.randn(E, dtype=torch.float32)
    shared = torch.randn(M, H, dtype=torch.float32)
    shard0 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_0": torch.randn(per, H, 2 * M, dtype=torch.float32),
        f"{PREFIX}.mlp.down_proj.shard_0": torch.randn(per, M, H, dtype=torch.float32),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H, dtype=torch.float32),
        f"{PREFIX}.mlp.gate.e_score_correction_bias": balancing,
        f"{PREFIX}.mlp.shared_experts.gate_proj.weight": shared,
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_1": torch.randn(per, H, 2 * M, dtype=torch.float32),
        f"{PREFIX}.mlp.down_proj.shard_1": torch.randn(per, M, H, dtype=torch.float32),
    }
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "laguna", [shard0, shard1], ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        merged = _load_merged(output_dir)

    assert f"{PREFIX}.mlp.experts.e_score_correction_bias" in merged
    assert f"{PREFIX}.mlp.shared_expert.gate_proj.weight" in merged
    assert not any(".gate.e_score_correction_bias" in k or ".shared_experts." in k for k in merged), (
        "module spellings leaked into the merged checkpoint"
    )
    bias = merged[f"{PREFIX}.mlp.experts.e_score_correction_bias"]
    assert bias.dtype == torch.float32
    assert torch.equal(bias, balancing)
    assert torch.equal(merged[f"{PREFIX}.mlp.shared_expert.gate_proj.weight"], shared)


def _reference_merge(input_dir, model_type):
    """The whole-dict merge pipeline: load every shard, concatenate ``.shard_N`` groups on dim 0,
    post-process. This is the semantics the streamed script must reproduce byte for byte."""
    all_weights = {}
    for fname in sorted(f for f in os.listdir(input_dir) if merge_mod.is_ep_shard(f)):
        all_weights.update(load_file(os.path.join(input_dir, fname)))
    shard_groups = defaultdict(dict)
    reference = {}
    for key, tensor in all_weights.items():
        match = EP_SHARD_KEY_RE.match(key)
        if match:
            shard_groups[match.group(1)][int(match.group(2))] = tensor
        else:
            reference[key] = tensor
    for base_name, shards in shard_groups.items():
        reference[base_name] = torch.cat([shards[i] for i in sorted(shards)], dim=0)
    return post_process_merged_weights(reference, model_type, verbose=False, had_expert_shards=bool(shard_groups))


def test_streamed_merge_is_key_and_value_identical_to_the_reference_merge():
    """The streamed merge must write exactly what the whole-dict pipeline produces — key set,
    shapes, dtypes and bytes — over multiple MoE layers, per-expert splits, routers, a bf16
    embedding and an fp32 norm, with the output forced into MULTIPLE parts by a tiny
    ``max_shard_size``. A streaming bug at any group or flush boundary shows up as a tensor diff,
    not just a key diff."""
    ep_size = 2
    per = E // ep_size
    shard0, shard1 = {}, {}
    for layer in range(3):
        p = f"model.layers.{layer}.mlp"
        gate = torch.randn(E, H, M, dtype=torch.bfloat16)
        up = torch.randn(E, H, M, dtype=torch.bfloat16)
        down = torch.randn(E, M, H, dtype=torch.bfloat16)
        shard0[f"{p}.gate_proj.shard_0"] = gate[:per].clone()
        shard0[f"{p}.up_proj.shard_0"] = up[:per].clone()
        shard0[f"{p}.down_proj.shard_0"] = down[:per].clone()
        shard0[f"{p}.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        shard1[f"{p}.gate_proj.shard_1"] = gate[per:].clone()
        shard1[f"{p}.up_proj.shard_1"] = up[per:].clone()
        shard1[f"{p}.down_proj.shard_1"] = down[per:].clone()
    shard0["model.embed_tokens.weight"] = torch.randn(16, H, dtype=torch.bfloat16)
    shard0["model.norm.weight"] = torch.randn(H, dtype=torch.float32)  # the writer keeps norms at trained dtype

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "qwen3_moe", [shard0, shard1], ep_size)
        reference = _reference_merge(input_dir, "qwen3_moe")

        merge_ep_shards(input_dir, output_dir, verbose=False, max_shard_size="100KB")
        merged = _load_merged(output_dir)

        assert set(merged) == set(reference), (
            f"streamed keys diverge from the reference merge:\n"
            f"  streamed-only:  {sorted(set(merged) - set(reference))[:6]}\n"
            f"  reference-only: {sorted(set(reference) - set(merged))[:6]}"
        )
        for key, expected in reference.items():
            assert merged[key].dtype == expected.dtype, key
            assert merged[key].shape == expected.shape, key
            assert torch.equal(merged[key], expected), f"{key} diverges from the reference merge"

        # The tiny cap must have actually sharded the output, with an index naming every key — a
        # single-part output would leave the flush boundaries untested.
        parts = sorted(f for f in os.listdir(output_dir) if f.endswith(".safetensors"))
        assert len(parts) > 1, "fixture too small to exercise multi-part streaming"
        with open(os.path.join(output_dir, "model.safetensors.index.json")) as fh:
            index = json.load(fh)
        assert set(index["weight_map"]) == set(reference)
        assert set(index["weight_map"].values()) == set(parts)
        # In-flight stream parts must have been renamed to HF-standard names.
        assert not any(f.startswith("model-streaming") for f in os.listdir(output_dir))


def _gptoss_shards():
    per = E // 2
    shard0 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_0": torch.randn(per, H, 2 * M),
        f"{PREFIX}.mlp.down_proj.shard_0": torch.randn(per, M, H),
        f"{PREFIX}.mlp.router.weight": torch.randn(E, H),
    }
    shard1 = {
        f"{PREFIX}.mlp.gate_up_proj.shard_1": torch.randn(per, H, 2 * M),
        f"{PREFIX}.mlp.down_proj.shard_1": torch.randn(per, M, H),
    }
    return [shard0, shard1]


def _assert_failed_merge_preserves_inputs(monkeypatch, writer_method, failing):
    """Inject ``failing`` into one StageShardWriter method and assert every input shard survives."""
    monkeypatch.setattr(StageShardWriter, writer_method, failing)
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", _gptoss_shards(), 2)
        shard_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))

        with pytest.raises(OSError, match="No space left"):
            merge_ep_shards(input_dir, output_dir, verbose=False, delete_input_shards=True)

        for f in shard_files:
            assert os.path.exists(os.path.join(input_dir, f)), f"input shard {f} destroyed by a failed merge"


def test_delete_input_shards_survives_midstream_write_failure(monkeypatch):
    """A merge that fails while streaming tensors out must leave every input shard on disk — the
    pre-streaming code once deleted each shard right after loading it, so a failed save destroyed
    the only copy of the trained weights."""

    def exploding_add(self, key, tensor):
        raise OSError("No space left on device")

    _assert_failed_merge_preserves_inputs(monkeypatch, "add", exploding_add)


def test_delete_input_shards_survives_failed_finalize(monkeypatch):
    """The finalize (last flush, HF renames, index write) is the final step before the artifact is
    complete; a failure there must still leave the inputs untouched."""

    def exploding_close(self):
        raise OSError("No space left on device")

    _assert_failed_merge_preserves_inputs(monkeypatch, "close_as_hf_checkpoint", exploding_close)


def test_delete_input_shards_after_successful_merge():
    """On success the flag still frees the input shards, and the merged output is complete."""
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", _gptoss_shards(), 2)
        shard_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))

        merge_ep_shards(input_dir, output_dir, verbose=False, delete_input_shards=True)

        for f in shard_files:
            assert not os.path.exists(os.path.join(input_dir, f))
        merged = _load_merged(output_dir)
        assert f"{PREFIX}.mlp.experts.gate_up_proj" in merged
        assert os.path.exists(os.path.join(output_dir, "config.json"))


def test_expert_suffixes_cover_every_class_declared_root():
    """Every expert-weight root any EP family declares (_EXPERT_WEIGHT_ATTR_ROOTS union) must be
    captured by the merge pattern — a missed root passes expert shards through as 'non-expert'
    weights and silently corrupts the merged checkpoint. Pre-derivation the hand list missed the
    GptOss ETP-layout biases (gate_proj_bias/up_proj_bias)."""
    roots = expert_weight_roots()
    assert roots, "EP layer classes declare no expert-weight roots — the subclass walk is broken"
    assert set(_EP_EXPERT_SUFFIXES) == set(roots)
    for root in sorted(roots):
        for container in ("mlp", "feed_forward"):
            key = f"model.layers.0.{container}.{root}"
            match = _EP_EXPERT_PATTERN.match(key)
            assert match is not None, f"expert root {root!r} not captured by the merge pattern"
            assert match.group(1) == f"model.layers.0.{container}"
            assert match.group(2) == root


def test_expert_suffixes_include_gptoss_family_roots():
    """Anchor the union against the GptOss family (grouped-GEMM + bias layouts): guards the layers
    import + subclass walk actually reaching the per-family declarations, which a same-source
    coverage test alone cannot (both sides would shrink together)."""
    assert {
        "gate_proj_gmm",
        "up_proj_gmm",
        "gate_proj_gmm_bias",
        "up_proj_gmm_bias",
        "gate_up_proj",
        "gate_up_proj_bias",
        "down_proj_bias",
    } <= set(_EP_EXPERT_SUFFIXES)


def test_expert_pattern_rejects_non_expert_keys():
    """Router / shared-expert / attention keys must never be captured as expert weights."""
    for key in (
        f"{PREFIX}.mlp.router.weight",
        f"{PREFIX}.mlp.gate.weight",
        f"{PREFIX}.mlp.shared_experts.gate_up_proj.weight",
        f"{PREFIX}.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        # Per-expert HUB keys (the merge OUTPUT) must not be re-captured on a second pass.
        f"{PREFIX}.mlp.experts.0.gate_proj.weight",
        f"{PREFIX}.feed_forward.experts.0.w1.weight",
    ):
        assert _EP_EXPERT_PATTERN.match(key) is None, f"non-expert key {key!r} captured as expert weight"


def _write_gathered_checkpoint(input_dir, model_type, state, *, metadata=None):
    """A checkpoint whose experts are ALREADY in HF layout: a gathered EP save, a PP save, or an
    earlier merge's output. Its fused ``...experts.gate_up_proj`` matches the expert pattern exactly
    as a per-rank shard key does, so nothing but the index marker distinguishes the two on disk."""
    with open(os.path.join(input_dir, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "num_local_experts": E}, f)
    save_file(state, os.path.join(input_dir, "model-00001-of-00001.safetensors"))
    index = {
        "metadata": metadata if metadata is not None else {"total_size": 1},
        "weight_map": dict.fromkeys(state, "model-00001-of-00001.safetensors"),
    }
    with open(os.path.join(input_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


@pytest.mark.parametrize(
    "metadata",
    [
        None,  # gathered EP save / an earlier merge's output
        {"total_size": 1},  # a PP save's index: standard HF metadata, no format marker
        {"ep_size": 2},  # ep_size without the format marker: not a per-rank save either
    ],
)
def test_merge_refuses_a_checkpoint_that_is_not_ep_sharded(metadata):
    """Without this gate the merge re-applies the layout transform to already-HF experts: it
    transposes them and writes them under a DOUBLED ``experts.experts.`` prefix, which
    from_pretrained then treats as missing and randomly initializes — a silently untrained model
    reported as a successful merge."""
    with tempfile.TemporaryDirectory() as tmp:
        input_dir, output_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(input_dir)
        _write_gathered_checkpoint(
            input_dir,
            "qwen3_5_moe",
            {
                f"{PREFIX}.mlp.experts.gate_up_proj": torch.randn(E, 2 * M, H),
                f"{PREFIX}.mlp.experts.down_proj": torch.randn(E, H, M),
                "model.embed_tokens.weight": torch.randn(8, H),
            },
            metadata=metadata,
        )
        with pytest.raises(ValueError, match="not a per-rank EP-sharded checkpoint"):
            merge_ep_shards(input_dir, output_dir, verbose=False)
        # Both gates precede output-dir creation, so a refusal leaves no trace on disk.
        assert not os.path.exists(output_dir)


def test_merge_refuses_to_write_into_its_own_input_dir():
    """save_sharded_state_dict deletes the ``model*.safetensors`` it does not own, so an in-place
    merge destroys the source shards — and ``--delete_input_shards`` would then remove the freshly
    written merged ones."""
    with tempfile.TemporaryDirectory() as tmp:
        shard = {f"{PREFIX}.mlp.gate_up_proj.shard_0": torch.randn(E, H, 2 * M)}
        _write_ep_checkpoint(tmp, "qwen3_5_moe", [shard], 1)
        before = sorted(os.listdir(tmp))
        with pytest.raises(ValueError, match="not in-place"):
            merge_ep_shards(tmp, tmp, verbose=False)
        assert sorted(os.listdir(tmp)) == before, "the refusal must not have touched the checkpoint"


def test_the_reader_accepts_exactly_what_the_writer_names():
    """The per-rank shard filename has ONE home — ``save_ep_model``'s ``ep_shard_filename``.

    A reader carrying its own copy of the pattern matches nothing after a writer-side rename (an
    empty merge on a full directory), and a loosened copy sweeps in a sibling adapter or a stale
    gathered save. Both directions are pinned here, off the writer's own name generator.
    """
    for rank, world in ((0, 1), (0, 8), (7, 8), (3, 16), (63, 64)):
        name = merge_mod.ep_shard_filename(rank, world)
        assert merge_mod.is_ep_shard(name), f"the reader rejects the writer's rank-{rank} filename {name}"
    for other in ("adapter_model.safetensors", "model.safetensors", "model-tp00000-of-00002.safetensors"):
        assert not merge_mod.is_ep_shard(other), f"{other} must stay out of the merge"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
