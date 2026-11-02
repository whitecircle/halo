#!/usr/bin/env python
"""CPU tests for merge_ep_shards.py post-processing logic.

Verifies that EP-internal weight formats are correctly converted to
HuggingFace checkpoint format for all supported MoE architectures. These
transforms are the contract between ``save_ep_sharded_checkpoint`` and
``from_pretrained`` — a wrong transpose or stale key here produces a
checkpoint that silently loads garbage (or fails to load at all), so the
tests assert exact shapes AND values, not just key presence.
"""

import json
import os
import shutil
import stat
import sys
import tempfile
from collections import defaultdict

import pytest
import torch
from safetensors.torch import load_file, save_file

from halo.checkpoint.format import (
    EP_SHARD_KEY_RE,
    OPTIMIZER_META_FILE,
    REFERENCE_LOGPS_FILE,
    ROUTER_BALANCING_BIASES_FILE,
    SCHEDULER_STATE_FILE,
    ep_shard_filename,
    is_ep_shard,
    optimizer_shard_filename,
)
from halo.checkpoint.parallel.resume_source import classify_resume_weights
from halo.checkpoint.tools.ep_shard_merge import (
    EP_EXPERT_PATTERN,
    EP_EXPERT_SUFFIXES,
    group_expert_weights,
    merge_ep_shards,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer
from halo.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer
from halo.distributed.expert_parallel.families.gemma4 import EPGemma4MoELayer
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.laguna import EPLagunaMoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.models.moe.expert_layout import to_hub_expert_tensors
from halo.models.moe.family_specs import expert_weight_roots, find_moe_family_spec
from scripts.after_training import merge_ep_shards as merge_script
from tests.common.checkpoint_resave import resave_in_place

# The whole-dict oracle the streamed merge is pinned against (test-only; see its module docstring).
from tests.common.ep_merge_oracle import post_process_merged_weights
from tests.common.ep_stubs import bare_ep_layer
from tests.common.file_links import OWNER_ONLY_MODE, mount_again, process_umask, refuse_hard_links


def _merge_transform(layer_cls):
    """The merge's per-layer transform for ``layer_cls``'s family: its ``ExpertLayout`` under the layer prefix."""

    def transform(prefix, params):
        hub = to_hub_expert_tensors(layer_cls.FAMILY.expert_layout, params, device="cpu")
        return {f"{prefix}.{key}": tensor for key, tensor in hub.items()}

    return transform


_transform_gptoss = _merge_transform(EPGptOssMoELayer)
_transform_qwen3 = _merge_transform(EPQwen3MoELayer)
_transform_individual = _merge_transform(EPBailingMoELayer)
_transform_fused_glu_experts = _merge_transform(EPQwen3_5MoELayer)

E, H, M = 8, 64, 32  # num_experts, hidden_size, intermediate_size
PREFIX = "model.layers.0"
# The prefix passed to a transform includes the MoE container: `mlp`, or `feed_forward` for LFM2.
PREFIX_MLP = f"{PREFIX}.mlp"
PREFIX_FF = f"{PREFIX}.feed_forward"
# The payload ``write_optimizer_meta`` writes beside the two ranks' optimizer shards of an ep2 save.
OPTIMIZER_META = {"num_ranks": 2, "fingerprint": {"world_size": 2, "ep_size": 2, "optimizer_class": "AdamWBF16"}}
# The per-rank resume files of that save and the meta vouching for them, which the merge carries itself.
EP2_RESUME_FILES = (
    optimizer_shard_filename(0),
    optimizer_shard_filename(1),
    "rng_state_0.pth",
    "rng_state_1.pth",
    OPTIMIZER_META_FILE,
)


def test_shared_expert_params_are_not_grouped_as_routed_experts():
    """Inkling keeps its shared experts as raw ``shared_experts.{gate,up,down}_proj`` parameters, the names
    of a separate-GLU family's routed experts. Grouped as an expert layer they would reach Inkling's fused
    merge, which refuses the unexpected names, and the whole merge would fail."""
    routed = {f"{PREFIX_MLP}.gate_up_proj": torch.zeros(1), f"{PREFIX_MLP}.down_proj": torch.zeros(1)}
    shared = {f"{PREFIX_MLP}.shared_experts.{name}": torch.zeros(1) for name in ("gate_proj", "up_proj", "down_proj")}
    groups, rest = group_expert_weights({**routed, **shared})
    assert set(groups) == {PREFIX_MLP}
    assert set(rest) == set(shared)


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


def testgroup_expert_weights():
    """Expert weights group by layer; router/attention/shared-expert stay non-expert."""
    weights = {
        f"{PREFIX}.mlp.gate_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.up_proj": torch.randn(E, H, M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),  # router
        f"{PREFIX}.self_attn.q_proj.weight": torch.randn(H, H),
        f"{PREFIX}.mlp.shared_expert.gate_proj.weight": torch.randn(M, H),
    }

    expert_groups, non_expert = group_expert_weights(weights)

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
    expert_groups, non_expert = group_expert_weights(weights)
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
    expert_groups, non_expert = group_expert_weights(weights)
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
    result = _merge_via_tool(weights, "qwen3_moe")

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
    result = _merge_via_tool(weights, "gpt_oss")

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
    result = _merge_via_tool(weights, "glm4_moe_lite")

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
    """LFM2 MoE: experts live under ``feed_forward`` (a merge requiring ``.mlp.``
    would pass LFM2 experts through unmerged → dropped on reload) and the hub layout is per-expert
    Llama-style names ``experts.{i}.w{1,3,2}.weight`` (w1 = gate, w3 = up, w2 = down) — the split
    order must match the spec's ``expert_layout.per_expert_keys`` exactly (a swap has identical
    shapes and corrupts the SwiGLU silently)."""
    gate_up = torch.randn(E, H, 2 * M)
    down = torch.randn(E, M, H)
    weights = {
        f"{PREFIX}.feed_forward.gate_up_proj": gate_up,
        f"{PREFIX}.feed_forward.down_proj": down,
        f"{PREFIX}.feed_forward.gate.weight": torch.randn(E, H),
    }
    result = _merge_via_tool(weights, "lfm2_moe")

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
        _merge_via_tool(weights, "glm4_moe_lite")


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
    result = _merge_via_tool(weights, "deepseek_v4")

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
    result = _merge_via_tool(weights, "bailing_moe")

    assert len([k for k in result if "experts." in k]) == 3 * E
    assert result[f"{PREFIX}.mlp.experts.0.gate_proj.weight"].shape == (M, H)
    assert result[f"{PREFIX}.mlp.experts.0.down_proj.weight"].shape == (H, M)


def test_post_process_no_expert_weights_is_noop():
    """A dense (non-MoE) state dict has no EP expert keys and passes through unchanged."""
    weights = {
        f"{PREFIX}.self_attn.q_proj.weight": torch.randn(H, H),
        f"{PREFIX}.mlp.gate.weight": torch.randn(E, H),
    }
    result = _merge_via_tool(weights, "unknown_model")
    assert set(result) == set(weights)
    assert all(torch.equal(result[key], tensor) for key, tensor in weights.items())


def test_post_process_unknown_type_with_experts_raises():
    """An unmapped model_type WITH expert weights must fail loudly, never emit an
    EP-internal checkpoint that from_pretrained() cannot load (regression guard)."""
    weights = {
        f"{PREFIX}.mlp.gate_up_proj": torch.randn(E, H, 2 * M),
        f"{PREFIX}.mlp.down_proj": torch.randn(E, M, H),
    }
    with pytest.raises(ValueError, match="no HF-layout transform"):
        _merge_via_tool(weights, "mistral4_moe")


def test_model_type_resolves_to_its_family_layout():
    """The merge resolves model_type -> family spec -> its ``ExpertLayout``, so every spelling a spec
    claims is mergeable — including Laguna, which declares GLM-4's layout."""
    assert find_moe_family_spec("glm4_moe_lite") is EPGlm4MoELayer.FAMILY
    assert find_moe_family_spec("laguna") is EPLagunaMoELayer.FAMILY
    assert EPLagunaMoELayer.FAMILY.expert_layout == EPGlm4MoELayer.FAMILY.expert_layout
    assert find_moe_family_spec("bailing_moe_linear") is find_moe_family_spec("bailing_moe")
    # No spelling heuristic: an unclaimed model_type is rejected, never guessed onto a layout.
    assert find_moe_family_spec("my_glm4_moe_variant") is None


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


def _merge_via_tool(weights: dict, model_type: str, ep_size: int = 2) -> dict:
    """``weights`` merged by ``merge_ep_shards`` from the layout the per-rank EP save writes: each
    expert tensor split on its expert dim into ``.shard_<rank>`` slices, every other tensor on rank 0."""
    expert_groups, non_expert = group_expert_weights(weights)
    shards = [dict(non_expert)] + [{} for _ in range(ep_size - 1)]
    for prefix, params in expert_groups.items():
        for suffix, tensor in params.items():
            for rank, part in enumerate(tensor.chunk(ep_size)):
                shards[rank][f"{prefix}.{suffix}.shard_{rank}"] = part.clone()
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, model_type, shards, ep_size)
        merge_ep_shards(input_dir, output_dir, verbose=False)
        return _load_merged(output_dir)


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
    pins from and would fold them to bf16 — GLM-5 Next's ``A_log``/``dt_bias``, Inkling's short
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


def test_the_cli_reports_the_merge_and_the_quiet_driver_prints_nothing(monkeypatch, capsys):
    """The driver is library code the oracle and the tests call: ``verbose=False`` keeps it off stdout, and the
    script's ``main`` is what tells the user where the merged checkpoint landed."""
    per = E // 2
    shards = [
        {
            f"{PREFIX}.mlp.gate_up_proj.shard_{rank}": torch.randn(per, H, 2 * M),
            f"{PREFIX}.mlp.down_proj.shard_{rank}": torch.randn(per, M, H),
        }
        for rank in range(2)
    ]
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as root:
        _write_ep_checkpoint(input_dir, "gpt_oss", shards, 2)
        merge_ep_shards(input_dir, os.path.join(root, "library"), verbose=False)
        assert capsys.readouterr().out == ""

        cli_output = os.path.join(root, "cli")
        monkeypatch.setattr(
            sys, "argv", ["merge_ep_shards.py", "--input_dir", input_dir, "--output_dir", cli_output, "--quiet"]
        )
        merge_script.main()
        assert f"Merged EP checkpoint saved to: {cli_output}" in capsys.readouterr().out


def test_the_merged_directory_is_the_resume_source_the_shards_are_not():
    """A resume refuses the per-rank shards and names this merge as the way back, so the merged
    directory must carry every resume file byte for byte. Without ``reference_logps.pt`` a
    precompute DPO/KTO resume, whose policy is built from these weights, has no untrained reference
    left to sweep and raises; without the optimizer shards and the meta vouching for them, a resume on
    the very topology that wrote them warm-restarts the optimizer."""
    # The meta stays last: it is torch-saved, every file before it written as its name's bytes.
    resume_files = (REFERENCE_LOGPS_FILE, SCHEDULER_STATE_FILE, ROUTER_BALANCING_BIASES_FILE, *EP2_RESUME_FILES)
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", _gptoss_shards(), 2)
        for name in resume_files[:-1]:
            with open(os.path.join(input_dir, name), "wb") as f:
                f.write(name.encode())
        torch.save(OPTIMIZER_META, os.path.join(input_dir, OPTIMIZER_META_FILE))
        assert classify_resume_weights(input_dir) == "invalid"

        merge_ep_shards(input_dir, output_dir, verbose=False)

        assert classify_resume_weights(output_dir) == "full"
        for name in resume_files:
            path = os.path.join(output_dir, name)
            assert os.path.isfile(path), f"{name} was dropped by the merge"
            with open(path, "rb") as merged, open(os.path.join(input_dir, name), "rb") as saved:
                assert merged.read() == saved.read(), f"{name} changed on its way through the merge"


def test_a_rank_whose_optimizer_shard_is_missing_fails_the_merge(tmp_path):
    """The meta vouches for every rank's optimizer shard; a merged directory missing one would load the
    whole model on resume only for the torn shard set to raise there."""
    input_dir, output_dir = tmp_path / "save", tmp_path / "merged"
    os.makedirs(input_dir)
    _write_ep_checkpoint(str(input_dir), "gpt_oss", _gptoss_shards(), 2)
    torch.save({"state": {}}, input_dir / optimizer_shard_filename(0))
    torch.save(OPTIMIZER_META, input_dir / OPTIMIZER_META_FILE)

    with pytest.raises(ValueError, match=r"optimizer shards of ranks \[1\]"):
        merge_ep_shards(str(input_dir), str(output_dir), verbose=False)
    assert not output_dir.exists()


def _save_with_resume_state(input_dir):
    """A sharded ep2 save at ``input_dir`` holding :data:`EP2_RESUME_FILES`, each rank's optimizer shard its own."""
    os.makedirs(input_dir)
    _write_ep_checkpoint(str(input_dir), "gpt_oss", _gptoss_shards(), 2)
    for rank in range(2):
        torch.save(
            {"state": {0: {"exp_avg": torch.full((4,), float(rank))}}}, input_dir / optimizer_shard_filename(rank)
        )
        torch.save({"python": rank}, input_dir / f"rng_state_{rank}.pth")
    torch.save(OPTIMIZER_META, input_dir / OPTIMIZER_META_FILE)
    return input_dir


def _assert_hard_linked(input_dir, output_dir):
    for name in EP2_RESUME_FILES:
        assert not (output_dir / name).is_symlink() and os.path.samefile(input_dir / name, output_dir / name), (
            f"{name} was copied, not hard-linked"
        )


def test_on_one_filesystem_the_resume_state_is_hard_linked_and_outlives_the_input(tmp_path):
    """Copied, every rank's optimizer shard would land on the volume twice. Linked, the merged directory
    stays whole through ``--delete_input_shards`` and the input's removal after it."""
    input_dir, output_dir = _save_with_resume_state(tmp_path / "save"), tmp_path / "merged"

    merge_ep_shards(str(input_dir), str(output_dir), verbose=False, delete_input_shards=True)

    _assert_hard_linked(input_dir, output_dir)
    shutil.rmtree(input_dir)
    assert classify_resume_weights(str(output_dir)) == "full"
    assert torch.load(output_dir / OPTIMIZER_META_FILE, weights_only=True) == OPTIMIZER_META
    shard = torch.load(output_dir / optimizer_shard_filename(1), weights_only=True)
    assert torch.equal(shard["state"][0]["exp_avg"], torch.full((4,), 1.0)), "rank 1's shard is not its own"


def test_across_filesystems_the_resume_state_is_copied_with_its_mode(tmp_path, monkeypatch):
    input_dir, output_dir = _save_with_resume_state(tmp_path / "save"), tmp_path / "merged"
    for name in EP2_RESUME_FILES:
        os.chmod(input_dir / name, OWNER_ONLY_MODE)
    refuse_hard_links(monkeypatch)

    with process_umask():
        merge_ep_shards(str(input_dir), str(output_dir), verbose=False)

    for name in EP2_RESUME_FILES:
        copied = (output_dir / name).stat()
        assert copied.st_ino != (input_dir / name).stat().st_ino
        assert stat.S_IMODE(copied.st_mode) == OWNER_ONLY_MODE, f"{name} took the umask's mode, not its source's"
        assert (output_dir / name).read_bytes() == (input_dir / name).read_bytes(), f"{name} changed on the way"


def test_a_merge_rerun_into_its_output_directory_links_the_resume_state_again(tmp_path):
    """A re-run finds the first run's links under the names it carries and must take them over rather than
    fail on them."""
    input_dir, output_dir = _save_with_resume_state(tmp_path / "save"), tmp_path / "merged"
    merge_ep_shards(str(input_dir), str(output_dir), verbose=False)

    merge_ep_shards(str(input_dir), str(output_dir), verbose=False)

    _assert_hard_linked(input_dir, output_dir)
    assert classify_resume_weights(str(output_dir)) == "full"


@pytest.mark.parametrize("resaved", ["input", "merged"])
def test_a_save_that_rewrites_one_linked_directory_leaves_the_other_as_merged(tmp_path, resaved):
    """A run saving a step it saved before rewrites that ``checkpoint-N`` in place, and the optimizer-shard and
    RNG writers truncate an existing name. The input re-saved after the merge, or the merged directory re-saved
    by a run resumed from it, must not change the other directory's linked resume state with it."""
    step_dir = tmp_path / "run" / "checkpoint-2"
    other = tmp_path / ("merged" if resaved == "input" else "save")
    input_dir, output_dir = (step_dir, other) if resaved == "input" else (other, step_dir)
    _save_with_resume_state(input_dir)
    merge_ep_shards(str(input_dir), str(output_dir), verbose=False)
    _assert_hard_linked(input_dir, output_dir)
    before = {name: (other / name).read_bytes() for name in EP2_RESUME_FILES}

    resave_in_place(step_dir, world_size=2)

    for name in (optimizer_shard_filename(0), "rng_state_0.pth", OPTIMIZER_META_FILE):
        assert (step_dir / name).read_bytes() != before[name], f"premise: the re-save rewrote {name}"
    changed = [name for name in EP2_RESUME_FILES if (other / name).read_bytes() != before[name]]
    assert not changed, f"the re-save of the {resaved} directory rewrote the other's {changed}"


def test_an_output_that_is_the_input_mounted_again_is_refused_and_the_input_survives(tmp_path, monkeypatch):
    """A second bind mount reaches the save by a path ``realpath`` keeps distinct. Merged there, the merge would
    replace the save's own per-rank files and sweep its weight shards, so it must be refused before any write."""
    input_dir = _save_with_resume_state(tmp_path / "save")
    saved = {path.name: path.read_bytes() for path in input_dir.iterdir()}
    mount_again(monkeypatch, input_dir, tmp_path / "mounted")

    with pytest.raises(ValueError, match="not in-place"):
        merge_ep_shards(str(input_dir), str(tmp_path / "mounted"), verbose=False)

    assert {path.name: path.read_bytes() for path in input_dir.iterdir()} == saved


def _gathered_reference(layer_cls, gate_up, down):
    """What the GATHERED save of the same weights emits: the family's real gather on one process. This is
    the invariant target for the sharded merge — vLLM's per-expert loaders and the hub conversion consume
    exactly this layout."""
    return bare_ep_layer(layer_cls, gate_up_proj=gate_up, down_proj=down).gather_expert_state_dict()


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
        f"{PREFIX}.{container}.{key}": tensor for key, tensor in _gathered_reference(layer_cls, gate_up, down).items()
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
        # Per-expert hub layouts (FUSED_SPLIT_PER_EXPERT).
        (EPGlm4MoELayer, "glm4_moe_lite", "mlp"),
        (EPLagunaMoELayer, "laguna", "mlp"),
        (EPLfm2MoELayer, "lfm2_moe", "feed_forward"),
        (EPQwen3NextMoELayer, "qwen3_next", "mlp"),
        # Fused hub layouts.
        (EPCohere2MoELayer, "cohere2_moe", "mlp"),
        (EPQwen3_5MoELayer, "qwen3_5_moe", "mlp"),
        (EPGlm5NextMoELayer, "glm5_next", "mlp"),
        (EPDeepseekV4MoELayer, "deepseek_v4", "mlp"),
        (EPMistral4MoELayer, "mistral4", "mlp"),
        (EPZayaMoELayer, "zaya", "mlp"),
        # Bare keys: the layer replaces the expert container itself.
        (EPGemma4MoELayer, "gemma4", "mlp.experts"),
        # No Qwen3.8-Flash-Next row: its sharded save is refused, since the merge has no n-gram table to carry.
    ],
)
def test_full_merge_matches_gathered_save(layer_cls, model_type, container):
    """merged-from-sharded == gathered, key and tensor, for every family with fused expert storage.

    Both apply the family's one ``ExpertLayout``; this asserts it end to end through the real merge
    script and the real gather."""
    _roundtrip_vs_gathered(layer_cls, model_type, container)


def test_merge_renames_module_spellings_to_hub_spellings():
    """Laguna is the roster's rename-bearing family (``export_key_renames``): the merge must
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
    for fname in sorted(f for f in os.listdir(input_dir) if is_ep_shard(f)):
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
    """Inject ``failing`` into one StreamingShardWriter method and assert every input shard survives."""
    monkeypatch.setattr(StreamingShardWriter, writer_method, failing)
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        _write_ep_checkpoint(input_dir, "gpt_oss", _gptoss_shards(), 2)
        shard_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))

        with pytest.raises(OSError, match="No space left"):
            merge_ep_shards(input_dir, output_dir, verbose=False, delete_input_shards=True)

        for f in shard_files:
            assert os.path.exists(os.path.join(input_dir, f)), f"input shard {f} destroyed by a failed merge"


def test_delete_input_shards_survives_midstream_write_failure(monkeypatch):
    """A merge that fails while streaming tensors out must leave every input shard on disk — a merge
    that deleted each shard right after loading it would let a failed save destroy the only copy of
    the trained weights."""

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
    """Every expert-weight root any EP family declares (the ``expert_weight_roots`` union) must be
    captured by the merge pattern — a missed root passes expert shards through as 'non-expert'
    weights and silently corrupts the merged checkpoint. A hand list misses roots such as the
    GptOss ETP-layout biases (gate_proj_bias/up_proj_bias)."""
    roots = expert_weight_roots()
    assert roots, "EP layer classes declare no expert-weight roots — the subclass walk is broken"
    assert set(EP_EXPERT_SUFFIXES) == set(roots)
    for root in sorted(roots):
        for container in ("mlp", "feed_forward"):
            key = f"model.layers.0.{container}.{root}"
            match = EP_EXPERT_PATTERN.match(key)
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
    } <= set(EP_EXPERT_SUFFIXES)


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
        assert EP_EXPERT_PATTERN.match(key) is None, f"non-expert key {key!r} captured as expert weight"


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
    """``StreamingShardWriter.close_as_hf_checkpoint`` deletes the ``model*.safetensors`` it did not
    write, so an in-place
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
    """The per-rank shard filename has ONE home — ``halo.checkpoint.format.ep_shard_filename``.

    A reader carrying its own copy of the pattern matches nothing after a writer-side rename (an
    empty merge on a full directory), and a loosened copy sweeps in a sibling adapter or a stale
    gathered save. Both directions are pinned here, off the writer's own name generator.
    """
    for rank, world in ((0, 1), (0, 8), (7, 8), (3, 16), (63, 64)):
        name = ep_shard_filename(rank, world)
        assert is_ep_shard(name), f"the reader rejects the writer's rank-{rank} filename {name}"
    for other in ("adapter_model.safetensors", "model.safetensors", "model-tp00000-of-00002.safetensors"):
        assert not is_ep_shard(other), f"{other} must stay out of the merge"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
