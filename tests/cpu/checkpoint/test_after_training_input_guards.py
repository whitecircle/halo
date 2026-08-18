#!/usr/bin/env python
"""CPU tests: every checkpoint conversion tool must REFUSE an input it cannot handle.

These tools (``scripts/after_training`` plus the ``scripts/before_training`` converters) are the last
stop before a checkpoint is served, and each failure mode here is silent — the tool prints success and
exits 0 while the experts are gone:

* **Per-rank EP/TP sharded input.** The index filename is the ordinary one but every expert tensor is
  one rank's partial slice under a ``.shard_N`` key. A ``from_pretrained``-based tool therefore sees
  the real expert keys as MISSING, and transformers RANDOMLY INITIALIZES them with a warning rather
  than an exception (``test_missing_keys_only_warn`` pins that premise). ``reset_sinks`` then
  overwrites the source in place by default, so the shards are unrecoverable.
* **Wrong model family.** ``experts.gate_up_proj`` / ``experts.down_proj`` are spelled identically at
  identical shapes across most of the roster, so ``unfuse_moe_experts`` must emit the projection
  names the checkpoint's OWN family declares — LFM-2 reads ``w1``/``w3``/``w2``, not GLM-4's
  ``gate_proj``/``up_proj``/``down_proj`` — and refuse a family that declares none.
* **The wrong merge tool.** An EP-sharded and a TP-sharded save are told apart only by their index's
  ``format`` marker, so a merge that infers its shard count from a defaulted ``tp_size`` diagnoses
  the wrong input as an incomplete shard set instead of naming the tool that owns it.
* **In-place output.** ``save_pretrained`` deletes the weight files it does not overwrite.
* **Asymmetric key sets in a merge.** A tensor present in a later model but not the reference is
  never visited and is dropped from the merged checkpoint.
* **A reader that re-types the writer's shard pattern.** An index-less sharded save is reachable
  only through the shard glob, so that glob is derived from the pattern the writer stamps; two
  spellings drift into "no weights found" on a checkpoint that is perfectly intact.

    python tests/cpu/checkpoint/test_after_training_input_guards.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.checkpoint.format import SAFETENSORS_INDEX_FILE, is_sharded_checkpoint, save_sharded_state_dict
from src.checkpoint.tool_io import checkpoint_shard_files, detect_model_type, reject_sharded_checkpoint
from src.distributed.expert_parallel.expert_weights import per_expert_hub_model_types


def _write_ep_sharded_checkpoint(path, *, model_type="gpt_oss"):
    """A per-rank EP-sharded save: experts under ``.shard_N`` keys, ``format: ep_sharded`` metadata."""
    os.makedirs(path, exist_ok=True)
    shards = {}
    for rank in range(2):
        name = f"model-{rank:05d}-of-00002.safetensors"
        tensors = {f"model.layers.0.mlp.experts.gate_up_proj.shard_{rank}": torch.ones(2, 4, 8) * (rank + 1)}
        if rank == 0:
            tensors["model.embed_tokens.weight"] = torch.zeros(16, 8)
        save_file(tensors, os.path.join(path, name), metadata={"format": "pt"})
        shards.update({key: name for key in tensors})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1, "ep_size": 2, "format": "ep_sharded"}, "weight_map": shards}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "hidden_size": 8, "intermediate_size": 4}, f)
    return path


def _write_gathered_checkpoint(path, tensors, *, model_type="qwen3_moe", with_index=False, config_extra=None):
    """An ordinary gathered single-file checkpoint.

    ``with_index`` adds the HF index a multi-shard gathered save also writes — standard metadata, no
    per-rank format marker — which is what a merge tool must tell apart from its own input.
    ``config_extra`` adds config fields a specific tool reads (e.g. ``moe_intermediate_size``).
    """
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    if with_index:
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": dict.fromkeys(tensors, "model.safetensors")}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "hidden_size": 8, "intermediate_size": 4, **(config_extra or {})}, f)
    return path


# Fused expert tensors for a hidden=8 / moe_intermediate=3 MoE layer: gate_up [E, 2I, H], down [E, H, I].
_FUSED_MOE_CONFIG = {"hidden_size": 8, "moe_intermediate_size": 3}


def _fused_moe_tensors(prefix="model.layers.0.mlp", num_experts=2):
    return {
        f"{prefix}.experts.gate_up_proj": torch.randn(num_experts, 6, 8),
        f"{prefix}.experts.down_proj": torch.randn(num_experts, 8, 3),
    }


def _write_pp_per_node_checkpoint(path, *, model_type="qwen3_moe", complete=False):
    """What one node's directory holds after a pipeline-parallel save on a non-shared output
    filesystem: the WORLD-WIDE index (standard HF metadata, no format marker) beside this node's own
    stage parts only. ``complete=True`` adds the other stage's part, as gathering the nodes' directories
    into one would."""
    os.makedirs(path, exist_ok=True)
    parts = {
        "model-pp00000-of-00002-00001.safetensors": {"model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 6, 8)},
        "model-pp00001-of-00002-00001.safetensors": {"model.layers.1.mlp.experts.gate_up_proj": torch.randn(2, 6, 8)},
    }
    weight_map = {}
    for name, tensors in parts.items():
        if complete or name.startswith("model-pp00000"):
            save_file(tensors, os.path.join(path, name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(path, SAFETENSORS_INDEX_FILE), "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, **_FUSED_MOE_CONFIG}, f)
    return path


def test_fixture_is_recognized_as_per_rank_sharded(tmp_path):
    """Anti-vacuity: the guards below are only meaningful if the fixture really trips the detector."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    assert is_sharded_checkpoint(str(ep))
    assert not is_sharded_checkpoint(str(_write_gathered_checkpoint(tmp_path / "ok", {"a": torch.zeros(2)})))


def test_reject_sharded_checkpoint_raises_and_names_the_merge_scripts(tmp_path):
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        reject_sharded_checkpoint(str(ep))


def test_an_index_naming_absent_shards_is_refused_and_names_them(tmp_path):
    """One node's directory from a PP save on a non-shared filesystem: the index is whole, the parts
    are not. A ``from_pretrained``-based tool would random-initialize the other stage's layers; a
    streaming one would die inside ``safe_open`` with its output directory already created. The
    refusal must name the absent file and the gather-the-nodes fix, and a gathered (complete)
    directory of the same shape must pass — the PP layout is loadable once every part is present."""
    node = _write_pp_per_node_checkpoint(tmp_path / "node0")
    with pytest.raises(ValueError, match="does not hold") as excinfo:
        reject_sharded_checkpoint(str(node))
    assert "model-pp00001-of-00002-00001.safetensors" in str(excinfo.value)
    assert "gather every node" in str(excinfo.value)
    with pytest.raises(ValueError, match="does not hold"):
        checkpoint_shard_files(str(node))

    gathered = _write_pp_per_node_checkpoint(tmp_path / "gathered", complete=True)
    reject_sharded_checkpoint(str(gathered))
    assert len(checkpoint_shard_files(str(gathered))) == 2


def _unfuse(src, out):
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    unfuse_checkpoint(src, out)


def _quantize(src, out):
    from scripts.after_training.quantize_to_lowp import quantize_checkpoint

    quantize_checkpoint(src, out, "mxfp8")


def _convert(src, out):
    from scripts.after_training.convert_to_bf16 import convert_to_bf16

    convert_to_bf16(src, out, "causal_lm")


@pytest.mark.parametrize("tool", [_unfuse, _quantize, _convert], ids=["unfuse", "quantize", "convert_to_bf16"])
def test_tools_refuse_a_per_node_pp_directory_before_writing(tmp_path, tool):
    """Every tool's first read of the source goes through the completeness gate, ahead of its
    ``os.makedirs`` — so the refusal leaves no half-made output for a pipeline to mistake for one."""
    node = _write_pp_per_node_checkpoint(tmp_path / "node0")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="does not hold"):
        tool(str(node), str(out))
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_reset_sinks_refuses_a_per_rank_sharded_checkpoint(tmp_path):
    """Without the guard this loads the dir via from_pretrained (no ``model.safetensors`` to shortcut
    on), re-initializes the experts, and writes the result back over the source shards."""
    from scripts.after_training.reset_sinks import reset_sinks

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    before = sorted(os.listdir(ep))
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        reset_sinks(str(ep))
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"


def test_convert_to_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path):
    """--verify cannot catch this one: it counts dtypes, not whether the weights are real."""
    from scripts.after_training.convert_to_bf16 import convert_to_bf16

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        convert_to_bf16(str(ep), str(out), "causal_lm")
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_unfuse_moe_experts_emits_the_familys_own_projection_names(tmp_path):
    """LFM-2's loader reads ``experts.{i}.w{1,3,2}.weight``. Emitting GLM-4's spelling instead exits 0
    over keys nothing reads, and transformers then re-initializes the whole expert bank on load."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    tensors = _fused_moe_tensors()
    src = _write_gathered_checkpoint(tmp_path / "lfm2", tensors, model_type="lfm2_moe", config_extra=_FUSED_MOE_CONFIG)
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    gate_up = tensors["model.layers.0.mlp.experts.gate_up_proj"]
    for i in range(gate_up.shape[0]):
        for name, expected in (("w1", gate_up[i, :3]), ("w3", gate_up[i, 3:])):
            key = f"model.layers.0.mlp.experts.{i}.{name}.weight"
            assert key in written, f"missing {key}; got {sorted(written)[:4]}"
            assert torch.equal(written[key], expected)
    assert not [k for k in written if "gate_proj" in k or "up_proj" in k], "emitted GLM-4's spelling"


def test_unfuse_moe_experts_emits_glm4_spelling_for_glm4(tmp_path):
    """Anti-over-rejection, and the other half of the derivation: the family that DOES declare
    gate/up/down still gets it."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / "glm4", _fused_moe_tensors(), model_type="glm4_moe_lite", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    for i in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{i}.{name}.weight" in written


def test_unfuse_moe_experts_converts_a_fused_qwen3_moe_checkpoint(tmp_path):
    """Qwen3 MoE's hub layout IS per-expert (transformers fuses on load and reverts on save), and a
    FUSED one is reachable — with ``use_grouped_gemm: false`` and no EP the gathered save writes the raw
    module state dict, never routing through ``save_pretrained``'s revert. Refusing it would turn away
    exactly the checkpoint this tool exists to repair."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / "qwen3", _fused_moe_tensors(), model_type="qwen3_moe", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    for i in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{i}.{name}.weight" in written


@pytest.mark.parametrize("model_type", ["mistral4", "zaya", "step3p7"])
def test_unfuse_moe_experts_refuses_a_fused_only_family(tmp_path, model_type):
    """Mistral4, Gemma4 and Zaya register no per-expert converter with transformers at all and serve
    from the fused halves, so a per-expert rewrite produces keys their loaders never read — and their
    fused tensors carry these very names at these very shapes, so nothing else would catch it.
    Step-3.7's hub layout is per-layer fused-but-split (``moe.gate_proj [E, M, H]``), which is not a
    per-expert spelling either — the same refusal applies."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / model_type, _fused_moe_tensors(), model_type=model_type, config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="does not store one tensor per expert"):
        unfuse_checkpoint(str(src), str(out))
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_unfuse_moe_experts_converts_a_qwen3_5_checkpoint(tmp_path):
    """Qwen3.5/3.6's hub layout IS per-expert — transformers fuses it on load and reverts on save — so a
    gathered save that bypassed that revert is exactly what this script exists to repair."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / "qwen35", _fused_moe_tensors(), model_type="qwen3_5_moe", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.0.mlp.experts.0.up_proj.weight" in keys
    assert "model.layers.0.mlp.experts.0.down_proj.weight" in keys


def test_unfuse_moe_experts_converts_a_glm5_next_composite_checkpoint(tmp_path):
    """GLM-5 Next's hub layout is per-expert under the composite wrapper's ``model.language_model``
    tree, with the MoE dimensions on ``text_config``: the EP-gathered artifact is fused, and the
    split must land under the family's ``gate/up/down_proj`` names at the wrapper prefix."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    prefix = "model.language_model.layers.1.mlp"
    src = _write_gathered_checkpoint(
        tmp_path / "glm5",
        _fused_moe_tensors(prefix=prefix),
        model_type="glm5_next",
        config_extra={"text_config": {"model_type": "glm5_next_text", **_FUSED_MOE_CONFIG}},
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert keys == {
        f"{prefix}.experts.{i}.{name}.weight" for i in range(2) for name in ("gate_proj", "up_proj", "down_proj")
    }


def test_unfuse_moe_experts_writes_deepseek_v4_under_its_own_w_names(tmp_path):
    """The layout is the checkpoint family's, not the roster's most common: DeepSeek-V4 stores
    ``w1``/``w3``/``w2``, and emitting GLM-4's spelling would re-initialize the whole expert bank."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / "dsv4", _fused_moe_tensors(), model_type="deepseek_v4", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert "model.layers.0.mlp.experts.0.w1.weight" in keys
    assert "model.layers.0.mlp.experts.0.w3.weight" in keys
    assert "model.layers.0.mlp.experts.0.w2.weight" in keys
    assert not any(key.endswith(".gate_proj.weight") for key in keys)


def test_unfuse_moe_experts_refuses_an_unregistered_model_type(tmp_path):
    """No registered family claims it, so nothing declares what its loader reads."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = _write_gathered_checkpoint(
        tmp_path / "unknown", _fused_moe_tensors(), model_type="some_future_moe", config_extra=_FUSED_MOE_CONFIG
    )
    with pytest.raises(ValueError, match="no registered EP family claims"):
        unfuse_checkpoint(str(src), str(tmp_path / "out"))


def test_unfuse_moe_experts_names_the_missing_config(tmp_path):
    """Without config.json the family is unresolvable; reporting model_type '' as an unclaimed family
    sends the user looking for a roster entry instead of the missing file."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    src = tmp_path / "no_config"
    os.makedirs(src)
    save_file(_fused_moe_tensors(), os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="no config.json"):
        unfuse_checkpoint(str(src), str(tmp_path / "out"))


def test_unfuse_moe_experts_diagnoses_a_per_rank_sharded_input_as_one(tmp_path):
    """Ordering: the sharded-input check owns this diagnosis and names the merge scripts. Resolving the
    family first answered "no per-expert hub layout" for a checkpoint whose real problem is that every
    expert tensor is one rank's slice."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="qwen3_5_moe")
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        unfuse_checkpoint(str(ep), str(tmp_path / "out"))


def test_unfuse_moe_experts_copies_through_an_already_per_expert_checkpoint(tmp_path):
    """Ordering, other half: a checkpoint with nothing fused needs no layout at all, so the family gate
    must not refuse a no-op — including for a family that has no per-expert hub layout to resolve."""
    from scripts.after_training.unfuse_moe_experts import unfuse_checkpoint

    tensors = {"model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(3, 8)}
    src = _write_gathered_checkpoint(
        tmp_path / "already", tensors, model_type="mistral4", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))
    assert set(load_file(os.path.join(out, "model.safetensors"))) == set(tensors)


def test_unfuse_moe_experts_convertible_set_is_the_declared_per_expert_families():
    """Anti-vacuity for the refusals: the advertised set has to be the families whose hub checkpoint
    really is per-expert — the two declaration paths (base-gather split, hub-layout declaration) unioned
    from the classes, never a list restated in the script.

    Membership is checked against transformers' own converter registry, which is what decides the names
    ``from_pretrained`` reads: a family with a ``mlp.experts.*.<proj>`` source pattern is convertible,
    and one with no converter at all serves from whatever its module layout is.
    """
    convertible = set(per_expert_hub_model_types())
    assert {
        "glm4_moe_lite",
        "laguna",
        "lfm2_moe",
        "qwen3_moe",
        "bailing_moe",
        "bailing_moe_linear",
        "qwen3_5_moe",
        "deepseek_v4",
        "cohere2_moe",
        "cohere2_vision",
    } <= convertible
    assert not convertible & {"zaya", "gpt_oss", "gemma4", "mistral4"}


def test_convert_mistral4_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """The one converter that read ``model.safetensors.index.json`` itself: an EP-sharded save was
    streamed through and re-indexed as if each partial expert slice were the whole tensor."""
    from scripts.before_training.convert_mistral4_bf16 import main

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="mistral4")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_mistral4_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_glm5_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """Same streaming class as the Mistral4 converter: without the refusal, each rank's partial
    expert slice would be block-dequantized and re-indexed as if it were the whole tensor."""
    from scripts.before_training.convert_glm5_bf16 import main

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="glm5_next")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_glm5_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_deepseek_v4_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """``--model_id`` also accepts a local directory, and ``from_pretrained`` cannot tell a per-rank
    EP save from a whole one: it reports the real expert keys as MISSING and randomly initializes
    them, warning only, so the converter would write a 420 GB checkpoint with no experts in it."""
    from scripts.before_training.convert_deepseek_v4_bf16 import main

    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="deepseek_v4")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_deepseek_v4_bf16_refuses_an_in_place_conversion(tmp_path, monkeypatch):
    """``save_pretrained`` clears the ``model*.safetensors`` it does not overwrite, so pointing
    ``--output_dir`` at ``--model_id`` destroys the FP8 source the run is still reading its tokenizer
    from — 100+ GiB, and the dequantized result is not a substitute for it."""
    from scripts.before_training.convert_deepseek_v4_bf16 import main

    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="deepseek_v4")
    before = sorted(os.listdir(src))
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(src), "--output_dir", str(src)])
    with pytest.raises(ValueError, match="input and output directory are the same"):
        main()
    assert sorted(os.listdir(src)) == before, "the refused input directory must be left untouched"


def test_convert_to_bf16_refuses_an_in_place_conversion(tmp_path, monkeypatch):
    """The one converter whose guard is newest. ``save_pretrained``
    clears the ``model*.safetensors`` it does not overwrite, so an in-place run destroys the source
    checkpoint it is still reading — the same reason every sibling converter refuses it."""
    from scripts.after_training.convert_to_bf16 import convert_to_bf16

    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="qwen3_moe")
    before = sorted(os.listdir(src))
    with pytest.raises(ValueError, match="input and output directory are the same"):
        convert_to_bf16(str(src), str(src), model_type="qwen3_moe")
    assert sorted(os.listdir(src)) == before, "the refused input directory must be left untouched"


def test_merge_ep_shards_refuses_a_sibling_adapter(tmp_path):
    """An adapter beside the shards must stop the merge BEFORE it writes: the aux copy carries
    ``adapter_config.json`` across but skips every weight file, so the merged directory would claim
    an adapter whose weights are missing — and the merge cannot tell whether the shards already hold
    the delta."""
    from scripts.after_training.merge_ep_shards import merge_ep_shards

    src = _write_ep_sharded_checkpoint(tmp_path / "ep")
    adapter = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 8)}
    save_file(adapter, os.path.join(src, "adapter_model.safetensors"))
    with open(os.path.join(src, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": 8}, f)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        merge_ep_shards(str(src), str(out), verbose=False)
    assert not out.exists(), "a refused merge must not create its output directory"


def test_detect_model_type_reads_config_and_tolerates_absence(tmp_path):
    src = _write_gathered_checkpoint(tmp_path / "m", {"a": torch.zeros(2)}, model_type="glm4_moe_lite")
    assert detect_model_type(str(src)) == "glm4_moe_lite"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_model_type(str(empty)) == ""


def test_index_less_shards_are_found_under_the_writers_own_pattern(tmp_path):
    """The reader's fallback glob must follow ``SAFETENSORS_SHARD_PATTERN``, not a second spelling.

    Written by the toolkit's own writer, then stripped of its index — the resolution path where the
    glob is the only thing left. A reader that re-types the pattern reports "no weights found" for a
    real checkpoint the moment the two spellings drift.
    """
    out = tmp_path / "sharded"
    out.mkdir()
    state = {f"model.layers.{i}.weight": torch.zeros(512) for i in range(3)}
    save_sharded_state_dict(state, str(out), max_shard_size="1KB")
    os.remove(out / SAFETENSORS_INDEX_FILE)

    found = checkpoint_shard_files(str(out))
    assert len(found) == 3, f"expected every shard, got {[os.path.basename(p) for p in found]}"
    recovered = {key for shard in found for key in load_file(shard)}
    assert recovered == set(state), "the glob found shards that do not add up to the checkpoint"


def test_merge_peft_adapters_refuses_in_place_output(tmp_path):
    """``--output_dir <adapter_dir>`` would have save_pretrained delete the adapter it just read."""
    from scripts.after_training.merge_peft_adapters import merge_peft_adapter

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="same path"):
        merge_peft_adapter(str(adapter), str(adapter))


def _merge(*, models, output_dir):
    """A linear merge with only the knobs linear consumes — an unused knob raises up front.

    ``allow_missing_tokenizer``: these fixtures are bare weight directories, and the merge otherwise
    refuses a tokenizer-less artifact — a separate contract, covered in test_after_training_tool_guards.
    """
    from scripts.after_training.merge_models import merge_models

    merge_models(models, output_dir, "linear", dtype="bfloat16", verbose=False, allow_missing_tokenizer=True)


def test_merge_models_refuses_asymmetric_key_sets(tmp_path):
    """A tensor only model B has is never visited by the reference-keyed loop — without this guard it
    is dropped from the merged checkpoint with no warning at all."""
    shared = {"model.embed_tokens.weight": torch.zeros(4, 8)}
    a = _write_gathered_checkpoint(tmp_path / "a", dict(shared))
    b = _write_gathered_checkpoint(tmp_path / "b", {**shared, "lm_head.weight": torch.ones(4, 8)})
    with pytest.raises(ValueError, match="absent from the merge's reference key set"):
        _merge(models=[str(a), str(b)], output_dir=str(tmp_path / "out"))


def test_merge_models_accepts_symmetric_key_sets(tmp_path):
    """Anti-over-rejection: identical key sets must still merge, and average correctly."""
    a = _write_gathered_checkpoint(tmp_path / "a", {"w": torch.zeros(4, 8)})
    b = _write_gathered_checkpoint(tmp_path / "b", {"w": torch.ones(4, 8) * 2})
    out = tmp_path / "out"
    _merge(models=[str(a), str(b)], output_dir=str(out))
    merged = load_file(os.path.join(out, "model.safetensors"))["w"]
    assert torch.allclose(merged.float(), torch.ones(4, 8), atol=1e-2), merged.float()[0, :2]


def _write_truncated_checkpoint(path):
    """A tiny causal-LM checkpoint missing every layer-1 tensor; returns (path, dropped_keys)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    os.makedirs(path, exist_ok=True)
    full_state = model.state_dict()
    state = {k: v for k, v in full_state.items() if "layers.1." not in k}
    save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(str(path))
    return str(path), sorted(set(full_state) - set(state))


def test_missing_keys_only_warn(tmp_path):
    """The premise behind every gate above: transformers does NOT raise on missing checkpoint keys.

    If a future transformers starts raising, these guards become belt-and-braces rather than the only
    thing standing between a sharded input and randomly initialized experts — and this test says so.
    """
    from transformers import AutoModelForCausalLM

    src, dropped = _write_truncated_checkpoint(tmp_path / "trunc")
    assert dropped, "fixture dropped no layer-1 tensors"

    reloaded = AutoModelForCausalLM.from_pretrained(src)  # must not raise
    live = reloaded.state_dict()
    assert all(k in live for k in dropped)
    assert not any(live[k].is_meta for k in dropped), "missing keys were left on meta, not re-initialized"


def test_patch_vocab_refuses_a_truncated_checkpoint(tmp_path, monkeypatch):
    """patch_vocab loads through the coverage gate (``auto_load_model``): a truncated source would
    otherwise be re-saved as a complete-looking patched checkpoint whose absent tensors are random
    (missing keys only warn — the premise test above)."""
    from scripts.before_training import patch_vocab as mod

    src, dropped = _write_truncated_checkpoint(tmp_path / "trunc")
    assert dropped
    out = tmp_path / "out"
    # The fixture ships no tokenizer; the load gate must fire before the tokenizer is ever used.
    monkeypatch.setattr(mod, "load_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(sys, "argv", ["patch_vocab.py", "--model_id", src, "--output_dir", str(out)])
    with pytest.raises(RuntimeError, match="randomly initialized"):
        mod.main()
    assert not out.exists(), "a refused patch must not create its output directory"


def test_patch_vocab_refuses_reset_sinks_on_a_family_that_has_none(tmp_path, monkeypatch):
    """``--reset_sinks`` on a sink-less family printed its banner and exited 0 over an unchanged
    checkpoint: the flag's entire effect dropped silently, and the artifact then reads to every
    downstream tool (provenance, the merge tools, the RL sink gate) as deliberately sink-free."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from scripts.before_training import patch_vocab as mod

    src, out = tmp_path / "src", tmp_path / "out"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).save_pretrained(str(src))
    # The fixture ships no tokenizer; with no --patterns it is never consulted.
    monkeypatch.setattr(mod, "load_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(
        sys, "argv", ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out), "--reset_sinks"]
    )

    with pytest.raises(ValueError, match="carries no attention sinks"):
        mod.main()
    assert not out.exists(), "a refused patch must not create its output directory"


def test_convert_deepseek_v4_bf16_loads_through_the_coverage_gate(tmp_path, monkeypatch):
    """The converter must route its load through ``from_pretrained_verified`` — a raw
    ``from_pretrained`` only warns on missing keys, and the tool would re-save random experts as a
    complete-looking 420 GB BF16 checkpoint. (The gate's raise behavior is pinned in
    test_checkpoint_coverage.py; the real FP8 load path needs the hub checkpoint, so this pins the
    wiring.)"""
    from scripts.before_training import convert_deepseek_v4_bf16 as mod

    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="deepseek_v4")
    out = tmp_path / "out"

    class _GateReached(Exception):
        pass

    def _gate(model_cls, model_id, **kwargs):
        assert model_id == str(src)
        raise _GateReached

    monkeypatch.setattr(mod, "from_pretrained_verified", _gate)
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(src), "--output_dir", str(out)])
    with pytest.raises(_GateReached):
        mod.main()
    assert not out.exists(), "nothing may be written before the gated load completes"


@pytest.mark.parametrize(
    "module",
    [
        "scripts.before_training.convert_mistral4_bf16",
        "scripts.before_training.convert_deepseek_v4_bf16",
        "scripts.before_training.convert_glm5_bf16",
    ],
)
def test_converter_progress_lines_survive_the_src_root_handler(module):
    """Importing anything under ``src`` installs a root handler at WARNING, and ``basicConfig`` is a
    no-op once the root has handlers — so an unforced call leaves these converters mute for the whole
    of a multi-hour, 100+ GiB run, with no way to tell progress from a hang.

    Checked in a subprocess: the root logger is process-wide, so an in-session assertion would read
    whatever an earlier test left behind.
    """
    probe = (
        "import importlib, logging; import src; "
        f"m = importlib.import_module({module!r}); "
        "print('INFO_ENABLED', m.logger.isEnabledFor(logging.INFO))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=_REPO_ROOT, check=True
    )
    assert "INFO_ENABLED True" in completed.stdout, completed.stdout + completed.stderr


def test_convert_to_bf16_refuses_merge_adapter_without_peft(tmp_path):
    """``--merge_adapter`` is read ONLY on the PEFT path, so without ``--peft`` it was a no-op.

    The tool loaded the source as a full model, never reached ``merge_and_unload``, and exited 0
    having written an UNMERGED bf16 checkpoint — the exact opposite of what a caller asking to merge
    an adapter wanted, with nothing in the output to tell the two apart. Refused before any I/O.
    """
    from scripts.after_training.convert_to_bf16 import convert_to_bf16

    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="qwen3_moe")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="--merge_adapter is only used with --peft"):
        convert_to_bf16(str(src), str(out), model_type="qwen3_moe", is_peft=False, merge_adapter=True)
    assert not out.exists(), "the refusal must land before anything is written"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
