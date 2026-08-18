#!/usr/bin/env python
"""What an exported artifact carries out of the run that trained it.

``convert_to_bf16`` and ``merge_peft_adapters`` both assemble a servable model from a run's output,
and both owe it the same two sidecars — the training provenance and the router-balancing state. The
sequence has one owner (``apply_training_sidecars`` / ``copy_training_sidecars``), which also closes
the dtype gap: every in-training save keeps the balancing tensors at the dtype they were TRAINED in
(``save_dtype_caster``), while a conversion loads the whole checkpoint at the target dtype — so a
per-tool sequence lets two exports of the same run disagree on the routing bias, and a bf16 round
trip quantizes away the ~1e-3 sign steps the balancing update writes.

    python tests/cpu/checkpoint/test_training_sidecar_transfer.py
"""

import json
import os
import types

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.checkpoint import tool_io
from src.checkpoint.format import (
    PROVENANCE_GPT_OSS_SINKS,
    ROUTER_BALANCING_BIASES_FILE,
    TRAINING_PROVENANCE_FILE,
)
from src.checkpoint.tool_io import (
    TRAINING_STATE_FILES,
    apply_training_sidecars,
    copy_training_sidecars,
)

# One of the registry-declared native slots (DeepSeek-V3-style ALF), so the key predicate under test
# is the shipped one rather than a spelling invented here.
BALANCING_KEY = "model.layers.0.mlp.gate.e_score_correction_bias"
# Values a bf16 round trip cannot represent: the sign steps a balancing run actually writes.
TRAINED_BIAS = torch.tensor([1e-3, -3e-3, 7.5e-4, -1.25e-3], dtype=torch.float32)


class _Gate(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(len(TRAINED_BIAS), dtype=dtype))


class _Model(nn.Module):
    """The shape of a hub MoE model as a conversion tool holds it: no EP wrappers, no adoption
    marks — only the checkpoint key spelling identifies the balancing tensor."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        # The provenance re-apply reads the family off the config, as it does on a real model.
        self.config = types.SimpleNamespace(model_type="deepseek_v4")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = _Gate(dtype)
        self.model.embed_tokens = nn.Linear(4, 4, bias=False, dtype=dtype)

    @property
    def bias(self) -> torch.Tensor:
        return self.model.layers[0].mlp.gate.e_score_correction_bias


def _checkpoint(tmp_path, *, bias: torch.Tensor):
    """A source checkpoint storing the balancing tensor at its trained fp32."""
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {BALANCING_KEY: bias, "model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16)},
        str(source / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return source


def test_the_balancing_tensor_survives_the_conversion_dtype(tmp_path):
    """The load downcast it to the conversion dtype; the export must carry the trained values, not
    their bf16 rounding — a flipped 1e-3 step is a routing decision that changes."""
    model = _Model()
    assert model.bias.dtype == torch.bfloat16, "premise: the load downcast the slot"

    apply_training_sidecars(model, str(_checkpoint(tmp_path, bias=TRAINED_BIAS)))

    assert model.bias.dtype == torch.float32
    assert torch.equal(model.bias, TRAINED_BIAS)


def test_a_checkpoint_without_balancing_state_is_left_alone(tmp_path):
    """No balancing tensor, nothing to restore: the helper must not touch an ordinary model."""
    source = tmp_path / "dense"
    source.mkdir()
    save_file(
        {"model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16)},
        str(source / "model.safetensors"),
        metadata={"format": "pt"},
    )
    model = nn.Linear(4, 4, dtype=torch.bfloat16)

    assert apply_training_sidecars(model, str(source)) == []
    assert model.weight.dtype == torch.bfloat16


def test_an_adapter_directory_gives_the_slot_fp32_storage_for_its_sidecar(tmp_path, monkeypatch):
    """A merge starts from BASE weights, so the adapter directory carries no model shards at all —
    and the sidecar copy writes at the SLOT's dtype, which would round the trained fp32 bias into
    the conversion dtype on the way in."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    sidecar = {"model.layers.0.mlp.gate": TRAINED_BIAS}
    torch.save(sidecar, adapter_dir / ROUTER_BALANCING_BIASES_FILE)
    model = _Model()
    seen: dict = {}

    def fake_apply(target, loaded):
        seen["dtype"] = target.bias.dtype
        seen["sidecar"] = loaded
        return ["model.layers.0.mlp.gate"], ["model.layers.9.mlp.gate"]

    monkeypatch.setattr(tool_io, "apply_router_balancing_sidecar", fake_apply)
    actions = apply_training_sidecars(model, str(adapter_dir))

    assert seen["dtype"] == torch.float32, "the sidecar would have been rounded into the slot's bf16"
    assert torch.equal(seen["sidecar"]["model.layers.0.mlp.gate"], TRAINED_BIAS)
    assert any("1 native slots" in action for action in actions)
    # The transient-bias warning is what tells the operator their export routes differently from
    # training; the two copies of this sequence had already lost half of it.
    assert any(action.startswith("WARNING:") and "TRANSIENT" in action for action in actions)


def test_a_source_with_its_own_weights_keeps_its_sidecar_as_a_travelling_copy(tmp_path, monkeypatch):
    """``router_balancing_biases.pt`` beside a full checkpoint is the RESUME copy — the trained bias
    is already in the weights. Applying it there re-decides a checkpoint that is correct, against a
    sidecar whose module names came from the trainer's tree (an EP run's is not the hub tree), so a
    plain conversion would refuse checkpoints it otherwise converts."""
    source = _checkpoint(tmp_path, bias=TRAINED_BIAS)
    torch.save({"model.layers.0.mlp.gate": TRAINED_BIAS}, source / ROUTER_BALANCING_BIASES_FILE)

    def refuse(*args, **kwargs):
        raise AssertionError("the sidecar must not be re-applied to a source that carries weights")

    monkeypatch.setattr(tool_io, "apply_router_balancing_sidecar", refuse)
    model = _Model()

    assert apply_training_sidecars(model, str(source)) == []
    assert torch.equal(model.bias, TRAINED_BIAS), "the weights' own trained bias must still be restored"


def test_the_provenance_action_is_reported_alongside(tmp_path, monkeypatch):
    """Both records come back through one call, so a tool cannot report one and drop the other."""
    source = _checkpoint(tmp_path, bias=TRAINED_BIAS)
    (source / TRAINING_PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_GPT_OSS_SINKS: "neutralized"}))
    monkeypatch.setattr(tool_io, "apply_sinks_policy", lambda *args, **kwargs: None)

    actions = apply_training_sidecars(_Model(), str(source))

    assert any("sinks" in action for action in actions), actions


def test_the_sidecars_are_carried_to_the_output(tmp_path):
    """The writes that do not go through save_full_checkpoint (an adapter-only save, a merge whose
    source_dir is the base model) must still ship both files, or the next tool in the chain reads a
    directory that has forgotten how the run was trained."""
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    output.mkdir()
    for name in TRAINING_STATE_FILES:
        (source / name).write_bytes(b"payload")

    copy_training_sidecars(str(source), str(output))

    assert sorted(os.listdir(output)) == sorted(TRAINING_STATE_FILES)


def test_carrying_sidecars_that_do_not_exist_is_a_no_op(tmp_path):
    """Most runs train no balancing bias and record no provenance."""
    source, output = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    output.mkdir()

    copy_training_sidecars(str(source), str(output))

    assert os.listdir(output) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
