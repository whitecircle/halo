#!/usr/bin/env python
"""A gathered Gemma-4 EP save must reload as the checkpoint a plain ``save_pretrained`` would write.

Gemma-4 is the family where the gathered writer has the least margin. Its EP wrapper replaces
``Gemma4TextExperts`` itself rather than an enclosing MoE block, so the base gather's ``experts.``
prefix would double into ``...experts.experts.gate_up_proj`` — keys that no loader claims, which
``from_pretrained`` reports as *missing* (silently randomly-initialized experts) plus *unexpected*
extras. The same save also has to carry the model's persistent buffers, which live outside
``named_parameters()`` and a parameters-only walk drops wholesale.

Neither failure raises. Both produce a directory that loads, so the gate has to be a key-set
comparison against the reference artifact — plain ``save_pretrained`` of the same weights — plus a
value comparison, plus an explicit unexpected/missing-key check from the reload itself.

Built in bf16 because the gathered writer casts non-norm floats to bf16 on the way out; an fp32
fixture would make every value comparison a dtype comparison.

    python tests/cpu/checkpoint/test_gemma4_tiny_save_roundtrip.py
"""

import pytest
import torch
from accelerate import PartialState
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

PartialState()  # save_ep_model logs through accelerate's logger

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.expert_parallel.saving import save_ep_model
from tests.common.checkpoint_io import written_keys
from tests.common.models import TINY_GEMMA4_MOE_CONFIG

LAYERS = TINY_GEMMA4_MOE_CONFIG["num_hidden_layers"]


def _tiny_gemma4_moe() -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    return Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG)).to(torch.bfloat16)


def _ep_patched(model: Gemma4ForCausalLM) -> Gemma4ForCausalLM:
    patch_moe_model_for_ep(model, EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False))
    wrapped = [name for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
    # Anti-vacuity: on an unpatched model every assertion below passes without exercising the gather.
    assert wrapped == [f"model.layers.{index}.experts" for index in range(LAYERS)], wrapped
    return model


def _saved_pair(tmp_path):
    """The EP-gathered artifact and the plain ``save_pretrained`` reference, from identical weights."""
    reference, model = _tiny_gemma4_moe(), _ep_patched(_tiny_gemma4_moe())
    ep_dir, reference_dir = str(tmp_path / "ep"), str(tmp_path / "reference")
    save_ep_model(model, ep_dir)
    reference.save_pretrained(reference_dir)
    return reference, ep_dir, reference_dir


def test_gathered_save_writes_the_same_keys_as_save_pretrained(tmp_path):
    _reference, ep_dir, reference_dir = _saved_pair(tmp_path)
    written, expected = written_keys(ep_dir), written_keys(reference_dir)

    assert any(key.endswith("experts.gate_up_proj") for key in expected), "fixture wrote no expert keys"
    assert written - expected == set(), f"gathered save invented keys: {sorted(written - expected)}"
    assert expected - written == set(), f"gathered save dropped keys: {sorted(expected - written)}"


def test_persistent_buffers_survive_the_gathered_save(tmp_path):
    """``layer_scalar`` is a persistent buffer, so it belongs in the checkpoint even though it is not
    a parameter; a writer built from ``named_parameters()`` alone silently drops it and the reload
    falls back to the constructor's value."""
    model = _ep_patched(_tiny_gemma4_moe())
    for index in range(LAYERS):
        model.model.layers[index].layer_scalar.fill_(0.25 * (index + 1))
    ep_dir = str(tmp_path / "ep")
    save_ep_model(model, ep_dir)

    written = written_keys(ep_dir)
    persistent = {name for name in model.state_dict() if name.endswith("layer_scalar")}
    assert persistent, "fixture carries no persistent buffer — the check below would be vacuous"
    assert persistent <= written, f"persistent buffers dropped: {sorted(persistent - written)}"

    reloaded = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16)
    for index in range(LAYERS):
        assert reloaded.model.layers[index].layer_scalar.item() == pytest.approx(0.25 * (index + 1))


def test_the_gathered_artifact_reloads_with_no_missing_or_unexpected_keys(tmp_path):
    """The user-facing end of the path: ``from_pretrained`` must claim every written key and find
    every key it needs — a doubled ``experts.experts.`` prefix shows up here as both at once."""
    reference, ep_dir, _reference_dir = _saved_pair(tmp_path)

    reloaded, loading_info = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16, output_loading_info=True)

    assert set(loading_info["missing_keys"]) == set()
    assert set(loading_info["unexpected_keys"]) == set()
    assert set(loading_info["mismatched_keys"]) == set()
    assert set(reloaded.state_dict()) == set(reference.state_dict())


def test_reloaded_values_match_the_source_weights(tmp_path):
    """Key parity alone would survive a gather that mis-slices or transposes an expert bank."""
    reference, ep_dir, _reference_dir = _saved_pair(tmp_path)
    reloaded = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16)

    expected = reference.state_dict()
    differing = [key for key, tensor in reloaded.state_dict().items() if not torch.equal(tensor, expected[key])]
    assert differing == []
    assert any(key.endswith("experts.gate_up_proj") for key in expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
