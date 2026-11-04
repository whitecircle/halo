#!/usr/bin/env python
"""A gathered Gemma-4 EP save reloads as the checkpoint a plain ``save_pretrained`` writes. A doubled
``experts.experts.`` prefix and dropped buffers both still load, silently."""

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.checkpoint_io import save_context, written_keys
from tests.common.models import TINY_GEMMA4_MOE_CONFIG
from tests.common.parallelism import single_process_ep_config

LAYERS = TINY_GEMMA4_MOE_CONFIG["num_hidden_layers"]


def _tiny_gemma4_moe() -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    return Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG)).to(torch.bfloat16)


def _ep_patched(model: Gemma4ForCausalLM) -> Gemma4ForCausalLM:
    apply_ep_layers(model, single_process_ep_config())
    wrapped = [name for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
    # Anti-vacuity: on an unpatched model every assertion passes without exercising the gather.
    assert wrapped == [f"model.layers.{index}.experts" for index in range(LAYERS)], wrapped
    return model


def _saved_pair(tmp_path):
    reference, model = _tiny_gemma4_moe(), _ep_patched(_tiny_gemma4_moe())
    ep_dir, reference_dir = str(tmp_path / "ep"), str(tmp_path / "reference")
    save_ep_gathered_checkpoint(save_context(model), ep_dir)
    reference.save_pretrained(reference_dir)
    return reference, ep_dir, reference_dir


def test_gathered_save_writes_the_same_keys_as_save_pretrained(tmp_path):
    _reference, ep_dir, reference_dir = _saved_pair(tmp_path)
    written, expected = written_keys(ep_dir), written_keys(reference_dir)

    assert any(key.endswith("experts.gate_up_proj") for key in expected), "fixture wrote no expert keys"
    assert written - expected == set(), f"gathered save invented keys: {sorted(written - expected)}"
    assert expected - written == set(), f"gathered save dropped keys: {sorted(expected - written)}"


def test_persistent_buffers_survive_the_gathered_save(tmp_path):
    """A writer built from ``named_parameters()`` drops it, and the reload falls back to the constructor's value."""
    model = _ep_patched(_tiny_gemma4_moe())
    for index in range(LAYERS):
        model.model.layers[index].layer_scalar.fill_(0.25 * (index + 1))
    ep_dir = str(tmp_path / "ep")
    save_ep_gathered_checkpoint(save_context(model), ep_dir)

    written = written_keys(ep_dir)
    persistent = {name for name in model.state_dict() if name.endswith("layer_scalar")}
    assert persistent, "fixture carries no persistent buffer — the check below would be vacuous"
    assert persistent <= written, f"persistent buffers dropped: {sorted(persistent - written)}"

    reloaded = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16)
    for index in range(LAYERS):
        assert reloaded.model.layers[index].layer_scalar.item() == pytest.approx(0.25 * (index + 1))


def test_the_gathered_artifact_reloads_with_no_missing_or_unexpected_keys(tmp_path):
    """A doubled ``experts.experts.`` prefix shows up here as both at once."""
    reference, ep_dir, _reference_dir = _saved_pair(tmp_path)

    reloaded, loading_info = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16, output_loading_info=True)

    assert set(loading_info["missing_keys"]) == set()
    assert set(loading_info["unexpected_keys"]) == set()
    assert set(loading_info["mismatched_keys"]) == set()
    assert set(reloaded.state_dict()) == set(reference.state_dict())


def test_reloaded_values_match_the_source_weights(tmp_path):
    """Key parity alone would survive a mis-sliced or transposed bank."""
    reference, ep_dir, _reference_dir = _saved_pair(tmp_path)
    reloaded = Gemma4ForCausalLM.from_pretrained(ep_dir, dtype=torch.bfloat16)

    expected = reference.state_dict()
    differing = [key for key, tensor in reloaded.state_dict().items() if not torch.equal(tensor, expected[key])]
    assert differing == []
    assert any(key.endswith("experts.gate_up_proj") for key in expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
