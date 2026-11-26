#!/usr/bin/env python
"""``frozen_quantized_experts`` on the lazy EP loader, on CPU with EP patching stubbed. An Inkling 2 release-layout
checkpoint (NVFP4 and MXFP8 experts) loads with each expert container holding this rank's slice of the packed tensors
as buffers, which dequantize to the reference weights; every other tensor loads as ``from_pretrained`` loads it. The
same checkpoint without the option is refused."""

import pytest
import torch
from transformers import AutoConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration

import halo.model_loading.parallel.ep as ep_lazy_loader
from halo.distributed.expert_parallel.config import EPConfig
from halo.kernels.lowp.frozen_bank import build_frozen_banks
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.frozen_experts import apply_frozen_bank_slots
from halo.model_loading.parallel.lazy_plan import LAZY_QUANTIZED, lazy_load_blocker
from halo.models.moe.family_specs import INKLING2
from tests.common.inkling2 import TINY_INKLING2_MXFP8_LAYERS, TINY_INKLING2_TEXT, write_tiny_inkling2_release

SLOTS = INKLING2.frozen_expert_slots
EXPERTS = TINY_INKLING2_TEXT["n_routed_experts"]
_EXPERTS_KEY = "model.language_model.layers.{layer}.mlp.experts.{weight}"


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    """The release directory, and the state dict of the same weights with the experts dequantized to bf16."""
    root = tmp_path_factory.mktemp("inkling2_frozen")
    release_dir, reference_dir = str(root / "release"), str(root / "reference")
    write_tiny_inkling2_release(release_dir, reference_dir)
    reference = Inkling2ForConditionalGeneration.from_pretrained(reference_dir, dtype=torch.bfloat16)
    return release_dir, {key: value.detach().clone() for key, value in reference.state_dict().items()}


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    """EP patching needs DeepEP and process groups; the load itself does not."""
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)


def _lazy(release_dir: str, ep_size: int):
    return load_ep_model_lazy(
        release_dir,
        EPConfig(ep_size=ep_size, stage_world_size=ep_size, nvlink_domain_size=ep_size, experts_fsdp_managed=False),
        AutoConfig.from_pretrained(release_dir),
        dtype=torch.bfloat16,
        trust_remote_code=False,
        model_class=Inkling2ForConditionalGeneration,
        frozen_quantized_experts=True,
    )


def _expert_containers(model) -> dict[int, torch.nn.Module]:
    return {
        int(name.split(".layers.")[1].split(".")[0]): module
        for name, module in model.named_modules()
        if name.endswith(".mlp.experts")
    }


def test_the_release_passes_the_lazy_gate_only_with_the_option(release):
    release_dir, _ = release
    assert lazy_load_blocker(release_dir) == LAZY_QUANTIZED
    assert lazy_load_blocker(release_dir, frozen_quantized_experts=True) is None


@pytest.mark.parametrize("ep_size", [1, 2])
def test_each_container_holds_this_ranks_packed_slice(release, ep_size):
    release_dir, reference = release
    containers = _expert_containers(_lazy(release_dir, ep_size))
    assert sorted(containers) == list(range(TINY_INKLING2_TEXT["num_hidden_layers"]))
    local = EXPERTS // ep_size
    for layer, experts in containers.items():
        assert not {slot.weight for slot in SLOTS} & set(experts._parameters)
        banks = build_frozen_banks(experts, SLOTS, 0, local, already_sharded=True)
        for slot in SLOTS:
            bank = banks[slot.weight]
            assert bank.quant_format == ("mxfp8" if layer in TINY_INKLING2_MXFP8_LAYERS else "nvfp4")
            assert bank.num_experts == local
            expected = reference[_EXPERTS_KEY.format(layer=layer, weight=slot.weight)][:local]
            assert torch.equal(bank.dequantize_linear(), expected), f"layer {layer} {slot.weight}"


def test_a_second_apply_leaves_the_frozen_slots_as_they_are(release):
    release_dir, _ = release
    model = _lazy(release_dir, ep_size=2)
    held = {name: (buffer.dtype, buffer.shape) for name, buffer in model.named_buffers()}

    apply_frozen_bank_slots(model, release_dir)

    assert {name: (buffer.dtype, buffer.shape) for name, buffer in model.named_buffers()} == held
    assert not any(SLOTS[0].weight in dict(module.named_parameters(recurse=False)) for module in model.modules())


def test_every_other_tensor_loads_as_from_pretrained_loads_it(release):
    release_dir, reference = release
    loaded = _lazy(release_dir, 2).state_dict()
    expert_keys = {
        _EXPERTS_KEY.format(layer=layer, weight=slot.weight)
        for layer in range(TINY_INKLING2_TEXT["num_hidden_layers"])
        for slot in SLOTS
    }
    others = set(reference) - expert_keys
    assert others and others <= set(loaded)
    assert not [key for key in sorted(others) if not torch.equal(loaded[key].to(reference[key].dtype), reference[key])]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
