#!/usr/bin/env python
"""A hub-namespace family's weight sync must forward exactly what its hub checkpoint carries.

Step-3.7's live module tree exists only behind transformers' from_pretrained-side conversion
(``model.language_model.layers.*.mlp.experts.gate_up_proj`` fused; the vision tower's split,
RoPE-permuted q/k/v), while vLLM 0.26.0 loads the HUB namespace (``model.layers.*.moe.gate_proj`` /
``up_proj`` stacks, ``share_expert.*``, ``vision_model.transformer.resblocks.*.attn.in_proj_weight``)
and silently drops every name it does not map. The sync therefore runs the same save-side revert the
gathered EP save applies (``_EXPORTS_HUB_NAMESPACE``). The oracle is transformers' own
``save_pretrained``: the forwarded keys must equal the on-disk keys — minus the router's frozen
selection-bias buffer, which the parameter-only payload never carries — and every forwarded tensor
must equal the on-disk one, so a rename that moved the wrong payload, a split on the wrong axis, or
an un-permuted q/k/v fusion all fail here. Pinned on both sync paths: the wrapper-less tree (the
registry resolves the family off ``config.model_type``) and an EP-wrapped tree (the expert gather).

Run: ``python tests/cpu/grpo/test_weight_sync_hub_namespace.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.models.structure import persistent_buffers
from src.trainers.grpo.rollout import weight_sync
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights
from tests.common.models import TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
from tests.common.utils import safetensors_state_dict

_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]
_NUM_EXPERTS = TINY_STEP3P7_CONFIG["n_routed_experts"]
_MOE_INTERMEDIATE = TINY_STEP3P7_CONFIG["moe_intermediate_size"]
_HIDDEN = TINY_STEP3P7_CONFIG["hidden_size"]


def _tiny_step3p7() -> Step3p7ForConditionalGeneration:
    torch.manual_seed(0)
    config = Step3p7Config(
        text_config=dict(TINY_STEP3P7_CONFIG), vision_config=dict(TINY_STEP3P7_VISION_CONFIG), image_token_id=2000
    )
    return Step3p7ForConditionalGeneration(config).to(torch.bfloat16)


def _ep_stub(block: nn.Module) -> EPStep3p7MoELayer:
    """An instance of the real wrapper class (its contract flags included) over a stock block's
    tensors, built without the EP process groups: experts held directly on the wrapper as the real
    one holds them, router and shared expert adopted by reference, and a gather that returns the
    fused F.linear layout the base gather produces. An INSTANCE, not a subclass — a subclass would
    join the EP family registry (a subclass walk) for every test collected in the same session."""
    layer = object.__new__(EPStep3p7MoELayer)
    nn.Module.__init__(layer)
    layer._expert_lora_attrs = frozenset()
    layer.gate_up_proj = block.experts.gate_up_proj
    layer.down_proj = block.experts.down_proj
    layer.gate = block.gate
    layer.shared_experts = block.shared_experts
    layer.forward = lambda hidden_states, **kwargs: hidden_states  # pragma: no cover - never called
    layer.expert_named_params = lambda: [("gate_up_proj", layer.gate_up_proj), ("down_proj", layer.down_proj)]
    layer.gather_expert_state_dict = lambda device="cpu", merge_lora=False, retain=True: (
        {
            "experts.gate_up_proj": layer.gate_up_proj.detach(),
            "experts.down_proj": layer.down_proj.detach(),
        }
        if retain
        else {}
    )
    return layer


def _ep_wrapped(model: Step3p7ForConditionalGeneration) -> Step3p7ForConditionalGeneration:
    for i in _SPARSE_LAYERS:
        layer = model.model.language_model.layers[i]
        layer.mlp = _ep_stub(layer.mlp)
    return model


class _RecordingSender:
    def __init__(self):
        self.sent: dict[str, torch.Tensor] = {}

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert name not in self.sent, f"{name} forwarded twice"
        self.sent[name] = weights

    def reset_prefix_cache(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _sync(model: nn.Module) -> dict[str, torch.Tensor]:
    sender = _RecordingSender()
    gather_and_send_weights(model, sender)
    return sender.sent


def _unsynced_buffer_keys(model: Step3p7ForConditionalGeneration) -> set[str]:
    """Hub keys of the persistent buffers — the only on-disk tensors a parameter payload omits."""
    return set(revert_weight_conversion(model, dict(persistent_buffers(model))))


def _assert_sync_equals_checkpoint(sent: dict[str, torch.Tensor], on_disk: dict[str, torch.Tensor], model) -> None:
    unsynced = _unsynced_buffer_keys(model)
    assert unsynced == {f"model.layers.{i}.moe.router_bias" for i in _SPARSE_LAYERS}, unsynced
    assert set(sent) == set(on_disk) - unsynced, (
        f"only-sync={sorted(set(sent) - set(on_disk))[:5]}, only-disk={sorted(set(on_disk) - unsynced - set(sent))[:5]}"
    )
    for key, tensor in sent.items():
        assert torch.equal(tensor, on_disk[key]), f"{key}: forwarded tensor differs from the checkpoint"


@pytest.fixture
def on_disk(tmp_path):
    """``save_pretrained`` of the seed-0 tiny model — the hub layout, from transformers itself."""
    _tiny_step3p7().save_pretrained(tmp_path)
    return safetensors_state_dict(tmp_path)


def test_wrapperless_sync_forwards_the_hub_checkpoint(on_disk):
    """``use_grouped_gemm: false`` at ``ep_size: 1`` leaves the stock tree; the family still resolves
    off the registry and the fused stock experts go through the dense path's held split."""
    model = _tiny_step3p7()
    _assert_sync_equals_checkpoint(_sync(model), on_disk, model)


def test_ep_wrapped_sync_forwards_the_hub_checkpoint(on_disk):
    """The expert gather's fused ``experts.gate_up_proj`` lands as the hub's two ``moe.*`` stacks."""
    model = _ep_wrapped(_tiny_step3p7())
    _assert_sync_equals_checkpoint(_sync(model), on_disk, model)


def test_peft_sync_forwards_base_names_in_the_hub_namespace(on_disk):
    """PEFT names are normalized BEFORE the revert (its patterns anchor on the base tree); the
    zero-initialized adapter merges to the base weight, so the merged q_proj equals the checkpoint."""
    model = get_peft_model(_ep_wrapped(_tiny_step3p7()), LoraConfig(r=2, target_modules=["q_proj"]))
    sent = _sync(model)
    assert not any("lora" in key for key in sent)
    _assert_sync_equals_checkpoint(sent, on_disk, model.get_base_model())


def test_the_held_converter_tensors_are_bounded_on_the_forwarding_rank(monkeypatch):
    """A tensor a many-to-one reverse converter claims is HELD until its sources are complete — on
    the forwarding rank's GPU, beside the gather that produced it.

    Bounded and refused by name: the shipped claims are megabytes, but a family whose converter
    claims a per-decoder-layer tensor would hold the whole stack on that one rank, which is the
    1x-model rank-local allocation the streamed sync exists to avoid. Silently, since the held set
    is invisible until the host OOMs.
    """
    monkeypatch.setattr(weight_sync, "_HELD_CONVERTER_BUDGET_BYTES", 512)

    with pytest.raises(RuntimeError, match="reverse WeightConverter"):
        _sync(_ep_wrapped(_tiny_step3p7()))


def test_hub_layouts_not_module_layouts():
    """The load-bearing conversions, named: the split expert stacks, the shared expert's hub spelling,
    the fused vision q/k/v, and no module-tree spelling anywhere in the payload."""
    sent = _sync(_ep_wrapped(_tiny_step3p7()))
    for i in _SPARSE_LAYERS:
        for proj in ("gate_proj", "up_proj"):
            assert sent[f"model.layers.{i}.moe.{proj}.weight"].shape == (_NUM_EXPERTS, _MOE_INTERMEDIATE, _HIDDEN)
        assert sent[f"model.layers.{i}.moe.down_proj.weight"].shape == (_NUM_EXPERTS, _HIDDEN, _MOE_INTERMEDIATE)
        assert f"model.layers.{i}.moe.gate.weight" in sent
        assert f"model.layers.{i}.share_expert.gate_proj.weight" in sent
    width = TINY_STEP3P7_VISION_CONFIG["hidden_size"]
    assert sent["vision_model.transformer.resblocks.0.attn.in_proj_weight"].shape == (3 * width, width)
    assert "vit_large_projector.weight" in sent and "lm_head.weight" in sent
    module_spellings = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector", "q_proj")
    leaked = sorted(
        key for key in sent if any(s in key for s in module_spellings) and not key.startswith("model.layers")
    )
    assert not leaked, leaked
    assert not any(".mlp.experts." in key or "language_model" in key or "shared_experts" in key for key in sent)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
