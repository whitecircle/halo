#!/usr/bin/env python
"""A hub-namespace family's (Step-3.7) weight sync must forward exactly what its hub checkpoint carries, since vLLM
silently drops every name it does not map. The oracle is transformers' own ``save_pretrained``."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers.core_model_loading import revert_weight_conversion
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.models.param_roles import persistent_buffers
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.grpo.rollout import weight_push
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.checkpoint_io import safetensors_state_dict
from tests.common.ep_stubs import ReadyGather
from tests.common.models import TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
from tests.common.repo import REPO_ROOT

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
    """An instance of the real wrapper class over a stock block's tensors, built without the EP process groups. An
    instance, since a subclass would join the EP family registry for every test in the session."""
    layer = object.__new__(EPStep3p7MoELayer)
    nn.Module.__init__(layer)
    layer._expert_lora_attrs = frozenset()
    layer.gate_up_proj = block.experts.gate_up_proj
    layer.down_proj = block.experts.down_proj
    layer.gate = block.gate
    layer.shared_experts = block.shared_experts
    layer.forward = lambda hidden_states, **kwargs: hidden_states  # pragma: no cover - never called
    layer.expert_named_params = lambda: [("gate_up_proj", layer.gate_up_proj), ("down_proj", layer.down_proj)]
    layer.start_gather_expert_state_dict = lambda **kwargs: ReadyGather(
        {
            "experts.gate_up_proj": layer.gate_up_proj.detach(),
            "experts.down_proj": layer.down_proj.detach(),
        }
        if kwargs["retain"]
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

    def finish_weight_update(self) -> None:  # pragma: no cover - the caller flushes, not the gather
        pass


def _sync(model: nn.Module) -> dict[str, torch.Tensor]:
    sender = _RecordingSender()
    gather_and_send_weights(model, sender)
    return sender.sent


def _unsynced_buffer_keys(model: Step3p7ForConditionalGeneration) -> set[str]:
    """Hub keys of the persistent buffers, the only on-disk tensors a parameter payload omits."""
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
    """``save_pretrained`` of the seed-0 tiny model: the hub layout, from transformers itself."""
    _tiny_step3p7().save_pretrained(tmp_path)
    return safetensors_state_dict(tmp_path)


def test_wrapperless_sync_forwards_the_hub_checkpoint(on_disk):
    """``use_grouped_gemm: false`` at ``ep_size: 1`` leaves the stock tree; the family resolves off the registry."""
    model = _tiny_step3p7()
    _assert_sync_equals_checkpoint(_sync(model), on_disk, model)


def test_ep_wrapped_sync_forwards_the_hub_checkpoint(on_disk):
    """The expert gather's fused ``experts.gate_up_proj`` lands as the hub's two ``moe.*`` stacks."""
    model = _ep_wrapped(_tiny_step3p7())
    _assert_sync_equals_checkpoint(_sync(model), on_disk, model)


def test_peft_sync_forwards_base_names_in_the_hub_namespace(on_disk):
    """PEFT names are normalized before the revert, whose patterns anchor on the base tree; the zero-initialized
    adapter merges to the base weight, so the merged q_proj equals the checkpoint."""
    model = get_peft_model(_ep_wrapped(_tiny_step3p7()), LoraConfig(r=2, target_modules=["q_proj"]))
    sent = _sync(model)
    assert not any("lora" in key for key in sent)
    _assert_sync_equals_checkpoint(sent, on_disk, model.get_base_model())


def test_the_dense_tensors_a_converter_holds_are_bounded_on_the_forwarding_rank(monkeypatch):
    """A dense tensor a many-to-one reverse converter claims is held until its sources are complete, on the forwarding
    rank's GPU, as a copy nothing else keeps. A converter claiming a per-decoder-layer tensor would hold a whole stack
    there, unseen until the host OOMs."""
    monkeypatch.setattr(weight_push, "WEIGHT_SYNC_CHUNK_BYTES", 512)

    with pytest.raises(RuntimeError, match="reverse WeightConverter") as refused:
        _sync(_ep_wrapped(_tiny_step3p7()))
    assert "vision" in str(refused.value), "the bound fired on something other than the vision q/k/v it guards"


def test_the_expert_stacks_stay_outside_the_held_bound(monkeypatch):
    """The gather already holds them and each layer flushes, so a bound at any size must reach the expert stacks whole.
    Only the vision q/k/v that follow may raise, which shows the bound is still armed."""
    monkeypatch.setattr(weight_push, "WEIGHT_SYNC_CHUNK_BYTES", 1)
    sender = _RecordingSender()
    with pytest.raises(RuntimeError, match="vision"):
        gather_and_send_weights(_ep_wrapped(_tiny_step3p7()), sender)
    for i in _SPARSE_LAYERS:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.{i}.moe.{proj}.weight" in sender.sent


def test_the_release_shape_goes_through_the_expert_path():
    """288 experts × 1280 × 4096 is a 5.6 GiB fused bank per layer: the first push of the real checkpoint raised on the
    first expert layer on every rank, past the 198B load. Meta tensors of the real shapes, the real forwarder."""
    raw = json.loads((REPO_ROOT / "docker/vllm/parity/fixtures/step3p7/config.json").read_text())
    config = Step3p7Config(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "model_type")})
    with init_empty_weights():
        model = Step3p7ForConditionalGeneration(config).to(torch.bfloat16)
    text = config.text_config
    sparse = [i for i, kind in enumerate(text.mlp_layer_types) if kind == "sparse"]
    for i in sparse:
        layer = model.model.language_model.layers[i]
        layer.mlp = _ep_stub(layer.mlp)
    ep_layers = dict(named_ep_layers(model))
    guard = DeferredRankFailure("release-shape push")
    sender = _RecordingSender()
    forwarder = weight_push._HubForwarder(sender, model, ep_layers, None, guard)
    weight_push._send_ep_expert_weights(ep_layers, forwarder, guard)

    assert guard.reason is None, guard.reason
    assert next(iter(ep_layers.values())).gate_up_proj.nbytes > weight_push.WEIGHT_SYNC_CHUNK_BYTES
    for i in sparse:
        for proj, shape in (
            ("gate_proj", (text.moe_num_experts, text.moe_intermediate_size, text.hidden_size)),
            ("up_proj", (text.moe_num_experts, text.moe_intermediate_size, text.hidden_size)),
            ("down_proj", (text.moe_num_experts, text.hidden_size, text.moe_intermediate_size)),
        ):
            assert sender.sent[f"model.layers.{i}.moe.{proj}.weight"].shape == shape


def test_hub_layouts_not_module_layouts():
    """The split expert stacks, the shared expert's hub spelling, the fused vision q/k/v, no module-tree spelling."""
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
