#!/usr/bin/env python
"""The weight sync folds each LoRA layer out of place into exactly what PEFT's merge would write.
The frozen base is never written; a layer the fold cannot reproduce, or a target the dense push skips, is refused."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from halo.distributed.lora_fold import lora_folded_data
from halo.models.peft_adapters import lora_fold_targets, lora_folded
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights, reject_unsyncable_model
from tests.common.adapters import adapter_param_items, randomize_adapters
from tests.common.ep_stubs import ReadyGatherMixin, StubEPLayerBase
from tests.common.gloo import run_gloo_ranks
from tests.common.weight_sync import RecordingSender, as_pushed, local_parameters, merged_by_peft, moved_parameters

_DIM = 16
_VOCAB = 32


def _patch_embed() -> nn.Conv3d:
    """A vision patch embedding's shape: a strided kernel over time and space."""
    return nn.Conv3d(3, _DIM, kernel_size=(2, 4, 4), stride=(2, 4, 4))


class _Net(nn.Module):
    """One of each layer type a LoRA config here targets, and a head the tied case shares."""

    def __init__(self, tied: bool = False, conv_groups: int = 1):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _DIM)
        self.conv = nn.Conv2d(_DIM, _DIM, 1, groups=conv_groups)
        self.conv1d = nn.Conv1d(_DIM, _DIM, 3)
        self.conv2d = nn.Conv2d(_DIM, _DIM, 3)
        self.conv3d = _patch_embed()
        self.proj = nn.Linear(_DIM, _DIM)
        self.lm_head = nn.Linear(_DIM, _VOCAB, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x):  # pragma: no cover - never called
        return x


class _Experts(nn.Module):
    """A 3-D expert bank, the parameter shape ``target_parameters`` wraps."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, _DIM, 2 * _DIM))

    def forward(self, x):  # pragma: no cover - never called
        return x


class _MoeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()

    def forward(self, x):  # pragma: no cover - never called
        return x


class _EPExperts(ReadyGatherMixin, StubEPLayerBase):
    """The same bank on an EP layer, which the expert gather sends as it is and the dense push skips."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, _DIM, 2 * _DIM))

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        return {"gate_up_proj": self.gate_up_proj.detach()}


class _EPMoeNet(nn.Module):
    """An EP expert bank beside a dense layer the push would send."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_DIM, _DIM)
        self.experts = _EPExperts()

    def forward(self, x):  # pragma: no cover - never called
        return x


_CASES = {
    "linear": (_Net, {"target_modules": ["proj"]}),
    "linear_lora_bias": (_Net, {"target_modules": ["proj"], "lora_bias": True}),
    "linear_dora": (_Net, {"target_modules": ["proj"], "use_dora": True}),
    "embedding": (_Net, {"target_modules": ["embed_tokens"]}),
    "embedding_dora": (_Net, {"target_modules": ["embed_tokens"], "use_dora": True}),
    "conv": (_Net, {"target_modules": ["conv"]}),
    "conv_dora": (_Net, {"target_modules": ["conv"], "use_dora": True}),
    "conv1d": (_Net, {"target_modules": ["conv1d"]}),
    "conv2d_kernel": (_Net, {"target_modules": ["conv2d"]}),
    "conv3d": (_Net, {"target_modules": ["conv3d"]}),
    "conv3d_dora": (_Net, {"target_modules": ["conv3d"], "use_dora": True}),
    "tied_head": (lambda: _Net(tied=True), {"target_modules": ["lm_head", "proj"]}),
    "tied_both": (lambda: _Net(tied=True), {"target_modules": ["embed_tokens", "lm_head"]}),
    "two_adapters": (_Net, {"target_modules": ["proj", "embed_tokens"]}),
    "param_wrapper": (_MoeNet, {"target_modules": [], "target_parameters": ["experts.gate_up_proj"]}),
}
_SECOND_ADAPTER = {"two_adapters"}


# (base, adapter) dtypes: the trainers' bf16 and fp32 alignments, and PEFT's default fp32 adapters on
# a bf16 base, where Linear's merge adds the delta unrounded and the other layers round it first.
_DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp32": (torch.float32, torch.float32),
    "bf16_base_fp32_adapters": (torch.bfloat16, torch.float32),
}


def _lora_model(case: str, dtypes: tuple[torch.dtype, torch.dtype]) -> nn.Module:
    """Adapter tensors drawn nonzero, since a ``lora_B`` at zero folds nothing; base and adapters in ``dtypes``."""
    torch.manual_seed(0)
    build, config = _CASES[case]
    base_dtype, adapter_dtype = dtypes
    model = get_peft_model(build().to(base_dtype), LoraConfig(r=4, lora_alpha=8, **config))
    if case in _SECOND_ADAPTER:
        model.add_adapter("second", LoraConfig(r=2, lora_alpha=4, **config))
        model.base_model.set_adapter(["default", "second"])
    for _, param in adapter_param_items(model):
        param.data = param.data.to(adapter_dtype)
    randomize_adapters(model, std=0.1)
    return model


@pytest.mark.parametrize("dtypes", sorted(_DTYPES))
@pytest.mark.parametrize("case", sorted(_CASES))
def test_lora_folded_is_peft_merge_bit_for_bit(case, dtypes):
    model = _lora_model(case, _DTYPES[dtypes])
    expected = merged_by_peft(model)
    before = local_parameters(model)

    targets = lora_fold_targets(model)
    folded = {
        name: lora_folded(param, targets[id(param)])
        for name, param in model.named_parameters()
        if id(param) in targets
    }

    assert folded, "premise: the case has a tensor to fold"
    moved_by_merge = {name for name in expected if not torch.equal(expected[name], before[name])}
    assert moved_by_merge == set(folded), "the fold targets are not the tensors PEFT's merge rewrites"
    wrong = sorted(name for name, value in folded.items() if not torch.equal(value, expected[name]))
    assert not wrong, f"folded out of place differently from PEFT's merge: {wrong}"
    assert not moved_parameters(before, local_parameters(model)), "the fold wrote the model's own weights"
    leaked = sorted(key for layers in targets.values() for layer in layers for key in layer._caches)
    assert not leaked, f"the fold left variant caches behind: {leaked}"


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_push_carries_the_peft_merge(case):
    """Each forwarded tensor is what PEFT's merge holds under that name; a tied head's delta rides on the embedding."""
    model = _lora_model(case, _DTYPES["bf16"])
    expected = as_pushed(model, merged_by_peft(model))
    before = local_parameters(model)

    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    pushed = {param.name: param.value for param in sender.params}

    assert pushed, "premise: the push forwarded tensors"
    assert set(pushed) == set(expected), (
        f"the push missed {sorted(set(expected) - set(pushed))} and forwarded {sorted(set(pushed) - set(expected))} "
        f"beyond PEFT's merge"
    )
    wrong = sorted(name for name, value in pushed.items() if not torch.equal(value, expected[name]))
    assert not wrong, f"pushed tensors differ from PEFT's merge: {wrong}"
    assert not moved_parameters(before, local_parameters(model)), "the push wrote the model's own weights"


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(_DIM, 2)

    def forward(self, x):  # pragma: no cover - never called
        return x


@pytest.mark.parametrize(
    "build, config",
    [
        (_Attention, {"target_modules": ["mha"]}),
        (_Net, {"target_modules": ["proj"], "trainable_token_indices": {"embed_tokens": [0, 1]}}),
        (_Net, {"target_modules": ["proj"], "alora_invocation_tokens": [1]}),
        (lambda: _Net(conv_groups=2), {"target_modules": ["conv"]}),
    ],
    ids=["multihead_attention", "trainable_tokens", "alora_variant", "grouped_conv"],
)
def test_a_layer_the_fold_cannot_reproduce_is_refused_at_construction(build, config):
    """The fold has no formula for these, so construction refuses them before their base is pushed unfolded."""
    model = get_peft_model(build(), LoraConfig(r=4, lora_alpha=8, **config))
    with pytest.raises(ValueError, match="cannot be folded out of place"):
        reject_unsyncable_model(model, "vllm")


def test_a_peft_target_the_dense_push_skips_is_refused():
    """The gather sends the bank without PEFT's delta; a push reaching it anyway raises on every rank at the end."""
    model = get_peft_model(_EPMoeNet(), LoraConfig(r=4, lora_alpha=8, target_parameters=["experts.gate_up_proj"]))
    with pytest.raises(ValueError, match="does not send through its dense push"):
        reject_unsyncable_model(model, "vllm")
    with pytest.raises(ValueError, match="does not send through its dense push"):
        gather_and_send_weights(model, RecordingSender())


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _patch_embed()

    def forward(self, x):  # pragma: no cover - never called
        return x


def _fold_sharded_conv(rank: int, use_dora: bool) -> None:
    """Base and adapters sharded on dim 0 across two ranks, as FSDP2 holds them."""
    torch.manual_seed(0)
    model = get_peft_model(_PatchEmbed(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], use_dora=use_dora))
    randomize_adapters(model, std=0.1)
    expected = merged_by_peft(copy.deepcopy(model))
    mesh = init_device_mesh("cpu", (2,))
    for module in model.modules():
        for key, param in list(module._parameters.items()):
            if param is None:
                continue
            module._parameters[key] = nn.Parameter(
                distribute_tensor(param.data, mesh, [Shard(0)]), requires_grad=param.requires_grad
            )
    targets = lora_fold_targets(model)
    for name, param in model.named_parameters():
        if id(param) in targets:
            folded = lora_folded_data(param, targets)
            assert isinstance(folded, DTensor), "the fold must stay sharded like its base"
            assert torch.equal(folded.full_tensor(), expected[name]), f"rank {rank}: {name} differs from PEFT's merge"


@pytest.mark.parametrize("use_dora", [False, True], ids=["lora", "dora"])
def test_a_sharded_conv_adapter_folds_to_peft_merge(use_dora):
    """A DTensor convolution convolves each rank's shard of the rank dimension, so the fold gathers the adapter first."""
    run_gloo_ranks(_fold_sharded_conv, 2, use_dora)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
