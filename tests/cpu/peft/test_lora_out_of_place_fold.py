#!/usr/bin/env python
"""The weight sync folds each LoRA layer out of place into exactly what PEFT's merge would write.
The frozen base is never written; a layer the fold cannot reproduce, or a target the dense push skips, is refused."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from halo.models.peft_adapters import lora_fold_targets, lora_folded
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights, reject_unsyncable_model
from tests.common.adapters import adapter_param_items, randomize_adapters
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.weight_sync import RecordingSender, as_pushed, local_parameters, merged_by_peft, moved_parameters

_DIM = 16
_VOCAB = 32


class _Net(nn.Module):
    """One of each layer type a LoRA config here targets, and a head the tied case shares."""

    def __init__(self, tied: bool = False, conv_groups: int = 1):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _DIM)
        self.conv = nn.Conv2d(_DIM, _DIM, 1, groups=conv_groups)
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


class _EPExperts(StubEPLayerBase):
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
