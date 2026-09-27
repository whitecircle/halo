#!/usr/bin/env python
"""The weight sync folds each LoRA layer out of place into exactly what PEFT's merge would write.

``lora_folded`` recomputes a merge one tensor at a time without touching the base, so the sync's
memory stays at one tensor's temporaries and the frozen base is never written. That only holds if it
reproduces PEFT's in-place ``merge_adapter`` bit for bit on every layer type it claims: plain Linear,
Embedding and Conv, a ``lora_bias`` fold into the base bias, DoRA's variant merge, a ``ParamWrapper``
(``target_parameters``), a LoRA'd head tied to the embedding (``named_parameters()`` lists the tensor
under the embedding's name), both halves of that tie adapted (two layers fold into one tensor, in
merge order), and two active adapters on one layer. Each case is pinned against PEFT's own merge of a
copy of the model, through ``lora_folded`` and through the whole push, with the model's own
parameters bit-identical afterwards. A layer whose merge the fold cannot reproduce must be refused by
the trainers' construction gate, not pushed unfolded.

Run: ``python tests/cpu/peft/test_lora_out_of_place_fold.py`` (or ``pytest -m cpu``).
"""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from src.models.structure import lora_fold_targets, lora_folded, normalize_peft_param_name
from src.trainers.grpo.rollout.weight_sync import gather_and_send_weights, validate_weight_sync_support
from tests.common.weight_sync import RecordingSender, local_parameters, moved_parameters

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
# Cases that also carry a second adapter, active alongside the first.
_SECOND_ADAPTER = {"two_adapters"}


# (base, adapter) dtypes: the trainers' bf16 and fp32 alignments, and PEFT's default fp32 adapters on
# a bf16 base, where Linear's merge adds the delta unrounded and the other layers round it first.
_DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp32": (torch.float32, torch.float32),
    "bf16_base_fp32_adapters": (torch.bfloat16, torch.float32),
}


def _lora_model(case: str, dtypes: tuple[torch.dtype, torch.dtype]) -> nn.Module:
    """The case's model with every adapter tensor drawn nonzero (``lora_B`` starts at zero, which
    folds nothing), base and adapters in ``dtypes``."""
    torch.manual_seed(0)
    build, config = _CASES[case]
    base_dtype, adapter_dtype = dtypes
    model = get_peft_model(build().to(base_dtype), LoraConfig(r=4, lora_alpha=8, **config))
    if case in _SECOND_ADAPTER:
        model.add_adapter("second", LoraConfig(r=2, lora_alpha=4, **config))
        model.base_model.set_adapter(["default", "second"])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = (torch.randn_like(param, dtype=torch.float32) * 0.1).to(adapter_dtype)
    return model


def _merged_by_peft(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter of a copy of ``model`` after PEFT's in-place ``merge_adapter``, by live name."""
    merged = copy.deepcopy(model)
    merged.merge_adapter()
    return {name: param.detach().clone() for name, param in merged.named_parameters()}


@pytest.mark.parametrize("dtypes", sorted(_DTYPES))
@pytest.mark.parametrize("case", sorted(_CASES))
def test_lora_folded_is_peft_merge_bit_for_bit(case, dtypes):
    model = _lora_model(case, _DTYPES[dtypes])
    expected = _merged_by_peft(model)
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


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_push_carries_the_peft_merge(case):
    """The whole push, names included: every forwarded tensor is the one PEFT's merge would hold
    under that base-model name. A tied head's delta rides on the embedding name the push sends."""
    model = _lora_model(case, _DTYPES["bf16"])
    expected = {
        normalized: value
        for name, value in _merged_by_peft(model).items()
        if (normalized := normalize_peft_param_name(name, model.prefix)) is not None
    }
    before = local_parameters(model)

    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    pushed = {param.name: param.value for param in sender.params}

    assert pushed, "premise: the push forwarded tensors"
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
    """PEFT's merge would fold (or refuse) these too; the out-of-place fold has no formula for them, so
    the trainers' construction gate refuses them rather than push their base unfolded."""
    model = get_peft_model(build(), LoraConfig(r=4, lora_alpha=8, **config))
    with pytest.raises(NotImplementedError, match="cannot be folded out of place"):
        validate_weight_sync_support(model, "vllm")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
