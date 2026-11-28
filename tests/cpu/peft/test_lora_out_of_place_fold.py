#!/usr/bin/env python
"""The weight sync folds each LoRA layer out of place into exactly what PEFT's merge would write.
The frozen base is never written; a layer the fold cannot reproduce, a target the dense push skips, or a LoRA on a tied
embedding pair is refused."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers import CLIPConfig, CLIPModel, Wav2Vec2Config, Wav2Vec2Model

from halo.distributed.lora_fold import lora_folded_data
from halo.models.peft_adapters import (
    adapted_tied_embeddings,
    lora_adapted_tied_embeddings,
    lora_fold_targets,
    lora_folded,
)
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

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, x):  # pragma: no cover - never called
        return x


class _SharedWeightNet(nn.Module):
    """Two projections holding one weight, a tie that is not the embedding pair."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_DIM, _DIM, bias=False)
        self.twin = nn.Linear(_DIM, _DIM, bias=False)
        self.twin.weight = self.proj.weight

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
    "head": (_Net, {"target_modules": ["lm_head"]}),
    "tied_pair_untouched": (lambda: _Net(tied=True), {"target_modules": ["proj"]}),
    # Both layers fold into the one tensor, which named_parameters() lists under the first owner's name alone.
    "shared_weight": (_SharedWeightNet, {"target_modules": ["proj", "twin"]}),
    "shared_weight_second_owner": (_SharedWeightNet, {"target_modules": ["twin"]}),
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


@pytest.mark.parametrize(
    ("case", "dtypes"),
    [
        (case, dtypes)
        for case in sorted(_CASES)
        for dtypes in sorted(_DTYPES)
        if dtypes != "fp32" or case in ("conv", "embedding", "linear")
    ],
)
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
    """Each forwarded tensor is what PEFT's merge holds under that name; a tied pair untouched by the adapters goes
    once, under the embedding's name."""
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


@pytest.mark.parametrize(
    ("targets", "roles"),
    [
        (["lm_head", "proj"], {"output head"}),
        (["embed_tokens"], {"input embedding"}),
        (["embed_tokens", "lm_head"], {"input embedding", "output head"}),
    ],
    ids=["head", "embedding", "both"],
)
def test_a_lora_on_a_tied_embedding_pair_is_refused(targets, roles):
    """Training adds the delta to one half; folded into the one shared tensor, the push would serve it on both."""
    model = get_peft_model(_Net(tied=True), LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    randomize_adapters(model, std=0.1)
    assert set(lora_adapted_tied_embeddings(model).values()) == roles
    with pytest.raises(ValueError, match="tied embedding pair") as refusal:
        lora_fold_targets(model)
    # It fires under the weight sync and merge_expert_lora_on_save, where the in-memory merge is no remedy.
    assert "lora_modules_to_save" in str(refusal.value) and "merge_peft_adapters" not in str(refusal.value)
    with pytest.raises(ValueError, match="tied embedding pair"):
        reject_unsyncable_model(model, "vllm")
    sender = RecordingSender()
    with pytest.raises(ValueError, match="tied embedding pair"):
        gather_and_send_weights(model, sender)
    assert not sender.params, "a refused push must send nothing"


def test_a_full_copy_of_a_tied_half_is_no_lora_to_fold():
    """A ``modules_to_save`` copy splits the pair too, but holds no delta for the fold to misplace."""
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], modules_to_save=["lm_head"])
    model = get_peft_model(_Net(tied=True), config)

    assert list(adapted_tied_embeddings(model).values()) == ["output head"]
    assert lora_adapted_tied_embeddings(model) == {}
    lora_fold_targets(model)


def test_an_untied_head_lora_reaches_the_head_alone():
    """The delta rides on ``lm_head`` and the input embedding goes out as the frozen base."""
    model = _lora_model("head", _DTYPES["fp32"])
    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    pushed = {param.name: param.value for param in sender.params}
    base = model.get_base_model()
    assert torch.equal(pushed["embed_tokens.weight"], base.embed_tokens.weight)
    delta = base.lm_head.get_delta_weight("default")
    assert torch.equal(pushed["lm_head.weight"], base.lm_head.base_layer.weight + delta)


def test_a_weight_two_modules_share_folds_both_owners_deltas():
    model = _lora_model("shared_weight", _DTYPES["fp32"])
    base = model.get_base_model()
    shared = base.proj.base_layer.weight
    assert shared is base.twin.base_layer.weight, "premise: the projections share one weight"

    targets = lora_fold_targets(model)
    folded = lora_folded(shared, targets[id(shared)])

    deltas = [base.proj.get_delta_weight("default"), base.twin.get_delta_weight("default")]
    assert all(delta.abs().max() > 0 for delta in deltas), "premise: both adapters move the weight"
    assert torch.equal(folded, shared + deltas[0] + deltas[1]), "the fold missed one owner's delta"
    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender)
    pushed = {param.name: param.value for param in sender.params}
    assert torch.equal(pushed["proj.weight"], folded)


@pytest.mark.parametrize("state", ["inactive", "merged"])
def test_a_head_lora_that_leaves_a_tied_pair_one_weight_is_not_refused(state):
    """An inactive adapter adds no delta, and a merged one already sits in the shared tensor both halves read."""
    model = get_peft_model(_Net(tied=True), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"]))
    model.add_adapter("head", LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"]))
    randomize_adapters(model, std=0.1)
    if state == "merged":
        model.base_model.set_adapter(["default", "head"])
        model.merge_adapter()
    head = model.get_base_model().lm_head
    assert isinstance(head, LoraLayer) and "head" in head.lora_A, "premise: the head carries the adapter"

    assert adapted_tied_embeddings(model) == {}
    assert lora_adapted_tied_embeddings(model) == {}
    lora_fold_targets(model)
    reject_unsyncable_model(model, "vllm")


def _tiny_clip() -> nn.Module:
    tower = {"hidden_size": _DIM, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 2 * _DIM}
    return CLIPModel(
        CLIPConfig(
            text_config={"vocab_size": _VOCAB, "max_position_embeddings": 8, **tower},
            vision_config={"image_size": 8, "patch_size": 4, **tower},
            projection_dim=_DIM,
        )
    )


def _tiny_wav2vec2() -> nn.Module:
    return Wav2Vec2Model(
        Wav2Vec2Config(
            hidden_size=_DIM,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=2 * _DIM,
            conv_dim=(8,),
            conv_stride=(2,),
            conv_kernel=(3,),
            num_conv_pos_embeddings=4,
            num_conv_pos_embedding_groups=2,
        )
    )


@pytest.mark.parametrize("build", [_tiny_clip, _tiny_wav2vec2], ids=["clip", "wav2vec2"])
@pytest.mark.parametrize("adapted", [False, True], ids=["bare", "lora"])
def test_a_model_with_no_resolvable_input_embedding_has_no_tied_pair(build, adapted):
    """transformers raises from get_input_embeddings here and answers None for the head, so there is no pair to read."""
    model = build()
    with pytest.raises(NotImplementedError):
        model.get_input_embeddings()
    if adapted:
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
        randomize_adapters(model, std=0.1)

    assert adapted_tied_embeddings(model) == {}
    assert bool(lora_fold_targets(model)) is adapted


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
