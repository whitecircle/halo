"""CPU test for the embedding save's LoRA fold (``_folded_backbone_items``).

The embedding script injects LoRA in place (the model stays a SentenceTransformer, not a PeftModel),
so the save must fold the adapters into the base weights — otherwise the checkpoint carries
``base_layer``/``lora_*`` keys that reload as random base weights. The save folds each tensor out of
place with the weight sync's fold (``src.models.structure.lora_folded``); this checks what the save
gets from it: the plain module's names and nothing else, PEFT's own merge for every target kind and
variant the fold covers (a linear, an embedding with its transposed delta, a conv; DoRA and
``lora_bias``), and a live model left untouched.

Run: python tests/cpu/peft/test_embedding_lora_merge.py
"""

import copy

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

from src.trainers.embedding.trainer import _folded_backbone_items
from tests.common.peft_helpers import merged_lora_targets

VOCAB = 11
HIDDEN = 6
LORA_R = 2
LORA_ALPHA = 8


class _Encoder(nn.Module):
    """An input embedding, a projection and a conv, plus a persistent buffer. ``VOCAB != HIDDEN``, so an
    embedding delta folded untransposed cannot match its base."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.conv = nn.Conv2d(HIDDEN, HIDDEN, kernel_size=3)
        self.register_buffer("steps", torch.arange(3.0))


def _trained_adapters(**lora) -> nn.Module:
    """``_Encoder`` with LoRA injected in place and every adapter tensor random, as after training
    (PEFT zero-inits one factor of each pair, which would make any fold look right)."""
    torch.manual_seed(0)
    config = {"target_modules": ["embed", "proj", "conv"], **lora}
    model = inject_adapter_in_model(LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, **config), _Encoder())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.normal_()
    return model


@pytest.mark.parametrize(
    "lora",
    # PEFT takes lora_bias on a linear target only.
    [{}, {"use_dora": True}, {"lora_bias": True, "target_modules": ["proj"]}],
    ids=["plain", "dora", "lora-bias"],
)
def test_the_save_writes_peft_merge_under_the_plain_names(lora):
    model = _trained_adapters(**lora)
    live_before = {name: value.clone() for name, value in model.state_dict().items()}
    merged_targets = merged_lora_targets(copy.deepcopy(model))
    expected = {name.replace(".base_layer", ""): value for name, value in live_before.items() if "lora_" not in name}
    expected.update(merged_targets)

    folded = dict(_folded_backbone_items(model))

    assert set(folded) == set(_Encoder().state_dict()), f"not the plain module's keys: {sorted(folded)}"
    unequal = [name for name, value in expected.items() if not torch.equal(folded[name], value)]
    assert not unequal, f"not PEFT's merge: {unequal}"
    bases = {name: live_before.get(name.replace(".", ".base_layer.", 1)) for name in expected}
    assert any(base is not None and not torch.equal(expected[name], base) for name, base in bases.items()), (
        "premise: the adapters move their bases"
    )
    assert all(torch.equal(value, live_before[name]) for name, value in model.state_dict().items()), (
        "the fold wrote into the live model"
    )


def test_the_folded_embedding_looks_tokens_up_as_the_adapted_one_does():
    model = _trained_adapters()
    tokens = torch.tensor([[0, 3, 10, 7]])

    plain = nn.Embedding(VOCAB, HIDDEN)
    plain.load_state_dict({"weight": dict(_folded_backbone_items(model))["embed.weight"]})

    assert torch.allclose(plain(tokens), model.embed(tokens).detach(), atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
