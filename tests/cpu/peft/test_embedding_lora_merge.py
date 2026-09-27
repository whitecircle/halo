"""CPU test for the embedding LoRA merge-on-save helper (``_merge_injected_lora_state_dict``).

The embedding script injects LoRA in place (the model stays a SentenceTransformer, not a PeftModel),
so the trainer must fold the adapter into the base weights at save time — otherwise the checkpoint
carries ``base_layer``/``lora_*`` keys that reload as random base weights. This verifies the merge
math against PEFT's own ``merge()`` for a linear and an embedding target (whose delta is the
transpose of ``B @ A``), the key rename/drop, and the raise on every adapter tensor the fold cannot
express (DoRA, ``lora_bias``, a conv target), which would otherwise be written as a stray key.

Run: python tests/cpu/peft/test_embedding_lora_merge.py
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

from src.trainers.embedding.trainer import _merge_injected_lora_state_dict

VOCAB = 11
HIDDEN = 6
LORA_R = 2
LORA_ALPHA = 8


class _Encoder(nn.Module):
    """An input embedding and a projection, the two target kinds the fold covers, plus a conv it does
    not. ``VOCAB != HIDDEN``, so an embedding delta folded untransposed cannot match its base."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.conv = nn.Conv2d(HIDDEN, HIDDEN, kernel_size=3)


def _trained_adapters(**lora) -> nn.Module:
    """``_Encoder`` with LoRA injected in place and every adapter tensor random, as after training
    (PEFT zero-inits one factor of each pair, which would make any fold look right)."""
    torch.manual_seed(0)
    model = inject_adapter_in_model(
        LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, **{"target_modules": ["embed", "proj"], **lora}), _Encoder()
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.normal_()
    return model


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_merge_matches_peft_own_merge_for_linear_and_embedding_targets():
    """The folded ``embed.weight`` / ``proj.weight`` are exactly what PEFT's ``merge()`` writes into the
    base layers, and a plain ``nn.Embedding`` holding the fold looks tokens up as the adapted one does."""
    model = _trained_adapters()
    tokens = torch.tensor([[0, 3, 10, 7]])
    adapted_lookup = model.embed(tokens).detach()
    folded = _merge_injected_lora_state_dict(_state(model), LORA_ALPHA / LORA_R)

    assert set(folded) == set(_Encoder().state_dict()), f"not the plain module's keys: {sorted(folded)}"
    base_embedding = model.embed.base_layer.weight.detach().clone()
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
    for name in ("embed", "proj"):
        peft_merge = getattr(model, name).base_layer.weight.detach()
        assert torch.allclose(folded[f"{name}.weight"], peft_merge, atol=1e-6), f"{name} is not PEFT's merge"
    assert not torch.allclose(folded["embed.weight"], base_embedding), "premise: the embedding delta is nonzero"
    plain = nn.Embedding(VOCAB, HIDDEN)
    plain.load_state_dict({"weight": folded["embed.weight"]})
    assert torch.allclose(plain(tokens), adapted_lookup, atol=1e-5), "the folded table looks tokens up differently"


def test_merge_folds_delta_and_renames_keys():
    """``<m>.base_layer.weight`` + LoRA → ``<m>.weight = base + scaling*(B@A)``; adapter keys dropped."""
    torch.manual_seed(0)
    base = torch.randn(8, 6)
    a = torch.randn(2, 6)  # lora_A [r, in]
    b = torch.randn(8, 2)  # lora_B [out, r]
    scaling = 2.0
    sd = {
        "m.q.base_layer.weight": base.clone(),
        "m.q.lora_A.default.weight": a.clone(),
        "m.q.lora_B.default.weight": b.clone(),
        "m.norm.weight": torch.ones(8),  # non-adapter key passes through
    }
    out = _merge_injected_lora_state_dict(sd, scaling)

    assert "m.q.weight" in out, "merged plain key missing"
    assert "m.q.base_layer.weight" not in out, "base_layer key must be dropped"
    assert not any(".lora_" in k for k in out), "adapter keys must be dropped"
    assert torch.allclose(out["m.q.weight"], base + scaling * (b @ a), atol=1e-5)
    assert torch.allclose(out["m.norm.weight"], torch.ones(8)), "non-adapter key must pass through"


def test_merge_base_only_module_renames_without_delta():
    """A base_layer with no adapter (and a base_layer bias) is renamed but unchanged."""
    base = torch.randn(4, 4)
    bias = torch.arange(4, dtype=torch.float32)
    out = _merge_injected_lora_state_dict(
        {"m.q.base_layer.weight": base.clone(), "m.q.base_layer.bias": bias.clone()}, 1.0
    )
    assert torch.allclose(out["m.q.weight"], base), "no LoRA → base weight unchanged"
    assert "m.q.base_layer.weight" not in out
    assert torch.allclose(out["m.q.bias"], bias), "base_layer.bias → bias"
    assert "m.q.base_layer.bias" not in out


@pytest.mark.parametrize(
    "lora",
    # PEFT takes lora_bias on a linear target only.
    [
        {"use_dora": True},
        {"lora_bias": True, "target_modules": ["proj"]},
        {"target_modules": ["embed", "proj", "conv"]},
    ],
    ids=["dora", "lora-bias", "conv-target"],
)
def test_merge_raises_on_an_adapter_tensor_it_cannot_fold(lora):
    """A DoRA magnitude, a ``lora_B`` bias and a conv adapter have no ``B @ A`` fold; written as stray
    keys they would be dropped at load, serving those modules' base weights."""
    with pytest.raises(NotImplementedError, match="have no such fold"):
        _merge_injected_lora_state_dict(_state(_trained_adapters(**lora)), LORA_ALPHA / LORA_R)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
