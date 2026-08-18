"""CPU test for the embedding LoRA merge-on-save helper (``_merge_injected_lora_state_dict``).

The embedding script injects LoRA in place (the model stays a SentenceTransformer, not a PeftModel),
so the trainer must fold the adapter into the base weights at save time — otherwise the checkpoint
carries ``base_layer``/``lora_*`` keys that reload as random base weights. This verifies the merge
math, the key rename/drop, and the DoRA rejection.

Run: python tests/cpu/peft/test_embedding_lora_merge.py
"""

import pytest
import torch

from src.trainers.embedding.trainer import _merge_injected_lora_state_dict


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


def test_merge_rejects_dora():
    """DoRA's magnitude reparam is non-linear → a plain B@A merge would be wrong; reject it."""
    sd = {
        "m.q.base_layer.weight": torch.randn(4, 4),
        "m.q.lora_A.default.weight": torch.randn(2, 4),
        "m.q.lora_B.default.weight": torch.randn(4, 2),
        "m.q.lora_magnitude_vector.default.weight": torch.randn(4),
    }
    with pytest.raises(NotImplementedError):
        _merge_injected_lora_state_dict(sd, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
