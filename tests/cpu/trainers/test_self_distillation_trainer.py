#!/usr/bin/env python
"""
CPU contract tests for the SDPG self-distillation pieces (arXiv:2606.04036):
trainer parallelism flags, the privileged-hint collator's hint injection, and the
positive-advantage/confidence gating math (a zero-weight sample contributes nothing).

The loss equations themselves are covered by test_distillation_shared_losses.py.

Run: python tests/cpu/trainers/test_self_distillation_trainer.py
"""

import pytest
import torch


def test_trainer_mro_and_flags():
    from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
    from src.trainers.sft import DistributedSFTTrainer

    assert issubclass(DistributedSelfDistillationTrainer, DistributedSFTTrainer)
    # Privileged teacher uses a second, longer sequence => CP unsupported; EP/TP fine.
    assert DistributedSelfDistillationTrainer._supports_ep is True
    assert DistributedSelfDistillationTrainer._supports_cp is False
    assert DistributedSelfDistillationTrainer._supports_tp is True


def test_args_defaults():
    from src.args.self_distill_args import SelfDistillationArguments

    a = SelfDistillationArguments()
    assert a.sdpg_loss == "reverse_kl"
    assert a.sdpg_beta_base == 1.0
    assert a.opd_exclude_eos is True
    assert "{answer}" in a.sdpg_hint_template


def test_collator_inject_hint_string_and_list():
    import types

    from src.data.collators.vlm import SelfDistillVLMDataCollator

    # _inject_hint needs only hint_template; the stub tokenizer just satisfies the base __init__,
    # which resolves the eos set and the image-token ids once at construction.
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0, get_vocab=dict)
    c = SelfDistillVLMDataCollator(None, tokenizer, hint_template="\n[Hint] {answer}\n")

    hist = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    out = c._inject_hint(hist, answer="B", solution=None)
    assert out[0]["content"] == "Q1"
    assert out[2]["content"] == "Q2\n[Hint] B\n"
    assert hist[2]["content"] == "Q2"  # input row not mutated

    hist2 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}]
    out2 = c._inject_hint(hist2, answer="7", solution=None)
    assert out2[0]["content"][-1] == {"type": "text", "text": "\n[Hint] 7\n"}


def test_opd_gating_zero_weight_contributes_nothing():
    """SDPG gates OPD on positive advantage; a zero-weight sample must add 0 to the mean."""
    from src.trainers.distillation.losses import masked_token_mean

    torch.manual_seed(0)
    per_token_vocab = torch.rand(2, 3, 5)  # [B, S, V]
    mask = torch.ones(2, 3)
    gated = masked_token_mean(per_token_vocab, mask, sample_weights=torch.tensor([0.0, 1.0]))
    # Equivalent to averaging only sample 1's contribution over the batch of 2.
    only1 = masked_token_mean(per_token_vocab[1:], mask[1:], sample_weights=torch.tensor([1.0]))
    assert torch.allclose(gated, only1 / 2), (gated.item(), only1.item())


def _vision_reuse_probe(model_type):
    """Run ``_maybe_setup_vision_reuse`` against a fake VLM of ``model_type``.

    Returns ``(activated, wrapper_installed)``.
    """
    import logging
    import types
    from unittest.mock import patch

    import src.trainers.distillation.self_distillation as sd
    from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer

    def original_get_image_features(*args, **kwargs):
        return "features"

    inner = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type=model_type),
        model=types.SimpleNamespace(get_image_features=original_get_image_features),
    )
    me = types.SimpleNamespace(
        _vision_reuse_setup=False,
        _vision_reuse_active=False,
        _vision_record=False,
        _vision_replay=False,
        _vision_cache=None,
    )
    # accelerate's logger needs a PartialState; a plain one keeps this call unit-level.
    with patch.object(sd, "logger", logging.getLogger(__name__)):
        activated = DistributedSelfDistillationTrainer._maybe_setup_vision_reuse(me, inner)
    return activated, inner.model.get_image_features is not original_get_image_features


def test_vision_reuse_only_for_validated_families():
    """The get_image_features cache replays the student's features on the teacher pass — sound only
    where both passes are known to call it exactly once with identical inputs (LFM2-VL). Any other
    VLM exposing get_image_features must be left alone."""
    assert _vision_reuse_probe("lfm2_vl") == (True, True)
    for model_type in ("qwen3_vl", "gemma4", "mistral3", None):
        assert _vision_reuse_probe(model_type) == (False, False), model_type


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
