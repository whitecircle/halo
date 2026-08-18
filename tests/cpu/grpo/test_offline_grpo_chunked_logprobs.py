#!/usr/bin/env python
"""Offline GRPO's chunked log-prob path and tokenization-level degenerate-group drop.

``use_chunked_grpo_logprobs`` must reproduce the full-logits path exactly — same ``(clamped, unclamped)``
log-probs, same ``min_log_prob`` negative-advantage clamping, same gradients — while never
materializing ``[B, T, vocab]`` logits, and must REFUSE an adapter on the output embedding rather
than silently drop its delta. ``drop_degenerate_groups`` must drop exactly the groups whose
advantages every method maps to zero (EXACT reward ties or < 2 completions) and nothing else —
a near-tie trains with full-scale advantages under the rank methods, so dropping it would change the
objective. These tests FAIL when either path drifts from the full-logits reference.

    python tests/cpu/grpo/test_offline_grpo_chunked_logprobs.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.trainers.grpo.offline import OfflineGRPOTrainer, tokenize_offline_grpo_rows
from tests.common.models import TINY_QWEN3_CONFIG

SEED = 17
PROMPT_LENS = (5, 3, 7)
COMPLETION_LENS = (6, 4, 2)


def _tiny_model():
    torch.manual_seed(SEED)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).float().train()


def _stub_trainer(chunked: bool) -> OfflineGRPOTrainer:
    """The logprob path needs only the routing flag, the temperature, and an unwrap — no full ctor."""
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    trainer._use_chunked_grpo_logprobs = chunked
    trainer.temperature = 1.0
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m, **kw: m)
    return trainer


def _collated_batch(vocab_size: int):
    """Collator-shaped inputs: prompts left-padded, completions right-padded, one span per row."""
    torch.manual_seed(SEED + 1)
    max_p, max_c = max(PROMPT_LENS), max(COMPLETION_LENS)
    input_ids = torch.zeros(len(PROMPT_LENS), max_p + max_c, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, (p_len, c_len) in enumerate(zip(PROMPT_LENS, COMPLETION_LENS, strict=True)):
        tokens = torch.randint(1, vocab_size, (p_len + c_len,))
        input_ids[i, max_p - p_len : max_p + c_len] = tokens
        attention_mask[i, max_p - p_len : max_p + c_len] = 1
    return input_ids, attention_mask, max_c


def _logps(model, chunked, advantages=None, min_log_prob=None):
    input_ids, attention_mask, logits_to_keep = _collated_batch(model.config.vocab_size)
    return _stub_trainer(chunked)._get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep, advantages=advantages, min_log_prob=min_log_prob
    )


def test_chunked_matches_full_path():
    model = _tiny_model()
    full_clamped, full_unclamped = _logps(model, chunked=False)
    chunk_clamped, chunk_unclamped = _logps(model, chunked=True)
    assert full_clamped is full_unclamped  # no clamp configured
    torch.testing.assert_close(chunk_clamped, full_clamped, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(chunk_unclamped, full_unclamped, atol=1e-4, rtol=1e-4)


def test_chunked_min_log_prob_clamp_matches_full_path():
    # The floor must bind only on negative-advantage rows, identically on both paths.
    model = _tiny_model()
    advantages = torch.tensor([1.0, -1.0, -0.5])
    floor = -2.0
    full_clamped, full_unclamped = _logps(model, chunked=False, advantages=advantages, min_log_prob=floor)
    chunk_clamped, chunk_unclamped = _logps(model, chunked=True, advantages=advantages, min_log_prob=floor)

    torch.testing.assert_close(chunk_clamped, full_clamped, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(chunk_unclamped, full_unclamped, atol=1e-4, rtol=1e-4)
    assert (chunk_clamped[1:] >= floor).all()  # negative-advantage rows are floored
    torch.testing.assert_close(chunk_clamped[0], chunk_unclamped[0])  # positive row untouched
    assert (chunk_unclamped < floor).any(), "fixture too easy: nothing fell below the floor"


def test_chunked_backward_matches_full_path():
    # Same loss surface: gradients through the head AND the backbone must match the full path.
    grads = {}
    for chunked in (False, True):
        model = _tiny_model()
        clamped, _ = _logps(model, chunked=chunked)
        clamped.sum().backward()
        grads[chunked] = {
            "lm_head": model.lm_head.weight.grad.clone(),
            "embed": model.model.embed_tokens.weight.grad.clone(),
            "q_proj": model.model.layers[0].self_attn.q_proj.weight.grad.clone(),
        }
    for name in grads[False]:
        torch.testing.assert_close(grads[True][name], grads[False][name], atol=1e-4, rtol=1e-4, msg=name)


def _lora_model(target_modules):
    model = get_peft_model(
        _tiny_model(), LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=target_modules)
    )
    # lora_B initializes to zero, making adapter-on ≡ adapter-off — a comparison that cannot fail
    # when the chunked path drops the adapters. Give the deltas real magnitude.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.05)
    return model


def test_chunked_matches_full_path_under_peft():
    # The chunked backbone forward must unwrap PeftModel the way TRL does (adapters stay ACTIVE).
    model = _lora_model(["q_proj", "v_proj"])
    full_clamped, _ = _logps(model, chunked=False)
    chunk_clamped, _ = _logps(model, chunked=True)
    # Fixture sensitivity: the adapters must actually move the log-probs, else this proves nothing.
    base_clamped, _ = _logps(_tiny_model(), chunked=False)
    assert (full_clamped - base_clamped).abs().max() > 1e-3, "LoRA deltas inert; test cannot catch a dropped adapter"
    torch.testing.assert_close(chunk_clamped, full_clamped, atol=1e-4, rtol=1e-4)


def test_chunked_rejects_lm_head_adapter():
    # get_output_embeddings() on an lm_head-targeted LoRA resolves to the BASE linear, so the chunked
    # matmul would silently drop the adapter delta — the path must refuse, not diverge.
    model = _lora_model(["q_proj", "lm_head"])
    _logps(model, chunked=False)  # the full-logits path applies the delta and stays supported
    with pytest.raises(ValueError, match="output embedding"):
        _logps(model, chunked=True)


# --- drop_degenerate_groups at tokenization ---


class _OneTokenTokenizer:
    bos_token_id = None
    eos_token_id = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}


_TOKENIZE_KWARGS = {
    "processing_class": _OneTokenTokenizer(),
    "max_prompt_length": None,
    "max_completion_length": None,
    "advantage_method": "z_norm",
    "best_completion_emphasis": 0.0,
    "is_encoder_decoder": False,
}


def _groups_batch():
    return {
        "prompt": ["graded", "exactly-tied", "singleton", "near-tied"],
        "completions": [["a", "b"], ["a", "b", "c"], ["a"], ["a", "b"]],
        "rewards": [[1.0, 0.0], [0.5, 0.5, 0.5], [1.0], [0.3, 0.3 + 5e-7]],
    }


def test_drop_degenerate_groups_drops_exact_ties_and_singletons_only():
    out = tokenize_offline_grpo_rows(_groups_batch(), [0, 1, 2, 3], drop_degenerate_groups=True, **_TOKENIZE_KWARGS)
    # The exactly-tied and singleton groups vanish; the graded pair AND the near-tie survive — the
    # rank advantage methods give a near-tie full-scale advantages (quantile_norm: ±0.674), so
    # dropping it would change the objective, not just save compute.
    assert out["group_id"] == [0, 0, 3, 3]
    assert out["group_size"] == [2, 2, 2, 2]
    assert len(out["advantage"]) == 4


def test_drop_degenerate_groups_off_keeps_everything():
    out = tokenize_offline_grpo_rows(_groups_batch(), [0, 1, 2, 3], drop_degenerate_groups=False, **_TOKENIZE_KWARGS)
    assert out["group_id"] == [0, 0, 1, 1, 1, 2, 3, 3]


def test_check_degenerate_drop_refuses_empty_dataset():
    from accelerate import PartialState

    PartialState()  # the accelerate logger the check reports through needs initialized state
    args = SimpleNamespace(drop_degenerate_groups=True)
    kept = Dataset.from_dict({"group_id": [0, 0]})
    OfflineGRPOTrainer._check_degenerate_drop(args, kept, num_groups_in=3, what="training")  # survivors: fine
    with pytest.raises(ValueError, match="removed every training group"):
        OfflineGRPOTrainer._check_degenerate_drop(
            args, Dataset.from_dict({"group_id": []}), num_groups_in=3, what="training"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
