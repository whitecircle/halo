#!/usr/bin/env python
"""Unit tests for the on-policy SDPG trainer's own logic (no GPU / vLLM).

DistributedSDPGTrainer layers a privileged-teacher reverse-KL OPD term onto the online GRPO loop.
The GRPO loop itself is covered elsewhere; here we test the two pieces SDPG adds and that must be
correct for the OPD to align: (1) ``_build_teacher_prompts`` — left-padded ``[prompt + hint]`` with
the hint revealing the gold answer; (2) the positive-advantage gate applied to the per-token OPD.

A real tokenizer is needed for the hint encoding; the test skips when none is cached.

Run: pytest tests/cpu/trainers/test_sdpg_trainer.py
"""

import pytest
import torch
from accelerate import PartialState

from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

PartialState()  # the trainer warns through accelerate's logger, which refuses to log without it


def _bare_trainer(tokenizer):
    """A DistributedSDPGTrainer shell with only the attributes the unit-under-test reads."""
    t = object.__new__(DistributedSDPGTrainer)
    t.processing_class = tokenizer
    t.sdpg_hint_template = "\n[Hint] answer: {answer}\n"
    t.sdpg_answer_field = "answer"
    t._warned_missing_answer = set()
    return t


def test_build_teacher_prompts_appends_hint_and_left_pads():
    tok = load_cached_tokenizer(QWEN3_0_6B)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t = _bare_trainer(tok)

    # Two left-padded prompts of different real lengths.
    p_a = tok.encode("What is 2+2?", add_special_tokens=False)
    p_b = tok.encode("Capital of France is", add_special_tokens=False)
    width = max(len(p_a), len(p_b)) + 2
    pad = tok.pad_token_id

    def left_pad(ids):
        return [pad] * (width - len(ids)) + ids

    prompt_ids = torch.tensor([left_pad(p_a), left_pad(p_b)])
    prompt_mask = (prompt_ids != pad).long()
    # ensure the masked positions are exactly the real tokens (handles pad==eos collisions)
    prompt_mask = torch.tensor([[0] * (width - len(p_a)) + [1] * len(p_a), [0] * (width - len(p_b)) + [1] * len(p_b)])

    answers = ["4", "Paris"]
    tids, tmask = t._build_teacher_prompts(prompt_ids, prompt_mask, answers)

    assert tids.shape == tmask.shape
    assert tids.size(0) == 2
    for i, (real, ans) in enumerate(zip([p_a, p_b], answers, strict=False)):
        expected = real + tok.encode(f"\n[Hint] answer: {ans}\n", add_special_tokens=False)
        got = tids[i][tmask[i].bool()].tolist()
        assert got == expected, f"row {i}: teacher prompt must be real-prompt + hint(answer)"
    # Left padding: the first real teacher token sits at column (width' - len) with mask 1.
    assert tmask[:, -1].tolist() == [1, 1]  # last column always real (left-padded)


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_a_missing_answer_yields_no_hint_instead_of_a_blank_one(answer):
    """The template STATES the answer, so a blank renders "answer: " as fact.

    The privileged teacher would then be misled rather than privileged, and the OPD term distils the
    student toward it. Dropping the hint leaves the teacher on the plain prompt, which is honest.
    """
    tok = load_cached_tokenizer(QWEN3_0_6B)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t = _bare_trainer(tok)

    real = tok.encode("What is 2+2?", add_special_tokens=False)
    prompt_ids = torch.tensor([real])
    prompt_mask = torch.ones_like(prompt_ids)

    tids, tmask = t._build_teacher_prompts(prompt_ids, prompt_mask, [answer])

    assert tids[0][tmask[0].bool()].tolist() == real, "an answer-less row must carry no hint at all"
    assert t._warned_missing_answer, "the degraded row must be warned about once"


def test_positive_advantage_gate_zeroes_nonpositive_rows():
    """The real gate, not a transcription of it: a zero advantage is a tied/unscorable group, so a
    ``>= 0`` there would pull the student toward the teacher on rows the verifier did not prefer."""
    from src.trainers.distillation.losses import positive_advantage_gate

    completion_mask = torch.ones(3, 2)
    advantages = torch.tensor([0.5, -0.3, 0.0])  # only row 0 is strictly positive
    gate = positive_advantage_gate(completion_mask, advantages, enabled=True)
    assert gate.tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

    # Disabled, the gate is the completion mask alone — every row contributes.
    ungated = positive_advantage_gate(completion_mask, advantages, enabled=False)
    assert ungated.sum().item() == 6.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
