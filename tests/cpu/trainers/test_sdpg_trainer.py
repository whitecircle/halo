#!/usr/bin/env python
"""Unit tests for the on-policy SDPG trainer's own logic (no GPU / vLLM).

DistributedSDPGTrainer layers a privileged-teacher reverse-KL OPD term onto the online GRPO loop.
The GRPO loop itself is covered elsewhere; here we test the pieces SDPG adds and that must be correct
for the OPD to align: ``_build_teacher_prompts`` (left-padded ``[prompt + hint]``), the completion
logits both forwards are read at, and the OPD term itself — the gate on positive advantages and on
the tokens the GRPO loss trains, its token mean and its grad-accum scaling.

A real tokenizer is needed for the hint encoding; the test skips when none is cached.

Run: pytest tests/cpu/trainers/test_sdpg_trainer.py
"""

import types
from unittest import mock

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.trainers.distillation.losses import reverse_kl_loss
from src.trainers.distillation.sdpg import DistributedSDPGTrainer, positive_advantage_gate
from src.trainers.grpo.online import DistributedGRPOTrainer
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
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
    completion_mask = torch.ones(3, 2)
    advantages = torch.tensor([0.5, -0.3, 0.0])  # only row 0 is strictly positive
    gate = positive_advantage_gate(completion_mask, advantages, enabled=True)
    assert gate.tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

    # Disabled, the gate is the completion mask alone — every row contributes.
    ungated = positive_advantage_gate(completion_mask, advantages, enabled=False)
    assert ungated.sum().item() == 6.0


def _construct_over(columns, **kwargs):
    """Run the real ctor over a train dataset of ``columns``, the GRPO parent stubbed out."""

    def parent_init(self, *args, **kw):
        self.train_dataset = Dataset.from_dict({name: ["x"] for name in columns})

    with mock.patch.object(DistributedGRPOTrainer, "__init__", parent_init):
        return DistributedSDPGTrainer(**kwargs)


def test_the_answer_column_is_demanded_only_where_the_hint_names_it():
    """A hint naming ``{answer}`` over a dataset without the column would assert an empty answer; a
    template naming no answer reads no column, so it must not be refused for lacking one."""
    with pytest.raises(ValueError, match="needs the gold answer in column 'answer'"):
        _construct_over(["prompt"], sdpg_hint_template="the answer is {answer}")
    _construct_over(["prompt"], sdpg_hint_template="think it through once more")
    _construct_over(["prompt", "answer"], sdpg_hint_template="the answer is {answer}")


@pytest.mark.parametrize(("beta", "raises"), [(0.5, True), (0.0, False)])
def test_a_batch_without_teacher_prompts_raises_while_opd_is_on(monkeypatch, beta, raises):
    """With the OPD term on, a batch missing its privileged teacher prompts would skip the term and
    train plain GRPO; with it off (``sdpg_beta_base: 0``) no teacher prompt is built or needed."""
    monkeypatch.setattr(DistributedGRPOTrainer, "_compute_loss", lambda self, model, inputs: torch.tensor(1.0))
    t = object.__new__(DistributedSDPGTrainer)
    t.sdpg_beta_base = beta
    if raises:
        with pytest.raises(RuntimeError, match="teacher_prompt_ids"):
            t._compute_loss(None, {})
    else:
        assert t._compute_loss(None, {}).item() == 1.0


def _left_padded(rows, pad=0):
    width = max(map(len, rows))
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows])
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    return ids, mask


def _right_padded(rows, pad=0):
    width = max(map(len, rows))
    return (
        torch.tensor([r + [pad] * (width - len(r)) for r in rows]),
        torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
    )


class _NoTrimForward(nn.Module):
    """A forward that declares no ``logits_to_keep`` and swallows extra kwargs, as remote-code Bailing
    does: handed one, it still returns every position."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.received = []

    def forward(self, input_ids, attention_mask=None, use_cache=None, **kwargs):
        self.received.append(set(kwargs))
        return self.inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)


@pytest.mark.parametrize("trims", [True, False], ids=["logits_to_keep", "full-forward"])
def test_completion_logits_match_each_rows_own_unpadded_forward(trims):
    """The rows kept must be the ones that predict each completion token, for left-padded prompts of
    different lengths, whether the head is trimmed (``logits_to_keep``) or the forward takes no such
    argument and returns every position."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")).eval()
    prompts, completions = [[5, 9, 3], [7, 2, 8, 4, 6]], [[11, 12, 13, 14], [21, 22]]
    prompt_ids, prompt_mask = _left_padded(prompts)
    completion_ids, completion_mask = _right_padded(completions)

    trainer = object.__new__(DistributedSDPGTrainer)
    trainer.model_kwarg_keys = {"input_ids", "attention_mask"} | ({"logits_to_keep"} if trims else set())
    forward = model if trims else _NoTrimForward(model)
    with torch.no_grad():
        logits = trainer._completion_logits(forward, prompt_ids, prompt_mask, completion_ids, completion_mask)
        if not trims:
            assert forward.received == [set()], "logits_to_keep must not reach a forward that does not declare it"
        assert logits.shape == (2, completion_ids.size(1), model.config.vocab_size)
        for row, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            alone = model(input_ids=torch.tensor([prompt + completion])).logits[0]
            expected = alone[len(prompt) - 1 : len(prompt) + len(completion) - 1]
            torch.testing.assert_close(logits[row, : len(completion)], expected, atol=1e-4, rtol=1e-4)


class _ContextTable(nn.Module):
    """Logits for each position from a random table keyed by (token id + 3 x prefix length), so the
    student and the hinted teacher forwards score the same completion token differently."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(64, 7, generator=torch.Generator().manual_seed(0)))

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        shift = 3 * attention_mask.sum(-1, keepdim=True)
        return types.SimpleNamespace(logits=self.table[(input_ids + shift) % 64][:, -logits_to_keep:])


def _opd_trainer(model, *, positive_advantage_only: bool) -> tuple[DistributedSDPGTrainer, dict]:
    """A bare SDPG trainer over ``model`` (beta 0.5, grad-accum 2) recording what it stores."""
    trainer = object.__new__(DistributedSDPGTrainer)
    trainer.model = model
    trainer.sdpg_beta_base, trainer.sdpg_beta_warmup_steps, trainer.sdpg_beta_decay_steps = 0.5, 0, 0
    trainer.sdpg_temperature = 1.0
    trainer.sdpg_loss_fn = reverse_kl_loss
    trainer.opd_positive_advantage_only = positive_advantage_only
    trainer.state = types.SimpleNamespace(global_step=3, max_steps=10)
    trainer.current_gradient_accumulation_steps = 2
    trainer.model_kwarg_keys = {"logits_to_keep"}
    stored = {}

    def _store(metrics, train_eval, rows=1):
        stored.update(metrics, rows=rows)

    trainer.store_metrics = _store
    return trainer, stored


def _opd_inputs() -> dict[str, torch.Tensor]:
    prompt_ids, prompt_mask = _left_padded([[1, 2], [3, 4, 5], [6]])
    teacher_ids, teacher_mask = _left_padded([[1, 2, 9, 9], [3, 4, 5, 9], [6, 9, 9]])
    completion_ids, completion_mask = _right_padded([[10, 11, 12], [13, 14], [15, 16, 17]])
    tool_mask = torch.tensor([[1, 0, 1], [1, 1, 1], [1, 1, 1]])
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "teacher_prompt_ids": teacher_ids,
        "teacher_prompt_mask": teacher_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "tool_mask": tool_mask,
        "advantages": torch.tensor([0.7, 0.2, -0.4]),
    }


def _per_token_opd(trainer, model, inputs) -> torch.Tensor:
    with torch.no_grad():
        student = trainer._completion_logits(
            model, inputs["prompt_ids"], inputs["prompt_mask"], inputs["completion_ids"], inputs["completion_mask"]
        )
        teacher = trainer._completion_logits(
            model,
            inputs["teacher_prompt_ids"],
            inputs["teacher_prompt_mask"],
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
    return reverse_kl_loss(student, teacher, 1.0).sum(-1)


def test_the_opd_term_is_gated_to_trained_tokens_of_positive_advantage_rows(monkeypatch):
    """``beta * mean_{gated tokens} sum_v KL`` divided by grad-accum, the gate being the GRPO loss's
    own token mask (a tool-output token is attention-valid but untrained) on rows with advantage > 0."""
    monkeypatch.setattr(DistributedGRPOTrainer, "_compute_loss", lambda self, model, inputs: torch.tensor(1.0))
    model = _ContextTable().train()
    trainer, stored = _opd_trainer(model, positive_advantage_only=True)
    inputs = _opd_inputs()
    loss = trainer._compute_loss(model, inputs)

    per_token = _per_token_opd(trainer, model, inputs)
    gated = [(0, 0), (0, 2), (1, 0), (1, 1)]
    opd = torch.stack([per_token[row, col] for row, col in gated]).mean()
    torch.testing.assert_close(loss.detach(), 1.0 + 0.5 * opd / 2)
    torch.testing.assert_close(stored["opd_loss"], opd)
    assert stored["opd_beta"] == 0.5
    assert stored["rows"] == 1, "a train micro-batch weighs 1, whatever its row count"


def test_an_eval_split_s_padding_rows_leave_the_opd_mean(monkeypatch):
    """Row 2 pads the eval split's final round: the OPD mean (no grad-accum division in eval) runs over
    rows 0 and 1's trained tokens alone, stored with their two rows as the weight."""
    monkeypatch.setattr(DistributedGRPOTrainer, "_compute_loss", lambda self, model, inputs: torch.tensor(1.0))
    model = _ContextTable().eval()
    trainer, stored = _opd_trainer(model, positive_advantage_only=False)
    trainer.eval_split_rows = lambda num_rows: 2
    inputs = _opd_inputs()
    loss = trainer._compute_loss(model, inputs)

    per_token = _per_token_opd(trainer, model, inputs)
    opd = torch.stack([per_token[row, col] for row, col in ((0, 0), (0, 2), (1, 0), (1, 1))]).mean()
    torch.testing.assert_close(loss.detach(), 1.0 + 0.5 * opd)
    torch.testing.assert_close(stored["opd_loss"], opd)
    assert stored["rows"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
