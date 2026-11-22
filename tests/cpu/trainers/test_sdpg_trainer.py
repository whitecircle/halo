#!/usr/bin/env python
"""The on-policy SDPG trainer's own logic (no GPU or vLLM): teacher prompts, the OPD term on the full-logits and the
chunked scorer, and the full-logits memory check. A real tokenizer is needed for the hint encoding; the test skips when
none is cached.
"""

import functools
import gc
import types
import weakref
from collections import defaultdict
from typing import get_args
from unittest import mock

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

import halo.trainers.distillation.sdpg as sdpg_module
import halo.trainers.grpo.logprob_scoring as logprob_scoring
from halo.configs.distillation import SDPGArguments, SelfDistillationLoss
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.kernels import logprobs
from halo.trainers.distillation.divergences import pooled_token_mean, require_divergence, reverse_kl_loss
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer, positive_advantage_gate
from halo.trainers.full_logits_memory import LogitsWidth
from halo.trainers.grpo.logprob_scoring import CompletionTerm
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.gloo import run_gloo_ranks
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.tokenizers import load_cached_tokenizer


def _bare_trainer(tokenizer):
    t = object.__new__(DistributedSDPGTrainer)
    t.processing_class = tokenizer
    t.sdpg = SDPGArguments(sdpg_hint_template="\n[Hint] answer: {answer}\n")
    t.sdpg_answer_field = "answer"
    t._warned_missing_answer = set()
    return t


def test_build_teacher_prompts_appends_hint_and_left_pads():
    tok = load_cached_tokenizer(QWEN3_0_6B)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t = _bare_trainer(tok)

    p_a = tok.encode("What is 2+2?", add_special_tokens=False)
    p_b = tok.encode("Capital of France is", add_special_tokens=False)
    width = max(len(p_a), len(p_b)) + 2
    pad = tok.pad_token_id

    def left_pad(ids):
        return [pad] * (width - len(ids)) + ids

    prompt_ids = torch.tensor([left_pad(p_a), left_pad(p_b)])
    prompt_mask = (prompt_ids != pad).long()
    # Exactly the real tokens, even where pad == eos.
    prompt_mask = torch.tensor([[0] * (width - len(p_a)) + [1] * len(p_a), [0] * (width - len(p_b)) + [1] * len(p_b)])

    answers = ["4", "Paris"]
    tids, tmask = t._build_teacher_prompts(prompt_ids, prompt_mask, answers)

    assert tids.shape == tmask.shape
    assert tids.size(0) == 2
    for i, (real, ans) in enumerate(zip([p_a, p_b], answers, strict=False)):
        expected = real + tok.encode(f"\n[Hint] answer: {ans}\n", add_special_tokens=False)
        got = tids[i][tmask[i].bool()].tolist()
        assert got == expected, f"row {i}: teacher prompt must be real-prompt + hint(answer)"
    assert tmask[:, -1].tolist() == [1, 1]  # last column always real (left-padded)


@pytest.mark.parametrize("answer", [None, "", "   "])
def test_a_missing_answer_yields_no_hint_instead_of_a_blank_one(answer):
    """The template states the answer, so a blank renders "answer: " as fact and the OPD term distils the student
    toward a misled teacher."""
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
    """A zero advantage is a tied or unscorable group, so a ``>= 0`` gate would pull the student toward the teacher on
    rows the verifier did not prefer."""
    completion_mask = torch.ones(3, 2)
    advantages = torch.tensor([0.5, -0.3, 0.0])
    gate = positive_advantage_gate(completion_mask, advantages, enabled=True)
    assert gate.tolist() == [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

    ungated = positive_advantage_gate(completion_mask, advantages, enabled=False)
    assert ungated.sum().item() == 6.0


def _construct_over(columns, **kwargs):
    def parent_init(self, *args, **kw):
        self.train_dataset = Dataset.from_dict({name: ["x"] for name in columns})

    with mock.patch.object(DistributedOnlineGRPOTrainer, "__init__", parent_init):
        return DistributedSDPGTrainer(sdpg_answer_field="answer", opd_positive_advantage_only=True, **kwargs)


def test_the_answer_column_is_demanded_only_where_the_hint_names_it():
    """A hint naming ``{answer}`` over a dataset without the column would assert an empty answer; a template naming no
    answer reads no column."""
    with pytest.raises(ValueError, match="needs the gold answer in column 'answer'"):
        _construct_over(["prompt"], sdpg_hint_template="the answer is {answer}")
    _construct_over(["prompt"], sdpg_hint_template="think it through once more")
    _construct_over(["prompt", "answer"], sdpg_hint_template="the answer is {answer}")


@pytest.mark.parametrize(("beta", "raises"), [(0.5, True), (0.0, False)])
def test_a_batch_without_teacher_prompts_raises_while_opd_is_on(monkeypatch, beta, raises):
    """With the OPD term on, a batch missing its teacher prompts would skip the term and train plain GRPO; with
    ``sdpg_beta_base: 0`` no teacher prompt is built."""
    monkeypatch.setattr(DistributedOnlineGRPOTrainer, "_compute_loss", lambda self, model, inputs: torch.tensor(1.0))
    t = object.__new__(DistributedSDPGTrainer)
    t.sdpg = SDPGArguments(sdpg_beta_base=beta)
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
    """Declares no ``logits_to_keep`` and swallows extra kwargs, as remote-code Bailing does; returns all positions."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.received = []

    def forward(self, input_ids, attention_mask=None, use_cache=None, **kwargs):
        self.received.append(set(kwargs))
        return self.inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)


@pytest.mark.parametrize("trims", [True, False], ids=["logits_to_keep", "full-forward"])
def test_completion_logits_match_each_rows_own_unpadded_forward(trims):
    """The rows kept must predict each completion token, whether the head is trimmed or every position comes back."""
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
    """Logits keyed by (token id + 3 x prefix length), so student and hinted teacher score a token differently."""

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn(64, 7, generator=torch.Generator().manual_seed(0)))

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        shift = 3 * attention_mask.sum(-1, keepdim=True)
        return types.SimpleNamespace(logits=self.table[(input_ids + shift) % 64][:, -logits_to_keep:])


def _scored_grpo_loss(self, model, inputs):
    """TRL's GRPO loss as far as SDPG sees it: one student forward through the full-logits scorer's capture and a
    constant loss."""
    with self._capturing_forward_logits(model, inputs["completion_ids"].size(1)):
        self._completion_logits(
            model, inputs["prompt_ids"], inputs["prompt_mask"], inputs["completion_ids"], inputs["completion_mask"]
        )
    return torch.tensor(1.0)


def _opd_trainer(model, *, positive_advantage_only: bool) -> tuple[DistributedSDPGTrainer, dict]:
    """A bare SDPG trainer over ``model`` (beta 0.5, grad-accum 2) recording what it stores."""
    trainer = object.__new__(DistributedSDPGTrainer)
    trainer.model = model
    trainer.sdpg = SDPGArguments(sdpg_beta_base=0.5, sdpg_temperature=1.0)
    trainer.sdpg_loss_fn = reverse_kl_loss
    trainer.opd_positive_advantage_only = positive_advantage_only
    trainer.state = types.SimpleNamespace(global_step=3, max_steps=10)
    trainer.current_gradient_accumulation_steps = 2
    trainer.model_kwarg_keys = {"logits_to_keep"}
    trainer._use_chunked_grpo_logprobs = False
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
    """``beta * mean_{gated tokens} sum_v KL`` over grad-accum. The gate is the GRPO loss's token mask (a tool-output
    token is attention-valid but untrained) on rows with advantage > 0."""
    monkeypatch.setattr(DistributedOnlineGRPOTrainer, "_compute_loss", _scored_grpo_loss)
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
    """Row 2 pads the eval split's final round, and eval has no grad-accum division."""
    monkeypatch.setattr(DistributedOnlineGRPOTrainer, "_compute_loss", _scored_grpo_loss)
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


def test_a_loss_forward_that_hands_back_no_logits_raises(monkeypatch):
    """A scorer path that stops capturing would leave OPD without a student; it raises, never reforwards."""
    monkeypatch.setattr(DistributedOnlineGRPOTrainer, "_compute_loss", lambda self, model, inputs: torch.tensor(1.0))
    trainer, _ = _opd_trainer(_ContextTable().train(), positive_advantage_only=True)
    with pytest.raises(RuntimeError, match="reads the student's completion rows off the GRPO loss forward"):
        trainer._compute_loss(trainer.model, _opd_inputs())


_GRAD_ACCUM = 2


def _real_loss_trainer(model, *, chunked: bool, batch_size: int) -> DistributedSDPGTrainer:
    """A bare SDPG trainer whose ``_compute_loss`` runs TRL's own GRPO loss through the toolkit's scorers."""
    trainer, _ = _opd_trainer(model, positive_advantage_only=True)
    trainer.args = types.SimpleNamespace(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to=[],
        delta=None,
        use_bias_correction_kl=False,
    )
    trainer.accelerator = types.SimpleNamespace(
        unwrap_model=lambda module: module,
        is_main_process=False,
        gather=lambda tensor: tensor,
        reduce=lambda tensor, reduction: tensor,
    )
    trainer._use_chunked_grpo_logprobs = chunked
    trainer.model_kwarg_keys = {"input_ids", "attention_mask", "logits_to_keep"}
    trainer.temperature = 0.9
    trainer._entropy_bonus_enabled = trainer.aux_loss_enabled = trainer.use_vllm = False
    trainer.top_entropy_quantile, trainer.off_policy_mask_threshold = 1.0, None
    trainer.importance_sampling_level, trainer.loss_type, trainer.beta = "token", "grpo", 0.0
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.current_gradient_accumulation_steps = _GRAD_ACCUM
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return trainer


def _real_inputs() -> dict[str, torch.Tensor]:
    inputs = _opd_inputs()
    inputs.pop("tool_mask")
    inputs["completion_ids"] = inputs["completion_ids"] + 100
    return inputs


@pytest.mark.parametrize(
    ("chunked", "batch_size"),
    [(False, 3), (True, 3), (True, 1)],
    ids=["full-logits", "chunked-padded", "chunked-dense"],
)
def test_the_student_runs_once_and_the_loss_matches_two_forwards(chunked, batch_size):
    """The backbone runs with grad once per scorer call (one padded batch, or one per dense row); OPD adds none."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")).train()
    inputs = _real_inputs()
    forwards = []
    model.model.register_forward_hook(lambda module, *_: forwards.append((torch.is_grad_enabled(), module.training)))

    trainer = _real_loss_trainer(model, chunked=chunked, batch_size=batch_size)
    loss = trainer._compute_loss(model, inputs)
    rows = inputs["completion_ids"].size(0)
    assert forwards.count((True, True)) == (rows if batch_size == 1 else 1)
    assert forwards.count((False, False)) == 1, "the privileged teacher forward, with no grad and no dropout"
    assert len(forwards) == forwards.count((True, True)) + 1
    loss.backward()
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad()

    reference = _real_loss_trainer(model, chunked=chunked, batch_size=batch_size)
    grpo = DistributedOnlineGRPOTrainer._compute_loss(reference, model, inputs)
    student = reference._completion_logits(
        model, inputs["prompt_ids"], inputs["prompt_mask"], inputs["completion_ids"], inputs["completion_mask"]
    )
    with torch.no_grad():
        teacher = reference._completion_logits(
            model.eval(),
            inputs["teacher_prompt_ids"],
            inputs["teacher_prompt_mask"],
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
    model.train()
    gate = positive_advantage_gate(inputs["completion_mask"].float(), inputs["advantages"], True)
    opd = pooled_token_mean(reverse_kl_loss(student, teacher, 1.0).sum(-1), gate)
    expected = grpo + 0.5 * opd / _GRAD_ACCUM
    torch.testing.assert_close(loss.detach(), expected.detach(), atol=1e-5, rtol=1e-4)
    expected.backward()
    for name, param in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(grads[name], param.grad, atol=1e-5, rtol=1e-3, msg=name)


_LOSSES = get_args(SelfDistillationLoss)
# fp32 rounding: the two paths group the head's matmul rows differently, and the dense rows run trimmed.
_RTOL = 1e-4


def _tiny_qwen3(dtype=torch.float32) -> nn.Module:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")).to(dtype).train()


def _sdpg_step(model, inputs, *, chunked: bool, batch_size: int = 3, loss: str = "reverse_kl", trainer=None):
    """``(loss, gradients, stored metrics)`` of one SDPG loss forward and backward through TRL's own GRPO loss."""
    trainer = trainer or _real_loss_trainer(model, chunked=chunked, batch_size=batch_size)
    trainer.sdpg = SDPGArguments(
        sdpg_loss=loss,
        sdpg_jsd_beta=0.3 if loss == "jensen_shannon" else 0.5,
        sdpg_beta_base=0.5,
        sdpg_temperature=1.3,
    )
    trainer.sdpg_loss_fn = require_divergence(loss, jsd_beta=trainer.sdpg.sdpg_jsd_beta)
    stored = {}
    trainer.store_metrics = lambda metrics, train_eval, rows=1: stored.update(metrics)
    model.zero_grad(set_to_none=True)
    value = trainer._compute_loss(model, dict(inputs))
    if torch.is_grad_enabled():
        value.backward()
    grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    return value.detach(), grads, stored


def _assert_paths_agree(model, inputs, *, batch_size: int = 3, loss: str = "reverse_kl", rtol: float = _RTOL):
    full_loss, full_grads, full_metrics = _sdpg_step(model, inputs, chunked=False, batch_size=batch_size, loss=loss)
    chunked_loss, chunked_grads, chunked_metrics = _sdpg_step(
        model, inputs, chunked=True, batch_size=batch_size, loss=loss
    )
    torch.testing.assert_close(chunked_loss, full_loss, rtol=rtol, atol=rtol * full_loss.abs())
    torch.testing.assert_close(chunked_metrics["opd_loss"], full_metrics["opd_loss"], rtol=rtol, atol=0.0)
    assert full_metrics["opd_loss"] > 0
    assert chunked_grads.keys() == full_grads.keys() and full_grads
    for name, grad in full_grads.items():
        floor = rtol * grad.abs().max()
        torch.testing.assert_close(chunked_grads[name], grad, rtol=rtol, atol=floor, msg=name)
    return full_grads


@pytest.mark.parametrize("batch_size", [3, 2, 1], ids=["padded", "padded-chunks", "dense"])
@pytest.mark.parametrize("loss", _LOSSES)
def test_the_chunked_opd_term_matches_the_full_logits_one_on_a_full_fine_tune(loss, batch_size):
    """The loss, the OPD metric and every gradient, ``lm_head.weight`` included: the student projects through the
    head with a gradient, the teacher without."""
    model = _tiny_qwen3()
    grads = _assert_paths_agree(model, _real_inputs(), batch_size=batch_size, loss=loss)
    assert grads["lm_head.weight"].abs().sum() > 0


def test_the_chunked_opd_term_matches_in_bf16():
    """bf16 heads on both sides: the term agrees to a few of bf16's 2**-8 mantissa steps on a sum over the
    vocabulary."""
    model = _tiny_qwen3(torch.bfloat16)
    _, _, full = _sdpg_step(model, _real_inputs(), chunked=False)
    _, _, chunked = _sdpg_step(model, _real_inputs(), chunked=True)
    assert full["opd_loss"] > 0
    torch.testing.assert_close(chunked["opd_loss"].float(), full["opd_loss"].float(), rtol=2e-2, atol=0.0)


def test_the_chunked_opd_term_matches_under_lora():
    model = get_peft_model(_tiny_qwen3(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.5)
    grads = _assert_paths_agree(model, _real_inputs())
    assert grads and all("lora_" in name for name in grads)


def test_a_batch_without_a_positive_advantage_projects_nothing(monkeypatch):
    """The gate drops every position, so the term projects none, and the step is the GRPO loss alone."""
    projections = []
    divergence = sdpg_module.projected_divergence
    monkeypatch.setattr(
        sdpg_module,
        "projected_divergence",
        lambda *args, **kwargs: projections.append(1) or divergence(*args, **kwargs),
    )
    inputs = {**_real_inputs(), "advantages": torch.tensor([-0.7, 0.0, -0.4])}
    model = _tiny_qwen3()
    loss, _, metrics = _sdpg_step(model, inputs, chunked=True)
    assert not projections and metrics["opd_loss"] == 0
    grpo = DistributedOnlineGRPOTrainer._compute_loss(
        _real_loss_trainer(model, chunked=True, batch_size=3), model, inputs
    )
    torch.testing.assert_close(loss, grpo.detach())


def test_the_teacher_side_projects_with_grad_disabled(monkeypatch):
    """The teacher's logits keep no graph, in the forward and in the backward's recompute, so its side holds no
    ``[chunk, V]`` activations for the backward."""
    modes = []
    divergence = sdpg_module.projected_divergence

    def recording(loss_fn, student, student_rows, teacher, *args, **kwargs):
        def project(rows):
            modes.append(torch.is_grad_enabled())
            return teacher(rows)

        return divergence(loss_fn, student, student_rows, project, *args, **kwargs)

    monkeypatch.setattr(sdpg_module, "projected_divergence", recording)
    _sdpg_step(_tiny_qwen3(), _real_inputs(), chunked=True)
    assert len(modes) >= 2 and not any(modes), modes


def test_an_evaluation_step_leaves_the_padding_rows_out_on_both_paths():
    """Row 2 pads the eval split's final round (2 of 3 rows are the split's own). Its advantage is positive, so the gate
    keeps it and only the eval split cuts it; eval has no grad-accum division."""
    model = _tiny_qwen3().eval()
    inputs = {**_real_inputs(), "advantages": torch.tensor([0.7, 0.2, 0.4])}
    results = []
    for chunked in (False, True):
        trainer = _real_loss_trainer(model, chunked=chunked, batch_size=3)
        trainer.eval_split_rows = lambda num_rows: 2
        stored = {}
        trainer.store_metrics = lambda metrics, train_eval, rows=1: stored.update(metrics, train_eval=train_eval)
        with torch.no_grad():
            results.append((trainer._compute_loss(model, dict(inputs)), stored))
    (full, full_metrics), (chunked, chunked_metrics) = results
    per_token = _per_token_opd(_real_loss_trainer(model, chunked=False, batch_size=3), model, inputs)
    split = inputs["completion_mask"][:2].bool()
    expected = per_token[:2][split].mean()
    assert not torch.isclose(expected, per_token[inputs["completion_mask"].bool()].mean()), "row 2 must move the mean"
    for metrics in (full_metrics, chunked_metrics):
        torch.testing.assert_close(metrics["opd_loss"], expected, rtol=_RTOL, atol=0.0)
        assert metrics["train_eval"] == "eval"
    torch.testing.assert_close(chunked, full, rtol=_RTOL, atol=_RTOL * full.abs())


def test_the_chunked_term_builds_no_batch_by_completion_by_vocab_plane(monkeypatch):
    """Every head projection of the term covers one chunk of the gated positions, never the completion window, and
    each gated position is projected once per side."""
    monkeypatch.setattr(logprobs, "LOGIT_CHUNK_ELEMENTS", 2 * TINY_QWEN3_CONFIG["vocab_size"])
    inputs = _real_inputs()
    model = _tiny_qwen3()
    trainer = _real_loss_trainer(model, chunked=True, batch_size=3)
    trainer._head_transform(model)  # resolved once, its probe forward is not the term's
    shapes = []
    linear = logprob_scoring.F.linear
    monkeypatch.setattr(
        logprob_scoring.F, "linear", lambda *args: shapes.append(tuple((out := linear(*args)).shape)) or out
    )
    with torch.no_grad():
        _sdpg_step(model, inputs, chunked=True, trainer=trainer)
    gated = int(positive_advantage_gate(inputs["completion_mask"], inputs["advantages"], True).sum())
    shapes = [shape for shape in shapes if shape[-1] == TINY_QWEN3_CONFIG["vocab_size"]]
    assert shapes and all(len(shape) == 2 and shape[0] <= 2 for shape in shapes), shapes
    assert sum(shape[0] for shape in shapes) == 2 * gated


def _term_alone(model, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """``(OPD values, head gradient)`` of a backward through the armed term alone, the GRPO log-probs left out of it,
    against a fixed teacher."""
    trainer = _real_loss_trainer(model, chunked=True, batch_size=3)
    trainer.sdpg = SDPGArguments(sdpg_beta_base=0.5, sdpg_temperature=1.3)
    completion = inputs["completion_ids"]
    teacher = torch.randn(
        *completion.shape, TINY_QWEN3_CONFIG["hidden_size"], generator=torch.Generator().manual_seed(1)
    )
    term = CompletionTerm(inputs["completion_mask"].float(), functools.partial(trainer._opd_scores, teacher))
    model.zero_grad(set_to_none=True)
    with trainer._scoring_with_completion_term(term):
        trainer._chunked_completion_logps(
            model,
            torch.cat([inputs["prompt_ids"], completion], 1),
            torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1),
            completion.size(1),
            3,
        )
    (values,) = term.per_token
    values.sum().backward()
    return values.detach(), model.lm_head.weight.grad


def _zero3_worker(rank: int, nprocs: int) -> None:
    """ZeRO-3 reshards the head after the loss forward. The term's values are outputs of that forward, so the backward
    gathers the head again before it recomputes them, also when no GRPO log-prob reaches the backward first."""
    inputs = _real_inputs()
    reference = _tiny_qwen3()
    expected_loss, expected_grads, _ = _sdpg_step(reference, inputs, chunked=True)
    expected_values, expected_head = _term_alone(reference, inputs)
    model = _tiny_qwen3()
    policy = mixed_precision_policy(torch.float32, fp32_reduce=True, cast_forward_inputs=True)
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (nprocs,)), policy, True, IdentityParamSet())
    loss, grads, _ = _sdpg_step(model, inputs, chunked=True)
    torch.testing.assert_close(loss, expected_loss, rtol=_RTOL, atol=0.0)
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        torch.testing.assert_close(grads[name].full_tensor(), expected_grads[name], rtol=_RTOL, atol=1e-6, msg=name)
    sweeps = []
    head_sweep = DistributedSDPGTrainer._head_sweep
    with mock.patch.object(
        DistributedSDPGTrainer,
        "_head_sweep",
        lambda self, unwrapped: (sweep := head_sweep(self, unwrapped), sweeps.append(weakref.ref(sweep)))[0],
    ):
        values, head = _term_alone(model, inputs)
    torch.testing.assert_close(values, expected_values, rtol=_RTOL, atol=0.0)
    torch.testing.assert_close(head.full_tensor(), expected_head, rtol=_RTOL, atol=1e-6)
    gc.collect()
    assert sweeps and all(sweep() is None for sweep in sweeps), "the gathered head outlived the term's backward"
    assert isinstance(model.lm_head.weight, DTensor), "the head was left gathered after the backward"


def test_the_chunked_term_backward_regathers_a_resharded_head():
    run_gloo_ranks(_zero3_worker, 2, 2)


@pytest.mark.parametrize(
    ("beta", "precision", "bytes_per_logit"),
    [(0.5, "bf16", 6), (0.5, "fp32", 8), (0.0, "bf16", 4)],
    ids=["opd-bf16", "opd-fp32", "grpo"],
)
def test_the_full_logits_check_weighs_the_teacher_plane_while_opd_is_on(monkeypatch, beta, precision, bytes_per_logit):
    """GRPO's plane, and the teacher's in the run's precision while the OPD term is on."""
    weighed = []
    monkeypatch.setattr(logprob_scoring, "verify_full_logits_fit", lambda *args, **kwargs: weighed.append(kwargs))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer, _ = _opd_trainer(_tiny_qwen3(), positive_advantage_only=True)
    trainer.sdpg = SDPGArguments(sdpg_beta_base=beta)
    trainer.args = types.SimpleNamespace(bf16=precision == "bf16", fp16=False)
    trainer.accelerator = types.SimpleNamespace(unwrap_model=lambda module: module, device="cpu")
    trainer._verify_full_logits_fit(LogitsWidth(64, "max_completion_length"))
    assert [call["bytes_per_logit"] for call in weighed] == [bytes_per_logit]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
