#!/usr/bin/env python
"""``DistributedDistillationTrainer``'s objective, teacher gate and PEFT wiring, on stub models.

The objective is ``alpha * L_distill + (1 - alpha) * L_clm``, each term one mean over the batch's
supervised (shifted) tokens. Every way it can go wrong keeps the loss finite: an ignored alpha or a
dropped CLM term, labels left unshifted, the teacher's logits replaced by the student's, the
hard-label gate skipped (or applied on top of SLIM's own weight), a per-row instead of a global
token mean. The stub models score each token through their own random table, so each of those lands
on a different number than the independent oracle below.

Run: pytest tests/cpu/trainers/test_teacher_distillation_trainer.py
"""

import contextlib
import inspect
import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig
from torch import nn
from torch.nn.functional import cross_entropy, log_softmax, softmax
from transformers import Trainer

import scripts.training.distillation.teacher_distill as teacher_distill_script
import src.trainers.distillation.teacher_distillation as teacher_distillation
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.losses import get_divergence
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from src.training.script_runner import ScriptRuntime
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

PartialState()  # the trainer's accelerate logger requires an initialized state

VOCAB = 11
TEMPERATURE = 2.0
ALPHA = 0.3
INPUT_IDS = torch.tensor([[1, 4, 2, 7, 3, 9], [5, 5, 8, 0, 6, 2]])
# Ragged supervision (4 and 2 shifted targets), so a per-row and a global token mean differ.
LABELS = torch.tensor(
    [
        [LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, 2, 7, 3, 9],
        [LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, 0, 6, LABEL_IGNORE_INDEX],
    ]
)


class _TokenTableModel(nn.Module):
    """Logits from a fixed random per-token table: each model scores a sequence its own way."""

    def __init__(self, seed: int, rows: int = VOCAB):
        super().__init__()
        self.table = nn.Parameter(torch.randn(VOCAB, rows, generator=torch.Generator().manual_seed(seed)))
        self.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(vocab_size=rows))
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        return types.SimpleNamespace(logits=self.table[input_ids])


class _Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)


TOKENS = {f"t{i}": i for i in range(VOCAB)}


def _trainer(distill_loss="kl_divergence", *, apply_hard_labels=False, alpha=ALPHA, peft_casted=False):
    """The real ``compute_loss`` on a bare trainer; returns ``(trainer, stored_metrics)``."""
    trainer = object.__new__(DistributedDistillationTrainer)
    trainer.model = _TokenTableModel(seed=0)
    trainer.teacher_model = _TokenTableModel(seed=1)
    trainer.args = types.SimpleNamespace(
        distill_loss=distill_loss,
        distill_alpha=alpha,
        distill_temperature=TEMPERATURE,
        apply_hard_labels=apply_hard_labels,
    )
    trainer.distillation_loss_fn = get_divergence(distill_loss)
    trainer._vocab_width = None
    trainer._peft_has_been_casted_to_bf16 = peft_casted
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    trainer._warned_empty_labels = True
    stored = {}

    def _store(metrics, train_eval, rows=1):
        stored.update(metrics, rows=rows)

    trainer.store_metrics = _store
    return trainer, stored


def _loss(trainer, input_ids: torch.Tensor = INPUT_IDS, labels: torch.Tensor = LABELS):
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}
    return trainer.compute_loss(trainer.model, inputs)


def _oracle(trainer, *, gate: bool = False, slim: bool = False):
    """``(loss, distill, clm)`` spelled from torch primitives, independently of the trainer's helpers."""
    student = trainer.model.table.detach()[INPUT_IDS][:, :-1]
    teacher = trainer.teacher_model.table.detach()[INPUT_IDS][:, :-1]
    target = LABELS[:, 1:]
    kept = target != LABEL_IGNORE_INDEX
    q = softmax(teacher / TEMPERATURE, dim=-1)
    per_token = (q * (q.log() - log_softmax(student / TEMPERATURE, dim=-1))).sum(-1) * TEMPERATURE**2
    gold = target.clamp_min(0).unsqueeze(-1)
    student_gold = softmax(student, dim=-1).gather(-1, gold).squeeze(-1)
    teacher_gold = softmax(teacher, dim=-1).gather(-1, gold).squeeze(-1)
    if gate:
        per_token = per_token * (1 - student_gold) * teacher_gold
    if slim:
        per_token = per_token * (1 - torch.exp(-teacher_gold / student_gold))
    distill = per_token[kept].mean()
    clm = cross_entropy(student[kept], target[kept])
    return ALPHA * distill + (1 - ALPHA) * clm, distill, clm


def test_the_loss_is_the_alpha_mix_of_two_global_token_means():
    trainer, stored = _trainer()
    expected, distill, clm = _oracle(trainer)
    torch.testing.assert_close(_loss(trainer), expected)
    torch.testing.assert_close(stored["distillation_loss"], distill)
    torch.testing.assert_close(stored["sft_loss"], clm)
    assert stored["rows"] == 1, "a train micro-batch weighs 1, whatever its row count"


def test_an_eval_split_s_padding_rows_leave_the_loss_and_its_metrics():
    """Row 2 pads the eval split's final round, a repeat of row 0 (``eval_split_rows`` keeps rows 0 and
    1): the loss and both metrics are the two real rows' global token means, stored with their two
    rows as the weight."""
    trainer, stored = _trainer()
    trainer.model.eval()
    trainer.eval_split_rows = lambda num_rows: 2
    expected, distill, clm = _oracle(trainer)

    torch.testing.assert_close(
        _loss(trainer, torch.cat([INPUT_IDS, INPUT_IDS[:1]]), torch.cat([LABELS, LABELS[:1]])), expected
    )
    torch.testing.assert_close(stored["distillation_loss"], distill)
    torch.testing.assert_close(stored["sft_loss"], clm)
    assert stored["rows"] == 2


def test_the_clm_term_is_logged_but_not_trained_at_alpha_one():
    trainer, stored = _trainer(alpha=1.0)
    _, distill, clm = _oracle(trainer)
    torch.testing.assert_close(_loss(trainer), distill)
    torch.testing.assert_close(stored["sft_loss"], clm)


def test_the_hard_label_gate_weights_the_divergence():
    trainer, stored = _trainer(apply_hard_labels=True)
    expected, _, _ = _oracle(trainer, gate=True)
    torch.testing.assert_close(_loss(trainer), expected)
    assert "distillation_coef" in stored


def test_slim_takes_its_own_weight_and_not_the_hard_label_gate_on_top():
    trainer, stored = _trainer("slim", apply_hard_labels=True)
    expected, _, _ = _oracle(trainer, slim=True)
    torch.testing.assert_close(_loss(trainer), expected)
    assert "distillation_coef" not in stored


def test_a_casted_peft_student_forwards_under_bf16_autocast():
    """``prepare_peft_model`` leaves a QLoRA student's adapters bf16 over fp32 activations; every
    sibling trainer feeds its cast flag to the autocast, so this one must too."""
    seen = []

    @contextmanager
    def recording(casted, device):
        seen.append(casted)
        yield

    trainer, _ = _trainer(peft_casted=True)
    with mock.patch.object(teacher_distillation, "peft_bf16_autocast", recording):
        _loss(trainer)
    assert seen == [True]


def test_the_trainer_wraps_peft_through_prepare_peft_model_and_keeps_its_cast_flag():
    """The PEFT wrap lives in the trainer, as in every sibling, so the cast flag reaches compute_loss."""
    peft_config, wrapped = LoraConfig(), nn.Linear(2, 2)

    def init_config(self, kwargs, **explicit):
        self.parallelism_config = explicit["parallelism_config"]
        return {**kwargs, "model": explicit["model"]}

    with (
        mock.patch.object(teacher_distillation, "load_model_from_pretrained", lambda model, *a, **k: (model, None)),
        mock.patch.object(DistributedDistillationTrainer, "_init_distributed_config", init_config),
        mock.patch.object(teacher_distillation, "prepare_peft_model", return_value=(wrapped, True)) as prepare,
        mock.patch.object(Trainer, "__init__", lambda self, model, **kwargs: setattr(self, "model", model)),
        mock.patch.object(nn.Linear, "add_model_tags", create=True),
        mock.patch.object(DistributedDistillationTrainer, "_setup_distributed_modes"),
        mock.patch.object(DistributedDistillationTrainer, "_setup_teacher_model"),
    ):
        trainer = DistributedDistillationTrainer(
            student_model=nn.Linear(2, 2),
            teacher_model=nn.Linear(2, 2),
            teacher_tokenizer=_Tokenizer(TOKENS),
            args=types.SimpleNamespace(distill_loss="kl_divergence"),
            processing_class=_Tokenizer(TOKENS),
            peft_config=peft_config,
            parallelism_config=ParallelismConfig(),
        )
    assert prepare.call_args.args[1] is peft_config
    assert trainer.model is wrapped
    assert trainer._peft_has_been_casted_to_bf16 is True


def test_the_teacher_tokenizer_is_keyword_only_so_positional_callers_keep_their_slots():
    parameter = inspect.signature(DistributedDistillationTrainer.__init__).parameters["teacher_tokenizer"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(DistributedDistillationTrainer.__init__).parameters)[1:4] == [
        "student_model",
        "teacher_model",
        "args",
    ]


def _setup(student_rows, teacher_rows, teacher_tokens=TOKENS):
    trainer = object.__new__(DistributedDistillationTrainer)
    trainer.model = _TokenTableModel(seed=0, rows=student_rows)
    trainer.teacher_model = _TokenTableModel(seed=1, rows=teacher_rows)
    trainer.processing_class = _Tokenizer(TOKENS)
    trainer._setup_teacher_model(_Tokenizer(teacher_tokens))
    return trainer


def test_a_teacher_of_another_tokenizer_is_refused_even_at_an_equal_vocab_size():
    swapped = {**TOKENS, "t1": 2, "t2": 1}
    with pytest.raises(ValueError, match=r"disagree on 4 \(token, id\) pair"):
        _setup(VOCAB, VOCAB, swapped)


def test_a_teacher_missing_logit_rows_for_tokenizer_ids_is_refused():
    with pytest.raises(ValueError, match="fewer logit rows than the tokenizer"):
        _setup(VOCAB, VOCAB - 1)


def test_embedding_padding_alone_is_compared_over_the_tokenizer_ids():
    """Same tokenizer, padded embeddings: both rows are sliced to the tokenizer's ids."""
    trainer = _setup(VOCAB, VOCAB + 5)
    assert trainer._vocab_width == VOCAB
    assert _setup(VOCAB, VOCAB)._vocab_width is None

    padded, _ = _trainer()
    padded.teacher_model = trainer.teacher_model
    padded._vocab_width = VOCAB
    reference, _ = _trainer()
    reference.teacher_model.table = nn.Parameter(trainer.teacher_model.table.detach()[:, :VOCAB])
    torch.testing.assert_close(_loss(padded), _loss(reference))
    assert not trainer.teacher_model.table.requires_grad, "the teacher must come out frozen"


class _TeacherWeightsLoaded(Exception):
    """Raised by the stubbed teacher weight load: the script got past every check before it."""


def _run_teacher_script(tmp_path, teacher_tokens):
    """Drive the script's ``main()`` with every load stubbed, the teacher's tokenizer and config included."""
    student = _TokenTableModel(seed=0)
    stubs = {
        "init_training_script": lambda *a, **k: ScriptRuntime(
            parallelism_config=ParallelismConfig(),
            mode_suffix="",
            local_rank=0,
            resume_checkpoint=None,
            model_source="stub/student",
        ),
        "load_script_datasets": lambda *a, **k: (None, False),
        "resolve_vlm_run": lambda *a, **k: False,
        "load_model_for_training": lambda *a, **k: (student, _Tokenizer(TOKENS), _Tokenizer(TOKENS), False),
        "apply_max_length": lambda config, args, model, tokenizer: tokenizer,
        "install_resolved_tokenizer": lambda processing_class, tokenizer: tokenizer,
        "enforce_text_path_padding_side": lambda *a, **k: None,
        "setup_peft_model": lambda *a, **k: None,
        "AutoTokenizer": types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer(teacher_tokens)),
        "AutoConfig": types.SimpleNamespace(from_pretrained=lambda *a, **k: student.config),
        "_load_distill_teacher": mock.Mock(side_effect=_TeacherWeightsLoaded),
    }
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/student\nteacher_model: stub/teacher\ndataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n"
    )
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        contextlib.ExitStack() as stack,
    ):
        for name, stub in stubs.items():
            stack.enter_context(mock.patch.object(teacher_distill_script, name, stub))
        teacher_distill_script.main()


def test_the_script_refuses_a_teacher_tokenizer_before_any_teacher_weight_loads(tmp_path):
    """A wrong-family teacher must not be downloaded and loaded on every rank before it is refused."""
    with pytest.raises(ValueError, match="disagree on"):
        _run_teacher_script(tmp_path, {**TOKENS, "t1": 2, "t2": 1})
    with pytest.raises(_TeacherWeightsLoaded):
        _run_teacher_script(tmp_path, TOKENS)


def test_the_text_collator_keeps_the_real_eos_label_when_pad_is_eos():
    """A plain LM collator masks every pad-valued label, the turn-ending EOS included when pad == eos,
    leaving the student no stop signal; both terms mask on these labels."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    tokenizer.pad_token = tokenizer.eos_token
    args = types.SimpleNamespace(train_on_completions_only=False, assistant_message_template=None)
    collator = teacher_distill_script._text_distill_collator(args, tokenizer, None)
    rows = [{"input_ids": [11, 12, 13, tokenizer.eos_token_id]}, {"input_ids": [21, tokenizer.eos_token_id]}]
    labels = collator(rows)["labels"]
    assert labels[:, -1].tolist() == [tokenizer.eos_token_id, LABEL_IGNORE_INDEX]
    assert labels[1, 1].item() == tokenizer.eos_token_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
