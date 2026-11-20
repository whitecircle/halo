#!/usr/bin/env python
"""``use_chunked_distillation_loss`` on the self-distillation trainer against its full-logits path, on tiny real models:
the loss, every trainable gradient and every metric agree, for each OPD loss, with and without the frozen reference,
the EOS exclusion and the confidence weights, with a LoRA adapter (and its dropout) on ``lm_head``, a
``modules_to_save`` head, a reference with padded vocabulary rows, a composite's image rows (LFM2-VL replaying its
image features on the teacher pass) and in eval mode. The default path's logits planes are weighed at construction."""

import logging
import types

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

import halo.kernels.logprobs as logprob_kernels
import halo.trainers.full_logits_memory as full_logits_memory
from halo.configs.distillation import SDPGArguments
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.distillation.divergences import require_divergence
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.hidden_state_forward import reject_metrics_reading_logits
from tests.common.composite_rows import composite_model_and_rows
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

TEMPERATURE = 1.5
BETA = 0.75
ALPHA = 0.4
EOS = 2
HINT = [11, 12, 11]
# Several chunks per batch and a remainder: 12 supervised positions at 5 rows a chunk.
CHUNK_ROWS = 5
TINY = {"num_hidden_layers": 2}
# A few fp32 ulps (2**-23 each).
FP32_RTOL = 1e-6
# Per row: (prompt, response). Responses end in EOS, one carries a mid-response EOS too, the last row is shorter.
ROWS = [([5, 6, 7, 30], [8, 9, 40, EOS]), ([4, 3], [10, EOS, 6, 7, 21, EOS]), ([9, 9, 9, 9, 9], [3, EOS])]
WEIGHTS = torch.tensor([0.5, 2.0, 1.25], dtype=torch.float64)


def _branch(rows, hinted: bool):
    """Right-padded ``(input_ids, attention_mask, labels)``, labels on the response tokens only."""
    sequences = [prompt + (HINT if hinted else []) + response for prompt, response in rows]
    width = max(map(len, sequences))
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, LABEL_IGNORE_INDEX)
    for row, (sequence, (_, response)) in enumerate(zip(sequences, rows, strict=True)):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = 1
        labels[row, len(sequence) - len(response) : len(sequence)] = torch.tensor(response, dtype=torch.long)
    return ids, mask, labels


def _batch(rows=ROWS, *, weights: bool = True) -> dict[str, torch.Tensor]:
    ids, mask, labels = _branch(rows, hinted=False)
    teacher_ids, teacher_mask, teacher_labels = _branch(rows, hinted=True)
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "teacher_input_ids": teacher_ids,
        "teacher_attention_mask": teacher_mask,
        "teacher_labels": teacher_labels,
    }
    if weights:
        batch["confidence_weights"] = WEIGHTS[: len(rows)]
    return batch


def _student(*, peft: bool, dtype=torch.float64) -> nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, overrides=TINY)
    if peft:
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.25, target_modules=["q_proj", "v_proj", "lm_head"])
        model = get_peft_model(model, config)
        # PEFT starts every lora_B at zero, which would leave the head's delta out of the comparison.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.5)
    return model.to(dtype).train()


def _reference(dtype=torch.float64) -> nn.Module:
    torch.manual_seed(1)
    reference = tiny_family_model(TINY_DENSE_FAMILY, overrides=TINY)
    return reference.to(dtype).eval().requires_grad_(False)


def _trainer(
    model,
    *,
    sdpg_loss: str = "reverse_kl",
    sdpg_jsd_beta: float = 0.5,
    beta: float = BETA,
    reference: nn.Module | None = None,
    reference_kl_loss: str = "unnormalized_kl",
    confidence_weight_opd: bool = True,
    opd_exclude_eos: bool = True,
) -> DistributedSelfDistillationTrainer:
    """The real ``compute_loss`` on a bare trainer."""
    trainer = object.__new__(DistributedSelfDistillationTrainer)
    trainer.model = model
    trainer.sdpg = SDPGArguments(
        sdpg_loss=sdpg_loss, sdpg_jsd_beta=sdpg_jsd_beta, sdpg_beta_base=beta, sdpg_temperature=TEMPERATURE
    )
    trainer.sdpg_loss_fn = require_divergence(sdpg_loss, jsd_beta=sdpg_jsd_beta)
    trainer.reference_kl_coef = ALPHA if reference is not None else 0.0
    trainer.reference_kl_loss_fn = require_divergence(reference_kl_loss) if reference is not None else None
    trainer._reference_model = reference
    trainer._reference_vocab_width = None
    trainer.confidence_weight_opd = confidence_weight_opd
    trainer.opd_exclude_eos = opd_exclude_eos
    trainer.use_chunked_distillation_loss = False
    trainer._stop_token_ids, trainer._stop_ids_tensor = {EOS}, None
    trainer.state = types.SimpleNamespace(global_step=4, max_steps=10)
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
    trainer._warned_empty_labels = True
    trainer._vision_reuse_setup, trainer._vision_reuse_active = True, False
    trainer.stored = {}
    trainer.store_metrics = lambda metrics, train_eval="train", rows=1: trainer.stored.update(metrics)
    return trainer


def _step(trainer, batch, *, chunked: bool):
    """``(loss, gradients, metrics)`` of one forward and backward, the dropout seeded alike for both paths."""
    trainer.use_chunked_distillation_loss = chunked
    trainer.model.zero_grad(set_to_none=True)
    trainer.stored = {}
    torch.manual_seed(3)
    loss = trainer.compute_loss(trainer.model, dict(batch))
    loss.backward()
    grads = {name: param.grad for name, param in trainer.model.named_parameters() if param.requires_grad}
    return loss.detach(), grads, dict(trainer.stored)


def _assert_paths_agree(trainer, batch, rtol: float = FP32_RTOL):
    """The model and head path run in fp64, while every divergence evaluates in fp32 by design, so the two paths agree
    to fp32 rounding. A dropped LoRA delta, dropout mask, misaligned teacher row or missing weight moves them by
    orders more."""
    full_loss, full_grads, full_metrics = _step(trainer, batch, chunked=False)
    loss, grads, metrics = _step(trainer, batch, chunked=True)
    torch.testing.assert_close(loss, full_loss, rtol=rtol, atol=1e-9)
    assert grads.keys() == full_grads.keys()
    # A gradient zero by symmetry (a key bias under the softmax) is rounding noise on both paths, so each error is
    # taken against at least ``rtol`` of the largest gradient.
    floor = rtol * max(grad.norm() for grad in full_grads.values() if grad is not None)
    for name, grad in grads.items():
        if full_grads[name] is None:
            assert grad is None, f"{name} got a gradient the full-logits path does not give it"
            continue
        assert grad is not None, f"{name} got no gradient on the chunked path"
        error = (grad - full_grads[name]).norm() / torch.maximum(full_grads[name].norm(), floor)
        assert error < 10 * rtol, f"{name}: relative gradient error {error:.3g}"
    assert metrics.keys() == full_metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(torch.as_tensor(value), torch.as_tensor(full_metrics[key]), rtol=rtol, atol=1e-9)
    return full_grads, full_metrics


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", CHUNK_ROWS * TINY_QWEN3_CONFIG["vocab_size"])


LOSS_CASES = [
    {"sdpg_loss": "reverse_kl"},
    {"sdpg_loss": "forward_kl"},
    {"sdpg_loss": "unnormalized_kl"},
    {"sdpg_loss": "jensen_shannon", "sdpg_jsd_beta": 0.3},
    {"sdpg_loss": "reverse_kl", "opd_exclude_eos": False},
    {"sdpg_loss": "reverse_kl", "confidence_weight_opd": False},
]


@pytest.mark.parametrize("with_reference", [False, True], ids=["no_reference", "reference"])
@pytest.mark.parametrize("case", LOSS_CASES, ids=lambda case: "-".join(f"{k}={v}" for k, v in case.items()))
def test_every_opd_loss_matches_the_full_logits_path_with_lora_on_the_head(case, with_reference):
    reference = _reference() if with_reference else None
    trainer = _trainer(_student(peft=True), reference=reference, **case)
    grads, metrics = _assert_paths_agree(trainer, _batch())
    assert {"sft_loss", "opd_loss", "opd_beta"} <= metrics.keys()
    assert ("reference_kl" in metrics) == with_reference
    head_delta = [name for name in grads if "lm_head.lora_B" in name]
    assert head_delta and all(grads[name].abs().sum() > 0 for name in head_delta)


@pytest.mark.parametrize("reference_kl_loss", ["reverse_kl", "forward_kl", "unnormalized_kl"])
def test_every_reference_loss_matches_with_the_opd_term_off(reference_kl_loss):
    """At ``sdpg_beta_base: 0`` no teacher branch is built, and the reference term stands beside the cross-entropy."""
    trainer = _trainer(_student(peft=True), beta=0.0, reference=_reference(), reference_kl_loss=reference_kl_loss)
    batch = {key: value for key, value in _batch().items() if not key.startswith("teacher_")}
    _, metrics = _assert_paths_agree(trainer, batch)
    assert "opd_loss" not in metrics and "reference_kl" in metrics


def test_the_cross_entropy_alone_matches_without_confidence_weights():
    trainer = _trainer(_student(peft=True), beta=0.0)
    batch = {key: value for key, value in _batch(weights=False).items() if not key.startswith("teacher_")}
    _, metrics = _assert_paths_agree(trainer, batch)
    assert metrics.keys() == {"sft_loss"}


def test_a_full_fine_tune_matches_the_full_logits_path():
    trainer = _trainer(_student(peft=False), reference=_reference())
    _assert_paths_agree(trainer, _batch())


def test_a_modules_to_save_head_matches_the_full_logits_path():
    """PEFT trains a copy of the head, set apart here from the frozen original, so the projection must read the copy."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, overrides=TINY)
    config = LoraConfig(r=4, lora_dropout=0.25, target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"])
    model = get_peft_model(model, config)
    with torch.no_grad():
        copy = model.get_output_embeddings().modules_to_save["default"].weight
        copy.add_(0.5 * torch.randn_like(copy))
    trainer = _trainer(model.to(torch.float64).train(), reference=_reference())
    grads, _ = _assert_paths_agree(trainer, _batch())
    head_copy = [name for name in grads if "lm_head.modules_to_save" in name]
    assert head_copy and all(grads[name].abs().sum() > 0 for name in head_copy)


def test_vocabulary_rows_past_the_compared_width_leave_the_reference_term():
    """The reference carries 5 more embedding-padding rows than the student, and both carry 4 past the tokenizer's ids,
    so the compared width cuts both sides of the reference term while the OPD term and the cross-entropy keep the
    student's whole row."""
    student, reference = _student(peft=True), _reference()
    vocab = student.config.vocab_size
    padded = nn.Linear(reference.lm_head.in_features, vocab + 5, bias=False, dtype=torch.float64)
    with torch.no_grad():
        padded.weight[:vocab] = reference.lm_head.weight
        padded.weight[vocab:] = 3.0
    reference.lm_head = padded.requires_grad_(False)
    reference.config.vocab_size = vocab + 5
    trainer = _trainer(student, reference=reference)
    trainer._reference_vocab_width = vocab - 4
    _assert_paths_agree(trainer, _batch())


def test_an_evaluation_step_matches_the_full_logits_path():
    """Eval mode turns the LoRA dropout off, and the row padding an eval split's final round leaves every term."""
    trainer = _trainer(_student(peft=True), reference=_reference())
    trainer.model.eval()
    batch = _batch()

    def evaluate(*, chunked: bool, real_rows: int):
        trainer.use_chunked_distillation_loss = chunked
        trainer.eval_split_rows = lambda rows: real_rows
        trainer.stored = {}
        with torch.no_grad():
            return trainer.compute_loss(trainer.model, dict(batch)), dict(trainer.stored)

    full_loss, full_metrics = evaluate(chunked=False, real_rows=2)
    loss, metrics = evaluate(chunked=True, real_rows=2)
    torch.testing.assert_close(loss, full_loss, rtol=FP32_RTOL, atol=1e-9)
    assert metrics.keys() == full_metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(
            torch.as_tensor(value), torch.as_tensor(full_metrics[key]), rtol=FP32_RTOL, atol=1e-9
        )
    padded_loss, _ = evaluate(chunked=True, real_rows=3)
    assert (padded_loss - loss).abs() > 1e-6, "the padding row must move the loss, or the comparison sees nothing"
    empty_loss, _ = evaluate(chunked=True, real_rows=0)
    assert empty_loss.item() == 0.0


def test_a_row_without_supervision_matches():
    rows = [ROWS[0], ([4, 3, 8], []), ROWS[2]]
    trainer = _trainer(_student(peft=True), reference=_reference())
    _assert_paths_agree(trainer, _batch(rows))


def test_a_micro_batch_without_supervision_still_reaches_every_trainable_parameter():
    """A rank whose micro-batch has no supervised position still runs its backward through the head and the
    backbone, so it joins every collective its peers' backward runs."""
    trainer = _trainer(_student(peft=True), reference=_reference())
    loss, grads, _ = _step(trainer, _batch([([5, 6], []), ([4, 3, 8], [])]), chunked=True)
    assert loss == 0
    assert all(grad is not None for grad in grads.values()), [name for name, grad in grads.items() if grad is None]


def test_backward_holds_no_vocabulary_wide_tensor():
    trainer = _trainer(_student(peft=True), reference=_reference())
    vocab = trainer.model.config.vocab_size
    trainer.use_chunked_distillation_loss = True
    saved = []

    def record(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        trainer.compute_loss(trainer.model, _batch())
    assert saved, "the hooks saw nothing, so the check would pass vacuously"
    planes = [shape for shape in saved if shape and shape[-1] == vocab]
    assert not planes, f"vocabulary-wide tensors held for backward: {planes}"


def test_misaligned_branches_raise_on_the_chunked_path():
    """Per row, not in total: equal totals would pair rows across samples without any raise."""
    trainer = _trainer(_student(peft=True))
    trainer.use_chunked_distillation_loss = True
    batch = _batch()
    first = int((batch["teacher_labels"][0] != LABEL_IGNORE_INDEX).nonzero()[0])
    batch["teacher_labels"][0, first - 1] = 5
    last = int((batch["teacher_labels"][1] != LABEL_IGNORE_INDEX).nonzero()[-1])
    batch["teacher_labels"][1, last] = LABEL_IGNORE_INDEX
    with pytest.raises(RuntimeError, match="response alignment contract"):
        trainer.compute_loss(trainer.model, batch)


def _image_batch(row: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The composite's image row as both branches, the last three tokens supervised. The teacher branch runs in eval
    mode, so its target still differs from the student's dropout-carrying distribution."""
    labels = torch.full_like(row["labels"], LABEL_IGNORE_INDEX)
    labels[:, -3:] = row["input_ids"][:, -3:]
    batch = {**row, "labels": labels}
    per_token = [key for key, value in row.items() if torch.is_tensor(value) and value.shape == row["input_ids"].shape]
    batch.update({f"teacher_{key}": batch[key].clone() for key in (*per_token, "labels")})
    return batch


def test_image_rows_match_the_full_logits_path():
    """A composite's backbone takes its image inputs on both passes. In fp32: the composites' vision and expert kernels
    take no fp64."""
    torch.manual_seed(0)
    model, image_row, _ = composite_model_and_rows("qwen3_5")
    model = get_peft_model(model, LoraConfig(r=4, lora_dropout=0.25, target_modules=["lm_head"]))
    with torch.no_grad():
        model.get_output_embeddings().lora_B["default"].weight.normal_(std=0.5)
    trainer = _trainer(model.train())
    batch = _image_batch(image_row)
    _assert_paths_agree(trainer, batch, rtol=1e-4)
    kept = ("input_ids", "attention_mask", "labels")
    text_only = {key: value for key, value in batch.items() if key.removeprefix("teacher_") in kept}
    without_pixels, _, _ = _step(trainer, text_only, chunked=True)
    with_pixels, _, _ = _step(trainer, batch, chunked=True)
    assert (with_pixels - without_pixels).abs() > 1e-3


def test_lfm2_vl_replays_the_student_s_image_features_on_the_teacher_pass():
    """Both paths compute the image features once per step and replay them on the privileged teacher pass."""
    torch.manual_seed(0)
    model, image_row, _ = composite_model_and_rows("lfm2_vl")
    trainer = _trainer(model.train())
    trainer._vision_reuse_setup = False
    calls = []
    original = model.model.get_image_features

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    model.model.get_image_features = counted
    batch = _image_batch(image_row)
    _assert_paths_agree(trainer, batch, rtol=1e-4)
    assert trainer._vision_reuse_active
    assert len(calls) == 2, f"one image-feature pass per step expected, saw {len(calls)} over two steps"


def test_metrics_reading_evaluation_logits_are_refused():
    with pytest.raises(ValueError, match="compute_metrics / preprocess_logits_for_metrics"):
        reject_metrics_reading_logits(lambda prediction: {}, None)
    reject_metrics_reading_logits(None, None)


_GIB = 1024**3


def _memory_host(*, chunked=False, beta=BETA, reference=True, max_length=4096, precision="bf16"):
    """At bf16 and 2 rows × 4096 tokens × 2**16 vocab, one plane is 1 GiB."""
    rows = types.SimpleNamespace(
        config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(vocab_size=2**16))
    )
    return types.SimpleNamespace(
        use_chunked_distillation_loss=chunked,
        sdpg=types.SimpleNamespace(sdpg_beta_base=beta),
        reference_kl_coef=ALPHA if reference else 0.0,
        model=rows,
        _reference_model=rows,
        args=types.SimpleNamespace(
            max_length=max_length,
            bf16=precision == "bf16",
            fp16=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            eval_strategy="no",
            eval_on_start=False,
        ),
        accelerator=types.SimpleNamespace(device="cuda:1", unwrap_model=lambda model: model),
    )


@pytest.fixture
def free_memory(monkeypatch):
    def _set(free: int):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, 8 * free))

    return _set


@pytest.mark.parametrize(
    ("beta", "reference", "precision", "bytes_per_logit"),
    [(BETA, True, "bf16", 8), (BETA, False, "bf16", 6), (0.0, False, "bf16", 4), (BETA, True, "fp32", 16)],
    ids=["opd+reference", "opd", "sft", "opd+reference-fp32"],
)
def test_the_warning_weighs_one_plane_per_model_and_the_student_gradient(
    free_memory, caplog, beta, reference, precision, bytes_per_logit
):
    """The student's logits and their gradient, the teacher's with OPD on, the reference's with its term on, each in
    the run's precision."""
    free_memory(_GIB)
    host = _memory_host(beta=beta, reference=reference, precision=precision)
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedSelfDistillationTrainer._warn_if_full_logits_large(host)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1, messages
    assert f"× {bytes_per_logit} B = {bytes_per_logit / 2:.1f} GiB" in messages[0], messages
    assert "use_chunked_distillation_loss: true" in messages[0] and "self-distillation loss" in messages[0]


@pytest.mark.parametrize(
    "host",
    [_memory_host(chunked=True), _memory_host(max_length=None), _memory_host(max_length=8)],
    ids=["chunked", "no_max_length", "fits"],
)
def test_the_warning_stays_quiet_when_nothing_is_large(free_memory, caplog, host):
    free_memory(_GIB)
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedSelfDistillationTrainer._warn_if_full_logits_large(host)
    assert not caplog.records


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
