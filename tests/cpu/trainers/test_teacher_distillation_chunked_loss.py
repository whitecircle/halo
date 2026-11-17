#!/usr/bin/env python
"""``use_chunked_distillation_loss`` against the full-logits path of the same trainer, on tiny real models: the loss and
every trainable gradient agree, with a LoRA adapter (and its dropout) on ``lm_head``, the families' head transforms and
a composite's image rows."""

import dataclasses
import types

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn

import halo.kernels.logprobs as logprob_kernels
import halo.trainers.distillation.teacher_distillation as teacher_distillation
from halo.configs.distillation import TeacherDistillationConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from halo.trainers.head_projection import resolve_output_embedding
from halo.trainers.hidden_state_forward import reject_backbone_without_image_inputs
from tests.common.composite_rows import composite_model_and_rows
from tests.common.models import TINY_INKLING_CONFIG, TINY_MINICPM3_CONFIG, TINY_QWEN3_CONFIG
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    tiny_family_model,
)

TEMPERATURE = 2.0
# Several chunks per batch and a remainder: 22 supervised positions at 4 rows a chunk.
CHUNK_ROWS = 4
TINY = {"num_hidden_layers": 2}
MINICPM3 = TinyFamily(
    lambda overrides: transformers.MiniCPM3ForCausalLM(
        transformers.MiniCPM3Config(**{**TINY_MINICPM3_CONFIG, **overrides})
    )
)
# A few fp32 ulps (2**-23 each).
FP32_RTOL = 1e-6
_LOSS_KNOBS = ("distill_loss", "apply_hard_labels", "distill_jsd_beta", "distill_topk", "distill_alpha")


def _loss_args(**overrides) -> types.SimpleNamespace:
    """The ``TeacherDistillationConfig`` loss knobs at their defaults unless overridden."""
    defaults = {f.name: f.default for f in dataclasses.fields(TeacherDistillationConfig) if f.name in _LOSS_KNOBS}
    return types.SimpleNamespace(**{**defaults, "distill_temperature": TEMPERATURE, **overrides})


def _student(family, *, peft: bool, overrides: dict | None = None, dtype=torch.float64) -> nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides={**TINY, **(overrides or {})})
    if peft:
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.25, target_modules=["q_proj", "v_proj", "lm_head"])
        model = get_peft_model(model, config)
        # PEFT starts every lora_B at zero, which would leave the head's delta out of the comparison.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.5)
    return model.to(dtype).train()


def _teacher(family, overrides: dict | None = None, dtype=torch.float64) -> nn.Module:
    """A narrower teacher than the student, so each model projects through its own head."""
    torch.manual_seed(1)
    shape = {"hidden_size": 64, "intermediate_size": 96} if overrides is None else overrides
    teacher = tiny_family_model(family, overrides={**TINY, **shape})
    return teacher.to(dtype).eval().requires_grad_(False)


def _trainer(student, teacher, **loss_overrides) -> DistributedTeacherDistillationTrainer:
    """The real ``compute_loss`` on a bare trainer."""
    trainer = object.__new__(DistributedTeacherDistillationTrainer)
    trainer.model = student
    trainer.teacher_model = teacher
    trainer.args = _loss_args(**loss_overrides)
    trainer.distillation_loss_fn = teacher_distillation.require_distill_loss(trainer.args)
    trainer._vocab_width = None
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
    trainer._warned_empty_labels = True
    trainer.stored = {}
    trainer.store_metrics = lambda metrics, train_eval, rows=1: trainer.stored.update(metrics)
    return trainer


def _batch(vocab: int, supervised: list[list[int]]) -> dict[str, torch.Tensor]:
    """Two 16-token rows; ``supervised`` lists each row's labelled positions."""
    input_ids = torch.randint(0, vocab, (2, 16), generator=torch.Generator().manual_seed(2))
    labels = torch.full_like(input_ids, LABEL_IGNORE_INDEX)
    for row, positions in enumerate(supervised):
        labels[row, positions] = input_ids[row, positions]
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


RAGGED = [list(range(3, 16)), list(range(7, 16))]


def _step(trainer, batch, *, chunked: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(loss, gradients, metrics)`` of one forward and backward, the dropout seeded alike for both paths."""
    trainer.args.use_chunked_distillation_loss = chunked
    trainer.model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    loss = trainer.compute_loss(trainer.model, dict(batch))
    loss.backward()
    grads = {name: param.grad for name, param in trainer.model.named_parameters() if param.requires_grad}
    return loss.detach(), grads, dict(trainer.stored)


def _assert_paths_agree(trainer, batch, rtol: float = FP32_RTOL):
    """The model and head path run in fp64, while every divergence evaluates in fp32 by design, so the two paths agree
    to fp32 rounding: the last-bit fp64 differences of two GEMM shapes can round one fp32 logit apart. A dropped LoRA
    delta, dropout mask or head transform moves them by orders more."""
    full_loss, full_grads, full_metrics = _step(trainer, batch, chunked=False)
    loss, grads, metrics = _step(trainer, batch, chunked=True)
    torch.testing.assert_close(loss, full_loss, rtol=rtol, atol=1e-9)
    assert grads.keys() == full_grads.keys()
    for name, grad in grads.items():
        # A merged or disabled adapter takes no gradient on the full path, and must take none here either.
        if full_grads[name] is None:
            assert grad is None, f"{name} got a gradient the full-logits path does not give it"
            continue
        assert grad is not None, f"{name} got no gradient on the chunked path"
        error = (grad - full_grads[name]).norm() / full_grads[name].norm().clamp_min(1e-12)
        assert error < 10 * rtol, f"{name}: relative gradient error {error:.3g}"
    assert metrics.keys() == full_metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(value, full_metrics[key], rtol=rtol, atol=1e-9)
    return full_grads


@pytest.fixture(autouse=True)
def _small_chunks(monkeypatch):
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", CHUNK_ROWS * TINY_QWEN3_CONFIG["vocab_size"])


LOSS_CASES = [
    *(
        {"distill_loss": loss}
        for loss in ("kl_divergence", "reverse_kl", "mse", "soft_cross_entropy", "cosine_similarity")
    ),
    {"distill_loss": "slim"},
    {"distill_loss": "jensen_shannon"},
    {"distill_loss": "jensen_shannon", "distill_jsd_beta": 0.3},
    {"distill_loss": "kl_divergence", "distill_topk": 5},
    {"distill_loss": "soft_cross_entropy", "distill_topk": 5},
    {"distill_loss": "kl_divergence", "apply_hard_labels": True},
    {"distill_loss": "kl_divergence", "distill_alpha": 0.5},
    {"distill_loss": "jensen_shannon", "distill_alpha": 0.5, "apply_hard_labels": True},
]


@pytest.mark.parametrize("loss_overrides", LOSS_CASES, ids=lambda case: "-".join(f"{k}={v}" for k, v in case.items()))
def test_every_loss_matches_the_full_logits_path_with_lora_on_the_head(loss_overrides):
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), **loss_overrides)
    batch = _batch(trainer.model.config.vocab_size, RAGGED)
    grads = _assert_paths_agree(trainer, batch)
    head_delta = [name for name in grads if "lm_head.lora_B" in name]
    assert head_delta and all(grads[name].abs().sum() > 0 for name in head_delta)


def test_a_full_fine_tune_matches_the_full_logits_path():
    student = _student(TINY_DENSE_FAMILY, peft=False)
    trainer = _trainer(student, _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    _assert_paths_agree(trainer, _batch(student.config.vocab_size, RAGGED))


def test_a_softcapped_head_matches_the_full_logits_path():
    """Gemma 2 caps its logits past the output embedding; the projection applies the family's verified transform."""
    gemma2 = TINY_DENSE_ATTENTION_FAMILIES["gemma2"]
    trainer = _trainer(_student(gemma2, peft=True), _teacher(gemma2), distill_alpha=0.5)
    _assert_paths_agree(trainer, _batch(trainer.model.config.vocab_size, RAGGED))


def test_a_scaled_hidden_state_matches_the_full_logits_path():
    """MiniCPM3 divides the hidden state before the head, so the LoRA dropout draws on the scaled input."""
    trainer = _trainer(_student(MINICPM3, peft=True), _teacher(MINICPM3, {}), distill_alpha=0.5)
    assert trainer._head_transform(trainer.model).hidden_scale is not None
    _assert_paths_agree(trainer, _batch(trainer.model.config.vocab_size, RAGGED))


def test_a_head_transform_vocabulary_cut_matches_the_full_logits_path():
    """Inkling cuts its padded vocabulary columns off the logits. Its grouped expert GEMM takes no fp64, so this runs
    in fp32, where the two GEMM shapes differ by fp32 rounding before the loss."""
    inkling = TINY_MOE_FAMILIES["inkling_text"]
    cut = {"unpadded_vocab_size": TINY_INKLING_CONFIG["vocab_size"] - 12}
    student = _student(inkling, peft=True, overrides=cut, dtype=torch.float32)
    trainer = _trainer(student, _teacher(inkling, cut, dtype=torch.float32), distill_alpha=0.5)
    assert trainer._head_transform(trainer.model).vocab_size == cut["unpadded_vocab_size"]
    _assert_paths_agree(trainer, _batch(cut["unpadded_vocab_size"], RAGGED), rtol=1e-4)


def test_a_row_without_supervision_and_padded_vocabulary_rows_match():
    """The teacher carries 5 embedding-padding rows past the student's, cut by the compared width."""
    student = _student(TINY_DENSE_FAMILY, peft=True)
    teacher = _teacher(TINY_DENSE_FAMILY)
    vocab = student.config.vocab_size
    padded = nn.Linear(teacher.lm_head.in_features, vocab + 5, bias=False, dtype=torch.float64)
    with torch.no_grad():
        padded.weight[:vocab] = teacher.lm_head.weight
    teacher.lm_head = padded.requires_grad_(False)
    teacher.config.vocab_size = vocab + 5
    trainer = _trainer(student, teacher, distill_alpha=0.5)
    trainer._vocab_width = vocab
    _assert_paths_agree(trainer, _batch(vocab, [[], list(range(2, 16))]))


def test_a_micro_batch_without_supervision_still_reaches_every_trainable_parameter():
    """A rank whose micro-batch has no supervised position still runs its backward through the head and the
    backbone, so it joins every collective its peers' backward runs."""
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    loss, grads, _ = _step(trainer, _batch(trainer.model.config.vocab_size, [[], []]), chunked=True)
    assert loss == 0
    assert all(grad is not None for grad in grads.values()), [name for name, grad in grads.items() if grad is None]


def test_backward_holds_no_vocabulary_wide_tensor():
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    vocab = trainer.model.config.vocab_size
    trainer.args.use_chunked_distillation_loss = True
    saved = []

    def record(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        trainer.compute_loss(trainer.model, _batch(vocab, RAGGED))
    assert saved, "the hooks saw nothing, so the check would pass vacuously"
    planes = [shape for shape in saved if shape and shape[-1] == vocab]
    assert not planes, f"vocabulary-wide tensors held for backward: {planes}"


def _trainer_with_a_trainable_norm():
    """With every adapter merged or disabled, the final norm is what carries a gradient through the head."""
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    trainer.model.get_base_model().model.norm.weight.requires_grad_(True)
    return trainer


def test_a_merged_head_adapter_matches_the_full_logits_path():
    """Merged, the adapters are in the base weights and PEFT's forward adds no delta, so neither path does."""
    trainer = _trainer_with_a_trainable_norm()
    trainer.model.merge_adapter()
    _assert_paths_agree(trainer, _batch(trainer.model.config.vocab_size, RAGGED))


def test_disabled_adapters_match_the_full_logits_path():
    trainer = _trainer_with_a_trainable_norm()
    with trainer.model.disable_adapter():
        _assert_paths_agree(trainer, _batch(trainer.model.config.vocab_size, RAGGED))


def test_a_head_adapter_merged_then_disabled_is_refused():
    """PEFT's forward unmerges it before scoring the base weight; the projection would score the merged weight."""
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    head = trainer.model.get_output_embeddings()
    head.merge()
    head.enable_adapters(False)
    trainer.args.use_chunked_distillation_loss = True
    with pytest.raises(RuntimeError, match="merged into its base weight while adapters are disabled"):
        trainer.compute_loss(trainer.model, _batch(trainer.model.config.vocab_size, RAGGED))


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4"])
def test_image_rows_match_the_full_logits_path(family):
    """A composite's backbone takes its image inputs, so an image row scores as through the composite's forward. The
    same row without its pixels scores differently, so the comparison sees the image. In fp32: the composites' vision
    and expert kernels take no fp64."""
    torch.manual_seed(0)
    student, image_row, _ = composite_model_and_rows(family)
    torch.manual_seed(0)
    teacher, _, _ = composite_model_and_rows(family)
    with torch.no_grad():
        for param in teacher.parameters():
            param.add_(0.05 * torch.randn_like(param))
    student = get_peft_model(student, LoraConfig(r=4, lora_dropout=0.25, target_modules=["lm_head"]))
    with torch.no_grad():
        student.get_output_embeddings().lora_B["default"].weight.normal_(std=0.5)
    trainer = _trainer(student.train(), teacher.eval().requires_grad_(False), distill_alpha=0.5)
    _assert_paths_agree(trainer, image_row, rtol=1e-4)
    text_only = {key: value for key, value in image_row.items() if key in ("input_ids", "attention_mask", "labels")}
    without_pixels, _, _ = _step(trainer, text_only, chunked=True)
    with_pixels, _, _ = _step(trainer, image_row, chunked=True)
    assert (with_pixels - without_pixels).abs() > 1e-3


def test_a_composite_that_builds_its_image_inputs_itself_is_refused():
    """Nemotron 3.5 Super VL embeds its images before its text model, which the hidden-state path enters directly."""
    torch.manual_seed(0)
    omni = tiny_family_model(TINY_MOE_VLM_FAMILIES["nemotron_h_omni"])
    with pytest.raises(ValueError, match="takes no pixel_values"):
        reject_backbone_without_image_inputs(omni)
    torch.manual_seed(0)
    reject_backbone_without_image_inputs(composite_model_and_rows("qwen3_5")[0])


def test_dora_on_the_head_is_refused():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, overrides=TINY)
    model = get_peft_model(model, LoraConfig(r=4, use_dora=True, target_modules=["lm_head"]))
    with pytest.raises(ValueError, match="carry a variant"):
        resolve_output_embedding(model, IDENTITY_HEAD_TRANSFORM)


def test_metrics_reading_evaluation_logits_are_refused():
    trainer = _trainer(_student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY))
    trainer.compute_metrics, trainer.preprocess_logits_for_metrics = (lambda prediction: {}), None
    with pytest.raises(ValueError, match="compute_metrics / preprocess_logits_for_metrics"):
        trainer._verify_chunkable_loss()
    trainer.compute_metrics = None
    trainer._verify_chunkable_loss()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
