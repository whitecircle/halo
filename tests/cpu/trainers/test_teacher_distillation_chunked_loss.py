#!/usr/bin/env python
"""``use_chunked_distillation_loss`` against the full-logits path of the same trainer, on tiny real models: the loss and
every trainable gradient agree, with a LoRA adapter (and its dropout) on ``lm_head``, a ``modules_to_save`` head, a tied
head, a vocab-parallel TP head, the families' head transforms, a composite's image rows and in eval mode. A batch the
default path refuses is refused here too, and the default path's logits plane is weighed at construction."""

import ast
import dataclasses
import inspect
import json
import logging
import math
import textwrap
import types

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import DistributedConfig

import halo.kernels.logprobs as logprob_kernels
import halo.trainers.distillation.teacher_distillation as teacher_distillation
import halo.trainers.full_logits_memory as full_logits_memory
from halo.configs.distillation import TeacherDistillationConfig
from halo.distributed.fsdp import apply_fsdp2_for_tp
from halo.distributed.mesh import build_dp_tp_mesh, tp_submesh
from halo.distributed.tensor_parallel import hf_plan
from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import apply_vision_options
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.parallel import tp as tp_loading
from halo.models.head_transform import IDENTITY_HEAD_TRANSFORM
from halo.models.introspection.module_tree import embedding_module
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from halo.trainers.head_projection import resolve_output_embedding
from halo.trainers.hidden_state_forward import reject_backbone_without_image_inputs
from tests.common.composite_rows import composite_model_and_rows
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_INKLING_CONFIG, TINY_MINICPM3_CONFIG, TINY_QWEN3_CONFIG
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    build_tiny_family_checkpoint,
    tiny_family_model,
)

TEMPERATURE = 2.0
# Several chunks per batch and a remainder: 22 supervised positions at 4 rows a chunk.
CHUNK_ROWS = 4
TINY = {"num_hidden_layers": 2}
TIED = {**TINY, "tie_word_embeddings": True}
TP = 2
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


def test_a_modules_to_save_head_matches_the_full_logits_path():
    """PEFT trains a copy of the head, set apart here from the frozen original, so the projection must read the copy."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, overrides=TINY)
    config = LoraConfig(r=4, lora_dropout=0.25, target_modules=["q_proj", "v_proj"], modules_to_save=["lm_head"])
    model = get_peft_model(model, config)
    with torch.no_grad():
        copy = model.get_output_embeddings().modules_to_save["default"].weight
        copy.add_(0.5 * torch.randn_like(copy))
    trainer = _trainer(model.to(torch.float64).train(), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    grads = _assert_paths_agree(trainer, _batch(trainer.model.config.vocab_size, RAGGED))
    head_copy = [name for name in grads if "lm_head.modules_to_save" in name]
    assert head_copy and all(grads[name].abs().sum() > 0 for name in head_copy)


def test_an_evaluation_step_matches_the_full_logits_path():
    """Eval mode turns the LoRA dropout off, and the row padding an eval split's final round leaves both losses."""
    trainer = _trainer(
        _student(TINY_DENSE_FAMILY, peft=True), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5, apply_hard_labels=True
    )
    trainer.model.eval()
    batch = _batch(trainer.model.config.vocab_size, RAGGED)

    def evaluate(*, chunked: bool, real_rows: int):
        trainer.args.use_chunked_distillation_loss = chunked
        trainer.eval_split_rows = lambda rows: real_rows
        with torch.no_grad():
            return trainer.compute_loss(trainer.model, dict(batch)), dict(trainer.stored)

    full_loss, full_metrics = evaluate(chunked=False, real_rows=1)
    loss, metrics = evaluate(chunked=True, real_rows=1)
    torch.testing.assert_close(loss, full_loss, rtol=FP32_RTOL, atol=1e-9)
    assert metrics.keys() == full_metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(value, full_metrics[key], rtol=FP32_RTOL, atol=1e-9)
    padded_loss, _ = evaluate(chunked=True, real_rows=2)
    assert (padded_loss - loss).abs() > 1e-6, "the padding row must move the loss, or the comparison sees nothing"


def test_a_tied_head_matches_the_full_logits_path():
    """The head and the input embedding share one parameter, whose gradient sums both uses on either path."""
    student = _student(TINY_DENSE_FAMILY, peft=False, overrides={"tie_word_embeddings": True})
    assert student.get_output_embeddings().weight is student.get_input_embeddings().weight
    trainer = _trainer(student, _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    batch = _batch(student.config.vocab_size, RAGGED)
    grads = _assert_paths_agree(trainer, batch)
    shared = grads["model.embed_tokens.weight"]
    # Rows of tokens the batch never feeds take their gradient from the head alone.
    unseen = torch.ones(student.config.vocab_size, dtype=torch.bool)
    unseen[batch["input_ids"].unique()] = False
    assert shared[unseen].abs().sum(-1).gt(0).all()


def _tp_worker(rank: int, checkpoint: str, dp: int, out_path: str) -> None:
    """Both paths on a student loaded under HF-native TP (``dp > 1``: also FSDP2-wrapped), its tied head one
    vocab-parallel DTensor. Writes the loss and each parameter's relative gradient error between the paths."""
    logprob_kernels.LOGIT_CHUNK_ELEMENTS = CHUNK_ROWS * TINY_QWEN3_CONFIG["vocab_size"]
    mesh = build_dp_tp_mesh(TP, dp, device_type="cpu")
    with tp_loading._toolkit_tp_mesh(mesh, tp_submesh(mesh)):
        student = TINY_DENSE_FAMILY.load_class.from_pretrained(
            checkpoint, distributed_config=DistributedConfig(tp_plan="auto", tp_size=TP), dtype=torch.float64
        )
    finalize_loaded_model(student, weights_dir=checkpoint)
    hf_plan.finish_hf_tp_load(student, mesh, TP)
    head = student.get_output_embeddings().weight
    tied = head is student.get_input_embeddings().weight
    vocab_parallel = isinstance(head, DTensor) and head.to_local().size(0) < head.size(0)
    if dp > 1:
        apply_fsdp2_for_tp(student, mesh, compute_dtype=torch.float64, fp32_reduce=False, dp_size=dp)
    trainer = _trainer(student.train(), _teacher(TINY_DENSE_FAMILY), distill_alpha=0.5)
    batch = _batch(TINY_QWEN3_CONFIG["vocab_size"], RAGGED)

    def full(grads):
        return {name: grad.full_tensor() if isinstance(grad, DTensor) else grad for name, grad in grads.items()}

    full_loss, full_grads, _ = _step(trainer, batch, chunked=False)
    loss, grads, _ = _step(trainer, batch, chunked=True)
    full_grads, grads = full(full_grads), full(grads)
    errors = {
        name: float((grads[name] - full_grads[name]).norm() / full_grads[name].norm().clamp_min(1e-12))
        for name in full_grads
    }
    result = {
        "tied": tied,
        "vocab_parallel": vocab_parallel,
        "loss": float(loss),
        "full_loss": float(full_loss),
        "gradients": sorted(grads),
        "full_gradients": sorted(full_grads),
        "errors": errors,
    }
    with open(f"{out_path}.{rank}", "w") as fh:
        json.dump(result, fh)


@pytest.mark.parametrize("dp", [1, 2], ids=["tp2", "dp2_tp2"])
def test_a_vocab_parallel_tp_head_matches_the_full_logits_path(dp, tmp_path):
    """Under HF-native TP the tied head is sharded on the vocabulary, and the projection reads it whole: a local shard,
    or a gradient summed over the TP group in place of sliced, moves the loss or the shared gradient."""
    checkpoint = str(tmp_path / "student")
    build_tiny_family_checkpoint(TINY_DENSE_FAMILY, checkpoint, overrides=TIED)
    out = str(tmp_path / "result")
    run_gloo_ranks(_tp_worker, dp * TP, checkpoint, dp, out)
    for rank in range(dp * TP):
        with open(f"{out}.{rank}") as fh:
            result = json.load(fh)
        assert result["tied"] and result["vocab_parallel"], f"rank {rank}: the head is not a vocab-parallel tie"
        assert result["gradients"] == result["full_gradients"], f"rank {rank}"
        assert "model.embed_tokens.weight" in result["errors"]
        assert result["loss"] == pytest.approx(result["full_loss"], rel=FP32_RTOL), f"rank {rank}"
        worst = {
            name: error
            for name, error in result["errors"].items()
            if not math.isfinite(error) or error >= 10 * FP32_RTOL
        }
        assert not worst, f"rank {rank}: relative gradient errors {worst}"


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


def test_a_batch_holding_no_image_is_refused_on_both_paths(monkeypatch):
    """A rank whose batch holds no image while the tower trains under data parallelism would hang in the reduce, so
    the wrapper's forward pre-hook refuses it; the chunked path reaches that forward too. A frozen input embedding
    with checkpointing off is a setup the reduces cannot pair in. One process stands in for the data-parallel axis,
    whose mesh is all the refusal reads of it."""
    torch.manual_seed(0)
    student, image_row, text_row = composite_model_and_rows("qwen3_5")
    torch.manual_seed(0)
    teacher, _, _ = composite_model_and_rows("qwen3_5")
    embedding_module(student, "input").requires_grad_(False)
    apply_vision_options(
        types.SimpleNamespace(freeze_vision_tower=False, min_pixels=None, max_pixels=None),
        student,
        types.SimpleNamespace(use_peft=False, lora_target_modules=None),
        None,
        is_vlm=True,
    )
    wide_mesh = types.SimpleNamespace(mesh_info=types.SimpleNamespace(mesh=types.SimpleNamespace(size=lambda: 2)))
    monkeypatch.setattr(data_setup, "fsdp2_param_groups", lambda model: [wide_mesh])
    trainer = _trainer(student, teacher.eval().requires_grad_(False), distill_alpha=0.5)
    for chunked in (False, True):
        trainer.args.use_chunked_distillation_loss = chunked
        with pytest.raises(ValueError, match="This rank's batch holds no image"):
            trainer.compute_loss(student, dict(text_row))
        assert trainer.compute_loss(student, dict(image_row)).isfinite()


_GIB = 1024**3


def _with_head_rows(rows: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(vocab_size=rows))
    )


def _memory_host(
    *,
    chunked: bool = False,
    max_length: int | None = 1024,
    precision: str = "bf16",
    student_rows: int = 2**17,
    teacher_rows: int = 2**17,
):
    """A trainer's state the full-logits check reads: in bf16, 2 rows × 1024 tokens × 2**17 vocab × 3 planes × 2 B is
    1.5 GiB."""
    return types.SimpleNamespace(
        model=_with_head_rows(student_rows),
        teacher_model=_with_head_rows(teacher_rows),
        args=types.SimpleNamespace(
            use_chunked_distillation_loss=chunked,
            max_length=max_length,
            bf16=precision == "bf16",
            fp16=precision == "fp16",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            eval_strategy="no",
            eval_on_start=False,
        ),
        accelerator=types.SimpleNamespace(device="cuda:3", unwrap_model=lambda model: model),
    )


@pytest.fixture
def free_memory(monkeypatch):
    """Pretend a CUDA device with ``free`` bytes available; records the device each read asked about."""
    asked: list = []

    def _set(free: int):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: asked.append(device) or (free, 8 * free))
        return asked

    return _set


def test_a_full_logits_plane_within_half_the_free_memory_stays_quiet(free_memory, caplog):
    asked = free_memory(3 * _GIB)
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedTeacherDistillationTrainer._warn_if_full_logits_large(_memory_host())
    assert not caplog.records
    assert asked == ["cuda:3"], "free memory must be read on the trainer's own device"


@pytest.mark.parametrize(
    ("precision", "bytes_per_logit"), [("bf16", 6), ("fp16", 6), ("fp32", 12)], ids=["bf16", "fp16", "fp32"]
)
def test_a_full_logits_plane_past_half_the_free_memory_warns(free_memory, caplog, precision, bytes_per_logit):
    """The three planes are in the run's precision, so an fp32 run weighs twice a bf16 one."""
    plane_gib = 1.5 * bytes_per_logit / 6
    free_memory(int(plane_gib / 0.75 * _GIB))
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedTeacherDistillationTrainer._warn_if_full_logits_large(_memory_host(precision=precision))
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and "75%" in messages[0], messages
    assert f"× {bytes_per_logit} B = {plane_gib:.1f} GiB" in messages[0], messages
    assert "use_chunked_distillation_loss: true" in messages[0] and "max_length" in messages[0]


def test_a_full_logits_plane_past_the_free_memory_only_warns(free_memory, caplog, monkeypatch):
    """``max_length`` is a cap: the context window when unset, while the rows pad only to the batch's longest. The
    warning takes no collective."""
    free_memory(_GIB)
    monkeypatch.setattr(full_logits_memory, "rank_consensus", lambda ok: pytest.fail("a warning takes no collective"))
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedTeacherDistillationTrainer._warn_if_full_logits_large(_memory_host())
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and "more than the 1.0 GiB" in messages[0], messages
    assert "use_chunked_distillation_loss: true" in messages[0]
    assert messages[0].endswith(
        "max_length is only an upper limit: a batch whose longest row is shorter builds a smaller plane."
    )


@pytest.mark.parametrize(
    ("student_rows", "teacher_rows"), [(2**17, 2**16), (2**16, 2**17)], ids=["student", "teacher"]
)
def test_the_larger_head_sets_the_weighed_vocabulary(free_memory, caplog, student_rows, teacher_rows):
    """Each model's forward returns all its head rows, more than the compared width when the two differ."""
    free_memory(2 * _GIB)
    with caplog.at_level(logging.WARNING, logger=full_logits_memory.__name__):
        DistributedTeacherDistillationTrainer._warn_if_full_logits_large(
            _memory_host(student_rows=student_rows, teacher_rows=teacher_rows)
        )
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and f"× {2**17} vocab" in messages[0], messages


@pytest.mark.parametrize(
    "host", [_memory_host(chunked=True), _memory_host(max_length=None)], ids=["chunked", "no_max_length"]
)
def test_the_chunked_path_or_an_unbounded_width_skip_the_check(free_memory, host):
    free_memory(1)
    DistributedTeacherDistillationTrainer._warn_if_full_logits_large(host)


def _call_lines(fn, name: str) -> list[int]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and name in (getattr(node.func, "attr", None), getattr(node.func, "id", None))
    ]


def test_construction_weighs_the_plane_after_the_teacher_is_placed():
    """The free memory read must already exclude the teacher's weights."""
    checks = _call_lines(DistributedTeacherDistillationTrainer.__init__, "_warn_if_full_logits_large")
    placements = _call_lines(DistributedTeacherDistillationTrainer.__init__, "place_frozen_scorer")
    assert checks and placements and min(checks) > max(placements)


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
        trainer._reject_unchunkable_loss()
    trainer.compute_metrics = None
    trainer._reject_unchunkable_loss()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
