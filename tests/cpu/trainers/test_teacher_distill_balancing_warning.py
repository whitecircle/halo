"""Teacher distillation against the student's own checkpoint warns on a resolved bias update."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest

from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import script as script_entry
from halo.entrypoints.script import ScriptContext, configure_balancing_and_build_callbacks
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from tests.common.repo import load_script_module

_WARNING = "teacher_model is the student's own checkpoint"


def _warns(requested: str, resolved: str, *, teacher_is_student: bool, caplog) -> bool:
    """Build the callbacks with balancing resolving ``requested`` to ``resolved``."""
    context = ScriptContext(
        parallelism_config=ParallelismConfig(),
        local_rank=0,
        resume_checkpoint=None,
        model_source="model",
        trainer_cls=DistributedTeacherDistillationTrainer,
    )
    args = SimpleNamespace(moe_balancing=requested, gradient_checkpointing_plan="off")
    caplog.clear()
    with (
        mock.patch.object(script_entry, "configure_moe_balancing", return_value=resolved),
        mock.patch.object(script_entry, "build_run_callbacks", return_value=[]),
        caplog.at_level(logging.WARNING),
    ):
        configure_balancing_and_build_callbacks(
            args, "config", "model", context, teacher_is_student_checkpoint=teacher_is_student
        )
    return _WARNING in caplog.text


@pytest.mark.parametrize("mode", ["bias_update", "bias_update_transient"])
def test_a_bias_update_against_the_students_own_checkpoint_warns(mode, caplog):
    assert _warns(mode, mode, teacher_is_student=True, caplog=caplog)


@pytest.mark.parametrize("mode", ["none", "aux_loss"])
def test_modes_that_leave_the_router_biases_alone_do_not_warn(mode, caplog):
    assert not _warns(mode, mode, teacher_is_student=True, caplog=caplog)


def test_a_different_teacher_does_not_warn(caplog):
    assert not _warns("bias_update", "bias_update", teacher_is_student=False, caplog=caplog)


class _Stop(Exception):
    """Raised by the stubbed callback builder, so ``main()`` runs only as far as it."""


# Everything ``main()`` calls before the callback builder, stubbed so only its wiring runs.
_MAIN_STEPS = (
    "parse_script_args",
    "init_training_script",
    "load_script_datasets",
    "resolve_vlm_run",
    "_load_teacher_metadata",
    "_reject_neutralized_teacher_sinks",
    "default_padded_attn_implementation",
    "reject_fa4_varlen_request",
    "rows_reach_varlen_kernel",
    "evaluation_batch_size",
    "resolve_training_dtype",
    "SinksPolicy",
    "load_policy_model",
    "apply_max_length",
    "install_resolved_tokenizer",
    "apply_text_path_padding_side",
    "configure_adapter_training",
    "apply_vision_options",
    "resolve_shared_vocab_width",
    "reject_topk_covering_vocab",
    "_load_distill_teacher",
    "_teacher_takes_image_class",
    "log_model_info",
    "prepare_chat_text_split",
    "_text_distill_collator",
    "log_script_dataset_examples",
    "configure_balancing_and_build_callbacks",
)


@pytest.mark.parametrize(("teacher", "expected"), [("Zyphra/ZAYA1-8B", True), ("Qwen/Qwen3-30B-A3B", False)])
def test_the_script_tells_the_builder_whether_the_teacher_is_the_student(teacher, expected):
    module = load_script_module("scripts/training/distillation/teacher_distill.py", "halo_test_teacher_distill_wiring")
    args = mock.MagicMock(teacher_model=teacher)
    model_config = SimpleNamespace(model_name_or_path="Zyphra/ZAYA1-8B")
    with mock.patch.multiple(module, **dict.fromkeys(_MAIN_STEPS, mock.DEFAULT)) as stubs:
        stubs["parse_script_args"].return_value = (args, mock.MagicMock(), model_config, None)
        stubs["load_script_datasets"].return_value = (None, False)
        stubs["resolve_vlm_run"].return_value = False
        stubs["_load_teacher_metadata"].return_value = (None, None)
        stubs["prepare_chat_text_split"].return_value = (None, None)
        stubs["configure_balancing_and_build_callbacks"].side_effect = _Stop
        with pytest.raises(_Stop):
            module.main()
    assert (
        stubs["configure_balancing_and_build_callbacks"].call_args.kwargs["teacher_is_student_checkpoint"] is expected
    )


@pytest.mark.parametrize(
    ("teacher", "student", "same"),
    [
        ("Zyphra/ZAYA1-8B", "Zyphra/ZAYA1-8B", True),
        ("./ckpt/", "ckpt", True),
        ("/models/a/../b", "/models/b", True),
        ("Zyphra/ZAYA1-8B", "Qwen/Qwen3-30B-A3B", False),
    ],
)
def test_the_script_compares_the_two_checkpoints_after_normalizing_paths(teacher, student, same):
    module = load_script_module(
        "scripts/training/distillation/teacher_distill.py", "halo_test_teacher_distill_balancing_warning"
    )
    args = SimpleNamespace(teacher_model=teacher)
    model_config = SimpleNamespace(model_name_or_path=student)
    assert module._teacher_is_student_checkpoint(args, model_config) is same


def test_a_relative_student_path_matches_its_absolute_teacher_spelling(tmp_path, monkeypatch):
    module = load_script_module(
        "scripts/training/distillation/teacher_distill.py", "halo_test_teacher_distill_balancing_warning"
    )
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(teacher_model=str(tmp_path / "ckpt"))
    model_config = SimpleNamespace(model_name_or_path="ckpt")
    assert module._teacher_is_student_checkpoint(args, model_config) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
