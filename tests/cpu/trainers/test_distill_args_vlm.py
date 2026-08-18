#!/usr/bin/env python
"""DistillScriptArguments must declare the fields the VLM distillation path uses.

scripts/training/distillation/teacher_distill.py, on the VLM branch (is_vlm=True),
dereferences these argument attributes:

  * args.system_prompt              (via _build_vlm_sft_processors)
  * args.model_supports_system_role  (via _build_vlm_sft_processors)
  * args.images_field               (dataset image column)
  * args.assistant_message_template (VLMDataCollator response template)
  * args.train_on_completions_only  (collator masking)
  * args.interleaved_thinking       (text-only flag; the VLM path fail-louds on it)

DistillScriptArguments extends CommonScriptArguments (NOT SFTScriptArguments), which
declares none of them: without these declarations, constructing DistillScriptArguments and reading
these attributes raises AttributeError -> the VLM teacher-distill path crashes at
dataset-prep time. This test pins that every field resolves, with defaults matching the
SFT/VLM-collator semantics so the text path is unaffected.

Run: python tests/cpu/trainers/test_distill_args_vlm.py  (or pytest)
"""

import pytest

from src.args.distill_args import DistillScriptArguments


def test_distill_args_vlm_field_defaults():
    args = DistillScriptArguments(teacher_model="dummy/teacher")
    # system_prompt off by default; model supports system role; thinking not preserved.
    assert args.system_prompt is None
    assert args.model_supports_system_role is True
    assert args.interleaved_thinking is False
    assert args.images_field is None
    # train_on_completions_only defaults on (mask the prompt); the assistant template has no default —
    # no marker fits every chat template, so the run must name the one its model renders.
    assert args.train_on_completions_only is True
    assert args.assistant_message_template is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
