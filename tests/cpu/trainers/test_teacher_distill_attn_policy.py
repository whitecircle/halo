#!/usr/bin/env python
"""CPU contract tests for the teacher load in off-policy distillation: revision pin, dtype, attention.

The teacher only runs forward, but its logits ARE the distillation target, so every knob the user
declares has to reach the hub fetch:

* ``teacher_model_revision`` pins the TEACHER repo (the student's ``model_revision`` names a commit in
  a different repo). Unpinned, a run silently trains against whatever ``main`` serves that day.
  The script owns that fetch — the trainer refuses a teacher path outright.
* The teacher runs in the run's own precision — an fp32 run compared against a bf16 teacher fits
  rounded targets.
* ``trust_remote_code`` is the run's, on both hub calls: a remote-code teacher is refused at the
  config fetch without it, and no teacher should import remote code the run did not opt into.
* On a live-sinks model (``reset_sinks: false``, the GptOss RL flow) a sink-dropping backend shifts
  every teacher logprob by nats while the student is restricted to sink-carrying impls, so the
  backend must be resolved under the SAME ``sinks_reset`` policy the student was loaded with.

Run: python tests/cpu/trainers/test_teacher_distill_attn_policy.py  (or pytest)
"""

import types
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForImageTextToText, PreTrainedTokenizerBase

import scripts.training.distillation.teacher_distill as distill_script
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from tests.common.frozen_loader import STUB_CONFIG, STUB_RESOLVED_ATTN, captured_load, stub_frozen_loader

PINNED_SHA = "970cfc9f5e7e5a4f5f6f0645955928a9b6a98415"

LOCAL_RANK = 3


def _script_teacher_load_capturing(
    *, revision=None, reset_sinks=True, is_vlm=False, bf16=True, fp16=False, pinned_attn=None, trust_remote_code=False
):
    """Run the shipped script's teacher load with the shared loader's hub calls stubbed."""
    args = types.SimpleNamespace(teacher_model="org/teacher", teacher_model_revision=revision)
    model_config = types.SimpleNamespace(attn_implementation=pinned_attn, trust_remote_code=trust_remote_code)
    training_config = types.SimpleNamespace(bf16=bf16, fp16=fp16)
    dist_args = types.SimpleNamespace(reset_sinks=reset_sinks)
    with stub_frozen_loader() as caps:
        teacher = distill_script._load_distill_teacher(
            args=args,
            model_config=model_config,
            training_config=training_config,
            dist_args=dist_args,
            is_vlm=is_vlm,
            local_rank=LOCAL_RANK,
        )
    captured = captured_load(caps, is_vlm=is_vlm)
    assert teacher is captured.model
    return captured


def test_a_teacher_path_is_refused_so_the_trainer_cannot_load_one_itself():
    """The trainer must not re-grow a second frozen-teacher loader.

    The one it used to carry resolved no backend (auto-detect), applied no VLM device placement and
    read the pin off a ctor knob no script passed — a silently different teacher from the shipped
    path's. A path handed to the trainer has to fail at construction, not be fetched.
    """
    with pytest.raises(TypeError, match="already-loaded module"):
        DistributedDistillationTrainer(
            student_model="org/student",
            teacher_model="org/teacher",
            processing_class=MagicMock(spec=PreTrainedTokenizerBase),
        )


def test_script_teacher_load_pins_the_config_and_weight_fetch():
    """The shipped launch path: a declared revision must reach the teacher's hub fetches.

    Unpinned, the run silently trains against whatever the teacher repo's ``main`` serves, and a
    pinned checkpoint paired with hub-main's config is a second way to get the wrong teacher.
    """
    caps = _script_teacher_load_capturing(revision=PINNED_SHA, reset_sinks=False)
    assert caps.config["revision"] == PINNED_SHA, "pinned config fetch"
    assert caps.load["revision"] == PINNED_SHA, "pinned weight fetch"


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_script_teacher_load_threads_the_run_remote_code_flag(trust_remote_code):
    """The run's ``trust_remote_code`` reaches BOTH teacher hub calls.

    A remote-code teacher (its architecture is the whole point of distilling from it) fails at the
    CONFIG fetch without the flag, which is also the fetch that imports the modeling file — so the
    two must carry the same value. Hardcoding it either way is the other failure: off refuses a repo
    the run opted into, on imports unreviewed code from one it did not.
    """
    caps = _script_teacher_load_capturing(trust_remote_code=trust_remote_code)
    assert caps.config["trust_remote_code"] is trust_remote_code, "the config fetch imports the modeling file"
    assert caps.load["trust_remote_code"] is trust_remote_code, "the weight fetch"


@pytest.mark.parametrize("unset", [None, ""])
def test_script_teacher_load_treats_an_empty_pin_as_unset(unset):
    """An empty string is an unset pin, not a branch name to send to the hub."""
    caps = _script_teacher_load_capturing(revision=unset)
    assert caps.config["revision"] is None
    assert caps.load["revision"] is None


def test_script_vlm_teacher_load_pins_the_weight_fetch():
    caps = _script_teacher_load_capturing(revision=PINNED_SHA, is_vlm=True)
    assert caps.load["revision"] == PINNED_SHA
    assert caps.load_positional[0] is AutoModelForImageTextToText, "the VLM teacher class must be pinned"
    # Only the VLM teacher is placed on the local device; the text teacher lands in host memory and
    # the trainer moves it, so a device_map leaking onto that branch would change where it loads.
    assert caps.load["device_map"] == {"": LOCAL_RANK}


def test_script_text_teacher_load_takes_no_device_map():
    assert _script_teacher_load_capturing().load["device_map"] is None


def test_script_teacher_fetch_is_coordinated_main_rank_first():
    """8 ranks fetching the same teacher repo at once is the download this tag exists to serialize."""
    caps = _script_teacher_load_capturing()
    caps.fetch_scope.assert_called_once_with("teacher_model")


def test_script_teacher_backend_is_resolved_against_the_teacher_config():
    """The per-family kernel limits and the fp32/FlashAttention guard are the TEACHER's, not the student's.

    The padded-workload SDPA default is only a *request*; skipping the resolver would hand an fp32 run
    the student's pinned FlashAttention, and a DeepSeek-V4 / Gemma-4 teacher a backend it cannot run.
    """
    caps = _script_teacher_load_capturing(pinned_attn="flash_attention_4", bf16=False, fp16=False)
    assert caps.resolver["model_config"] == STUB_CONFIG
    assert caps.resolver["dtype"] is torch.float32
    assert caps.resolver["attn_implementation"] == "flash_attention_4"
    assert caps.load["attn_implementation"] == STUB_RESOLVED_ATTN, "the resolved backend, not the request"


def test_script_teacher_defaults_to_the_padded_workload_backend():
    """Nothing pinned: SDPA is requested (FA4 runs padded batches through its slow varlen path)."""
    assert _script_teacher_load_capturing().resolver["attn_implementation"] == "sdpa"


@pytest.mark.parametrize("reset_sinks", [True, False])
def test_script_teacher_resolution_mirrors_the_student_sinks_policy(reset_sinks):
    """With live sinks the resolver must reject the sink-dropping backends for the teacher too."""
    assert _script_teacher_load_capturing(reset_sinks=reset_sinks).resolver["sinks_reset"] is reset_sinks


@pytest.mark.parametrize(
    ("bf16", "fp16", "expected"),
    [(True, False, torch.bfloat16), (False, True, torch.float16), (False, False, torch.float32)],
)
def test_script_teacher_load_uses_the_run_precision(bf16, fp16, expected):
    """The teacher's logits are the target the student is fit to — it must not assume bf16."""
    caps = _script_teacher_load_capturing(bf16=bf16, fp16=fp16)
    assert caps.load["dtype"] is expected


@pytest.mark.parametrize("is_vlm", [False, True])
@pytest.mark.parametrize("reset_sinks", [True, False])
def test_script_teacher_load_applies_the_sinks_policy_it_resolved_under(reset_sinks, is_vlm):
    """Resolving the backend is only half the policy — the sinks themselves have to be applied.

    ``resolve_attn_implementation(..., sinks_reset=True)`` approves a sink-DROPPING backend (sdpa) on
    the premise that the caller then resets the sinks. A loader that resolves under that premise and
    never applies it leaves a GptOss teacher running sdpa over live sinks, shifting every teacher
    logprob — the distillation target — by nats with nothing logged. This is the shipped path: the
    script preloads the teacher — the only path there is — so nothing else can cover it.
    """
    caps = _script_teacher_load_capturing(reset_sinks=reset_sinks, is_vlm=is_vlm)

    applied, skipped = (caps.reset_sinks, caps.freeze_sinks) if reset_sinks else (caps.freeze_sinks, caps.reset_sinks)
    applied.assert_called_once()
    skipped.assert_not_called()
    # The policy must land on the model that is actually returned, against the TEACHER's config.
    assert applied.call_args.args[0] is caps.model
    assert applied.call_args.args[1] == STUB_CONFIG


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
