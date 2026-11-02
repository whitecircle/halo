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
  backend must be resolved under the SAME ``SinksPolicy`` the student was loaded with.
* Student and teacher load on ONE padded-workload request, decided on both checkpoints, so the two
  forwards compared token by token resolve from the same label.
* A student loaded on FA4 whose padded batches reach FA4's varlen kernels, which a softcap stays off (Gemma 2 at
  more than one row per batch), is refused before the teacher loads, not in the first step.

Run: python tests/cpu/trainers/test_teacher_distill_attn_policy.py  (or pytest)
"""

import contextlib
import functools
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import (
    AutoModelForImageTextToText,
    Gemma2Config,
    GemmaConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    Qwen3Config,
)

import scripts.training.distillation.teacher_distill as distill_script
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import model_setup
from halo.models.attention import selection
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from tests.common.frozen_loader import STUB_CONFIG, STUB_RESOLVED_ATTN, captured_load, stub_frozen_loader

PINNED_SHA = "970cfc9f5e7e5a4f5f6f0645955928a9b6a98415"

LOCAL_RANK = 3


class _RemoteCodeConfig(PretrainedConfig):
    """A config no Auto mapping knows, as a remote-code teacher's is (Bailing's)."""

    model_type = "halo_test_remote_code"


def _script_teacher_load_capturing(
    *,
    revision=None,
    reset_sinks=True,
    is_vlm=False,
    bf16=True,
    fp16=False,
    requested_attn="sdpa",
    trust_remote_code=False,
):
    """Run the shipped script's teacher load with the shared loader's hub calls stubbed."""
    args = types.SimpleNamespace(teacher_model="org/teacher", teacher_model_revision=revision, reset_sinks=reset_sinks)
    model_config = types.SimpleNamespace(trust_remote_code=trust_remote_code)
    training_config = types.SimpleNamespace(bf16=bf16, fp16=fp16)
    with stub_frozen_loader() as caps:
        teacher = distill_script._load_distill_teacher(
            args=args,
            model_config=model_config,
            training_config=training_config,
            requested_attn=requested_attn,
            is_vlm=is_vlm,
            local_rank=LOCAL_RANK,
        )
    captured = captured_load(caps, is_vlm=is_vlm)
    assert teacher is captured.model
    return captured


def test_a_teacher_path_is_refused_so_the_trainer_cannot_load_one_itself():
    """The trainer must not hold a second frozen-teacher loader.

    A trainer-local loader resolves no backend (auto-detect), applies no VLM device placement and
    reads the pin off a ctor knob no script passes — a silently different teacher from the shipped
    path's. A path handed to the trainer has to fail at construction, not be fetched.
    """
    with pytest.raises(TypeError, match="already-loaded module"):
        DistributedTeacherDistillationTrainer(
            student_model="org/student",
            teacher_model="org/teacher",
            teacher_tokenizer=MagicMock(spec=PreTrainedTokenizerBase),
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
    caps = _script_teacher_load_capturing(revision=PINNED_SHA)
    caps.fetch_scope.assert_called_once_with("org/teacher", PINNED_SHA, tag="teacher_model")


def test_script_teacher_backend_is_resolved_against_the_teacher_config():
    """The per-family kernel limits and the fp32/FlashAttention guard are the TEACHER's, not the student's.

    The padded-workload SDPA default is only a *request*; skipping the resolver would hand an fp32 run
    the student's pinned FlashAttention, and a DeepSeek-V4 / Gemma-4 teacher a backend it cannot run.
    """
    caps = _script_teacher_load_capturing(requested_attn="flash_attention_4", bf16=False, fp16=False)
    assert caps.resolver["model_config"] == STUB_CONFIG
    assert caps.resolver["dtype"] is torch.float32
    assert caps.resolver["attn_implementation"] == "flash_attention_4"
    assert caps.load["attn_implementation"] == STUB_RESOLVED_ATTN, "the resolved backend, not the request"


class _StopAtTeacherLoad(Exception):
    """Raised by the stubbed teacher load: main() runs only as far as the two requests under test."""


def _main_requests(
    tmp_path, *, student_config, teacher_config, blackwell=True, yaml_body="", student=None, requests=None
):
    """The (student, teacher) attention requests the shipped ``main()`` hands its two loads, on a host that is
    Blackwell when ``blackwell`` is set, for checkpoints with these configs; ``student`` is the loaded model, and
    ``requests`` collects each request as its load is reached."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: org/student\nteacher_model: org/teacher\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{yaml_body}"
    )
    runtime = types.SimpleNamespace(parallelism_config=ParallelismConfig(), local_rank=0)
    requests = {} if requests is None else requests
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

    def fake_policy_load(context, training_config, model_config, args, *, vlm_run, attn_default):
        requests["student"] = attn_default
        return types.SimpleNamespace(
            model=student or MagicMock(), tokenizer=tokenizer, processing_class=tokenizer, is_vlm=False
        )

    def fake_teacher_load(*, requested_attn, **_):
        requests["teacher"] = requested_attn
        raise _StopAtTeacherLoad

    stubs = {
        "init_training_script": MagicMock(return_value=runtime),
        "load_script_datasets": MagicMock(return_value=(None, False)),
        "resolve_vlm_run": MagicMock(return_value=False),
        "_load_teacher_metadata": MagicMock(return_value=(tokenizer, teacher_config)),
        "load_policy_model": fake_policy_load,
        "apply_max_length": MagicMock(return_value=tokenizer),
        "configure_adapter_training": MagicMock(return_value=None),
        "resolve_shared_vocab_width": MagicMock(),
        "_load_distill_teacher": fake_teacher_load,
    }
    with contextlib.ExitStack() as stack:
        for name, stub in stubs.items():
            stack.enter_context(patch.object(distill_script, name, stub))
        stack.enter_context(patch.object(selection, "is_blackwell_gpu", lambda: blackwell))
        stack.enter_context(patch.object(model_setup, "fetch_checkpoint_config", return_value=student_config))
        stack.enter_context(patch("halo.entrypoints.script.install_log_tee"))
        stack.enter_context(patch.object(sys, "argv", ["prog", str(config)]))
        stack.enter_context(pytest.raises(_StopAtTeacherLoad))
        distill_script.main()
    return requests["student"], requests["teacher"]


def _softcapped() -> Gemma2Config:
    """Gemma 2 27B's attention: 128-wide heads, which FA4's general kernel runs, and a logit softcap."""
    return Gemma2Config(hidden_size=4608, num_attention_heads=32, num_key_value_heads=16, head_dim=128)


_NATIVE, _REMOTE, _HEAD_DIM_256, _SOFTCAPPED = Qwen3Config(), _RemoteCodeConfig(), GemmaConfig(), _softcapped()


@pytest.mark.parametrize(
    ("student_config", "teacher_config", "expected"),
    [
        (_NATIVE, _NATIVE, None),
        (_NATIVE, _REMOTE, "sdpa"),
        (_REMOTE, _NATIVE, "sdpa"),
        (_NATIVE, _HEAD_DIM_256, None),
        (_HEAD_DIM_256, _NATIVE, None),
        (_NATIVE, _SOFTCAPPED, "sdpa"),
        (_SOFTCAPPED, _NATIVE, "sdpa"),
    ],
    ids=[
        "native-native",
        "native-remote",
        "remote-native",
        "native-hd256",
        "hd256-native",
        "native-softcapped",
        "softcapped-native",
    ],
)
def test_student_and_teacher_load_on_one_request_decided_on_both_checkpoints(
    student_config, teacher_config, expected, tmp_path
):
    """On Blackwell the request is the loader's own pick only when neither checkpoint rules it out, and both
    loads take it: a request decided on one checkpoint alone would hand the other model a label its class
    cannot be read for (remote code) or a kernel its padded batches cannot run (a softcap on FA4's varlen
    kernels, a forward-only teacher's included)."""
    requests = _main_requests(tmp_path, student_config=student_config, teacher_config=teacher_config)
    assert requests == (expected, expected)


@pytest.mark.parametrize(
    ("blackwell", "yaml_body", "expected"),
    [(False, "", "sdpa"), (True, "attn_implementation: eager\n", "eager"), (True, "reset_sinks: false\n", None)],
    ids=["off-blackwell", "pinned", "live-sinks"],
)
def test_the_shared_request_keeps_the_padded_default_rules(blackwell, yaml_body, expected, tmp_path):
    requests = _main_requests(
        tmp_path, student_config=_NATIVE, teacher_config=_NATIVE, blackwell=blackwell, yaml_body=yaml_body
    )
    assert requests == (expected, expected)


@pytest.mark.parametrize(("batch_size", "refused"), [(2, True), (1, False)], ids=["padded-bs2", "unpadded-bs1"])
def test_a_softcapped_student_on_fa4_is_refused_on_padded_rows_before_any_weight_loads(batch_size, refused, tmp_path):
    """More than one row per batch pads, and a padded batch reaches FA4's varlen kernels, which a softcap stays off: a
    pinned FA4 the student load would keep is refused before the student loads; one unpadded row per batch runs
    FA4's dense kernel and reaches both loads."""
    run = functools.partial(
        _main_requests,
        tmp_path,
        student_config=_SOFTCAPPED,
        teacher_config=_NATIVE,
        yaml_body=f"attn_implementation: flash_attention_4\nper_device_train_batch_size: {batch_size}\n",
        student=types.SimpleNamespace(config=_SOFTCAPPED),
    )
    reached: dict = {}
    # A half-precision run, which a GPU-less TrainingArguments cannot construct: fp32 keeps FA4 off the student.
    with patch.object(distill_script, "resolve_training_dtype", return_value=torch.bfloat16):
        if refused:
            with pytest.raises(ValueError, match="which are not validated with attention logit softcapping"):
                run(requests=reached)
            assert reached == {}, "the student load was reached"
        else:
            assert run(requests=reached) == ("flash_attention_4", "flash_attention_4")


@pytest.mark.parametrize("reset_sinks", [True, False])
def test_script_teacher_resolution_mirrors_the_student_sinks_policy(reset_sinks):
    """With live sinks the resolver must reject the sink-dropping backends for the teacher too."""
    resolver = _script_teacher_load_capturing(reset_sinks=reset_sinks).resolver
    assert resolver["sinks_policy"] is SinksPolicy.from_flags(reset_sinks=reset_sinks)


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

    ``resolve_attn_implementation`` under a neutralized policy approves a sink-DROPPING backend (sdpa) on
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
