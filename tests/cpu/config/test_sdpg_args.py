#!/usr/bin/env python
"""Parse-time validation of the SDPG argument block (``SDPGArguments``) on both OPD arms.

The hint template is a ``str.format`` string whose placeholders each arm's formatter fills from a
fixed set: the on-policy trainer the gold answer alone, the self-distillation collator a reference
solution too. A placeholder outside that set would raise ``KeyError`` at the first teacher prompt,
after the model load, and an unbounded numeric knob would NaN or invert the OPD term; both are
refused at parse time. A null template is refused naming the field — except beside RLVR's closed
``use_sdpg`` gate, where the script's gate refusal names it as a value set beside the gate.

Run: pytest tests/cpu/config/test_sdpg_args.py
"""

import math
import re
import sys
from unittest import mock

import pytest
import torch
from accelerate import PartialState

from src.args.mixins import PRIVILEGED_HINT_TEMPLATE, SDPGArguments, format_field_names
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.data.collators.self_distill import inject_privileged_hint
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from tests.common.utils import load_script_module

PartialState()  # the RLVR script logs through accelerate, which refuses to log without it

ONLINE_ARMS = (SDPGArguments, RLVROnlineGRPOScriptArguments)
ALL_ARMS = (*ONLINE_ARMS, SelfDistillationArguments)


def _placeholders(names) -> str:
    return " ".join(f"{{{name}}}" for name in sorted(names))


class _CharTokenizer:
    """One id per character, so a teacher prompt decodes back to the exact hint text."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [ord(c) for c in text]


def _online_teacher_hint(template: str, answer: str) -> str:
    """The hint text the on-policy trainer's own formatter appends to a one-token prompt."""
    trainer = object.__new__(DistributedSDPGTrainer)
    trainer.processing_class = _CharTokenizer()
    trainer.sdpg_hint_template = template
    trainer.sdpg_answer_field = "answer"
    trainer._warned_missing_answer = set()
    prompt_ids = torch.tensor([[ord("Q")]])
    ids, mask = trainer._build_teacher_prompts(prompt_ids, torch.ones_like(prompt_ids), [answer])
    return "".join(chr(i) for i in ids[0][mask[0].bool()].tolist()[1:])


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
def test_the_pinned_default_template_is_valid_on_every_arm(arm):
    assert arm().sdpg_hint_template == PRIVILEGED_HINT_TEMPLATE
    assert format_field_names(PRIVILEGED_HINT_TEMPLATE) == {"answer"}


def test_each_formatter_fills_every_placeholder_its_arm_admits():
    """The declared sets are the contract the parse-time check enforces; a name added to one that its
    formatter does not supply would pass the check and raise ``KeyError`` mid-run."""
    online = _placeholders(SDPGArguments.HINT_PLACEHOLDERS)
    assert _online_teacher_hint(online, "42") == online.replace("{answer}", "42")

    offline = _placeholders(SelfDistillationArguments.HINT_PLACEHOLDERS)
    history = inject_privileged_hint([{"role": "user", "content": "Q"}], offline, answer="42", solution="6 * 7")
    assert history[-1]["content"] == "Q" + offline.replace("{answer}", "42").replace("{solution}", "6 * 7")


@pytest.mark.parametrize("arm", ONLINE_ARMS, ids=lambda c: c.__name__)
def test_the_online_arm_refuses_the_solution_placeholder(arm):
    """Only self-distillation reads a solution column; the online trainer formats ``answer=`` alone."""
    with pytest.raises(ValueError, match=r"sdpg_hint_template names \['\{solution\}'\]"):
        arm(sdpg_hint_template="{answer} {solution}")
    with pytest.raises(KeyError, match="solution"):
        _online_teacher_hint("{answer} {solution}", "42")


def test_self_distillation_admits_the_solution_placeholder():
    assert SelfDistillationArguments(sdpg_hint_template="{answer} {solution}").sdpg_hint_template


def test_the_online_trainer_refuses_it_at_construction_too():
    """The trainer rebuilds the block from its kwargs, so a directly built trainer is held to it."""
    with pytest.raises(ValueError, match="solution"):
        DistributedSDPGTrainer(sdpg_hint_template="{solution}")


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("template", "named"),
    [
        ("answer: {answr}", "{answr}"),
        ("answer: {}", "{}"),
        ("answer: {0}", "{0}"),
        ("answer: {answer.real}", "{answer.real}"),
        ("answer: {answer:{width}}", "{width}"),
    ],
)
def test_an_unfilled_placeholder_is_refused_by_name(arm, template, named):
    with pytest.raises(ValueError, match=re.escape(f"'{named}'")):
        arm(sdpg_hint_template=template)


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("template", ["answer: {answer", "answer} {answer}"])
def test_a_malformed_template_is_refused(arm, template):
    with pytest.raises(ValueError, match="not a valid str.format template"):
        arm(sdpg_hint_template=template)


@pytest.mark.parametrize(
    "build",
    [SDPGArguments, SelfDistillationArguments, lambda **kw: RLVROnlineGRPOScriptArguments(use_sdpg=True, **kw)],
    ids=["SDPGArguments", "SelfDistillationArguments", "RLVROnlineGRPOScriptArguments-use_sdpg"],
)
def test_a_null_template_is_refused_naming_the_field(build):
    """A YAML ``sdpg_hint_template: null`` parses to None, which ``string.Formatter`` meets with a bare
    ``TypeError`` naming no field."""
    with pytest.raises(ValueError, match=r"^sdpg_hint_template must be a str.format template string, got None$"):
        build(sdpg_hint_template=None)


def _run_rlvr_main(tmp_path, yaml_tail: str):
    config = tmp_path / "rlvr.yaml"
    config.write_text(
        f"model_name_or_path: dummy/model\noutput_dir: {tmp_path / 'out'}\nbf16: false\nuse_cpu: true\n{yaml_tail}"
    )
    script = load_script_module("scripts/training/online_grpo/rlvr.py", "halo_test_rlvr_sdpg_gate")
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
        mock.patch.object(script, "init_training_script", side_effect=AssertionError("reached the load")),
    ):
        script.main()


def test_a_null_template_beside_the_closed_rlvr_gate_is_refused_by_the_gate(tmp_path):
    """With ``use_sdpg`` off the template is inert, so the refusal that speaks is the one naming it as set
    beside the closed gate — not a parse error asking for a template the run would never read."""
    with pytest.raises(ValueError, match=r"use_sdpg off does not support .*\['sdpg_hint_template'\]"):
        _run_rlvr_main(tmp_path, "sdpg_hint_template: null\n")


def test_a_null_template_under_the_open_rlvr_gate_is_refused_at_parse(tmp_path):
    with pytest.raises(ValueError, match="^sdpg_hint_template must be a str.format template string"):
        _run_rlvr_main(tmp_path, "use_sdpg: true\nsdpg_hint_template: null\n")


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
def test_escaped_braces_and_format_specs_pass(arm):
    """Only the looked-up names are checked: a literal ``{{...}}`` and a spec on a filled name are fine."""
    arm(sdpg_hint_template="{{answer}} is literal; the answer is {answer!s:>4}")


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sdpg_temperature", 0.0),
        ("sdpg_temperature", -1.0),
        ("sdpg_temperature", math.nan),
        ("sdpg_temperature", math.inf),
        ("sdpg_beta_base", -0.5),
        ("sdpg_beta_base", math.nan),
        ("sdpg_beta_base", math.inf),
        ("sdpg_beta_warmup_steps", -1),
        ("sdpg_beta_decay_steps", -1),
        ("sdpg_beta_decay_steps", 2.5),
        ("sdpg_loss", "kl"),
    ],
)
def test_an_out_of_range_tunable_is_refused(arm, field_name, value):
    with pytest.raises(ValueError, match=field_name):
        arm(**{field_name: value})


@pytest.mark.parametrize("arm", ALL_ARMS, ids=lambda c: c.__name__)
def test_the_boundary_values_still_parse(arm):
    """``sdpg_beta_base: 0`` is the documented way to drop the term, and 0-step ramps are the defaults."""
    parsed = arm(sdpg_beta_base=0.0, sdpg_beta_warmup_steps=0, sdpg_beta_decay_steps=10, sdpg_temperature=0.5)
    assert (parsed.sdpg_beta_base, parsed.sdpg_beta_decay_steps, parsed.sdpg_temperature) == (0.0, 10, 0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
