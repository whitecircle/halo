#!/usr/bin/env python
"""``rollout_thinking: false``: ``enable_thinking: false`` reaches the engine request and the trainer's renders alike,
every setting that reads reasoning is refused, and a chat template that ignores the switch is refused.

Run: python tests/cpu/grpo/test_rollout_thinking_off.py  (or pytest)
"""

import ast
from dataclasses import replace

import pytest
import yaml
from transformers.utils.chat_template_utils import _compile_jinja_template

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.registry import build_environment
from halo.environments.rollout_config import verify_template_closes_thinking
from halo.environments.rollouts.engine_wire import generation_control_fields
from halo.trainers.grpo.environmental.trajectory_tokenize import rollout_template_kwargs
from tests.common.repo import REPO_ROOT

# A thinking template with no switch: the generation prompt opens the block whatever the variables.
ALWAYS_THINKING = (
    "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}<think>\n{% endif %}"
)


class _TemplateTokenizer:
    def __init__(self, template: str):
        self._template = _compile_jinja_template(template)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, bos_token="<s>", **kwargs
        )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_thinking_off_reaches_the_request_and_the_trainers_render(backend):
    """vLLM sets ``enable_thinking`` from the top-level level unless the request's own kwargs carry it, so the switch
    has to travel nested, where the trainer's renders read it too."""
    run = AsyncRolloutConfig(
        rollout_backend=backend, rollout_thinking=False, rollout_chat_template_kwargs={"preserve_thinking": True}
    )
    fields = generation_control_fields(run.get_rollout_config(), "low")
    assert fields["chat_template_kwargs"]["enable_thinking"] is False
    assert rollout_template_kwargs(run.rollout_chat_template_kwargs, "low") == {
        "preserve_thinking": True,
        "enable_thinking": False,
        "reasoning_effort": "low",
    }
    assert replace(run).rollout_chat_template_kwargs == run.rollout_chat_template_kwargs, "validation is idempotent"
    thinking = AsyncRolloutConfig(rollout_backend=backend, rollout_chat_template_kwargs={"preserve_thinking": True})
    assert (
        "enable_thinking"
        not in generation_control_fields(thinking.get_rollout_config(), "low")["chat_template_kwargs"]
    ), "a thinking run's requests and renders are unchanged"


@pytest.mark.parametrize(("rollout_thinking", "written"), [(True, False), (True, True), (False, True)])
def test_enable_thinking_is_spelled_rollout_thinking(rollout_thinking, written):
    with pytest.raises(ValueError, match="must not carry 'enable_thinking': set rollout_thinking"):
        AsyncRolloutConfig(
            rollout_thinking=rollout_thinking, rollout_chat_template_kwargs={"enable_thinking": written}
        )


@pytest.mark.parametrize(
    "knob",
    [
        {"reasoning_floor": 0.25},
        {"reasoning_price_cap": 0.2},
        {"reasoning_terms_in_failed_groups": True},
        {"rollout_max_thinking_tokens": 1024},
        {"rollout_max_answer_tokens": 1024},
    ],
    ids=lambda knob: next(iter(knob)),
)
def test_thinking_off_refuses_each_reasoning_knob(knob):
    AsyncRolloutConfig(**knob)
    with pytest.raises(ValueError, match=rf"{next(iter(knob))} is set with rollout_thinking: false"):
        AsyncRolloutConfig(rollout_thinking=False, **knob)


def test_thinking_off_refuses_an_environment_that_reads_reasoning():
    build_environment(
        "native_math", {"reasoning_effort": "random", "reasoning_effort_profiles": {}}
    ).reject_settings_without_thinking()
    with pytest.raises(ValueError, match="carry_reasoning: true with rollout_thinking: false"):
        build_environment("native_math", {"carry_reasoning": True}).reject_settings_without_thinking()
    for profile in ({"reasoning_budget": 0}, {"reasoning_budget": 40000, "reasoning_price": 0.008}):
        env = build_environment(
            "native_math", {"reasoning_effort": "medium", "reasoning_effort_profiles": {"low": profile}}
        )
        with pytest.raises(ValueError, match=r"reasoning_effort_profiles \['low'\] set reasoning_budget"):
            env.reject_settings_without_thinking()


# The shipped examples that sample without thinking: each must pass the gates its run's startup applies.
_NO_THINKING_EXAMPLES = sorted(
    path
    for path in (REPO_ROOT / "examples").rglob("*.yaml")
    if isinstance(config := yaml.safe_load(path.read_text()), dict) and config.get("rollout_thinking") is False
)


def test_some_example_samples_without_thinking():
    """Guards the sweep below: an empty one would pass asserting nothing."""
    assert _NO_THINKING_EXAMPLES


@pytest.mark.parametrize("path", _NO_THINKING_EXAMPLES, ids=lambda p: p.name)
def test_a_no_thinking_example_passes_the_startup_gates(path):
    config = yaml.safe_load(path.read_text())
    build_environment(
        config["environment_type"], config.get("environment_kwargs", {})
    ).reject_settings_without_thinking()
    tokenizer = _TemplateTokenizer((REPO_ROOT / config["chat_template"]).read_text())
    verify_template_closes_thinking(tokenizer, {"enable_thinking": False, "reasoning_effort": "low"})


def test_the_script_checks_the_template_before_it_probes_the_servers():
    """Dropped, a template that ignores the switch on both sides passes the server probe, which compares the renders."""
    first_line = {}
    for node in ast.walk(ast.parse((REPO_ROOT / "scripts/training/grpo/environmental.py").read_text())):
        if isinstance(node, ast.Call):
            first_line.setdefault(ast.unparse(node.func), node.lineno)
    assert first_line["verify_template_closes_thinking"] < first_line["verify_served_template"]


def test_a_template_that_ignores_the_switch_is_refused():
    with pytest.raises(ValueError, match="renders the same generation prompt either way. Set chat_template:"):
        verify_template_closes_thinking(_TemplateTokenizer(ALWAYS_THINKING), {"enable_thinking": False})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
