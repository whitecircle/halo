#!/usr/bin/env python
"""``assistant_message_template`` must actually occur in the chat template it is paired with — once
per assistant turn.

``train_on_completions_only`` finds the assistant span by searching the RENDERED conversation for
the marker string. When the marker does not occur — a template swapped for another family's, a
marker copied from a sibling model — the collator masks the whole row
(``src/data/collators/packing.py``: ``batch["labels"][i, :] = ignore_index``) and the run trains on
nothing, reporting loss 0 instead of failing. A marker that occurs only ONCE on a multi-turn row is
the same failure at 1/n scale: every earlier assistant turn is masked, silently. Both are invisible
in a parse-only check: every one of these configs parses clean.

Only pairings pinning a repo-owned ``jinja-templates/*.jinja`` are covered — a hub-bundled template
is not on disk here. The scan reads the ``examples/`` YAML **and** the ```yaml fences of the two doc
trees, so a doc recipe that pairs a marker with a template is held to the same contract as a shipped
config. Rendering is real (the same Jinja environment transformers uses), so a marker the template
assembles from parts is matched exactly as the collator would match it.

Run: pytest tests/cpu/config/test_assistant_marker_matches_template.py
"""

import re
from pathlib import Path

import pytest
from jinja2.exceptions import TemplateError
from ruamel.yaml import YAML
from transformers.utils.chat_template_utils import _compile_jinja_template

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Doc trees whose ```yaml fences are held to the same pairing contract as examples/.
_DOC_TREES = ("human-docs", "agent-docs")
_YAML_FENCE = re.compile(r"^```ya?ml\n(.*?)^```", re.M | re.S)

# Single turn: every shipped template accepts it, including the ``*-instruct`` ones whose own guard
# rejects multi-turn data. One assistant turn is all the occurrence check needs.
_CONVERSATION = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
# Three plain pairs (no reasoning channel, no tools): the shape a multi-turn SFT corpus is made of.
_TURNS = 3
_MULTI_TURN = [
    {"role": role, "content": f"{role[0]}{i}"} for i in range(1, _TURNS + 1) for role in ("user", "assistant")
]

# The pairing the gpt-oss multiturn template exists to make trainable: every assistant turn carries
# the final-channel marker AND is terminated by <|return|>, the id the completion span ends on.
_GPT_OSS_TEMPLATE = "jinja-templates/gpt-oss/gpt-oss-multiturn.jinja"
_GPT_OSS_MARKER = "<|start|>assistant<|channel|>final<|message|>"
_GPT_OSS_TERMINATOR = "<|return|>"

_yaml = YAML(typ="safe")


def _pairing(source: str, data: object) -> tuple[str, str, str] | None:
    """``(source, template path, marker)`` when ``data`` pins both and the template is repo-owned."""
    if not isinstance(data, dict):
        return None
    template, marker = data.get("chat_template"), data.get("assistant_message_template")
    if isinstance(template, str) and template.endswith(".jinja") and marker:
        return (source, template, marker)
    return None


def _marker_configs() -> list[tuple[str, str, str]]:
    """Every pairing in the tree: the ``examples/`` YAML plus the doc trees' ```yaml fences."""
    found = []
    for config in sorted((PROJECT_ROOT / "examples").rglob("*.yaml")):
        pairing = _pairing(str(config.relative_to(PROJECT_ROOT)), _yaml.load(config))
        if pairing:
            found.append(pairing)
    for tree in _DOC_TREES:
        for page in sorted((PROJECT_ROOT / tree).rglob("*.md")):
            for index, block in enumerate(_YAML_FENCE.findall(page.read_text())):
                try:
                    data = _yaml.load(block)
                except Exception:  # a fence that is not parseable YAML is not a config
                    continue
                pairing = _pairing(f"{page.relative_to(PROJECT_ROOT)}#yaml[{index}]", data)
                if pairing:
                    found.append(pairing)
    return found


_MARKER_CONFIGS = _marker_configs()


def _render(template_path: str, messages: list[dict]) -> str:
    source = (PROJECT_ROOT / template_path).read_text()
    compiled = _compile_jinja_template(source)
    return compiled.render(messages=messages, add_generation_prompt=False, bos_token="<bos>", eos_token="<eos>")


def _case_id(source: str) -> str:
    return source.replace("/", ".").replace("#", ".")


def test_some_config_pins_a_repo_template():
    """Guards the scan itself: an empty list would make every case below vacuously pass."""
    assert _MARKER_CONFIGS, "no example or doc recipe pins a jinja-templates/* chat_template — the scan is broken"


def test_the_scan_reaches_the_doc_trees():
    """The doc half of the scan is the reason a broken cookbook recipe used to survive a green suite."""
    assert any(source.endswith("]") for source, _, _ in _MARKER_CONFIGS), (
        f"no ```yaml fence under {_DOC_TREES} pairs a chat_template with an assistant_message_template — "
        f"the doc half of the scan is dead"
    )


@pytest.mark.parametrize(
    ("source", "template", "marker"),
    _MARKER_CONFIGS,
    ids=[_case_id(s) for s, _, _ in _MARKER_CONFIGS],
)
def test_assistant_marker_occurs_in_its_rendered_template(source, template, marker):
    rendered = _render(template, _CONVERSATION)
    assert marker in rendered, (
        f"{source} sets assistant_message_template={marker!r}, which does not occur in the rendered "
        f"{template}. train_on_completions_only would mask every row and the run would train on "
        f"nothing (loss 0). Rendered: {rendered!r}"
    )


@pytest.mark.parametrize(
    ("source", "template", "marker"),
    _MARKER_CONFIGS,
    ids=[_case_id(s) for s, _, _ in _MARKER_CONFIGS],
)
def test_every_assistant_turn_of_a_multi_turn_row_carries_the_marker(source, template, marker):
    """A marker that matches only the LAST assistant turn masks every earlier one — the same silent
    loss of signal as a non-matching marker, just partial. A template whose own guard refuses
    multi-turn data is exempt: its render raises instead of pairing badly."""
    try:
        rendered = _render(template, _MULTI_TURN)
    except TemplateError as e:
        pytest.skip(f"{template} refuses multi-turn data by design ({e})")
    assert rendered.count(marker) == _TURNS, (
        f"{source}: {template} renders the marker {rendered.count(marker)}x on a {_TURNS}-turn row, "
        f"so completion-only masking trains {rendered.count(marker)} of {_TURNS} assistant turns and "
        f"silently drops the rest. Rendered: {rendered!r}"
    )


def test_gpt_oss_multiturn_terminates_every_assistant_turn():
    """Each marked span must close on <|return|>: the harmony template ends non-final turns with
    <|end|>, which is not a terminator, so a span would run through the following user turn and train
    on its text."""
    rendered = _render(_GPT_OSS_TEMPLATE, _MULTI_TURN)
    spans = rendered.split(_GPT_OSS_MARKER)[1:]
    assert len(spans) == _TURNS, f"{_GPT_OSS_TEMPLATE} marked {len(spans)} of {_TURNS} turns: {rendered!r}"
    for index, span in enumerate(spans):
        turn = span.split("<|start|>")[0]
        assert turn.endswith(_GPT_OSS_TERMINATOR), (
            f"assistant turn {index + 1} of {_TURNS} ends {turn[-12:]!r}, not {_GPT_OSS_TERMINATOR!r}: "
            f"its completion span would run past the turn. Rendered: {rendered!r}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
