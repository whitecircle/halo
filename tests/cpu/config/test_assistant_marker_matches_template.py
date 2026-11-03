#!/usr/bin/env python
"""``assistant_message_template`` must occur in its paired chat template once per assistant turn: a miss masks the
row, so the run trains on nothing at loss 0, or on 1/n of a multi-turn row. Covers pairings pinning a repo-owned
``jinja-templates/*.jinja``, in ``examples/`` and the docs' yaml fences."""

import re

import pytest
from jinja2.exceptions import TemplateError
from ruamel.yaml import YAML
from transformers.utils.chat_template_utils import _compile_jinja_template

from tests.common.repo import REPO_ROOT

_DOC_TREES = ("human-docs", "agent-docs", "skills")
_YAML_FENCE = re.compile(r"^```ya?ml\n(.*?)^```", re.M | re.S)

# Single turn: every shipped template accepts it, including the ``*-instruct`` ones whose own guard
# rejects multi-turn data. One assistant turn is all the occurrence check needs.
_CONVERSATION = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
# Three plain pairs (no reasoning channel, no tools): the shape a multi-turn SFT corpus is made of.
_TURNS = 3
_MULTI_TURN = [
    {"role": role, "content": f"{role[0]}{i}"} for i in range(1, _TURNS + 1) for role in ("user", "assistant")
]

# What a single-turn template's own guard raises. Any other TemplateError is a template that broke,
# not one refusing the shape by design, and must fail rather than skip.
_SINGLE_TURN_REFUSAL = "only supports single-turn data"

# The pairing the gpt-oss multiturn template exists to make trainable: every assistant turn carries the final-channel
# marker and ends on <|return|>, the id the completion span ends on.
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


def _page_fences(source: str, text: str) -> list[tuple[str, object]]:
    """``(source#yaml[i], parsed)`` for every ```yaml fence of one page; an unparseable fence is skipped."""
    fences = []
    for index, block in enumerate(_YAML_FENCE.findall(text)):
        try:
            data = _yaml.load(block)
        except Exception:  # a fence that is not parseable YAML is not a config
            continue
        fences.append((f"{source}#yaml[{index}]", data))
    return fences


def _doc_fences() -> list[tuple[str, object]]:
    return [
        fence
        for tree in _DOC_TREES
        for page in sorted((REPO_ROOT / tree).rglob("*.md"))
        for fence in _page_fences(str(page.relative_to(REPO_ROOT)), page.read_text())
    ]


def _marker_configs() -> list[tuple[str, str, str]]:
    sources = [
        (str(config.relative_to(REPO_ROOT)), _yaml.load(config))
        for config in sorted((REPO_ROOT / "examples").rglob("*.yaml"))
    ]
    return [pairing for source, data in sources + _DOC_FENCES if (pairing := _pairing(source, data))]


_DOC_FENCES = _doc_fences()
_MARKER_CONFIGS = _marker_configs()


def _render(template_path: str, messages: list[dict]) -> str:
    source = (REPO_ROOT / template_path).read_text()
    compiled = _compile_jinja_template(source)
    return compiled.render(messages=messages, add_generation_prompt=False, bos_token="<bos>", eos_token="<eos>")


def _case_id(source: str) -> str:
    return source.replace("/", ".").replace("#", ".")


def test_some_config_pins_a_repo_template():
    assert _MARKER_CONFIGS, "no example or doc recipe pins a jinja-templates/* chat_template — the scan is broken"


def test_a_doc_fence_pairing_is_held_to_the_contract():
    """A doc recipe that pins a repo template with a marker is checked like a shipped config."""
    page = (
        "Intro.\n\n```yaml\n"
        f"chat_template: {_GPT_OSS_TEMPLATE}\nassistant_message_template: '{_GPT_OSS_MARKER}'\n"
        "```\n\n```yaml\nnot: [valid\n```\n"
    )
    pairings = [pairing for source, data in _page_fences("page.md", page) if (pairing := _pairing(source, data))]
    assert pairings == [("page.md#yaml[0]", _GPT_OSS_TEMPLATE, _GPT_OSS_MARKER)]


@pytest.mark.parametrize("tree", _DOC_TREES)
def test_the_scan_reads_each_doc_tree(tree):
    """Counted per tree: ``rglob`` on a missing or renamed root yields nothing, so one dead root would
    otherwise hide behind the other's fences."""
    assert (REPO_ROOT / tree).is_dir(), f"{tree} is not a directory — that half of the scan is dead"
    assert any(source.startswith(f"{tree}/") for source, _ in _DOC_FENCES), (
        f"no ```yaml fence found under {tree} — that half of the scan is dead"
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
    """A marker matching only the last assistant turn masks every earlier one. A template whose own guard refuses
    multi-turn data is exempt, since its render raises."""
    try:
        rendered = _render(template, _MULTI_TURN)
    except TemplateError as e:
        if _SINGLE_TURN_REFUSAL not in str(e):
            raise
        pytest.skip(f"{template} refuses multi-turn data by design ({e})")
    assert rendered.count(marker) == _TURNS, (
        f"{source}: {template} renders the marker {rendered.count(marker)}x on a {_TURNS}-turn row, "
        f"so completion-only masking trains {rendered.count(marker)} of {_TURNS} assistant turns and "
        f"silently drops the rest. Rendered: {rendered!r}"
    )


def test_gpt_oss_multiturn_terminates_every_assistant_turn():
    """The harmony template ends non-final turns with <|end|>, which is no terminator, so a span would run through
    the following user turn and train on its text."""
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
    raise SystemExit(pytest.main([__file__, "-v"]))
