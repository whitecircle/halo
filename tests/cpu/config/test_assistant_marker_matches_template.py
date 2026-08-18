#!/usr/bin/env python
"""``assistant_message_template`` must actually occur in the chat template it is paired with.

``train_on_completions_only`` finds the assistant span by searching the RENDERED conversation for
the marker string. When the marker does not occur — a template swapped for another family's, a
marker copied from a sibling model — the collator masks the whole row
(``src/data/collators/packing.py``: ``batch["labels"][i, :] = ignore_index``) and the run trains on
nothing, reporting loss 0 instead of failing. That is the failure this file exists to catch, and it
is invisible in a parse-only check: every one of these configs parses clean.

Only configs pinning a repo-owned ``jinja-templates/*.jinja`` are covered — a hub-bundled template
is not on disk here. Rendering is real (the same Jinja environment transformers uses), so a marker
the template assembles from parts is matched exactly as the collator would match it.

Run: pytest tests/cpu/config/test_assistant_marker_matches_template.py
"""

from pathlib import Path

import pytest
from ruamel.yaml import YAML
from transformers.utils.chat_template_utils import _compile_jinja_template

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Single turn: every shipped template accepts it, including the ``*-instruct`` ones whose own guard
# rejects multi-turn data. One assistant turn is all a marker check needs.
_CONVERSATION = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]

_yaml = YAML(typ="safe")


def _marker_configs() -> list[tuple[str, str, str]]:
    """``(config path, template path, marker)`` for every example pinning both, from the tree."""
    found = []
    for config in sorted((PROJECT_ROOT / "examples").rglob("*.yaml")):
        data = _yaml.load(config) or {}
        template, marker = data.get("chat_template"), data.get("assistant_message_template")
        if isinstance(template, str) and template.endswith(".jinja") and marker:
            found.append((str(config.relative_to(PROJECT_ROOT)), template, marker))
    return found


_MARKER_CONFIGS = _marker_configs()


def _render(template_path: str) -> str:
    source = (PROJECT_ROOT / template_path).read_text()
    compiled = _compile_jinja_template(source)
    return compiled.render(messages=_CONVERSATION, add_generation_prompt=False, bos_token="<bos>", eos_token="<eos>")


def test_some_config_pins_a_repo_template():
    """Guards the scan itself: an empty list would make every case below vacuously pass."""
    assert _MARKER_CONFIGS, "no example pins a jinja-templates/* chat_template — the scan is broken"


@pytest.mark.parametrize(
    ("config", "template", "marker"),
    _MARKER_CONFIGS,
    ids=[f"{Path(c).stem}" for c, _, _ in _MARKER_CONFIGS],
)
def test_assistant_marker_occurs_in_its_rendered_template(config, template, marker):
    rendered = _render(template)
    assert marker in rendered, (
        f"{config} sets assistant_message_template={marker!r}, which does not occur in the rendered "
        f"{template}. train_on_completions_only would mask every row and the run would train on "
        f"nothing (loss 0). Rendered: {rendered!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
