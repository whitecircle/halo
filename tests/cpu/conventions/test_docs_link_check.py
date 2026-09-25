#!/usr/bin/env python
"""The docs link gate must check heading anchors the way GitHub generates them.

``scripts/docs/check_links.sh`` is the only gate on the doc trees' cross-references. A fragment it
does not validate, or validates against a slug GitHub never produces, passes CI while the link lands
at the top of the page. The expected slugs below are the ids GitHub's renderer emits for the same
headings, including a ``{#id}`` attribute, which GitHub renders as heading text.

Run: python tests/cpu/conventions/test_docs_link_check.py
"""

import subprocess

import pytest

from tests.common.utils import REPO_ROOT, load_script_module

check_links = load_script_module("scripts/docs/check_links.py")


def _anchors(markdown: str) -> set[str]:
    return set(check_links.parse_markdown(markdown).anchors)


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("EP / CP / TP behavior {#ep-cp-tp-behavior}", "ep--cp--tp-behavior-ep-cp-tp-behavior"),
        ("The EP8 dispatch ceiling: ~64k tokens/rank", "the-ep8-dispatch-ceiling-64k-tokensrank"),
        ("`__init__` and _emph_ and snake_case — 8× B300 → ok", "__init__-and-emph-and-snake_case--8-b300--ok"),
        ("SFTConfig — `X` ([TRL docs](https://x.y/z#a))", "sftconfig--x-trl-docs"),
        (r"a &amp; b &lt;c&gt; \_esc\_ <sup>up</sup> ![img](x.png) end", "a--b-c-_esc_-up--end"),
        ("Ünïcode Straße ½ ² café", "ünïcode-straße---café"),
        ("🚀 Launch", "-launch"),
        ("` spaced code `", "spaced-code"),
        ("Closing ##", "closing"),
    ],
)
def test_heading_slug_matches_github(heading, slug):
    assert _anchors(f"## {heading}\n") == {slug}


def test_repeated_headings_are_numbered_like_github():
    assert _anchors("## Foo\n\n## Foo\n\n## Foo 1\n\n## Foo\n") == {"foo", "foo-1", "foo-1-1", "foo-2"}


def test_only_rendered_headings_and_explicit_anchors_count():
    markdown = (
        "---\nname: front matter\n# not a heading\n---\n"
        "```bash\n# comment in a fence\n```\n"
        "````md\n```\n## still fenced\n```\n````\n"
        "~~~\n## tilde fence\n~~~\n"
        "    ## four-space indent is code\n"
        "> ## Quoted\n"
        '<a id="pinned-id"></a>\n<a name="named-id"></a>\n'
        '<h3 align="center">Html Heading</h3>\n'
        "## Real\n"
    )
    assert _anchors(markdown) == {"quoted", "pinned-id", "named-id", "html-heading", "real"}


def test_heading_id_attribute_is_refused():
    document = check_links.parse_markdown("# Title\n\n## Section {#section}\n\n## Uses `{#x}` in code\n")
    assert document.heading_attributes == ((3, "{#section}"),)


def _run_gate(tree):
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/docs/check_links.sh"), str(tree)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_gate_fails_on_a_missing_anchor_and_passes_the_generated_one(tmp_path):
    (tmp_path / "target.md").write_text("# Target\n\n## EP / CP behavior\n\n```\n## Fenced\n```\n")
    good = (
        "[a](target.md#ep--cp-behavior) [b](#local-section) [c](script.py#L3) "
        "[d](https://example.com/x.md#nowhere) `[e](target.md#in-code)`\n\n## Local section\n"
    )
    (tmp_path / "script.py").write_text("")
    (tmp_path / "good.md").write_text(good)
    result = _run_gate(tmp_path)
    assert result.returncode == 0, result.stdout

    (tmp_path / "bad.md").write_text("intro\n[x](target.md#ep-cp-behavior) [y](target.md#fenced) [z](#gone)\n")
    result = _run_gate(tmp_path)
    assert result.returncode == 1
    broken = sorted(line for line in result.stdout.splitlines() if line.startswith("BROKEN ANCHOR"))
    bad = tmp_path / "bad.md"
    assert broken == [
        f"BROKEN ANCHOR  {bad}:2  ->  #gone",
        f"BROKEN ANCHOR  {bad}:2  ->  target.md#ep-cp-behavior",
        f"BROKEN ANCHOR  {bad}:2  ->  target.md#fenced",
    ]


def test_a_fragment_on_a_directory_link_is_checked_against_its_readme(tmp_path):
    """GitHub renders a directory's ``README.md`` under its listing, so ``sub/#x`` lands in that page."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "README.md").write_text("# Sub\n\n## Deep section\n")
    page = tmp_path / "page.md"
    page.write_text("[a](sub/#deep-section) [b](sub#deep-section)\n")
    result = _run_gate(tmp_path)
    assert result.returncode == 0, result.stdout

    page.write_text("[a](sub/#gone) [b](sub#gone)\n")
    result = _run_gate(tmp_path)
    assert result.returncode == 1
    broken = sorted(line for line in result.stdout.splitlines() if line.startswith("BROKEN"))
    assert broken == [f"BROKEN ANCHOR  {page}:1  ->  sub#gone", f"BROKEN ANCHOR  {page}:1  ->  sub/#gone"]


def test_gate_fails_on_a_missing_target_and_a_heading_id(tmp_path):
    (tmp_path / "page.md").write_text("## Section {#section}\n\n[x](missing.md)\n")
    result = _run_gate(tmp_path)
    assert result.returncode == 1
    page = tmp_path / "page.md"
    assert f"HEADING ID  {page}:1  {{#section}}" in result.stdout
    assert f"BROKEN  {page}:3  ->  missing.md" in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
