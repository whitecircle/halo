"""The voice skill's scanner keys its maps by exactly their Literals, exits as ``skills/voice/SKILL.md`` says, and
applies the limits ``skills/voice/patterns.md`` states."""

from typing import get_args

import pytest

from tests.common.repo import load_script_module

# Registered: its dataclasses resolve their string annotations through sys.modules.
scan = load_script_module("skills/voice/scripts/scan.py", "voice_scan", register=True)


def _row(text: str, name: str, *, message: bool = False):
    return next((row for row in scan.scan(text, message).rows if row.name == name), None)


def _hits(text: str, name: str, *, message: bool = False) -> list[int]:
    row = _row(text, name, message=message)
    return [] if row is None else row.hits


def test_the_maps_keyed_by_a_literal_hold_exactly_its_values():
    """``report`` walks ``_GROUP_TITLES``, so a kind missing from it would drop its rows without an error."""
    assert set(scan._BLOCK_KINDS) == set(get_args(scan.Where))
    assert set(scan._GROUP_TITLES) == set(get_args(scan.Kind))


def test_exit_status_is_1_only_on_a_strong_tell_and_2_on_a_file_it_cannot_read(tmp_path, capsys):
    clean = tmp_path / "clean.md"
    clean.write_text("The job writes its log to the scratch volume.\n", encoding="utf-8")
    tell = tmp_path / "tell.md"
    tell.write_text("The loader is robust.\n", encoding="utf-8")
    over = tmp_path / "over.md"
    over.write_text("## Attention — the fix\n", encoding="utf-8")
    binary = tmp_path / "binary.md"
    binary.write_bytes(b"\xff\xfe\x00text")

    assert scan.main([str(clean)]) == 0
    assert scan.main([str(over)]) == 0
    assert "OVER" in capsys.readouterr().out
    assert scan.main([str(clean), str(tell)]) == 1
    assert "1 strong tell," in capsys.readouterr().out
    for unreadable in (tmp_path / "missing.md", binary):
        with pytest.raises(SystemExit) as raised:
            scan.main([str(clean), str(unreadable)])
        assert raised.value.code == 2
        assert capsys.readouterr().out == "", "a report was printed before the unreadable file stopped the run"


def test_a_message_allows_no_bold():
    text = "The **reshard** step frees the gathered weights.\n"
    assert _row(text, "bold").kind == "read"
    assert not _row(text, "bold").over
    assert _row(text, "bold", message=True).kind == "density"
    assert _row(text, "bold", message=True).over


def test_quoted_and_code_text_is_not_counted():
    assert _hits("The loader is robust.\n", "banned word") == [1]
    assert _hits('The flag `robust` and the word "robust" are mentions.\n', "banned word") == []
    assert _hits("```\nrobust\n```\n", "banned word") == []


@pytest.mark.parametrize("heading", ["## Attention—the fix", "##### Attention — the fix", "## Attention: the fix"])
def test_a_dash_or_colon_in_a_heading_of_any_level_is_a_tagline(heading):
    assert _hits(f"{heading}\n", "heading with a tagline") == [1]


def test_a_plain_heading_is_no_tagline():
    assert _hits("## Attention and FlashAttention\n", "heading with a tagline") == []


def test_a_dash_after_a_bold_or_code_label_is_over_and_another_dash_is_left_to_the_rate():
    text = "- **Label** — the text of the item.\n- `--flag` — what it sets.\n\nRun the job — then read the log.\n"
    assert _hits(text, "dash after a label") == [1, 2]
    assert _row(text, "dash after a label").over
    assert _hits(text, "em dash") == [1, 2, 4]


def test_an_emoji_counts_in_a_heading_and_in_prose_but_not_as_a_legend_mark():
    assert _hits("## 🚀 Launch the job\n", "emoji") == [1]
    assert _hits("✅ Tests pass on the new head.\n", "emoji") == [1]
    assert _hits("Legend: ✅ = patched, ❌= refused, ⚠\ufe0f = partial, ✓ plain, ✗ plain.\n", "emoji") == []
    assert _hits("| Family | EP |\n|---|---|\n| Qwen3 | ✅ |\n", "emoji") == []


def test_the_word_lists_take_just_call_and_a_capitalized_thats_why_and_count_wins_big_once():
    assert _hits("You can just call the tool.\n", "banned word") == [1]
    assert _hits("That's why it holds.\n", "is why") == [1]
    wins_big = "The grouped launch wins big on this shape.\n"
    assert _hits(wins_big, "absolute or color word") == [1]
    assert _hits(wins_big, "wish or appetite") == []


def test_an_inline_comment_is_listed_on_its_line_and_its_words_are_not_prose():
    text = (
        "A first line of the paragraph.\n"
        "Text with a note <!-- WAITS: a robust check --> in the middle.\n"
        "A last line that is robust.\n"
        "\n"
        "Start <!-- a comment that opens here\n"
        "robust inside it\n"
        "--> robust after it.\n"
        "\n"
        "The marker `<!-- not a comment -->` is code.\n"
    )
    assert _hits(text, "HTML comment") == [2, 5]
    assert _hits(text, "banned word") == [3, 7]


def test_two_contrasts_in_one_paragraph_are_over_whatever_the_page_rate():
    filler = "The job writes its log to the scratch volume every step. " * 200
    text = (
        f"{filler}\n\n"
        "A row is dropped, not truncated, and a node runs one process rather than one per GPU.\n\n"
        "A row is dropped, not truncated.\n"
    )
    assert not _row(text, "contrast").over
    assert _hits(text, "paragraph with two contrasts") == [3]
    assert _row(text, "paragraph with two contrasts").over


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
