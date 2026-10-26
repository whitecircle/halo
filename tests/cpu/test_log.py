"""``preview`` renders a list for a message: the first items, then how many it left out.

python tests/cpu/test_log.py
"""

import pytest

from src.log import KEY_PREVIEW_COUNT, preview


@pytest.mark.parametrize(
    ("items", "n", "expected"),
    [
        ([], 3, ""),
        (["a"], 3, "a"),
        (["a", "b", "c"], 3, "a, b, c"),
        (["a", "b", "c", "d"], 3, "a, b, c, ... (+1 more)"),
        ([1, 2, 3, 4, 5, 6, 7], 2, "1, 2, ... (+5 more)"),
        (("x.weight", "y.bias"), 1, "x.weight, ... (+1 more)"),
    ],
)
def test_preview_lists_the_first_items_and_counts_the_rest(items, n, expected):
    assert preview(items, n) == expected


def test_preview_shows_key_preview_count_items_by_default():
    keys = [f"layer.{index}.weight" for index in range(KEY_PREVIEW_COUNT + 2)]
    assert preview(keys) == ", ".join(keys[:KEY_PREVIEW_COUNT]) + ", ... (+2 more)"
    assert preview(keys[:KEY_PREVIEW_COUNT]) == ", ".join(keys[:KEY_PREVIEW_COUNT])


@pytest.mark.parametrize("n", [0, -1])
def test_preview_refuses_a_count_below_one(n):
    """``n=0`` would open the message on a bare separator, and a negative ``n`` slices from the end."""
    with pytest.raises(ValueError, match="n >= 1"):
        preview(["a", "b"], n)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
