"""SGLang's DSA page for GLM-5 Next is ``64 * index_kpool`` tokens, and its hybrid-linear radix cache asserts that
``mamba_track_interval`` is a multiple of the page size (``validate_mamba_extra_buffer``). The documented launch flag has
to satisfy that rule for the checkpoint's ``index_kpool``, and the engine's default interval must not, or the flag is
dead weight."""

import re

import pytest

from tests.common.models import GLM5_ENGINE_GATED_DIMS
from tests.common.repo import REPO_ROOT

DSA_PAGE_TOKENS = 64
SGLANG_DEFAULT_MAMBA_TRACK_INTERVAL = 256
PAGE_SIZE = DSA_PAGE_TOKENS * GLM5_ENGINE_GATED_DIMS["index_kpool"]


def _documented(path: str, flag: str) -> str:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(flag)} (\d+)", text)
    assert match, f"{path} documents no {flag}"
    return match.group(1)


def test_engine_default_interval_trips_the_assert() -> None:
    assert SGLANG_DEFAULT_MAMBA_TRACK_INTERVAL % PAGE_SIZE != 0


def test_documented_interval_is_a_multiple_of_the_page_size() -> None:
    for path in ("agent-docs/models/glm5-next.md", "tests/gpu/trainers/grpo/test_weight_push_exact.py"):
        assert int(_documented(path, "--mamba-track-interval")) % PAGE_SIZE == 0, path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
