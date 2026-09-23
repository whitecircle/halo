#!/usr/bin/env python
"""The Hub seed and the cache-miss policy that makes a repo outside it fail instead of skip.

``tests.common.hub_seed`` derives what the hosted CPU tier downloads; ``HALO_TEST_REQUIRE_HUB_CACHE``
turns a cache miss on any Hub repo the seed could hold into a failure, so a test that starts reading a
new repo cannot pass that tier as a skip. Gated repos and local checkpoint paths still skip.

Run: python tests/cpu/conventions/test_hub_seed.py
"""

import pytest

from tests.common.hub_seed import GATED_REPOS, example_repos, is_hub_repo, roster_repos, seed
from tests.common.models import GEMMA3_4B_IT, GPT_OSS_20B_PATCHED, PINNED_REVISIONS, QWEN3_0_6B
from tests.common.tokenizers import REQUIRE_HUB_CACHE, require_cached, try_cached_tokenizer

# Well-formed and certainly absent from any cache.
_UNSEEDED = "halo-tests/no-such-repo"


@pytest.mark.parametrize(
    ("reference", "hub"),
    [
        (QWEN3_0_6B, True),
        (GPT_OSS_20B_PATCHED, False),
        ("checkpoints/run/checkpoint-200", False),
        ("gpt-oss-20b", False),
    ],
)
def test_hub_ids_are_told_from_local_checkpoints(reference, hub):
    assert is_hub_repo(reference) is hub


def test_the_seed_fetches_examples_roster_and_pins_but_no_gated_repo():
    fetched = {repo for repo, revision in seed() if revision is None}
    assert example_repos() <= fetched | GATED_REPOS and roster_repos() <= fetched | GATED_REPOS
    assert not fetched & GATED_REPOS, f"the seed fetches gated repos: {fetched & GATED_REPOS}"
    assert set(PINNED_REVISIONS.items()) <= set(seed())


def test_a_miss_skips_by_default(monkeypatch):
    monkeypatch.delenv(REQUIRE_HUB_CACHE, raising=False)
    assert try_cached_tokenizer(_UNSEEDED) is None


def test_a_miss_fails_where_the_cache_must_be_complete(monkeypatch):
    monkeypatch.setenv(REQUIRE_HUB_CACHE, "1")
    with pytest.raises(pytest.fail.Exception, match="hub_seed"):
        try_cached_tokenizer(_UNSEEDED)


@pytest.mark.parametrize("reference", [GEMMA3_4B_IT, GPT_OSS_20B_PATCHED])
def test_gated_repos_and_local_paths_may_miss_where_the_cache_must_be_complete(monkeypatch, reference):
    monkeypatch.setenv(REQUIRE_HUB_CACHE, "1")
    assert require_cached(reference, "tokenizer") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
