#!/usr/bin/env python
"""The Hub seed and the cache-miss policy that makes a repo outside it fail instead of skip.

``tests.common.hub_seed`` derives what ``make seed-hf-cache`` downloads; ``HALO_TEST_REQUIRE_HUB_CACHE``
turns a cache miss on any Hub repo the seed could hold into a failure, so a test that starts reading a
new repo cannot pass an offline CPU-tier run over the seeded cache as a skip. Gated repos and local
checkpoint paths still skip.

Run: python tests/cpu/conventions/test_hub_seed.py
"""

import json

import pytest
import yaml

from tests.common.hub_seed import GATED_REPOS, example_repos, is_hub_repo, roster_repos, seed
from tests.common.models import GEMMA3_4B_IT, GEMMA4_26B_A4B, GPT_OSS_20B_PATCHED, PINNED_REVISIONS, QWEN3_0_6B
from tests.common.tokenizers import REQUIRE_HUB_CACHE, require_cached, try_cached_config, try_cached_tokenizer
from tests.common.utils import REPO_ROOT

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


def _example_references() -> set[str]:
    """Every example's ``model_name_or_path``, read here rather than through the seed's own parser."""
    return {
        yaml.safe_load(path.read_text(encoding="utf-8"))["model_name_or_path"]
        for path in (REPO_ROOT / "examples").rglob("*.yaml")
    }


def test_the_seed_holds_every_example_hub_id_and_no_local_checkpoint():
    references = _example_references()
    fetched = {repo for repo, revision in seed() if revision is None}
    local = {ref for ref in references if ref.startswith(("/", "checkpoints/"))}
    assert GEMMA4_26B_A4B in references and local, "precondition: the examples train a Hub id and a local path"
    assert GEMMA4_26B_A4B in fetched, f"the seed misses {GEMMA4_26B_A4B}, which an example trains"
    assert not local & fetched, f"the seed would fetch local checkpoints: {sorted(local & fetched)}"
    assert references - local <= fetched | GATED_REPOS, sorted(references - local - fetched - GATED_REPOS)


def test_the_seed_holds_the_roster_and_the_pins_but_no_gated_repo():
    assert example_repos() and roster_repos(), "a seed source parsed to nothing"
    assert QWEN3_0_6B in roster_repos()
    fetched = {repo for repo, revision in seed() if revision is None}
    assert roster_repos() - GATED_REPOS <= fetched
    assert not fetched & GATED_REPOS, f"the seed fetches gated repos: {fetched & GATED_REPOS}"
    assert set(PINNED_REVISIONS.items()) <= set(seed())


def test_a_miss_skips_by_default(monkeypatch):
    monkeypatch.delenv(REQUIRE_HUB_CACHE, raising=False)
    assert try_cached_tokenizer(_UNSEEDED) is None


def test_a_miss_fails_where_the_cache_must_be_complete(monkeypatch):
    monkeypatch.setenv(REQUIRE_HUB_CACHE, "1")
    with pytest.raises(pytest.fail.Exception, match="not in the local HF cache"):
        try_cached_tokenizer(_UNSEEDED)


@pytest.mark.parametrize(
    ("reference", "revision", "remedy"),
    [
        (QWEN3_0_6B, None, "run `make seed-hf-cache`"),
        (*next(iter(PINNED_REVISIONS.items())), "run `make seed-hf-cache`"),
        (QWEN3_0_6B, "0" * 40, "in PINNED_REVISIONS"),
        (_UNSEEDED, None, "in tests/common/models.py"),
    ],
    ids=["seeded", "pinned", "unpinned-revision", "unseeded"],
)
def test_the_failure_names_the_step_that_fills_the_cache(monkeypatch, reference, revision, remedy):
    """A repo the seed already holds needs a re-seed, not a roster edit, and the reverse."""
    monkeypatch.setenv(REQUIRE_HUB_CACHE, "1")
    with pytest.raises(pytest.fail.Exception) as failure:
        require_cached(reference, "tokenizer", revision)
    assert remedy in str(failure.value), str(failure.value)


@pytest.mark.parametrize("reference", [GEMMA3_4B_IT, GPT_OSS_20B_PATCHED])
def test_gated_repos_and_local_paths_may_miss_where_the_cache_must_be_complete(monkeypatch, reference):
    monkeypatch.setenv(REQUIRE_HUB_CACHE, "1")
    assert require_cached(reference, "tokenizer") is None


def test_a_broken_config_raises_instead_of_reading_as_not_cached(tmp_path):
    """Only a miss is a miss: an unknown ``model_type`` must not skip as "not in the cache"."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "halo-no-such-model-type"}))
    with pytest.raises(ValueError, match="halo-no-such-model-type"):
        try_cached_config(str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
