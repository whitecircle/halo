"""Tokenizer, processor and config loading from the local HF cache, for CPU tests needing a real one. With
``local_files_only``, an ``OSError`` means not cached and every other failure propagates. A miss skips, except under
``HALO_TEST_REQUIRE_HUB_CACHE``, where a miss on a seedable repo fails. Gated repos and local paths still skip."""

from pathlib import Path
from typing import NoReturn

import pytest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from halo.env import env_flag
from tests.common.hub_seed import GATED_REPOS, is_hub_repo, seed_repos

REQUIRE_HUB_CACHE = "HALO_TEST_REQUIRE_HUB_CACHE"


def _remedy(reference: str, revision: str | None) -> str:
    """How to get ``reference`` into the cache: re-seed, pin the revision, or add the repo to the roster."""
    seeded = seed_repos()
    spelled = reference if revision is None else f"{reference}@{revision}"
    if (reference, revision) in seeded:
        return f"The seed holds {spelled}: run `make seed-hf-cache`."
    if revision is not None and any(repo == reference for repo, _ in seeded):
        return f"Pin {spelled} in PINNED_REVISIONS (tests/common/models.py) so `make seed-hf-cache` fetches it."
    return f"Name {reference} in tests/common/models.py so `make seed-hf-cache` fetches it."


def fail_if_cache_required(reference: str, what: str, revision: str | None = None) -> None:
    if env_flag(REQUIRE_HUB_CACHE) and is_hub_repo(reference) and reference not in GATED_REPOS:
        pytest.fail(
            f"{what} {reference} is not in the local HF cache, which {REQUIRE_HUB_CACHE} requires. "
            + _remedy(reference, revision)
        )


def skip_uncached(reference: str, what: str, revision: str | None = None) -> NoReturn:
    fail_if_cache_required(reference, what, revision)
    where = "in the local HF cache" if is_hub_repo(reference) else "on this host"
    pytest.skip(f"{what} {reference} is not {where}")


def try_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local cache or None, for roster probes that must keep scanning rather than skip."""
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        fail_if_cache_required(name, "tokenizer", kwargs.get("revision"))
        return None


def load_cached_tokenizer(name: str, **kwargs):
    tokenizer = try_cached_tokenizer(name, **kwargs)
    if tokenizer is None:
        skip_uncached(name, "tokenizer", kwargs.get("revision"))
    return tokenizer


def _try_cached_processor(name: str, **kwargs):
    """The processor from the local cache or None; a processor-only snapshot is enough to render and tokenize."""
    try:
        return AutoProcessor.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        return None


def load_cached_processor(name: str, **kwargs):
    processor = _try_cached_processor(name, **kwargs)
    if processor is None:
        skip_uncached(name, "processor", kwargs.get("revision"))
    return processor


def try_cached_config(reference: str, **kwargs):
    """The model config from the local cache or None; ``reference`` may be a local path, absent paths read None too."""
    if not is_hub_repo(reference) and not Path(reference).exists():
        return None
    try:
        return AutoConfig.from_pretrained(reference, local_files_only=True, **kwargs)
    except OSError:
        fail_if_cache_required(reference, "config", kwargs.get("revision"))
        return None


def load_cached_config(reference: str, **kwargs):
    config = try_cached_config(reference, **kwargs)
    if config is None:
        skip_uncached(reference, "config", kwargs.get("revision"))
    return config
