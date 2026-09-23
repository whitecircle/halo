"""Tokenizer/processor loading from the local HF cache, for CPU tests that need a real one.

``local_files_only=True`` keeps the load off the network, which narrows the ``except OSError``: with
no fetch to fail, an ``OSError`` means the repo is not in the cache. Every other failure (a broken
chat template, an unreadable config, a missing ``trust_remote_code``) propagates rather than being
reported as "not cached".

A cache miss skips the test, except under ``HALO_TEST_REQUIRE_HUB_CACHE``, set where the cache was
seeded from :mod:`tests.common.hub_seed`: there a miss on a repo the seed could hold fails, so a test
that reads a repo outside the seed cannot pass as a skip. Gated repos and local paths still skip.
"""

from typing import NoReturn

import pytest
from transformers import AutoProcessor, AutoTokenizer

from src.env import env_flag
from tests.common.hub_seed import GATED_REPOS, is_hub_repo

REQUIRE_HUB_CACHE = "HALO_TEST_REQUIRE_HUB_CACHE"


def require_cached(reference: str, what: str) -> None:
    """Fail the test when the cache must hold ``reference`` and does not; otherwise return."""
    if env_flag(REQUIRE_HUB_CACHE) and is_hub_repo(reference) and reference not in GATED_REPOS:
        pytest.fail(
            f"{what} {reference} is not in the local HF cache, which {REQUIRE_HUB_CACHE} requires. Name the "
            "repo in tests/common/models.py so the seed (python -m tests.common.hub_seed) fetches it."
        )


def skip_uncached(reference: str, what: str) -> NoReturn:
    """Skip a test whose Hub files are not cached, or fail it where the cache must be complete."""
    require_cached(reference, what)
    pytest.skip(f"{what} {reference} is not in the local HF cache")


def try_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, or ``None`` when it is not cached.

    For roster probes that must keep scanning the other models rather than skip.
    """
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        require_cached(name, "tokenizer")
        return None


def load_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, skipping the test when it is not cached."""
    tokenizer = try_cached_tokenizer(name, **kwargs)
    if tokenizer is None:
        skip_uncached(name, "tokenizer")
    return tokenizer


def _try_cached_processor(name: str, **kwargs):
    """The multimodal processor from the local HF cache, or ``None`` when it is not cached.

    A processor-only snapshot is enough — no weights are needed to render and tokenize.
    """
    try:
        return AutoProcessor.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        return None


def load_cached_processor(name: str, **kwargs):
    """The multimodal processor from the local HF cache, skipping the test when it is not cached."""
    processor = _try_cached_processor(name, **kwargs)
    if processor is None:
        skip_uncached(name, "processor")
    return processor
