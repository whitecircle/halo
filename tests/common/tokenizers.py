"""Tokenizer, processor and config loading from the local HF cache, for CPU tests that need a real one.

``local_files_only=True`` keeps the load off the network, which narrows the ``except OSError``: with
no fetch to fail, an ``OSError`` means the repo is not in the cache. Every other failure (a broken
chat template, an unknown ``model_type``, a missing ``trust_remote_code``) propagates rather than
being reported as "not cached".

A cache miss skips the test, except under ``HALO_TEST_REQUIRE_HUB_CACHE``, set where the cache was
seeded from :mod:`tests.common.hub_seed`: there a miss on a repo the seed could hold fails, so a test
that reads a repo outside the seed cannot pass as a skip. Gated repos and local paths still skip.
"""

from pathlib import Path
from typing import NoReturn

import pytest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from src.env import env_flag
from tests.common.hub_seed import GATED_REPOS, is_hub_repo, seed

REQUIRE_HUB_CACHE = "HALO_TEST_REQUIRE_HUB_CACHE"


def _remedy(reference: str, revision: str | None) -> str:
    """How to get ``reference`` into the cache: re-seed, pin the revision, or add the repo to the roster."""
    seeded = seed()
    spelled = reference if revision is None else f"{reference}@{revision}"
    if (reference, revision) in seeded:
        return f"The seed holds {spelled}: run `make seed-hf-cache`."
    if revision is not None and any(repo == reference for repo, _ in seeded):
        return f"Pin {spelled} in PINNED_REVISIONS (tests/common/models.py) so `make seed-hf-cache` fetches it."
    return f"Name {reference} in tests/common/models.py so `make seed-hf-cache` fetches it."


def require_cached(reference: str, what: str, revision: str | None = None) -> None:
    """Fail the test when the cache must hold ``reference`` and does not; otherwise return."""
    if env_flag(REQUIRE_HUB_CACHE) and is_hub_repo(reference) and reference not in GATED_REPOS:
        pytest.fail(
            f"{what} {reference} is not in the local HF cache, which {REQUIRE_HUB_CACHE} requires. "
            + _remedy(reference, revision)
        )


def skip_uncached(reference: str, what: str, revision: str | None = None) -> NoReturn:
    """Skip a test whose Hub files are not cached, or fail it where the cache must be complete."""
    require_cached(reference, what, revision)
    where = "in the local HF cache" if is_hub_repo(reference) else "on this host"
    pytest.skip(f"{what} {reference} is not {where}")


def try_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, or ``None`` when it is not cached.

    For roster probes that must keep scanning the other models rather than skip.
    """
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        require_cached(name, "tokenizer", kwargs.get("revision"))
        return None


def load_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, skipping the test when it is not cached."""
    tokenizer = try_cached_tokenizer(name, **kwargs)
    if tokenizer is None:
        skip_uncached(name, "tokenizer", kwargs.get("revision"))
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
        skip_uncached(name, "processor", kwargs.get("revision"))
    return processor


def try_cached_config(reference: str, **kwargs):
    """The model config from the local HF cache, or ``None`` when it is not cached.

    ``reference`` may be a local checkpoint path, which a host without it resolves to ``None`` too.
    """
    if not is_hub_repo(reference) and not Path(reference).exists():
        return None
    try:
        return AutoConfig.from_pretrained(reference, local_files_only=True, **kwargs)
    except OSError:
        require_cached(reference, "config", kwargs.get("revision"))
        return None


def load_cached_config(reference: str, **kwargs):
    """The model config from the local HF cache, skipping the test when it is not cached."""
    config = try_cached_config(reference, **kwargs)
    if config is None:
        skip_uncached(reference, "config", kwargs.get("revision"))
    return config
