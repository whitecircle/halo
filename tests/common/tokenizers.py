"""Tokenizer/processor loading from the local HF cache, for CPU tests that need a real one.

``local_files_only=True`` keeps the load off the network, which is what makes the narrow
``except OSError`` honest: with no fetch to fail, an ``OSError`` means the repo is not in the cache
and nothing else. Every other failure — a broken chat template, an unreadable config, a missing
``trust_remote_code`` — propagates instead of being silently reported as "not cached".
"""

import pytest
from transformers import AutoProcessor, AutoTokenizer


def try_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, or ``None`` when it is not cached.

    For roster probes that must keep scanning the other models rather than skip.
    """
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True, **kwargs)
    except OSError:
        return None


def load_cached_tokenizer(name: str, **kwargs):
    """The tokenizer from the local HF cache, skipping the test when it is not cached."""
    tokenizer = try_cached_tokenizer(name, **kwargs)
    if tokenizer is None:
        pytest.skip(f"tokenizer {name} is not in the local HF cache")
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
        pytest.skip(f"processor {name} is not in the local HF cache")
    return processor
