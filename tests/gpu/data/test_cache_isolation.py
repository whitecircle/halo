#!/usr/bin/env python
"""
Test dataset cache isolation: different tokenizers produce different cache files.

Validates:
1. get_function_identifier() is deterministic (no memory addresses)
2. _get_kwargs_fingerprint() differs for different tokenizers
3. HF_DATASETS_CACHE env var is respected by ensure_cache_dir()
4. Two parallel processes with different tokenizers don't share stale cache
5. coordinated_map produces distinct cache files when fn_kwargs differ

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/data/test_cache_isolation.py
"""

import os
import shutil
import tempfile

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from src.data.pipeline.processing import (
    _build_cache_file_name,
    _get_kwargs_fingerprint,
    coordinated_map,
    ensure_cache_dir,
    get_function_identifier,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log

# Configuration

# Use two distinct tokenizers to test cache isolation
TOKENIZER_A_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER_B_NAME = "openai-community/gpt2"

NUM_SAMPLES = 20
SEED = 42


# Utilities


def create_test_dataset(num_samples: int = NUM_SAMPLES) -> Dataset:
    """Create a small synthetic dataset for cache testing."""
    data = []
    for i in range(num_samples):
        data.append(
            {
                "text": f"Sample {i}: The quick brown fox jumps over the lazy dog.",
                "id": i,
            }
        )
    return Dataset.from_list(data)


# Test Functions


def test_function_identifier_determinism() -> None:
    """Two function objects with the same code get the same identifier, and it carries no address.

    Two separate objects stand in for two processes: an identifier that folded in anything
    object-specific would key a different cache on every rank.
    """

    def make_transform():
        def my_transform(example):
            return {"processed": example["text"].upper()}

        return my_transform

    # Both alive at once: a temporary freed before the second is built hands it the same address.
    first, second = make_transform(), make_transform()
    id1 = get_function_identifier(first)
    id2 = get_function_identifier(second)

    assert id1 == id2, f"Identifiers differ across two objects of one function: {id1!r} vs {id2!r}"
    assert "0x" not in id1, f"Memory address found in identifier: {id1!r}"
    log(f"    Deterministic identifier = {id1!r}")


def test_function_identifier_uniqueness() -> None:
    """Test that different functions produce different identifiers."""

    def transform_a(example):
        return {"processed": example["text"].upper()}

    def transform_b(example):
        return {"processed": example["text"].lower()}

    id_a = get_function_identifier(transform_a)
    id_b = get_function_identifier(transform_b)

    assert id_a != id_b, f"Different functions produced same identifier: {id_a!r}"
    log(f"    transform_a={id_a!r}, transform_b={id_b!r}")


def test_kwargs_fingerprint_differs_for_tokenizers() -> None:
    """Different tokenizers fingerprint differently; two loads of one tokenizer fingerprint alike."""
    tokenizer_a = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(TOKENIZER_B_NAME, trust_remote_code=True)

    fp_a = _get_kwargs_fingerprint({"tokenizer": tokenizer_a, "max_length": 512})
    fp_b = _get_kwargs_fingerprint({"tokenizer": tokenizer_b, "max_length": 512})
    assert fp_a != fp_b, f"Same fingerprint for different tokenizers: {fp_a!r}"

    # A separately loaded instance, as every rank and every restart holds its own.
    tokenizer_a_reloaded = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    fp_a2 = _get_kwargs_fingerprint({"tokenizer": tokenizer_a_reloaded, "max_length": 512})
    assert fp_a == fp_a2, f"Two loads of one tokenizer produced different fingerprints: {fp_a!r} vs {fp_a2!r}"

    log(f"    fp_a={fp_a!r}, fp_b={fp_b!r}")


def test_kwargs_fingerprint_empty() -> None:
    """Empty and None kwargs produce an empty fingerprint."""
    fp_empty = _get_kwargs_fingerprint({})
    fp_none = _get_kwargs_fingerprint(None)

    assert fp_empty == "", f"Empty kwargs should produce empty string, got {fp_empty!r}"
    assert fp_none == "", f"None kwargs should produce empty string, got {fp_none!r}"


def test_ensure_cache_dir_respects_env() -> None:
    """Test that ensure_cache_dir respects HF_DATASETS_CACHE env var."""
    temp_dir = tempfile.mkdtemp(prefix="test_cache_dir_")
    original_env = os.environ.get("HF_DATASETS_CACHE")

    try:
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        cache_dir = ensure_cache_dir()

        assert cache_dir == temp_dir, f"Expected {temp_dir}, got {cache_dir}"
        assert os.path.exists(cache_dir), f"Cache directory was not created: {cache_dir}"
        log(f"    Cache dir = {cache_dir}")

    finally:
        # Restore original env
        if original_env is not None:
            os.environ["HF_DATASETS_CACHE"] = original_env
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_coordinated_map_cache_isolation() -> None:
    """The TOKENIZER alone must change the cache key — the known cross-model token-ID collision.

    Both calls use the SAME ``desc`` and the same map function, so the tokenizer identity is the only
    thing that can distinguish the two cache files. (With different ``desc`` strings the keys would
    differ for that reason alone and this would prove nothing about tokenizer keying.) A regression
    that drops the tokenizer from the key makes the second call load the FIRST tokenizer's tokens.
    """
    cache_dir = tempfile.mkdtemp(prefix="test_cache_iso_")
    original_env = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    try:
        dataset = create_test_dataset(10)

        tokenizer_a = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
        if tokenizer_a.pad_token is None:
            tokenizer_a.pad_token = tokenizer_a.eos_token

        tokenizer_b = AutoTokenizer.from_pretrained(TOKENIZER_B_NAME, trust_remote_code=True)
        if tokenizer_b.pad_token is None:
            tokenizer_b.pad_token = tokenizer_b.eos_token

        def tokenize_fn(example, tokenizer=None):
            """Simple tokenization function used as the map_fn."""
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=64,
            )

        # Same desc for both: the cache name may only diverge on the tokenizer.
        shared_desc = "tokenize"
        name_a = _build_cache_file_name(
            "map", tokenize_fn, dataset, shared_desc, {"fn_kwargs": {"tokenizer": tokenizer_a}}
        )
        name_b = _build_cache_file_name(
            "map", tokenize_fn, dataset, shared_desc, {"fn_kwargs": {"tokenizer": tokenizer_b}}
        )
        assert name_a != name_b, f"both tokenizers key the SAME cache file {name_a!r}: cross-model token reuse"

        result_a = coordinated_map(
            dataset, tokenize_fn, desc=shared_desc, num_proc=1, fn_kwargs={"tokenizer": tokenizer_a}
        )
        result_b = coordinated_map(
            dataset, tokenize_fn, desc=shared_desc, num_proc=1, fn_kwargs={"tokenizer": tokenizer_b}
        )

        ids_a = result_a[0]["input_ids"]
        ids_b = result_b[0]["input_ids"]

        assert ids_a and ids_b, "Empty token IDs"
        assert ids_a != ids_b, "second call served the first tokenizer's cached token IDs"
        # The reference encodings prove which tokenizer each result actually came from.
        expected_a = tokenizer_a(dataset[0]["text"], truncation=True, padding="max_length", max_length=64)
        expected_b = tokenizer_b(dataset[0]["text"], truncation=True, padding="max_length", max_length=64)
        assert ids_a == expected_a["input_ids"], "result_a does not match tokenizer A's own encoding"
        assert ids_b == expected_b["input_ids"], "result_b does not match tokenizer B's own encoding (stale cache)"

        written = sorted(f for f in os.listdir(cache_dir) if f.endswith(".arrow"))
        assert {name_a, name_b}.issubset(written), f"expected both {name_a!r} and {name_b!r} in {written}"
        log(f"    one desc, two tokenizers -> two cache files ({len(written)} .arrow files)")

    finally:
        if original_env is not None:
            os.environ["HF_DATASETS_CACHE"] = original_env
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_same_class_tokenizers_key_distinct_caches() -> None:
    """Two tokenizers of the SAME class must still key different caches.

    The dangerous real case: two checkpoints of one family (or one tokenizer mutated in place by
    ``--force_chat_template`` / added tokens). A fingerprint that degrades to the kwarg's *type name*
    collides here — both would hash to "Qwen2TokenizerFast" and the second run would train on the
    first model's token IDs. ``_tokenizer_identity`` therefore folds in vocab size, ``len()`` and a
    chat-template hash; this pins each of those signals with a same-class pair.
    """
    dataset = create_test_dataset(4)

    def tokenize_fn(example, tokenizer=None):
        return tokenizer(example["text"], truncation=True, max_length=32)

    def key(tokenizer) -> str:
        return _build_cache_file_name(
            "map", tokenize_fn, dataset, "same_class", {"fn_kwargs": {"tokenizer": tokenizer}}
        )

    base = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    base_key = key(base)

    grown = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    grown.add_tokens(["<halo_cache_probe>"])  # same class + name_or_path, different len()
    assert key(grown) != base_key, (
        f"an added token did not change the cache key ({base_key!r}): vocab drift reuses stale tokens"
    )

    retemplated = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    # In-place template swap, exactly what --force_chat_template does; name_or_path never changes.
    retemplated.chat_template = "{% for m in messages %}<halo>{{ m['content'] }}{% endfor %}"
    assert key(retemplated) != base_key, f"an in-place chat_template swap did not change the cache key ({base_key!r})"


def test_kwargs_fingerprint_with_scalars() -> None:
    """Test that scalar kwargs are included in the fingerprint."""
    fp1 = _get_kwargs_fingerprint({"max_length": 512, "truncation": True})
    fp2 = _get_kwargs_fingerprint({"max_length": 1024, "truncation": True})
    fp3 = _get_kwargs_fingerprint({"max_length": 512, "truncation": True})

    assert fp1 == fp3, f"Same kwargs gave different fingerprints: {fp1!r} vs {fp3!r}"
    assert fp1 != fp2, f"Different kwargs gave same fingerprint: {fp1!r}"
    log(f"    fp(512)={fp1!r}, fp(1024)={fp2!r}")


def run(ctx) -> dict:
    """Run every cache-isolation check; one failing check never skips the rest."""
    log(f"\n{'=' * 70}")
    log("  Dataset Cache Isolation Tests")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}\n")

    checks: dict[str, bool] = {}
    record_check(checks, "function_identifier_determinism", test_function_identifier_determinism)
    record_check(checks, "function_identifier_uniqueness", test_function_identifier_uniqueness)
    record_check(checks, "kwargs_fingerprint_differs_for_tokenizers", test_kwargs_fingerprint_differs_for_tokenizers)
    record_check(checks, "kwargs_fingerprint_empty", test_kwargs_fingerprint_empty)
    record_check(checks, "kwargs_fingerprint_with_scalars", test_kwargs_fingerprint_with_scalars)
    record_check(checks, "ensure_cache_dir_respects_env", test_ensure_cache_dir_respects_env)
    record_check(checks, "coordinated_map_cache_isolation", test_coordinated_map_cache_isolation)
    record_check(checks, "same_class_tokenizers_key_distinct_caches", test_same_class_tokenizers_key_distinct_caches)
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="cache_isolation", partial_state=False)(run)

if __name__ == "__main__":
    main()
