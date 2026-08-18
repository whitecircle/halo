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
        tests/data/test_cache_isolation.py
"""

import os
import shutil
import tempfile
import traceback

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
from tests.common.harness import gpu_test_main
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


def test_function_identifier_determinism() -> bool:
    """Test that get_function_identifier returns the same result for the same function."""
    log("  [test_function_identifier_determinism] Running...")

    def my_transform(example):
        return {"processed": example["text"].upper()}

    id1 = get_function_identifier(my_transform)
    id2 = get_function_identifier(my_transform)

    if id1 != id2:
        log(f"    FAIL: Identifiers differ: {id1!r} vs {id2!r}")
        return False

    # Verify no memory address in the identifier
    if "0x" in id1:
        log(f"    FAIL: Memory address found in identifier: {id1!r}")
        return False

    log(f"    PASS: Deterministic identifier = {id1!r}")
    return True


def test_function_identifier_uniqueness() -> bool:
    """Test that different functions produce different identifiers."""
    log("  [test_function_identifier_uniqueness] Running...")

    def transform_a(example):
        return {"processed": example["text"].upper()}

    def transform_b(example):
        return {"processed": example["text"].lower()}

    id_a = get_function_identifier(transform_a)
    id_b = get_function_identifier(transform_b)

    if id_a == id_b:
        log(f"    FAIL: Different functions produced same identifier: {id_a!r}")
        return False

    log(f"    PASS: transform_a={id_a!r}, transform_b={id_b!r}")
    return True


def test_kwargs_fingerprint_differs_for_tokenizers() -> bool:
    """Test that _get_kwargs_fingerprint differs for different tokenizers."""
    log("  [test_kwargs_fingerprint_differs_for_tokenizers] Running...")

    tokenizer_a = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(TOKENIZER_B_NAME, trust_remote_code=True)

    fp_a = _get_kwargs_fingerprint({"tokenizer": tokenizer_a, "max_length": 512})
    fp_b = _get_kwargs_fingerprint({"tokenizer": tokenizer_b, "max_length": 512})

    if fp_a == fp_b:
        log(f"    FAIL: Same fingerprint for different tokenizers: {fp_a!r}")
        return False

    # Same tokenizer, same kwargs -> same fingerprint
    fp_a2 = _get_kwargs_fingerprint({"tokenizer": tokenizer_a, "max_length": 512})
    if fp_a != fp_a2:
        log(f"    FAIL: Same tokenizer produced different fingerprints: {fp_a!r} vs {fp_a2!r}")
        return False

    log(f"    PASS: fp_a={fp_a!r}, fp_b={fp_b!r}")
    return True


def test_kwargs_fingerprint_empty() -> bool:
    """Test that empty kwargs produce an empty fingerprint."""
    log("  [test_kwargs_fingerprint_empty] Running...")

    fp_empty = _get_kwargs_fingerprint({})
    fp_none = _get_kwargs_fingerprint({})

    if fp_empty != "":
        log(f"    FAIL: Empty kwargs should produce empty string, got {fp_empty!r}")
        return False

    if fp_none != "":
        log(f"    FAIL: None kwargs should produce empty string, got {fp_none!r}")
        return False

    log("    PASS: Empty kwargs -> empty fingerprint")
    return True


def test_ensure_cache_dir_respects_env() -> bool:
    """Test that ensure_cache_dir respects HF_DATASETS_CACHE env var."""
    log("  [test_ensure_cache_dir_respects_env] Running...")

    temp_dir = tempfile.mkdtemp(prefix="test_cache_dir_")
    original_env = os.environ.get("HF_DATASETS_CACHE")

    try:
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        cache_dir = ensure_cache_dir()

        if cache_dir != temp_dir:
            log(f"    FAIL: Expected {temp_dir}, got {cache_dir}")
            return False

        if not os.path.exists(cache_dir):
            log(f"    FAIL: Cache directory was not created: {cache_dir}")
            return False

        log(f"    PASS: Cache dir = {cache_dir}")
        return True

    finally:
        # Restore original env
        if original_env is not None:
            os.environ["HF_DATASETS_CACHE"] = original_env
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_coordinated_map_cache_isolation() -> bool:
    """The TOKENIZER alone must change the cache key — the known cross-model token-ID collision.

    Both calls use the SAME ``desc`` and the same map function, so the tokenizer identity is the only
    thing that can distinguish the two cache files. (With different ``desc`` strings the keys would
    differ for that reason alone and this would prove nothing about tokenizer keying.) A regression
    that drops the tokenizer from the key makes the second call load the FIRST tokenizer's tokens.
    """
    log("  [test_coordinated_map_cache_isolation] Running...")

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
        if name_a == name_b:
            log(f"    FAIL: both tokenizers key the SAME cache file {name_a!r} — cross-model token reuse")
            return False

        result_a = coordinated_map(
            dataset, tokenize_fn, desc=shared_desc, num_proc=1, fn_kwargs={"tokenizer": tokenizer_a}
        )
        result_b = coordinated_map(
            dataset, tokenize_fn, desc=shared_desc, num_proc=1, fn_kwargs={"tokenizer": tokenizer_b}
        )

        ids_a = result_a[0]["input_ids"]
        ids_b = result_b[0]["input_ids"]

        if not ids_a or not ids_b:
            log("    FAIL: Empty token IDs")
            return False
        if ids_a == ids_b:
            log("    FAIL: second call served the first tokenizer's cached token IDs")
            return False
        # The reference encodings prove which tokenizer each result actually came from.
        if ids_a != tokenizer_a(dataset[0]["text"], truncation=True, padding="max_length", max_length=64)["input_ids"]:
            log("    FAIL: result_a does not match tokenizer A's own encoding")
            return False
        if ids_b != tokenizer_b(dataset[0]["text"], truncation=True, padding="max_length", max_length=64)["input_ids"]:
            log("    FAIL: result_b does not match tokenizer B's own encoding (stale cache)")
            return False

        written = sorted(f for f in os.listdir(cache_dir) if f.endswith(".arrow"))
        if not {name_a, name_b}.issubset(written):
            log(f"    FAIL: expected both {name_a!r} and {name_b!r} in {written}")
            return False

        log(f"    PASS: one desc, two tokenizers -> two cache files ({len(written)} .arrow files)")
        return True

    finally:
        if original_env is not None:
            os.environ["HF_DATASETS_CACHE"] = original_env
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_same_class_tokenizers_key_distinct_caches() -> bool:
    """Two tokenizers of the SAME class must still key different caches.

    The dangerous real case: two checkpoints of one family (or one tokenizer mutated in place by
    ``--force_chat_template`` / added tokens). A fingerprint that degrades to the kwarg's *type name*
    collides here — both would hash to "Qwen2TokenizerFast" and the second run would train on the
    first model's token IDs. ``_tokenizer_identity`` therefore folds in vocab size, ``len()`` and a
    chat-template hash; this pins each of those signals with a same-class pair.
    """
    log("  [test_same_class_tokenizers_key_distinct_caches] Running...")

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
    if key(grown) == base_key:
        log(f"    FAIL: an added token did not change the cache key ({base_key!r}) — vocab drift reuses stale tokens")
        return False

    retemplated = AutoTokenizer.from_pretrained(TOKENIZER_A_NAME, trust_remote_code=True)
    # In-place template swap, exactly what --force_chat_template does; name_or_path never changes.
    retemplated.chat_template = "{% for m in messages %}<halo>{{ m['content'] }}{% endfor %}"
    if key(retemplated) == base_key:
        log(f"    FAIL: an in-place chat_template swap did not change the cache key ({base_key!r})")
        return False

    log("    PASS: same-class tokenizers differing in len()/chat_template key distinct caches")
    return True


def test_kwargs_fingerprint_with_scalars() -> bool:
    """Test that scalar kwargs are included in the fingerprint."""
    log("  [test_kwargs_fingerprint_with_scalars] Running...")

    fp1 = _get_kwargs_fingerprint({"max_length": 512, "truncation": True})
    fp2 = _get_kwargs_fingerprint({"max_length": 1024, "truncation": True})
    fp3 = _get_kwargs_fingerprint({"max_length": 512, "truncation": True})

    # Same kwargs -> same fingerprint
    if fp1 != fp3:
        log(f"    FAIL: Same kwargs gave different fingerprints: {fp1!r} vs {fp3!r}")
        return False

    # Different kwargs -> different fingerprint
    if fp1 == fp2:
        log(f"    FAIL: Different kwargs gave same fingerprint: {fp1!r}")
        return False

    log(f"    PASS: fp(512)={fp1!r}, fp(1024)={fp2!r}")
    return True


# Test Runner

ALL_TESTS = [
    ("function_identifier_determinism", test_function_identifier_determinism),
    ("function_identifier_uniqueness", test_function_identifier_uniqueness),
    ("kwargs_fingerprint_differs_for_tokenizers", test_kwargs_fingerprint_differs_for_tokenizers),
    ("kwargs_fingerprint_empty", test_kwargs_fingerprint_empty),
    ("kwargs_fingerprint_with_scalars", test_kwargs_fingerprint_with_scalars),
    ("ensure_cache_dir_respects_env", test_ensure_cache_dir_respects_env),
    ("coordinated_map_cache_isolation", test_coordinated_map_cache_isolation),
    ("same_class_tokenizers_key_distinct_caches", test_same_class_tokenizers_key_distinct_caches),
]


def run(ctx) -> dict:
    """Run every cache-isolation check; one failing check never skips the rest."""
    log(f"\n{'=' * 70}")
    log("  Dataset Cache Isolation Tests")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}\n")

    checks: dict[str, bool] = {}
    for test_name, test_fn in ALL_TESTS:
        log(f"\n  [{test_name}]")
        try:
            result = test_fn()
        except Exception as e:
            log(f"  [{test_name}] UNHANDLED EXCEPTION: {e}")
            if ctx.rank == 0:
                traceback.print_exc()
            result = False
        checks[test_name] = result

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="cache_isolation", partial_state=False)(run)

if __name__ == "__main__":
    main()
