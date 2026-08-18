#!/usr/bin/env python
"""
Tests for _build_cache_file_name and datasets caching collision avoidance.

Ensures that cache file names differ when any significant input changes
(operation type, function identity, dataset content/shape, fn_kwargs,
description) and remain stable across repeated calls.

Run: python tests/cpu/data/test_cache_file_naming.py
"""

import ast
import inspect
from pathlib import Path

import pytest
from datasets import Dataset
from transformers import ProcessorMixin

from src.data.collators.self_distill import SelfDistillTextCollator
from src.data.pipeline import processing
from src.data.pipeline.processing import (
    _build_cache_file_name,
    _get_closure_fingerprint,
    _get_kwargs_fingerprint,
    coordinated_filter,
    coordinated_map,
    get_function_identifier,
)
from src.data.spans import COLLATOR_SPAN_POLICY, PACKED_SPAN_POLICY

# Helper functions & datasets


def _add_one(x):
    return x + 1


def _multiply_two(x):
    return x * 2


def _make_dataset(n: int, seed: int = 0) -> Dataset:
    """Create a small in-memory dataset with a deterministic fingerprint."""
    return Dataset.from_dict({"value": list(range(seed, seed + n))})


SELF_DISTILL_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/training/distillation/self_distill.py"


# _build_cache_file_name — determinism


def test_deterministic_output():
    """Calling with identical inputs must return the exact same cache name."""
    ds = _make_dataset(10)
    name1 = _build_cache_file_name("map", _add_one, ds, "desc", {})
    name2 = _build_cache_file_name("map", _add_one, ds, "desc", {})
    assert name1 == name2, f"Non-deterministic: '{name1}' vs '{name2}'"


def test_cache_name_format():
    """Cache name must be cache-<hex>.arrow."""
    ds = _make_dataset(5)
    name = _build_cache_file_name("map", _add_one, ds, None, {})
    assert name.startswith("cache-"), f"Bad prefix: {name}"
    assert name.endswith(".arrow"), f"Bad suffix: {name}"
    hex_part = name[len("cache-") : -len(".arrow")]
    assert len(hex_part) == 32, f"Expected 32 hex chars, got {len(hex_part)}"
    int(hex_part, 16)  # Raises ValueError if not valid hex


# _build_cache_file_name — collision avoidance


def test_different_operation_types():
    """'map' vs 'filter' with otherwise identical inputs must differ."""
    ds = _make_dataset(10)
    name_map = _build_cache_file_name("map", _add_one, ds, "desc", {})
    name_filter = _build_cache_file_name("filter", _add_one, ds, "desc", {})
    assert name_map != name_filter, "map and filter produced same cache name"


def test_different_functions():
    """Different functions must produce different cache names."""
    ds = _make_dataset(10)
    name_a = _build_cache_file_name("map", _add_one, ds, "desc", {})
    name_b = _build_cache_file_name("map", _multiply_two, ds, "desc", {})
    assert name_a != name_b, "Different functions produced same cache name"


def test_different_desc():
    """Different descriptions must produce different cache names."""
    ds = _make_dataset(10)
    name1 = _build_cache_file_name("map", _add_one, ds, "tokenize", {})
    name2 = _build_cache_file_name("map", _add_one, ds, "format", {})
    assert name1 != name2, "Different descriptions produced same cache name"


def test_different_dataset_sizes():
    """Datasets with different lengths must produce different cache names."""
    ds_small = _make_dataset(5)
    ds_large = _make_dataset(20)
    name_small = _build_cache_file_name("map", _add_one, ds_small, "d", {})
    name_large = _build_cache_file_name("map", _add_one, ds_large, "d", {})
    assert name_small != name_large, "Different-sized datasets produced same cache name"


def test_different_dataset_content_same_size():
    """Datasets with same length but different content must differ (via _fingerprint)."""
    ds_a = _make_dataset(10, seed=0)
    ds_b = _make_dataset(10, seed=100)
    # HF datasets assigns different _fingerprint to different data
    assert ds_a._fingerprint != ds_b._fingerprint, "Test setup error: datasets have same fingerprint"
    name_a = _build_cache_file_name("map", _add_one, ds_a, "d", {})
    name_b = _build_cache_file_name("map", _add_one, ds_b, "d", {})
    assert name_a != name_b, "Same-size different-content datasets produced same cache name"


def test_different_fn_kwargs_tokenizer():
    """Different fn_kwargs (e.g. different tokenizer name_or_path) must differ."""
    ds = _make_dataset(10)

    class MockTokenizerA:
        name_or_path = "model-A"

    class MockTokenizerB:
        name_or_path = "model-B"

    name_a = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"tokenizer": MockTokenizerA()}})
    name_b = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"tokenizer": MockTokenizerB()}})
    assert name_a != name_b, "Different tokenizers produced same cache name"


def test_fn_kwargs_vs_no_fn_kwargs():
    """Having fn_kwargs vs not having them must differ."""
    ds = _make_dataset(10)
    name_no_kwargs = _build_cache_file_name("map", _add_one, ds, "d", {})
    name_with_kwargs = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 512}})
    assert name_no_kwargs != name_with_kwargs, "With/without fn_kwargs produced same cache name"


def test_different_fn_kwargs_primitives():
    """Different primitive fn_kwargs values must differ."""
    ds = _make_dataset(10)
    name_a = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 512}})
    name_b = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 1024}})
    assert name_a != name_b, "Different max_length values produced same cache name"


def test_different_fn_kwargs_keys():
    """Different fn_kwargs key names must differ."""
    ds = _make_dataset(10)
    name_a = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"alpha": 1}})
    name_b = _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"beta": 1}})
    assert name_a != name_b, "Different kwarg keys produced same cache name"


# _build_cache_file_name — DatasetDict handling


def test_dataset_dict_includes_split_keys():
    """DatasetDict cache name must depend on split names."""
    from datasets import DatasetDict

    dd_ab = DatasetDict(
        {
            "train": _make_dataset(5),
            "test": _make_dataset(3),
        }
    )
    dd_cd = DatasetDict(
        {
            "validation": _make_dataset(5),
            "test": _make_dataset(3),
        }
    )
    name_ab = _build_cache_file_name("map", _add_one, dd_ab, "d", {})
    name_cd = _build_cache_file_name("map", _add_one, dd_cd, "d", {})
    assert name_ab != name_cd, "Different DatasetDict splits produced same cache name"


def test_dataset_vs_dataset_dict():
    """A Dataset and a DatasetDict (even with same total size) must differ."""
    from datasets import DatasetDict

    ds = _make_dataset(10)
    dd = DatasetDict({"train": _make_dataset(10)})
    name_ds = _build_cache_file_name("map", _add_one, ds, "d", {})
    name_dd = _build_cache_file_name("map", _add_one, dd, "d", {})
    assert name_ds != name_dd, "Dataset and DatasetDict produced same cache name"


def test_dataset_dict_different_content_same_splits():
    """DatasetDicts with identical split names/sizes but different content must differ.

    A cache key relying on `len(dataset)` (= number of splits) and
    `getattr(dataset, '_fingerprint', '')` (missing on DatasetDict) lets an
    updated underlying source (e.g., re-uploaded S3 dataset) silently reuse the
    old tokenized cache.
    """
    from datasets import DatasetDict

    dd_v1 = DatasetDict(
        {
            "train": _make_dataset(10, seed=0),
            "test": _make_dataset(4, seed=0),
        }
    )
    dd_v2 = DatasetDict(
        {
            "train": _make_dataset(10, seed=100),
            "test": _make_dataset(4, seed=100),
        }
    )
    assert dd_v1["train"]._fingerprint != dd_v2["train"]._fingerprint, (
        "Test setup error: per-split fingerprints should differ"
    )
    name_v1 = _build_cache_file_name("map", _add_one, dd_v1, "d", {})
    name_v2 = _build_cache_file_name("map", _add_one, dd_v2, "d", {})
    assert name_v1 != name_v2, (
        "DatasetDicts with different underlying content collided — cache will "
        "serve stale tokenization after a source dataset update."
    )


def test_dataset_dict_same_content_stable():
    """Identical DatasetDicts must produce the same cache name (determinism)."""
    from datasets import DatasetDict

    dd_a = DatasetDict(
        {
            "train": _make_dataset(10, seed=7),
            "test": _make_dataset(3, seed=7),
        }
    )
    dd_b = DatasetDict(
        {
            "train": _make_dataset(10, seed=7),
            "test": _make_dataset(3, seed=7),
        }
    )
    name_a = _build_cache_file_name("map", _add_one, dd_a, "d", {})
    name_b = _build_cache_file_name("map", _add_one, dd_b, "d", {})
    assert name_a == name_b, "Identical DatasetDicts produced different cache names"


# _build_cache_file_name — lambda and closure differentiation


def test_closures_with_different_captured_values():
    """Closures capturing different scalar values MUST disambiguate via the closure fingerprint.

    get_function_identifier hashes only the source, which is identical for two closures from
    the same factory. _get_closure_fingerprint is what keys on the captured free variables —
    so make_adder(1) and make_adder(2), whose only difference is the captured int n, must
    produce DIFFERENT closure fingerprints, hence different cache names. The comparisons below stay
    unconditional: an ``if name_1 == name_2: pass`` branch would assert nothing about the closure
    path.
    """

    def make_adder(n):
        def adder(x):
            return x + n

        return adder

    ds = _make_dataset(10)
    add_1 = make_adder(1)
    add_2 = make_adder(2)

    # The captured int scalar must move the closure fingerprint.
    fp_1 = _get_closure_fingerprint(add_1)
    fp_2 = _get_closure_fingerprint(add_2)
    assert fp_1 != fp_2, (
        f"captured int n=1 vs n=2 produced identical closure fingerprints ({fp_1!r}) — "
        "the closure-fingerprint path no longer disambiguates scalar captures"
    )

    # ...and therefore the full cache name must differ even with identical source + no fn_kwargs.
    name_1 = _build_cache_file_name("map", add_1, ds, "d", {})
    name_2 = _build_cache_file_name("map", add_2, ds, "d", {})
    assert name_1 != name_2, (
        "closures capturing different scalars collided on one cache name despite differing closure fingerprints"
    )

    # And fn_kwargs remains an independent tiebreaker.
    name_1_kw = _build_cache_file_name("map", add_1, ds, "d", {"fn_kwargs": {"n": 1}})
    name_2_kw = _build_cache_file_name("map", add_2, ds, "d", {"fn_kwargs": {"n": 2}})
    assert name_1_kw != name_2_kw, "fn_kwargs should differentiate closures with different captured values"


# _build_cache_file_name — functions with same name, different code


def test_same_name_different_body():
    """Two functions with the same __name__ but different bodies must differ."""
    ds = _make_dataset(10)

    def process(x):
        return x + 1

    name_a = _build_cache_file_name("map", process, ds, "d", {})

    # Redefine 'process' with different code
    def process(x):
        return x * 3 + 7

    name_b = _build_cache_file_name("map", process, ds, "d", {})
    assert name_a != name_b, "Same-name functions with different code produced same cache name"


# _get_kwargs_fingerprint — collision edge cases


def test_kwargs_different_tokenizer_names():
    """Two tokenizers with different name_or_path must fingerprint differently."""

    class TokA:
        name_or_path = "org/model-v1"

    class TokB:
        name_or_path = "org/model-v2"

    fp_a = _get_kwargs_fingerprint({"processing_class": TokA()})
    fp_b = _get_kwargs_fingerprint({"processing_class": TokB()})
    assert fp_a != fp_b, "Different tokenizer name_or_path produced same fingerprint"


def test_kwargs_different_vocab_sizes():
    """Two configs with different vocab_size must fingerprint differently."""

    class CfgA:
        vocab_size = 32000

    class CfgB:
        vocab_size = 128256

    fp_a = _get_kwargs_fingerprint({"config": CfgA()})
    fp_b = _get_kwargs_fingerprint({"config": CfgB()})
    assert fp_a != fp_b, "Different vocab_size produced same fingerprint"


def test_kwargs_bool_vs_int_distinction():
    """bool and int values that look similar must fingerprint differently."""
    fp_true = _get_kwargs_fingerprint({"flag": True})
    fp_one = _get_kwargs_fingerprint({"flag": 1})
    # True and 1 are different values in the repr, so should differ
    assert fp_true != fp_one, "True and 1 produced same fingerprint"


def test_kwargs_name_or_path_takes_priority():
    """Object with both name_or_path and vocab_size should use name_or_path."""

    class ModelA:
        name_or_path = "model-A"
        vocab_size = 32000

    class ModelB:
        name_or_path = "model-B"
        vocab_size = 32000  # same vocab_size, different name

    fp_a = _get_kwargs_fingerprint({"model": ModelA()})
    fp_b = _get_kwargs_fingerprint({"model": ModelB()})
    assert fp_a != fp_b, "Objects with different name_or_path but same vocab_size collided"


def test_kwargs_scalar_lists_distinct():
    """Scalar lists/tuples serialize by value: two different eos_token_id lists must key
    different caches (collapsing to the type name served one list's cache for every other)."""
    fp_a = _get_kwargs_fingerprint({"eos_token_ids": [1, 2, 3]})
    fp_b = _get_kwargs_fingerprint({"eos_token_ids": [4, 5, 6]})
    assert fp_a != fp_b, "Different scalar lists must produce different fingerprints"

    fp_tuple_a = _get_kwargs_fingerprint({"eos_token_ids": (1, 2, 3)})
    fp_tuple_b = _get_kwargs_fingerprint({"eos_token_ids": (9,)})
    assert fp_tuple_a != fp_tuple_b, "Different scalar tuples must produce different fingerprints"

    # Stability: the same list value keys the same cache.
    assert fp_a == _get_kwargs_fingerprint({"eos_token_ids": [1, 2, 3]})

    # Non-scalar lists still fall back to the type name (unhashable/heavy contents).
    fp_obj_a = _get_kwargs_fingerprint({"objs": [object()]})
    fp_obj_b = _get_kwargs_fingerprint({"objs": [object()]})
    assert fp_obj_a == fp_obj_b, "Lists of non-scalars keep the type-name fallback"


# _build_cache_file_name — dataset fingerprint edge cases


def test_shuffled_dataset_different_fingerprint():
    """A shuffled dataset must produce a different cache name."""
    ds = _make_dataset(20)
    ds_shuffled = ds.shuffle(seed=42)
    name_orig = _build_cache_file_name("map", _add_one, ds, "d", {})
    name_shuffled = _build_cache_file_name("map", _add_one, ds_shuffled, "d", {})
    assert name_orig != name_shuffled, "Shuffled dataset produced same cache name as original"


# _build_cache_file_name — multiple kwargs interaction


def test_fn_kwargs_subset_differs():
    """Subsets of fn_kwargs must produce different fingerprints."""
    ds = _make_dataset(10)
    name_a = _build_cache_file_name(
        "map",
        _add_one,
        ds,
        "d",
        {"fn_kwargs": {"max_length": 512, "truncation": True}},
    )
    name_b = _build_cache_file_name(
        "map",
        _add_one,
        ds,
        "d",
        {"fn_kwargs": {"max_length": 512}},
    )
    assert name_a != name_b, "Different number of fn_kwargs produced same cache name"


def test_output_affecting_map_kwargs_change_cache_name():
    """Output-affecting map kwargs (remove_columns, batched, batch_size, with_indices) MUST be
    part of the cache key: two calls differing only in remove_columns produce different rows, and
    a shared key would silently serve one call's cached rows to the other (stale-cache reuse).

    Ignoring these kwargs collides the two names.
    """
    ds = _make_dataset(10)
    base = {"remove_columns": ["value"], "batched": True}
    name_base = _build_cache_file_name("map", _add_one, ds, "d", dict(base))

    for variant in (
        {"remove_columns": ["other"], "batched": True},
        {"remove_columns": ["value"], "batched": False},
        {"remove_columns": ["value"], "batched": True, "batch_size": 7},
        {"remove_columns": ["value"], "batched": True, "with_indices": True},
    ):
        name_variant = _build_cache_file_name("map", _add_one, ds, "d", variant)
        assert name_variant != name_base, f"cache name must change for {variant}"


# _get_closure_fingerprint — the cross-model token-ID collision. The processors built by
# create_llm_processor capture the *tokenizer* and *max_length* as closure free variables, NOT as
# fn_kwargs: source-hashing (get_function_identifier) is identical across tokenizers and the
# fn_kwargs fingerprint never sees them. Without the closure fingerprint, two runs sharing a
# processor factory + dataset collide on one cache file and load another model's token IDs (a
# 262k-vocab tokenizer's IDs for a 201k-vocab model → out-of-bounds embedding gather, crash at step 0).


class _MockTokenizer:
    """Mimics the attributes _get_closure_fingerprint keys on."""

    def __init__(self, name_or_path, vocab_size):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size


def _make_processor(tokenizer, max_length):
    """Factory whose returned closure captures tokenizer + max_length (like create_llm_processor)."""

    def process(example):
        # references the free variables so they live in __closure__
        return {"ids": [tokenizer.vocab_size, max_length]}

    return process


def test_closure_fingerprint_differs_by_tokenizer():
    """Two processors from the same factory but different tokenizers → different closure fp."""
    tok_a = _MockTokenizer("org/model-262k", 262000)
    tok_b = _MockTokenizer("org/model-201k", 201000)
    fp_a = _get_closure_fingerprint(_make_processor(tok_a, 4096))
    fp_b = _get_closure_fingerprint(_make_processor(tok_b, 4096))
    assert fp_a and fp_b, "closure fingerprint should be non-empty when a tokenizer is captured"
    assert fp_a != fp_b, "different captured tokenizers must produce different closure fingerprints"


def test_closure_fingerprint_differs_by_max_length():
    """Same tokenizer, different captured max_length → different closure fp."""
    tok = _MockTokenizer("org/model", 32000)
    fp_512 = _get_closure_fingerprint(_make_processor(tok, 512))
    fp_4096 = _get_closure_fingerprint(_make_processor(tok, 4096))
    assert fp_512 != fp_4096, "different captured max_length must change the closure fingerprint"


def test_closure_fingerprint_empty_for_no_closure():
    """A plain function with no closure cells yields an empty fingerprint."""
    assert _get_closure_fingerprint(_add_one) == ""


def test_cache_name_breaks_cross_tokenizer_collision():
    """End-to-end regression: identical factory + dataset + tokenizer-name but
    different captured tokenizers must NOT share a cache file.

    This is the exact scenario that corrupted token IDs across models — the
    map function source and fn_kwargs are identical, only the closure differs.
    """
    ds = _make_dataset(10)
    proc_262k = _make_processor(_MockTokenizer("org/model-262k", 262000), 4096)
    proc_201k = _make_processor(_MockTokenizer("org/model-201k", 201000), 4096)
    name_262k = _build_cache_file_name("map", proc_262k, ds, "tokenize", {})
    name_201k = _build_cache_file_name("map", proc_201k, ds, "tokenize", {})
    assert name_262k != name_201k, (
        "Different closure-captured tokenizers collided on one cache file — "
        "this is the cross-model token-ID corruption regression."
    )


def test_cache_name_same_tokenizer_stable():
    """Same captured tokenizer + max_length → identical cache name (reuse works)."""
    ds = _make_dataset(10)
    tok = _MockTokenizer("org/model", 32000)
    name1 = _build_cache_file_name("map", _make_processor(tok, 4096), ds, "tokenize", {})
    name2 = _build_cache_file_name("map", _make_processor(tok, 4096), ds, "tokenize", {})
    assert name1 == name2, "Identical closure capture must reuse the same cache file"


# get_function_identifier — additional collision tests


def test_function_id_method_vs_function():
    """A method and a standalone function with same body should differ."""

    class Processor:
        def transform(self, x):
            return x + 1

    def transform(x):
        return x + 1

    id_method = get_function_identifier(Processor.transform)
    id_func = get_function_identifier(transform)
    # They should differ because source includes class context
    assert id_method != id_func, "Method and standalone function with same name produced same ID"


def test_function_id_builtin_fallback():
    """Built-in functions (no source code) should still produce a valid ID."""
    fid = get_function_identifier(len)
    assert isinstance(fid, str)
    assert len(fid) > 0


# _tokenizer_identity — processors and special tokens


class _LeafTok:
    """Minimal tokenizer stub carrying every field the identity folds in."""

    def __init__(self, name="org/model", vocab=32000, template="{{ messages }}", eos=2, bos=1, pad=0, side="right"):
        self.name_or_path = name
        self.vocab_size = vocab
        self.chat_template = template
        self.eos_token_id = eos
        self.bos_token_id = bos
        self.pad_token_id = pad
        self.padding_side = side
        self.truncation_side = "right"

    def __len__(self):
        return self.vocab_size


class _Processor(ProcessorMixin):
    """Processor stub: no name_or_path/vocab_size of its own — identity must come from the descent.

    A real ``ProcessorMixin``, because that type is what the descent is gated on.
    """

    def __init__(self, tokenizer, chat_template):
        self.tokenizer = tokenizer
        self.chat_template = chat_template


def test_processor_identity_breaks_the_class_name_collision():
    """Two checkpoints sharing a processor class but differing in chat template key different caches.

    Collapsed to the bare class name they share one cache file, and the second run silently trains
    on the first model's render.
    """
    fp_a = _get_kwargs_fingerprint({"processing_class": _Processor(_LeafTok(), "template-A")})
    fp_b = _get_kwargs_fingerprint({"processing_class": _Processor(_LeafTok(), "template-B")})
    assert fp_a != fp_b, "same processor class with different chat templates must key different caches"


def test_processor_identity_tracks_the_inner_tokenizer():
    fp_a = _get_kwargs_fingerprint({"processing_class": _Processor(_LeafTok(name="org/base"), "t")})
    fp_b = _get_kwargs_fingerprint({"processing_class": _Processor(_LeafTok(name="org/instruct"), "t")})
    assert fp_a != fp_b, "different inner tokenizers must key different caches"


def test_processor_dict_chat_template_is_fingerprinted():
    """Multi-template processors carry chat_template as a dict — it must key by content, not fail."""
    fp_a = _get_kwargs_fingerprint({"p": _Processor(_LeafTok(), {"default": "A", "tool_use": "T"})})
    fp_b = _get_kwargs_fingerprint({"p": _Processor(_LeafTok(), {"default": "B", "tool_use": "T"})})
    assert fp_a != fp_b, "dict chat templates must fingerprint by content"


def test_nested_collection_values_fingerprint_by_content():
    """A value the scalar branches cannot read must still key on its CONTENT, not its type name.

    A dict or a list of pairs is exactly what a caller threads through ``cache_key_extras`` to name
    a policy; keyed as the literal ``dict``/``list`` every value of that knob shares one cache file
    and the second run silently loads the first run's rows.
    """
    assert _get_kwargs_fingerprint({"policy": [("a", True), ("b", False)]}) != _get_kwargs_fingerprint(
        {"policy": [("a", True), ("b", True)]}
    ), "a list of pairs must key by content"
    assert _get_kwargs_fingerprint({"policy": {"a": True, "b": False}}) != _get_kwargs_fingerprint(
        {"policy": {"a": True, "b": True}}
    ), "a dict value must key by content"
    assert _get_kwargs_fingerprint({"policy": {"a": True, "b": False}}) == _get_kwargs_fingerprint(
        {"policy": {"b": False, "a": True}}
    ), "the same dict content must key one cache whatever the insertion order"


def test_the_two_completion_span_policies_key_different_caches():
    """The bake picks its span policy by whether the artifact is packed, and the two policies differ
    in a single flag. Sharing a cache file means a packed bake loads the UNPACKED bake's labels —
    silently training a terminator-less turn on the wrong tokens."""
    ds = _make_dataset(4)
    assert COLLATOR_SPAN_POLICY != PACKED_SPAN_POLICY, "the two policies must actually differ"
    unpacked = _build_cache_file_name(
        "map", _add_one, ds, "d", {}, cache_key_extras={"span_policy": COLLATOR_SPAN_POLICY}
    )
    packed = _build_cache_file_name("map", _add_one, ds, "d", {}, cache_key_extras={"span_policy": PACKED_SPAN_POLICY})
    assert unpacked != packed, "the two span policies must key different cache files"


def test_object_without_tokenizer_surface_keeps_the_type_fallback():
    """A random object is still just its type name — the descent must not invent identities."""

    class Opaque:
        pass

    assert _get_kwargs_fingerprint({"x": Opaque()}) == _get_kwargs_fingerprint({"x": Opaque()})


def test_non_processor_carrying_a_tokenizer_is_not_keyed_as_that_tokenizer():
    """A collator is not a processor: holding ``.tokenizer`` must not buy it a borrowed identity.

    The descent exists because a processor IS its inner tokenizer plus its own template. For
    anything else that claim is false — a collator's render knobs are invisible to it — so a
    signature built from the inner tokenizer reads as specific while ignoring everything that
    actually shapes the map. Such an object keys by class name; its knobs go through
    ``cache_key_extras``.
    """

    class Collator:
        def __init__(self, tokenizer, hint):
            self.tokenizer = tokenizer
            self.hint_template = hint

    # The discriminating case: an untyped descent makes these two differ — the borrowed identity,
    # specific-looking yet blind to every knob the collator actually renders with.
    assert _get_kwargs_fingerprint({"c": Collator(_LeafTok(name="org/base"), "hint")}) == _get_kwargs_fingerprint(
        {"c": Collator(_LeafTok(name="org/instruct"), "hint")}
    ), "a non-processor must not fingerprint as its inner tokenizer"

    # The other half of the contract, through the REAL key (which fingerprints fn_kwargs and
    # cache_key_extras separately): a class-name entry is blind to those knobs, so they reach the
    # cache only when the caller threads them through cache_key_extras.
    ds = _make_dataset(4)
    a, b = Collator(_LeafTok(), "hint A"), Collator(_LeafTok(), "hint B")
    assert _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"c": a}}) == _build_cache_file_name(
        "map", _add_one, ds, "d", {"fn_kwargs": {"c": b}}
    ), "a collator keys by class name, so a render knob it hides cannot invalidate the cache alone"
    assert _build_cache_file_name(
        "map", _add_one, ds, "d", {"fn_kwargs": {"c": a}}, cache_key_extras=vars(a)
    ) != _build_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"c": b}}, cache_key_extras=vars(b)), (
        "the knobs it hides must key the cache once threaded through cache_key_extras"
    )


@pytest.mark.parametrize(
    "knob,value",
    [
        ("max_length", 4096),
        ("conversation_field", "dialogue"),
        ("hint_template", "the answer is {answer}"),
        ("answer_field", "gold"),
        ("solution_field", None),
        ("confidence_field", "conf"),
        ("confidence_power", 2.0),
        ("response_prompt_template", "<|assistant|>"),
        ("train_on_completions_only", False),
        ("system_prompt", "be terse"),
        ("model_supports_system_role", False),
        ("tools_field", "tools"),
        ("interleaved_thinking", True),
    ],
)
def test_self_distill_cache_signature_covers_every_render_knob(knob, value):
    """The self-distill audit map keys on ``collator.cache_signature()``; a knob missing from it
    lets a stale cache skip the audit, and the over-length raise reverts to collate time — where it
    is rank-local and hangs the peers in the step's collectives."""
    baseline = {
        "tokenizer": _LeafTok(),
        "max_length": 2048,
        "conversation_field": "messages",
        "hint_template": "hint: {answer}",
        "answer_field": "answer",
        "solution_field": "solution",
        "confidence_field": None,
        "confidence_power": 4.0,
        # A marker on the baseline: completion-only masking without one is refused at construction.
        "response_prompt_template": "<|im_start|>assistant",
        "train_on_completions_only": True,
        "system_prompt": None,
        "model_supports_system_role": True,
        "tools_field": None,
        "interleaved_thinking": False,
    }
    assert baseline[knob] != value, "the variant must actually differ from the baseline"
    fp_base = _get_kwargs_fingerprint(SelfDistillTextCollator(**baseline).cache_signature())
    fp_variant = _get_kwargs_fingerprint(SelfDistillTextCollator(**{**baseline, knob: value}).cache_signature())
    assert fp_base != fp_variant, f"{knob} changes what the audit accepts but not the cache key"


def test_the_self_distill_audit_map_threads_the_collator_signature_into_its_cache_key():
    """The signature above only protects the audit if the script actually passes it.

    ``audit_self_distill_row`` takes the collator through ``fn_kwargs``, where it fingerprints by
    class name alone — so without ``cache_key_extras`` a stale cache skips the audit after any
    render knob changes, and the over-length raise reverts to collate time, where it is rank-local
    and hangs the peers in the step's collectives.
    """
    calls = [
        node
        for node in ast.walk(ast.parse(SELF_DISTILL_SCRIPT.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "coordinated_map"
        and any(kw.arg == "fn_kwargs" and "collator" in ast.unparse(kw.value) for kw in node.keywords)
    ]
    assert calls, "no self-distill audit map takes the collator through fn_kwargs"
    for call in calls:
        # Presence, not spelling: hoisting the signature to a local, or adding the same guard to
        # another audit map, is a correct refactor that an exact-source pin would fail.
        assert any(kw.arg == "cache_key_extras" for kw in call.keywords), (
            f"the audit map at line {call.lineno} does not key on the collator's render knobs"
        )


def test_the_worker_count_is_a_named_parameter_not_a_cache_keyed_kwarg():
    """Every kwarg reaching ``_build_cache_file_name`` is keyed, so ``num_proc`` must never arrive as
    one: it steers HOW an op runs, not what it produces. Absorbed into ``**kwargs`` it would key the
    cache — a re-run at a different worker count misses, and ranks that derive different counts
    (the default is host-derived) write different cache files, breaking the cross-rank transport."""
    for operation in (coordinated_map, coordinated_filter):
        assert "num_proc" in inspect.signature(operation).parameters, (
            f"{operation.__name__} must take num_proc as a named parameter, else it lands in "
            f"**kwargs and the worker count keys the dataset cache"
        )


@pytest.mark.parametrize("knob", ["load_from_cache_file", "keep_in_memory"])
@pytest.mark.parametrize("operation", [coordinated_map, coordinated_filter])
def test_managed_execution_kwargs_are_refused_not_dropped(operation, knob):
    """Passing an execution knob the machinery owns must raise, never be silently discarded —
    dropping it leaves a call site reading as if it steered caching when it cannot."""
    with pytest.raises(TypeError, match=knob):
        operation(_make_dataset(4), _add_one, desc="d", **{knob: True})


def test_in_vocab_special_token_override_changes_the_identity():
    """An in-vocab --eos-token override changes neither name, vocab, len, nor template text, yet
    shifts the ids a map bakes into every row — it must change the cache key."""
    fp_a = _get_kwargs_fingerprint({"tokenizer": _LeafTok(eos=2)})
    fp_b = _get_kwargs_fingerprint({"tokenizer": _LeafTok(eos=7)})
    assert fp_a != fp_b, "an eos_token_id override must key a different cache"


def test_padding_side_changes_the_identity():
    fp_a = _get_kwargs_fingerprint({"tokenizer": _LeafTok(side="right")})
    fp_b = _get_kwargs_fingerprint({"tokenizer": _LeafTok(side="left")})
    assert fp_a != fp_b, "padding_side shapes padded map output and must key the cache"


def test_closure_captured_processor_is_fingerprinted():
    """A map closing over a processor must not hit the unfingerprintable-cell skip warning — its
    template changes then silently reuse the cache. The descent makes the cell identity-bearing."""

    def make_fn(processor):
        def fn(row):
            return processor.chat_template

        return fn

    fp_a = _get_closure_fingerprint(make_fn(_Processor(_LeafTok(), "A")))
    fp_b = _get_closure_fingerprint(make_fn(_Processor(_LeafTok(), "B")))
    assert fp_a and fp_a != fp_b, "a captured processor's template must reach the closure fingerprint"


# _build_cache_file_name — third-party version stamp


def test_render_library_versions_key_the_cache(monkeypatch):
    """A transformers/TRL bump can change render/packing output with our code untouched; pre-bump
    arrow caches must not be reused across it."""
    ds = _make_dataset(4)
    name_now = _build_cache_file_name("map", _add_one, ds, "d", {})
    monkeypatch.setattr(processing, "_RENDER_LIBRARY_VERSIONS", "tf0.0.0-trl0.0.0")
    name_bumped = _build_cache_file_name("map", _add_one, ds, "d", {})
    assert name_now != name_bumped, "a dependency bump must invalidate the map cache"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
