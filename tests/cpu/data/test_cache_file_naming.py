#!/usr/bin/env python
import ast
import inspect

import pytest
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import ProcessorMixin

from halo.data.chat_render import render_conversation, tokenize_rendered
from halo.data.collators.self_distill import (
    SelfDistillTextCollator,
    encode_self_distill_row,
    privileged_hint,
    require_aligned_responses,
    teacher_history,
)
from halo.data.completion_spans import PACKED_SPAN_POLICY, PADDED_SPAN_POLICY, build_training_labels
from halo.data.pipeline import map_cache_key
from halo.data.pipeline.coordinated_map import coordinated_filter, coordinated_map
from halo.data.pipeline.map_cache_key import (
    CACHE_KEY_ATTR,
    _closure_fingerprint,
    _tokenizer_content_sig,
    function_cache_id,
    kwargs_fingerprint,
    map_cache_file_name,
)
from tests.common.models import QWEN3_0_6B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_tokenizer


def _add_one(x):
    return x + 1


def _multiply_two(x):
    return x * 2


def _make_dataset(n: int, seed: int = 0) -> Dataset:
    return Dataset.from_dict({"value": list(range(seed, seed + n))})


SELF_DISTILL_SCRIPT = REPO_ROOT / "scripts/training/distillation/self_distill.py"


def test_deterministic_output():
    ds = _make_dataset(10)
    name1 = map_cache_file_name("map", _add_one, ds, "desc", {})
    name2 = map_cache_file_name("map", _add_one, ds, "desc", {})
    assert name1 == name2, f"Non-deterministic: '{name1}' vs '{name2}'"


def test_cache_name_format():
    ds = _make_dataset(5)
    name = map_cache_file_name("map", _add_one, ds, None, {})
    assert name.startswith("cache-"), f"Bad prefix: {name}"
    assert name.endswith(".arrow"), f"Bad suffix: {name}"
    hex_part = name[len("cache-") : -len(".arrow")]
    assert len(hex_part) == 32, f"Expected 32 hex chars, got {len(hex_part)}"
    int(hex_part, 16)


def test_different_operation_types():
    ds = _make_dataset(10)
    name_map = map_cache_file_name("map", _add_one, ds, "desc", {})
    name_filter = map_cache_file_name("filter", _add_one, ds, "desc", {})
    assert name_map != name_filter, "map and filter produced same cache name"


def test_different_functions():
    ds = _make_dataset(10)
    name_a = map_cache_file_name("map", _add_one, ds, "desc", {})
    name_b = map_cache_file_name("map", _multiply_two, ds, "desc", {})
    assert name_a != name_b, "Different functions produced same cache name"


def test_different_desc():
    ds = _make_dataset(10)
    name1 = map_cache_file_name("map", _add_one, ds, "tokenize", {})
    name2 = map_cache_file_name("map", _add_one, ds, "format", {})
    assert name1 != name2, "Different descriptions produced same cache name"


def test_different_dataset_sizes():
    ds_small = _make_dataset(5)
    ds_large = _make_dataset(20)
    name_small = map_cache_file_name("map", _add_one, ds_small, "d", {})
    name_large = map_cache_file_name("map", _add_one, ds_large, "d", {})
    assert name_small != name_large, "Different-sized datasets produced same cache name"


def test_different_dataset_content_same_size():
    ds_a = _make_dataset(10, seed=0)
    ds_b = _make_dataset(10, seed=100)
    assert ds_a._fingerprint != ds_b._fingerprint, "Test setup error: datasets have same fingerprint"
    name_a = map_cache_file_name("map", _add_one, ds_a, "d", {})
    name_b = map_cache_file_name("map", _add_one, ds_b, "d", {})
    assert name_a != name_b, "Same-size different-content datasets produced same cache name"


def test_different_fn_kwargs_tokenizer():
    ds = _make_dataset(10)

    class MockTokenizerA:
        name_or_path = "model-A"

    class MockTokenizerB:
        name_or_path = "model-B"

    name_a = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"tokenizer": MockTokenizerA()}})
    name_b = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"tokenizer": MockTokenizerB()}})
    assert name_a != name_b, "Different tokenizers produced same cache name"


def test_fn_kwargs_vs_no_fn_kwargs():
    ds = _make_dataset(10)
    name_no_kwargs = map_cache_file_name("map", _add_one, ds, "d", {})
    name_with_kwargs = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 512}})
    assert name_no_kwargs != name_with_kwargs, "With/without fn_kwargs produced same cache name"


def test_different_fn_kwargs_primitives():
    ds = _make_dataset(10)
    name_a = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 512}})
    name_b = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"max_length": 1024}})
    assert name_a != name_b, "Different max_length values produced same cache name"


def test_different_fn_kwargs_keys():
    ds = _make_dataset(10)
    name_a = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"alpha": 1}})
    name_b = map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"beta": 1}})
    assert name_a != name_b, "Different kwarg keys produced same cache name"


def test_dataset_dict_includes_split_keys():
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
    name_ab = map_cache_file_name("map", _add_one, dd_ab, "d", {})
    name_cd = map_cache_file_name("map", _add_one, dd_cd, "d", {})
    assert name_ab != name_cd, "Different DatasetDict splits produced same cache name"


def test_dataset_vs_dataset_dict():
    ds = _make_dataset(10)
    dd = DatasetDict({"train": _make_dataset(10)})
    name_ds = map_cache_file_name("map", _add_one, ds, "d", {})
    name_dd = map_cache_file_name("map", _add_one, dd, "d", {})
    assert name_ds != name_dd, "Dataset and DatasetDict produced same cache name"


def test_dataset_dict_different_content_same_splits():
    """A key relying on ``len(dataset)`` and ``_fingerprint`` (missing on DatasetDict) lets a re-uploaded source
    silently reuse the old tokenized cache."""
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
    name_v1 = map_cache_file_name("map", _add_one, dd_v1, "d", {})
    name_v2 = map_cache_file_name("map", _add_one, dd_v2, "d", {})
    assert name_v1 != name_v2, (
        "DatasetDicts with different underlying content collided — cache will "
        "serve stale tokenization after a source dataset update."
    )


def test_dataset_dict_same_content_stable():
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
    name_a = map_cache_file_name("map", _add_one, dd_a, "d", {})
    name_b = map_cache_file_name("map", _add_one, dd_b, "d", {})
    assert name_a == name_b, "Identical DatasetDicts produced different cache names"


def test_closures_with_different_captured_values():
    """``function_cache_id`` hashes only the source, identical for two closures from one factory, so
    ``_closure_fingerprint`` must key on the captured free variables."""

    def make_adder(n):
        def adder(x):
            return x + n

        return adder

    ds = _make_dataset(10)
    add_1 = make_adder(1)
    add_2 = make_adder(2)

    fp_1 = _closure_fingerprint(add_1)
    fp_2 = _closure_fingerprint(add_2)
    assert fp_1 != fp_2, (
        f"captured int n=1 vs n=2 produced identical closure fingerprints ({fp_1!r}) — "
        "the closure-fingerprint path no longer disambiguates scalar captures"
    )

    name_1 = map_cache_file_name("map", add_1, ds, "d", {})
    name_2 = map_cache_file_name("map", add_2, ds, "d", {})
    assert name_1 != name_2, (
        "closures capturing different scalars collided on one cache name despite differing closure fingerprints"
    )

    name_1_kw = map_cache_file_name("map", add_1, ds, "d", {"fn_kwargs": {"n": 1}})
    name_2_kw = map_cache_file_name("map", add_2, ds, "d", {"fn_kwargs": {"n": 2}})
    assert name_1_kw != name_2_kw, "fn_kwargs should differentiate closures with different captured values"


def test_same_name_different_body():
    ds = _make_dataset(10)

    def process(x):
        return x + 1

    name_a = map_cache_file_name("map", process, ds, "d", {})

    def process(x):
        return x * 3 + 7

    name_b = map_cache_file_name("map", process, ds, "d", {})
    assert name_a != name_b, "Same-name functions with different code produced same cache name"


def test_kwargs_different_tokenizer_names():
    class TokA:
        name_or_path = "org/model-v1"

    class TokB:
        name_or_path = "org/model-v2"

    fp_a = kwargs_fingerprint({"processing_class": TokA()})
    fp_b = kwargs_fingerprint({"processing_class": TokB()})
    assert fp_a != fp_b, "Different tokenizer name_or_path produced same fingerprint"


def test_kwargs_different_vocab_sizes():
    class CfgA:
        vocab_size = 32000

    class CfgB:
        vocab_size = 128256

    fp_a = kwargs_fingerprint({"config": CfgA()})
    fp_b = kwargs_fingerprint({"config": CfgB()})
    assert fp_a != fp_b, "Different vocab_size produced same fingerprint"


def test_kwargs_bool_vs_int_distinction():
    fp_true = kwargs_fingerprint({"flag": True})
    fp_one = kwargs_fingerprint({"flag": 1})
    assert fp_true != fp_one, "True and 1 produced same fingerprint"


def test_kwargs_list_order_is_part_of_the_key_and_set_order_is_not():
    """A list's order is content (``label_list`` maps position to class id), so two orders must key
    two caches; a set's iteration order is hash-seeded, so it must not."""
    assert kwargs_fingerprint({"label_list": ["a", "b"]}) != kwargs_fingerprint({"label_list": ["b", "a"]})
    assert kwargs_fingerprint({"ids": {"a", "b"}}) == kwargs_fingerprint({"ids": {"b", "a"}})


def test_kwargs_name_or_path_takes_priority():
    class ModelA:
        name_or_path = "model-A"
        vocab_size = 32000

    class ModelB:
        name_or_path = "model-B"
        vocab_size = 32000

    fp_a = kwargs_fingerprint({"model": ModelA()})
    fp_b = kwargs_fingerprint({"model": ModelB()})
    assert fp_a != fp_b, "Objects with different name_or_path but same vocab_size collided"


def test_kwargs_tokenizer_content_beats_path():
    """The same vocab reloaded from a resume checkpoint keys the same cache (a path term would re-tokenize on every
    resume leg), while a different vocab at the same path keys another."""

    class Tok:
        vocab_size = 4

        def __init__(self, name_or_path, vocab):
            self.name_or_path = name_or_path
            self._vocab = vocab

        def get_vocab(self):
            return self._vocab

    vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
    fp_source = kwargs_fingerprint({"processing_class": Tok("org/base-model", vocab)})
    fp_resumed = kwargs_fingerprint({"processing_class": Tok("checkpoints/run/checkpoint-750", vocab)})
    fp_other = kwargs_fingerprint({"processing_class": Tok("org/base-model", {"a": 0, "b": 1, "c": 2, "e": 3})})
    assert fp_source == fp_resumed, "Same tokenizer content at a checkpoint path missed the cache"
    assert fp_source != fp_other, "Different vocab content at one path shared a fingerprint"


def test_tokenizer_content_sig_ignores_per_call_padding_and_truncation():
    """``PreTrainedTokenizerFast.__call__`` rewrites its serialized padding/truncation state, so the raw
    ``to_str()`` keys on whatever was tokenized last. At ``num_proc <= 1`` only the writer runs a map's fn, so
    its key diverges and every peer re-runs the next map, writing one arrow file concurrently."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    assert tokenizer.is_fast and tokenizer.backend_tokenizer is not None, "the probe needs a fast tokenizer"

    pristine = _tokenizer_content_sig(tokenizer)
    assert pristine, "a fast tokenizer must produce a content signature"

    tokenizer("halo halo halo", truncation=True, max_length=4)
    assert _tokenizer_content_sig(tokenizer) == pristine, "a truncating call re-keyed the cache"

    tokenizer("halo", padding="max_length", max_length=8)
    assert _tokenizer_content_sig(tokenizer) == pristine, "a padding call re-keyed the cache"

    # The whole identity moves with it, not just the content term.
    tokenizer("halo halo", truncation=True, max_length=2, padding="max_length")
    assert kwargs_fingerprint({"tokenizer": tokenizer}) == kwargs_fingerprint(
        {"tokenizer": load_cached_tokenizer(QWEN3_0_6B)}
    ), "a used tokenizer keyed a different cache than a freshly loaded one"


def test_tokenizer_content_sig_still_separates_different_vocabs():
    """Two backends differing only in vocab still key different caches, or dropping the state trades a race for a
    cross-model collision."""

    class _Fast:
        def __init__(self, backend):
            self.backend_tokenizer = backend

    def _backend(vocab):
        return Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    shared = {"[UNK]": 0, "a": 1, "b": 2}
    same = _tokenizer_content_sig(_Fast(_backend(dict(shared))))
    other = _tokenizer_content_sig(_Fast(_backend({**shared, "c": 3})))
    assert same and other
    assert same == _tokenizer_content_sig(_Fast(_backend(dict(shared)))), "same vocab must key one cache"
    assert same != other, "different vocabs must key different caches"

    truncating = _backend(dict(shared))
    truncating.enable_truncation(max_length=2)
    assert _tokenizer_content_sig(_Fast(truncating)) == same, "enabling truncation re-keyed the cache"

    # A post-processor wrapping the sequence changes the ids a map writes, so widening the dropped keys past the
    # per-call pair must re-key here.
    templated = _backend(dict(shared))
    templated.post_processor = TemplateProcessing(single="a $A", special_tokens=[("a", 1)])
    assert _tokenizer_content_sig(_Fast(templated)) != same, "a post-processor change must key a different cache"


def test_kwargs_scalar_lists_distinct():
    """Collapsing to the type name would serve one list's cache for every other."""
    fp_a = kwargs_fingerprint({"eos_token_ids": [1, 2, 3]})
    fp_b = kwargs_fingerprint({"eos_token_ids": [4, 5, 6]})
    assert fp_a != fp_b, "Different scalar lists must produce different fingerprints"

    fp_tuple_a = kwargs_fingerprint({"eos_token_ids": (1, 2, 3)})
    fp_tuple_b = kwargs_fingerprint({"eos_token_ids": (9,)})
    assert fp_tuple_a != fp_tuple_b, "Different scalar tuples must produce different fingerprints"

    assert fp_a == kwargs_fingerprint({"eos_token_ids": [1, 2, 3]})

    # Non-scalar lists still fall back to the type name (unhashable/heavy contents).
    fp_obj_a = kwargs_fingerprint({"objs": [object()]})
    fp_obj_b = kwargs_fingerprint({"objs": [object()]})
    assert fp_obj_a == fp_obj_b, "Lists of non-scalars keep the type-name fallback"


def test_shuffled_dataset_different_fingerprint():
    ds = _make_dataset(20)
    ds_shuffled = ds.shuffle(seed=42)
    name_orig = map_cache_file_name("map", _add_one, ds, "d", {})
    name_shuffled = map_cache_file_name("map", _add_one, ds_shuffled, "d", {})
    assert name_orig != name_shuffled, "Shuffled dataset produced same cache name as original"


def test_fn_kwargs_subset_differs():
    ds = _make_dataset(10)
    name_a = map_cache_file_name(
        "map",
        _add_one,
        ds,
        "d",
        {"fn_kwargs": {"max_length": 512, "truncation": True}},
    )
    name_b = map_cache_file_name(
        "map",
        _add_one,
        ds,
        "d",
        {"fn_kwargs": {"max_length": 512}},
    )
    assert name_a != name_b, "Different number of fn_kwargs produced same cache name"


def test_output_affecting_map_kwargs_change_cache_name():
    """Two calls differing only in remove_columns produce different rows, and a shared key would silently serve one
    call's rows to the other."""
    ds = _make_dataset(10)
    base = {"remove_columns": ["value"], "batched": True}
    name_base = map_cache_file_name("map", _add_one, ds, "d", dict(base))

    for variant in (
        {"remove_columns": ["other"], "batched": True},
        {"remove_columns": ["value"], "batched": False},
        {"remove_columns": ["value"], "batched": True, "batch_size": 7},
        {"remove_columns": ["value"], "batched": True, "with_indices": True},
    ):
        name_variant = map_cache_file_name("map", _add_one, ds, "d", variant)
        assert name_variant != name_base, f"cache name must change for {variant}"


# The processors from build_chat_row_processor capture tokenizer and max_length as free variables, which source hashing
# and the fn_kwargs fingerprint never see. Without the closure fingerprint two runs collide on one cache file and load
# another model's token IDs (out-of-bounds embedding gather, crash at step 0).


class _MockTokenizer:
    def __init__(self, name_or_path, vocab_size):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size


def _make_processor(tokenizer, max_length):
    """Factory whose returned closure captures tokenizer + max_length (like build_chat_row_processor)."""

    def process(example):
        # references the free variables so they live in __closure__
        return {"ids": [tokenizer.vocab_size, max_length]}

    return process


def test_closure_fingerprint_differs_by_tokenizer():
    tok_a = _MockTokenizer("org/model-262k", 262000)
    tok_b = _MockTokenizer("org/model-201k", 201000)
    fp_a = _closure_fingerprint(_make_processor(tok_a, 4096))
    fp_b = _closure_fingerprint(_make_processor(tok_b, 4096))
    assert fp_a and fp_b, "closure fingerprint should be non-empty when a tokenizer is captured"
    assert fp_a != fp_b, "different captured tokenizers must produce different closure fingerprints"


def test_closure_fingerprint_differs_by_max_length():
    tok = _MockTokenizer("org/model", 32000)
    fp_512 = _closure_fingerprint(_make_processor(tok, 512))
    fp_4096 = _closure_fingerprint(_make_processor(tok, 4096))
    assert fp_512 != fp_4096, "different captured max_length must change the closure fingerprint"


def test_closure_fingerprint_empty_for_no_closure():
    assert _closure_fingerprint(_add_one) == ""


def test_cache_name_breaks_cross_tokenizer_collision():
    """A shared file would load another model's token IDs."""
    ds = _make_dataset(10)
    proc_262k = _make_processor(_MockTokenizer("org/model-262k", 262000), 4096)
    proc_201k = _make_processor(_MockTokenizer("org/model-201k", 201000), 4096)
    name_262k = map_cache_file_name("map", proc_262k, ds, "tokenize", {})
    name_201k = map_cache_file_name("map", proc_201k, ds, "tokenize", {})
    assert name_262k != name_201k, (
        "Different closure-captured tokenizers collided on one cache file — "
        "this is the cross-model token-ID corruption regression."
    )


def test_cache_name_same_tokenizer_stable():
    ds = _make_dataset(10)
    tok = _MockTokenizer("org/model", 32000)
    name1 = map_cache_file_name("map", _make_processor(tok, 4096), ds, "tokenize", {})
    name2 = map_cache_file_name("map", _make_processor(tok, 4096), ds, "tokenize", {})
    assert name1 == name2, "Identical closure capture must reuse the same cache file"


def test_function_id_method_vs_function():
    class Processor:
        def transform(self, x):
            return x + 1

    def transform(x):
        return x + 1

    id_method = function_cache_id(Processor.transform)
    id_func = function_cache_id(transform)
    assert id_method != id_func, "Method and standalone function with same name produced same ID"


def test_function_id_builtin_fallback():
    fid = function_cache_id(len)
    assert isinstance(fid, str)
    assert len(fid) > 0


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
    """A real ``ProcessorMixin``, the type the descent is gated on; no name_or_path or vocab_size of its own."""

    def __init__(self, tokenizer, chat_template):
        self.tokenizer = tokenizer
        self.chat_template = chat_template


def test_processor_identity_breaks_the_class_name_collision():
    """Collapsed to the class name they share one cache file, and the second run silently trains on the first
    model's render."""
    fp_a = kwargs_fingerprint({"processing_class": _Processor(_LeafTok(), "template-A")})
    fp_b = kwargs_fingerprint({"processing_class": _Processor(_LeafTok(), "template-B")})
    assert fp_a != fp_b, "same processor class with different chat templates must key different caches"


def test_processor_identity_tracks_the_inner_tokenizer():
    fp_a = kwargs_fingerprint({"processing_class": _Processor(_LeafTok(name="org/base"), "t")})
    fp_b = kwargs_fingerprint({"processing_class": _Processor(_LeafTok(name="org/instruct"), "t")})
    assert fp_a != fp_b, "different inner tokenizers must key different caches"


def test_processor_dict_chat_template_is_fingerprinted():
    fp_a = kwargs_fingerprint({"p": _Processor(_LeafTok(), {"default": "A", "tool_use": "T"})})
    fp_b = kwargs_fingerprint({"p": _Processor(_LeafTok(), {"default": "B", "tool_use": "T"})})
    assert fp_a != fp_b, "dict chat templates must fingerprint by content"


def test_nested_collection_values_fingerprint_by_content():
    """A dict or list of pairs is what a caller threads through ``cache_key_extras`` to name a policy; keyed as its
    type name, every value of that knob shares one cache file."""
    assert kwargs_fingerprint({"policy": [("a", True), ("b", False)]}) != kwargs_fingerprint(
        {"policy": [("a", True), ("b", True)]}
    ), "a list of pairs must key by content"
    assert kwargs_fingerprint({"policy": {"a": True, "b": False}}) != kwargs_fingerprint(
        {"policy": {"a": True, "b": True}}
    ), "a dict value must key by content"
    assert kwargs_fingerprint({"policy": {"a": True, "b": False}}) == kwargs_fingerprint(
        {"policy": {"b": False, "a": True}}
    ), "the same dict content must key one cache whatever the insertion order"


def test_the_two_completion_span_policies_key_different_caches():
    """The two policies differ in a single flag, and a shared file has a packed bake load the unpacked bake's
    labels."""
    ds = _make_dataset(4)
    assert PADDED_SPAN_POLICY != PACKED_SPAN_POLICY, "the two policies must actually differ"
    unpacked = map_cache_file_name("map", _add_one, ds, "d", {}, cache_key_extras={"span_policy": PADDED_SPAN_POLICY})
    packed = map_cache_file_name("map", _add_one, ds, "d", {}, cache_key_extras={"span_policy": PACKED_SPAN_POLICY})
    assert unpacked != packed, "the two span policies must key different cache files"


def test_object_without_tokenizer_surface_keeps_the_type_fallback():
    class Opaque:
        pass

    assert kwargs_fingerprint({"x": Opaque()}) == kwargs_fingerprint({"x": Opaque()})


def test_non_processor_carrying_a_tokenizer_is_not_keyed_as_that_tokenizer():
    """A processor is its inner tokenizer plus its own template; a collator's render knobs are invisible to that
    signature. Such an object keys by class name, and its knobs go through ``cache_key_extras``."""

    class Collator:
        def __init__(self, tokenizer, hint):
            self.tokenizer = tokenizer
            self.hint_template = hint

    # An untyped descent makes these two differ: the borrowed identity, blind to every knob the collator renders with.
    assert kwargs_fingerprint({"c": Collator(_LeafTok(name="org/base"), "hint")}) == kwargs_fingerprint(
        {"c": Collator(_LeafTok(name="org/instruct"), "hint")}
    ), "a non-processor must not fingerprint as its inner tokenizer"

    # Through the real key: a class-name entry is blind to those knobs, so they reach the cache only through
    # cache_key_extras.
    ds = _make_dataset(4)
    a, b = Collator(_LeafTok(), "hint A"), Collator(_LeafTok(), "hint B")
    assert map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"c": a}}) == map_cache_file_name(
        "map", _add_one, ds, "d", {"fn_kwargs": {"c": b}}
    ), "a collator keys by class name, so a render knob it hides cannot invalidate the cache alone"
    assert map_cache_file_name(
        "map", _add_one, ds, "d", {"fn_kwargs": {"c": a}}, cache_key_extras=vars(a)
    ) != map_cache_file_name("map", _add_one, ds, "d", {"fn_kwargs": {"c": b}}, cache_key_extras=vars(b)), (
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
        ("confidence_normalizer", 0.25),
        ("response_prompt_template", "<|assistant|>"),
        ("train_on_completions_only", False),
        ("system_prompt", "be terse"),
        ("model_supports_system_role", False),
        ("tools_field", "tools"),
        ("interleaved_thinking", True),
    ],
)
def test_self_distill_cache_signature_covers_every_render_knob(knob, value):
    """A knob missing from ``collator.cache_signature()`` serves ids under its previous value, or lets a stale cache
    skip the VLM audit, whose refusals then revert to collate time, rank-local, hanging the peers."""
    baseline = {
        "tokenizer": _LeafTok(),
        "max_length": 2048,
        "conversation_field": "messages",
        "hint_template": "hint: {answer}",
        "answer_field": "answer",
        "solution_field": "solution",
        "confidence_field": None,
        "confidence_power": 4.0,
        "confidence_normalizer": 0.5,
        # A marker on the baseline: completion-only masking without one is refused at construction.
        "response_prompt_template": "<|im_start|>assistant",
        "train_on_completions_only": True,
        "system_prompt": None,
        "model_supports_system_role": True,
        "tools_field": None,
        "interleaved_thinking": False,
    }
    assert baseline[knob] != value, "the variant must actually differ from the baseline"
    fp_base = kwargs_fingerprint(SelfDistillTextCollator(**baseline).cache_signature())
    fp_variant = kwargs_fingerprint(SelfDistillTextCollator(**{**baseline, knob: value}).cache_signature())
    assert fp_base != fp_variant, f"{knob} changes what the prep map writes or accepts but not the cache key"


def test_the_self_distill_audit_map_threads_the_collator_signature_into_its_cache_key():
    """The collator fingerprints by class name through ``fn_kwargs``, so without ``cache_key_extras`` a stale cache
    serves the previous knobs' encoding or skips the audit."""
    calls = [
        node
        for node in ast.walk(ast.parse(SELF_DISTILL_SCRIPT.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "coordinated_map"
        and any(kw.arg == "fn_kwargs" and "collator" in ast.unparse(kw.value) for kw in node.keywords)
    ]
    assert calls, "no self-distill prep map takes the collator through fn_kwargs"
    for call in calls:
        # Presence, not spelling: hoisting the signature to a local is a correct refactor.
        assert any(kw.arg == "cache_key_extras" for kw in call.keywords), (
            f"the prep map at line {call.lineno} does not key on the collator's render knobs"
        )


@pytest.mark.parametrize(
    "helper",
    [
        render_conversation,
        tokenize_rendered,
        build_training_labels,
        teacher_history,
        privileged_hint,
        require_aligned_responses,
    ],
    ids=lambda helper: helper.__name__,
)
def test_the_self_distill_encode_map_key_covers_the_encoder_source(helper, monkeypatch):
    """The key folds in only the source of helpers the function reaches by bare name: an encoder behind a collator
    method drops out, and an edited render rule would serve the previous rows."""
    before = function_cache_id(encode_self_distill_row)
    getsource = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda func: getsource(func) + ("\n# edited" if func is helper else ""))
    assert function_cache_id(encode_self_distill_row) != before, f"editing {helper.__name__} keeps the cache key"


def test_the_worker_count_is_a_named_parameter_not_a_cache_keyed_kwarg():
    """Every kwarg reaching ``map_cache_file_name`` is keyed, and ``num_proc`` steers how an op runs. Keyed, a
    re-run at another worker count misses, and ranks deriving different counts write different files."""
    for operation in (coordinated_map, coordinated_filter):
        assert "num_proc" in inspect.signature(operation).parameters, (
            f"{operation.__name__} must take num_proc as a named parameter, else it lands in "
            f"**kwargs and the worker count keys the dataset cache"
        )


def _bump(row):
    return {"value": row["value"] + 1}


def test_coordinated_map_stamps_the_cache_file_it_wrote_as_the_rank_stable_key(tmp_path, monkeypatch):
    """Every downstream key reads this stamp in place of HF's per-rank fingerprint, so it names the file the map
    wrote: a Dataset's own, a DatasetDict's per split."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    single = coordinated_map(_make_dataset(4), _bump, desc="stamp", num_proc=1)
    assert getattr(single, CACHE_KEY_ATTR).endswith(".arrow")
    assert (tmp_path / getattr(single, CACHE_KEY_ATTR)).is_file()

    splits = coordinated_map(
        DatasetDict({"train": _make_dataset(4), "test": _make_dataset(2, seed=9)}), _bump, desc="stamp", num_proc=1
    )
    for split, split_ds in splits.items():
        assert getattr(split_ds, CACHE_KEY_ATTR).endswith(f"_{split}.arrow")
        assert (tmp_path / getattr(split_ds, CACHE_KEY_ATTR)).is_file()


@pytest.mark.parametrize("knob", ["load_from_cache_file", "keep_in_memory"])
@pytest.mark.parametrize("operation", [coordinated_map, coordinated_filter])
def test_managed_execution_kwargs_are_refused_not_dropped(operation, knob):
    """Dropping it leaves a call site reading as if it steered caching."""
    with pytest.raises(TypeError, match=knob):
        operation(_make_dataset(4), _add_one, desc="d", **{knob: True})


def test_in_vocab_special_token_override_changes_the_identity():
    """An in-vocab --eos_token override changes neither name, vocab, len nor template text, yet shifts the ids a map
    bakes into every row."""
    fp_a = kwargs_fingerprint({"tokenizer": _LeafTok(eos=2)})
    fp_b = kwargs_fingerprint({"tokenizer": _LeafTok(eos=7)})
    assert fp_a != fp_b, "an eos_token_id override must key a different cache"


def test_padding_side_changes_the_identity():
    fp_a = kwargs_fingerprint({"tokenizer": _LeafTok(side="right")})
    fp_b = kwargs_fingerprint({"tokenizer": _LeafTok(side="left")})
    assert fp_a != fp_b, "padding_side shapes padded map output and must key the cache"


def test_closure_captured_processor_is_fingerprinted():
    """A map closing over a processor must not hit the unfingerprintable-cell skip, or template changes silently
    reuse the cache."""

    def make_fn(processor):
        def fn(row):
            return processor.chat_template

        return fn

    fp_a = _closure_fingerprint(make_fn(_Processor(_LeafTok(), "A")))
    fp_b = _closure_fingerprint(make_fn(_Processor(_LeafTok(), "B")))
    assert fp_a and fp_a != fp_b, "a captured processor's template must reach the closure fingerprint"


def test_render_library_versions_key_the_cache(monkeypatch):
    """A transformers/TRL bump can change render or packing output with our code untouched."""
    ds = _make_dataset(4)
    name_now = map_cache_file_name("map", _add_one, ds, "d", {})
    monkeypatch.setattr(map_cache_key, "_RENDER_LIBRARY_VERSIONS", "tf0.0.0-trl0.0.0")
    name_bumped = map_cache_file_name("map", _add_one, ds, "d", {})
    assert name_now != name_bumped, "a dependency bump must invalidate the map cache"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
