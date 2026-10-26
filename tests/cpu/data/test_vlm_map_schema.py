#!/usr/bin/env python3
"""VLM map Arrow-schema stability on mixed datasets + cache-key coverage of captured helpers.

Shard-wise type inference diverges when image rows and text-only rows land in different map worker
shards (struct vs Json content, null vs Image lists). `vlm_map_features` pins the schema so the
multiprocess map succeeds for every mix; `_closure_fingerprint` must cover closure-captured
helper functions so editing them invalidates the dataset cache.

Usage:
    python tests/cpu/data/test_vlm_map_schema.py
"""

import importlib.util

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from halo.data.pipeline import map_cache_key
from halo.data.pipeline.map_cache_key import _closure_fingerprint, function_cache_id
from halo.data.pipeline.vlm_rows import build_vlm_row_processor, vlm_map_features


def _image(color: int) -> Image.Image:
    return Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8))


def _mixed_dataset() -> Dataset:
    """Image rows first, text-only rows second — clusters modalities into separate map shards."""
    rows = {
        "texts": (
            [[{"user": f"describe image {i}", "assistant": f"answer {i}"}] for i in range(4)]
            + [[{"user": f"text question {i}", "assistant": f"text answer {i}"}] for i in range(4)]
        ),
        "images": [[_image(i)] for i in range(4)] + [[] for _ in range(4)],
    }
    return Dataset.from_dict(rows)


def test_mixed_modality_multiproc_map_with_pinned_features():
    ds = _mixed_dataset()
    process = build_vlm_row_processor(conversation_field="texts", images_field="images")
    out = ds.map(
        process,
        num_proc=2,
        remove_columns=["texts"],
        features=vlm_map_features(),
        load_from_cache_file=False,
    )
    assert len(out) == 8
    image_row, text_row = out[0], out[7]
    assert image_row["history"][0]["content"][0] == {"type": "image"}
    assert isinstance(image_row["images"][0], Image.Image), "Image feature must round-trip to PIL"
    assert text_row["images"] == []
    assert text_row["history"][0]["content"] == [{"type": "text", "text": "text question 3"}], (
        "text rows must share the parts-form schema"
    )


def _probe_wrapper(tmp_path, module_name: str, helper_body: str):
    """``wrapper`` calling a module-global ``helper``, loaded under ``module_name`` from a real file: the
    identifier hashes ``inspect.getsource``, which exec()'d code lacks."""
    path = tmp_path / f"{module_name.replace('.', '_')}.py"
    path.write_text(f"def helper(x):\n    return {helper_body}\n\n\ndef wrapper(x):\n    return helper(x)\n")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wrapper


def test_function_identifier_covers_module_global_helpers(tmp_path):
    """Editing a module-global helper a map function calls must change the caller's cache identity."""
    root = map_cache_key.__name__.partition(".")[0]
    id_a = function_cache_id(_probe_wrapper(tmp_path, f"{root}.fake_cache_probe_a", "x + 1"))
    id_b = function_cache_id(_probe_wrapper(tmp_path, f"{root}.fake_cache_probe_b", "x + 2"))
    id_a2 = function_cache_id(_probe_wrapper(tmp_path, f"{root}.fake_cache_probe_a2", "x + 1"))
    assert id_a != id_b, "identical wrapper source with a different helper body must not collide"
    assert id_a == id_a2, "the identifier must stay deterministic across rebuilds"


def test_helper_fold_follows_the_first_party_prefixes(tmp_path, monkeypatch):
    """After a package rename a helper under the new root must still be folded, and one outside every
    first-party root never is: the fold reads the derived prefixes, not a spelled-out package name."""
    monkeypatch.setattr(map_cache_key, "_FIRST_PARTY_PREFIXES", ("renamed_pkg.", "scripts."))
    renamed = [
        function_cache_id(_probe_wrapper(tmp_path, f"renamed_pkg.fake_cache_probe_{i}", body))
        for i, body in enumerate(("x + 1", "x + 2"))
    ]
    foreign = [
        function_cache_id(_probe_wrapper(tmp_path, f"thirdparty.fake_cache_probe_{i}", body))
        for i, body in enumerate(("x + 1", "x + 2"))
    ]
    assert renamed[0] != renamed[1], "a helper under the renamed package root was not folded into the key"
    assert foreign[0] == foreign[1], "a third-party helper's source entered the key"


def test_closure_fingerprint_covers_captured_helpers():
    def make(fn):
        def wrapper(x):
            return fn(x)

        return wrapper

    def helper_a(x):
        return x + 1

    def helper_b(x):
        return x + 2

    fp_a = _closure_fingerprint(make(helper_a))
    fp_b = _closure_fingerprint(make(helper_b))
    assert fp_a and fp_b, "captured callables must contribute to the fingerprint"
    assert fp_a != fp_b, "editing a captured helper must change the cache fingerprint"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
