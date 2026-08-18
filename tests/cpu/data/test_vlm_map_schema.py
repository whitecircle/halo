#!/usr/bin/env python3
"""VLM map Arrow-schema stability on mixed datasets + cache-key coverage of captured helpers.

Shard-wise type inference diverges when image rows and text-only rows land in different map worker
shards (struct vs Json content, null vs Image lists). `vlm_map_features` pins the schema so the
multiprocess map succeeds for every mix; `_get_closure_fingerprint` must cover closure-captured
helper functions so editing them invalidates the dataset cache.

Usage:
    python tests/cpu/data/test_vlm_map_schema.py
"""

import importlib.util
import sys

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image

from src.data.pipeline.processing import _get_closure_fingerprint, get_function_identifier
from src.data.pipeline.row_processors import create_vlm_processor
from src.data.pipeline.vlm_dataset import vlm_map_features


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
    process = create_vlm_processor(conversation_field="texts", images_field="images")
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


def test_function_identifier_covers_module_global_helpers(tmp_path):
    """Editing a module-global helper a map function calls must change the caller's cache identity."""

    def build(helper_body: str, name: str):
        # Real files on disk: the identifier hashes inspect.getsource, which exec()'d code lacks.
        path = tmp_path / f"{name}.py"
        path.write_text(f"def helper(x):\n    return {helper_body}\n\n\ndef wrapper(x):\n    return helper(x)\n")
        spec = importlib.util.spec_from_file_location(f"src.fake_cache_probe_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.wrapper

    id_a = get_function_identifier(build("x + 1", "a"))
    id_b = get_function_identifier(build("x + 2", "b"))
    id_a2 = get_function_identifier(build("x + 1", "a2"))
    assert id_a != id_b, "identical wrapper source with a different helper body must not collide"
    assert id_a == id_a2, "the identifier must stay deterministic across rebuilds"


def test_closure_fingerprint_covers_captured_helpers():
    def make(fn):
        def wrapper(x):
            return fn(x)

        return wrapper

    def helper_a(x):
        return x + 1

    def helper_b(x):
        return x + 2

    fp_a = _get_closure_fingerprint(make(helper_a))
    fp_b = _get_closure_fingerprint(make(helper_b))
    assert fp_a and fp_b, "captured callables must contribute to the fingerprint"
    assert fp_a != fp_b, "editing a captured helper must change the cache fingerprint"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
