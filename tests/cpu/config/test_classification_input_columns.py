#!/usr/bin/env python
"""classification.py must validate its input columns BEFORE the model load.

``tokenize_classification_row`` reads a pre-built ``prompt`` conversation, falling back to
``text_field`` — with neither column present the failure surfaces as a per-row ``KeyError`` deep in
the dataset map, after the full (possibly multi-node) model load. The config-time guard raises
first, naming what is missing.

Run: pytest tests/cpu/config/test_classification_input_columns.py
"""

import sys

import pytest

from tests.common.utils import load_script_module


@pytest.fixture(scope="module")
def classification():
    return load_script_module("scripts/training/classification.py", "halo_test_classification_script")


def test_prompt_column_passes(classification):
    classification.require_prompt_or_text_column(["prompt", "label"], text_field=None)


def test_text_field_naming_an_existing_column_passes(classification):
    classification.require_prompt_or_text_column(["text", "label"], text_field="text")


def test_neither_prompt_nor_text_field_raises(classification):
    with pytest.raises(ValueError, match="text_field is not set"):
        classification.require_prompt_or_text_column(["document", "label"], text_field=None)


def test_text_field_naming_a_missing_column_raises_with_the_name(classification):
    with pytest.raises(ValueError, match="text_field='body' names no existing column"):
        classification.require_prompt_or_text_column(["document", "label"], text_field="body")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
