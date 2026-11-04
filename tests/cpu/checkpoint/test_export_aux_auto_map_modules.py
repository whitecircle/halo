#!/usr/bin/env python
"""A processor or tokenizer config in an export names remote code by ``auto_map``. A tool copies those configs raw from
the release, and a trusted load then reads the modules beside them, so an export without them cannot be served."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from halo.checkpoint import config_export

_PROCESSOR_MAP = {"auto_map": {"AutoProcessor": "processing_vendor.VendorProcessor"}}
_IMAGE_MAP = {"auto_map": {"AutoImageProcessor": "image_processing_vendor.VendorImageProcessor"}}
_TOKENIZER_MAP = {
    "auto_map": {
        "AutoProcessor": "processing_vendor.VendorProcessor",
        "AutoTokenizer": ["tokenization_vendor.Slow", None],
    }
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release(root: Path) -> Path:
    release = root / "release"
    release.mkdir()
    (release / "processing_vendor.py").write_text("from .processing_shared import SHARED\n", encoding="utf-8")
    (release / "processing_shared.py").write_text("SHARED = 1\n", encoding="utf-8")
    (release / "image_processing_vendor.py").write_text("IMAGE = 1\n", encoding="utf-8")
    (release / "tokenization_vendor.py").write_text("TOKENIZER = 1\n", encoding="utf-8")
    return release


def _export(root: Path) -> Path:
    export = root / "export"
    export.mkdir()
    _write_json(export / "config.json", {"model_type": "qwen3"})
    return export


def _finalize(export: Path, source: Path | None) -> None:
    config_export.finalize_exported_config(
        SimpleNamespace(model_type="qwen3"), str(export), source=None if source is None else str(source)
    )


def test_every_module_the_copied_configs_name_lands_with_its_sibling_imports(tmp_path):
    release, export = _release(tmp_path), _export(tmp_path)
    _write_json(export / "processor_config.json", _PROCESSOR_MAP)
    _write_json(export / "preprocessor_config.json", _IMAGE_MAP)
    _write_json(export / "tokenizer_config.json", _TOKENIZER_MAP)

    _finalize(export, release)

    for module in ("processing_vendor", "processing_shared", "image_processing_vendor", "tokenization_vendor"):
        assert (export / f"{module}.py").read_text() == (release / f"{module}.py").read_text(), module


def test_a_repo_qualified_reference_resolves_by_itself(tmp_path):
    release, export = _release(tmp_path), _export(tmp_path)
    _write_json(export / "processor_config.json", {"auto_map": {"AutoProcessor": "org/repo--processing_vendor.P"}})

    _finalize(export, release)

    assert not (export / "processing_vendor.py").exists()


@pytest.mark.parametrize("source_kind", ["lacks_the_module", "no_source"])
def test_a_module_nobody_carries_warns_and_leaves_the_save_alive(tmp_path, source_kind):
    export = _export(tmp_path)
    source = tmp_path / "empty"
    source.mkdir()
    _write_json(export / "processor_config.json", _PROCESSOR_MAP)

    with patch.object(config_export.logger, "warning") as warning:
        _finalize(export, source if source_kind == "lacks_the_module" else None)

    assert any("trust_remote_code" in str(call) for call in warning.call_args_list)


def test_an_export_with_no_remote_processor_copies_nothing(tmp_path):
    release, export = _release(tmp_path), _export(tmp_path)
    _write_json(export / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast"})
    before = sorted(path.name for path in export.iterdir())

    _finalize(export, release)

    copied = sorted(path.name for path in export.iterdir() if path.suffix == ".py")
    assert copied == [], before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
