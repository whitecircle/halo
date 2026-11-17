#!/usr/bin/env python
"""The stale-figure gate fails when a figure's page, the data it reads, the visuals kit, a render script or the PNG
itself changed since ``make diagrams`` wrote the manifest, and when a page was never rendered."""

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT

_SCRIPT = "scripts/docs/figure_manifest.sh"
_CHARTS = Path("agent-docs/assets") / "charts"
_PAGE = _CHARTS / "speed.html"
_DATA = _CHARTS / "runs" / "summary.tsv"
_KIT_STYLE = Path("skills/visuals/kit/figure.css")
_RENDER = Path("skills/visuals/scripts/render.sh")


def _write_sample_tree(root: Path) -> None:
    files = {
        _PAGE: "<script>fetch('runs/summary.tsv')</script>",
        _PAGE.with_suffix(".png"): "png",
        _DATA: "1\t2\n",
        _CHARTS / "raw" / "trace.json": "{}",
        _KIT_STYLE: "body {}",
        Path("skills/visuals/kit/figure.js"): "",
        _RENDER: "",
        Path("skills/visuals/scripts/canvas-size.sh"): "",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    script = root / _SCRIPT
    script.parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / _SCRIPT, script)


def _run(root: Path, mode: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(root / _SCRIPT), mode], capture_output=True, text=True, check=False)


@pytest.fixture
def rendered(tmp_path):
    _write_sample_tree(tmp_path)
    assert _run(tmp_path, "write").returncode == 0
    result = _run(tmp_path, "check")
    assert result.returncode == 0, result.stderr
    return tmp_path


@pytest.mark.parametrize("edited", [_PAGE, _DATA, _KIT_STYLE, _RENDER, _PAGE.with_suffix(".png")])
def test_an_edit_since_the_last_render_fails_naming_the_file(rendered, edited):
    (rendered / edited).write_text("changed")
    result = _run(rendered, "check")
    assert result.returncode == 1
    assert f"{edited}: FAILED" in result.stderr


def test_a_data_folder_no_page_names_is_not_an_input(rendered):
    (rendered / _CHARTS / "raw" / "trace.json").write_text("changed")
    result = _run(rendered, "check")
    assert result.returncode == 0, result.stderr


def test_a_page_never_rendered_fails_naming_it_and_its_png(rendered):
    page = _CHARTS / "memory.html"
    (rendered / page).write_text("")
    result = _run(rendered, "check")
    assert result.returncode == 1
    assert f"{page}\n{page.with_suffix('.png')}\n" in result.stderr


def test_a_render_after_the_edit_clears_the_gate(rendered):
    (rendered / _PAGE).write_text("changed")
    assert _run(rendered, "write").returncode == 0
    result = _run(rendered, "check")
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
