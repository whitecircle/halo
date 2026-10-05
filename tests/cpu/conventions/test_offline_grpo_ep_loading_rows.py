"""The EP+CP GRPO tier runs both checkpoint-loader implementations."""

import shlex

import pytest

from tests.gpu.manifest import MANIFEST
from tests.gpu.trainers.grpo.test_offline_grpo_ep_cp import ep_cp_parser


def test_ep_cp_manifest_runs_lazy_and_eager_loaders():
    spec = MANIFEST["trainers/grpo/test_offline_grpo_ep_cp.py"]
    rows = [ep_cp_parser().parse_args(shlex.split(row)).ep_loading for row in spec.args_matrix]
    assert sorted(rows) == ["eager", "lazy"]
    assert spec.nproc == 8


@pytest.mark.parametrize("loading", ["eager", "lazy"])
def test_explicit_ep_loader_row_overrides_environment_default(monkeypatch, loading):
    monkeypatch.setenv("HALO_TEST_OFFLINE_GRPO_EP_LAZY", "1" if loading == "eager" else "0")
    assert ep_cp_parser().parse_args(["--ep-loading", loading]).ep_loading == loading


@pytest.mark.parametrize("enabled,loading", [("0", "eager"), ("1", "lazy")])
def test_ep_loader_default_follows_environment(monkeypatch, enabled, loading):
    monkeypatch.setenv("HALO_TEST_OFFLINE_GRPO_EP_LAZY", enabled)
    assert ep_cp_parser().parse_args([]).ep_loading == loading


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
