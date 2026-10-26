"""Each offline-GRPO EP tier runs both checkpoint-loader implementations, one manifest row each."""

import shlex

import pytest

from tests.common.offline_grpo import EP_LOADINGS, ep_loading_parser
from tests.common.source_sweep import imports_name
from tests.gpu.manifest import MANIFEST, script_path

EP_SUITES = ("trainers/grpo/test_offline_grpo_ep_cp.py", "trainers/grpo/test_offline_grpo_reference.py")


@pytest.mark.parametrize("suite", EP_SUITES)
def test_ep_manifest_rows_run_lazy_and_eager_loaders(suite):
    assert imports_name(script_path(suite), ep_loading_parser.__name__), f"{suite} must read the shared flag"
    ep_rows = [(args, spec) for args, spec in MANIFEST[suite].launches() if spec.nproc == 8]
    loadings = [ep_loading_parser().parse_known_args(shlex.split(args))[0].ep_loading for args, _ in ep_rows]
    assert sorted(loadings) == sorted(EP_LOADINGS)
    assert all("--ep-loading" in args for args, _ in ep_rows), "an EP row must name its loader"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
