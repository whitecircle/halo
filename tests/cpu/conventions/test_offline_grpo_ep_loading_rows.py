"""Each offline-GRPO EP tier runs both checkpoint-loader implementations, one manifest row each."""

import shlex

import pytest

from tests.common.offline_grpo import EP_LOADINGS, ep_loading_parser
from tests.common.utils import imports_name
from tests.gpu.manifest import MANIFEST, script_path

EP_SUITES = ("trainers/grpo/test_offline_grpo_ep_cp.py", "trainers/grpo/test_offline_grpo_ep_reference.py")


@pytest.mark.parametrize("suite", EP_SUITES)
def test_ep_manifest_rows_run_lazy_and_eager_loaders(suite):
    spec = MANIFEST[suite]
    assert imports_name(script_path(suite), ep_loading_parser.__name__), f"{suite} must read the shared flag"
    rows = [ep_loading_parser().parse_args(shlex.split(row)).ep_loading for row in spec.args_matrix]
    assert sorted(rows) == sorted(EP_LOADINGS)
    assert spec.nproc == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
