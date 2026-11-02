#!/usr/bin/env python
"""A trainer handed its model as a path string loads it with the family patches whether or not a backend is pinned.

``resolve_trainer_model`` builds the model a trainer constructor was given as a path (SMPO, offline GRPO, a
teacher-distillation student). Zaya's patches key on the family alone: the router's load recording, which its native
``balancing_biases`` buffer needs, and the gradient-checkpointing refusal (backward recompute through the CCA convs
faults in cuDNN). A load that applied them only under a pinned ``attn_implementation`` left an unpinned run without
either. Checked in a fresh interpreter, since the patches are process-wide and idempotent and an earlier test in this
process may already have installed them.

    python tests/cpu/model_loading/test_trainer_path_family_patches.py
"""

import json
import subprocess
import sys

import pytest

from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint

_PROBE = """
import json, sys
from types import SimpleNamespace
import transformers.models.zaya.modeling_zaya as zaya_modeling
from halo.model_loading.parallel.whole_model import resolve_trainer_model
before = zaya_modeling.ZayaPreTrainedModel.supports_gradient_checkpointing
args = SimpleNamespace(model_init_kwargs=json.loads(sys.argv[2]), gradient_checkpointing=False)
model = resolve_trainer_model(sys.argv[1], args, parallelism_config=None)
print(json.dumps({
    "checkpointing_before": before,
    "checkpointing": type(model).supports_gradient_checkpointing,
    "load_recording": bool(getattr(zaya_modeling.ZayaRouter.forward, "_halo_load_recording", False)),
}))
"""


@pytest.fixture(scope="module")
def zaya_checkpoint(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("zaya"))
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["zaya"], directory)
    return directory


@pytest.mark.parametrize("model_init_kwargs", [{}, {"attn_implementation": "eager"}], ids=["unpinned", "pinned"])
def test_a_trainer_path_zaya_load_installs_the_family_patches(zaya_checkpoint, model_init_kwargs):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, zaya_checkpoint, json.dumps(model_init_kwargs)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    assert probe["checkpointing_before"] is not False, "premise: the fresh interpreter starts unpatched"
    assert probe == {
        "checkpointing_before": probe["checkpointing_before"],
        "checkpointing": False,
        "load_recording": True,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
