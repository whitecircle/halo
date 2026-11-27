#!/usr/bin/env python
"""The exact-replay harnesses train and replay with Flash-Next's QSA on its tiled path, since the block-sparse
kernel's dK and dV do not repeat, and put the knob back after. Each harness runs here up to its first load, which
records the knob and stops the run."""

import os

import pytest
import torch

from tests.common import merged_resume_e2e, resume_export
from tests.common.resume_export import ResumeExportRoute

KNOB = "HALO_QSA_KERNEL"


class _FirstLoad(Exception):
    """Raised by the stand-in for a harness's first load."""


def _run_resume_export() -> None:
    route = ResumeExportRoute(lambda *args, **kwargs: None, object, torch.bfloat16, "sdpa")
    resume_export.run_resume_export(None, route, None, base="base", root="root", save_step=1, total_steps=2)


def _run_merged_resume() -> None:
    # ep_size 1, so the run leaves the process-wide DeepEP dispatch seam alone.
    merged_resume_e2e.run_merged_resume(None, family="qwen4_exp_text", adapters="expert", ep_size=1, cp_size=1)


HARNESSES = {
    "run_resume_export": (resume_export, "load_route_model", _run_resume_export),
    "run_merged_resume": (merged_resume_e2e, "shared_output_dir", _run_merged_resume),
}


@pytest.mark.parametrize("previous", [None, "auto", "on"])
@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_a_replay_harness_runs_off_the_qsa_kernel_and_puts_the_knob_back(monkeypatch, harness, previous):
    module, first_load, run = HARNESSES[harness]
    if previous is None:
        monkeypatch.delenv(KNOB, raising=False)
    else:
        monkeypatch.setenv(KNOB, previous)
    seen = []

    def record_and_stop(*args, **kwargs):
        seen.append(os.environ.get(KNOB))
        raise _FirstLoad

    monkeypatch.setattr(module, first_load, record_and_stop)
    with pytest.raises(_FirstLoad):
        run()
    assert seen == ["off"]
    assert os.environ.get(KNOB) == previous


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
