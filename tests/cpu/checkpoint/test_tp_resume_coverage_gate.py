#!/usr/bin/env python
"""The TP resume coverage gate: a checkpoint that matches too little of the live model must raise.

``CheckpointLoader._load_tp`` writes only the keys the checkpoint and the live model share, so
nothing else reports a bad checkpoint: a foreign key set applies as a no-op and the run trains on
base weights while logging a successful resume. The gate is the same numel-weighted verdict the
FSDP2 path uses; these tests drive it through :meth:`CheckpointLoader.load_model` so the wiring is
covered, not just the predicate.

Run: ``python tests/cpu/checkpoint/test_tp_resume_coverage_gate.py`` (or ``pytest -m cpu``).
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader


class _Model(nn.Module):
    """A large and a small parameter, so a majority-coverage checkpoint can miss the small one."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(16, 16, bias=False)
        self.head = nn.Linear(2, 2, bias=False)


def _tp_ctx(model):
    return CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=True,
        has_ep_layers=False,
        fsdp_wrapped=False,
        tp_rank=0,
        tp_size=2,
        super_load_from_checkpoint=lambda *a, **k: None,
        super_load_optimizer_and_scheduler=lambda *a, **k: None,
    )


def test_a_foreign_checkpoint_raises_and_leaves_the_model_untouched(tmp_path):
    save_file({"totally.wrong.key": torch.ones(16, 16)}, os.path.join(tmp_path, "model.safetensors"))
    model = _Model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    with pytest.raises(RuntimeError, match="fewer than half"):
        CheckpointLoader(_tp_ctx(model)).load_model(str(tmp_path), model)

    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name}: a refused resume must not half-apply the checkpoint"


def test_a_checkpoint_covering_a_minority_of_the_parameters_raises(tmp_path):
    """The verdict weighs elements, not keys: the head alone is 4 of 260 parameters."""
    save_file({"head.weight": torch.full((2, 2), 3.0)}, os.path.join(tmp_path, "model.safetensors"))
    model = _Model()

    with pytest.raises(RuntimeError, match="fewer than half"):
        CheckpointLoader(_tp_ctx(model)).load_model(str(tmp_path), model)


def test_a_majority_checkpoint_loads_what_it_has_and_keeps_the_rest(tmp_path):
    """A legitimately partial checkpoint (tied heads, task heads) is a real resume."""
    save_file({"body.weight": torch.full((16, 16), 3.0)}, os.path.join(tmp_path, "model.safetensors"))
    model = _Model()
    head_before = model.head.weight.detach().clone()

    CheckpointLoader(_tp_ctx(model)).load_model(str(tmp_path), model)

    assert torch.equal(model.body.weight, torch.full((16, 16), 3.0))
    assert torch.equal(model.head.weight, head_before)


def test_a_matching_checkpoint_loads_every_key(tmp_path):
    save_file(
        {"body.weight": torch.full((16, 16), 3.0), "head.weight": torch.full((2, 2), 5.0)},
        os.path.join(tmp_path, "model.safetensors"),
    )
    model = _Model()

    CheckpointLoader(_tp_ctx(model)).load_model(str(tmp_path), model)

    assert torch.equal(model.body.weight, torch.full((16, 16), 3.0))
    assert torch.equal(model.head.weight, torch.full((2, 2), 5.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
