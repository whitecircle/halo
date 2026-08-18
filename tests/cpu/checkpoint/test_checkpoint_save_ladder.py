#!/usr/bin/env python
"""CPU test: which saver a checkpoint context routes to.

``select_checkpoint_saver`` follows one precedence exactly — PP before everything (a stage is a
partial model), EP (incl. EP+TP / EP+CP / EP+ETP) before CP, CP before TP, TP before plain FSDP2,
and ``None`` when no mode owns the save. A drift here silently routes a save through the wrong
gather path, so this asserts the ROUTE — the function that will run — for every mode combination.

    python tests/cpu/checkpoint/test_checkpoint_save_ladder.py
"""

import pytest
import torch.nn as nn

import src.distributed.checkpoint.save as save_mod
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.checkpoint.save import (
    save_checkpoint,
    save_cp_checkpoint,
    save_ep_checkpoint,
    save_fsdp2_checkpoint,
    save_pp_checkpoint,
    save_tp_checkpoint,
    select_checkpoint_saver,
)


def _ctx(**flags) -> CheckpointContext:
    """Build a CheckpointContext with all flags False/defaults, overridden by kwargs."""
    base = {
        "model": nn.Linear(2, 2),
        "parallelism_config": None,
        "is_pp_mode": False,
        "is_cp_mode": False,
        "is_tp_mode": False,
        "is_ep_tp_mode": False,
        "has_ep_layers": False,
        "fsdp_wrapped": False,
        "accelerate_manages_fsdp": False,
        "is_save_rank": True,
        "max_shard_size": "5GB",
        "save_sharded_ep": False,
        "has_expert_lora": False,
        "merge_expert_lora_on_save": False,
        "cp_wrapper": None,
        "tokenizer": None,
    }
    base.update(flags)
    return CheckpointContext(**base)


# (description, flags, saver that must run) — precedence + every mode. None = nobody owns the save.
CASES = [
    ("PP", {"is_pp_mode": True}, save_pp_checkpoint),
    (
        "PP beats EP/CP/TP/FSDP2 (a stage is a partial model)",
        {"is_pp_mode": True, "has_ep_layers": True, "is_cp_mode": True, "is_tp_mode": True, "fsdp_wrapped": True},
        save_pp_checkpoint,
    ),
    ("EP-only", {"has_ep_layers": True}, save_ep_checkpoint),
    ("EP+TP", {"has_ep_layers": True, "is_ep_tp_mode": True, "is_tp_mode": True}, save_ep_checkpoint),
    ("EP+CP", {"has_ep_layers": True, "is_cp_mode": True}, save_ep_checkpoint),
    ("EP+ETP (is_ep_tp_mode)", {"is_ep_tp_mode": True}, save_ep_checkpoint),
    ("CP-only", {"is_cp_mode": True, "cp_wrapper": object()}, save_cp_checkpoint),
    ("CP beats TP", {"is_cp_mode": True, "cp_wrapper": object(), "is_tp_mode": True}, save_cp_checkpoint),
    ("CP with no live wrapper falls through", {"is_cp_mode": True}, None),
    ("TP-only", {"is_tp_mode": True}, save_tp_checkpoint),
    ("TP+DP (fsdp_wrapped)", {"is_tp_mode": True, "fsdp_wrapped": True}, save_tp_checkpoint),
    ("FSDP2 dense", {"fsdp_wrapped": True}, save_fsdp2_checkpoint),
    ("accelerate-managed FSDP", {"fsdp_wrapped": False, "accelerate_manages_fsdp": True}, None),
    ("plain / DDP", {}, None),
]


@pytest.mark.parametrize("desc,flags,expected", CASES)
def test_ladder_routes_each_mode(desc, flags, expected):
    saver = select_checkpoint_saver(_ctx(**flags))
    assert saver is expected, f"{desc}: routed to {getattr(saver, '__name__', saver)}"


@pytest.mark.parametrize("desc,flags,expected", CASES)
def test_save_checkpoint_reports_whether_a_mode_owned_the_save(desc, flags, expected, monkeypatch):
    """The bool the trainers branch on: False sends ``save_model`` to the base Trainer, and a mode
    that owns the save must never return it — under TP the base save gathers every DTensor from the
    one rank it runs on and then barriers the WORLD, hanging the job."""
    ran = []
    for name in (
        "save_pp_checkpoint",
        "save_ep_checkpoint",
        "save_cp_checkpoint",
        "save_tp_checkpoint",
        "save_fsdp2_checkpoint",
    ):
        monkeypatch.setattr(save_mod, name, lambda ctx, out, _n=name: ran.append(_n))

    handled = save_checkpoint(_ctx(**flags), "/never/written")

    assert handled is (expected is not None), desc
    assert ran == ([expected.__name__] if expected is not None else []), desc


def test_find_peft_model_none_for_plain_module():
    assert find_peft_model(nn.Linear(2, 2)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
