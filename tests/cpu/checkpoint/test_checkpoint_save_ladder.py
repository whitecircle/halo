#!/usr/bin/env python
"""Which saver a checkpoint context routes to: ``select_checkpoint_saver``'s precedence, asserted as the function that
will run. A drift silently routes a save through the wrong gather path."""

import pytest
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import halo.checkpoint.parallel.save as save_mod
from halo.checkpoint.parallel.adapters import save_expert_lora_adapters, save_peft_adapters
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint, save_ep_sharded_checkpoint
from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.parallel.save import (
    save_checkpoint,
    save_cp_checkpoint,
    save_fsdp2_checkpoint,
    save_tp_checkpoint,
    save_unsharded_checkpoint,
    select_checkpoint_saver,
)
from halo.models.external_tables import ExternalTable
from halo.models.introspection.unwrapping import find_peft_model
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config

_PP = {"world_size": 4, "gpus_per_node": 2, "pp_size": 2}
_CP = {"world_size": 2, "gpus_per_node": 2, "cp_size": 2}
_TP = {"world_size": 2, "gpus_per_node": 2, "tp_size": 2}
_TP_DP = {"world_size": 4, "gpus_per_node": 4, "tp_size": 2}
_EP_TP = {"world_size": 4, "gpus_per_node": 4, "ep_size": 2, "tp_size": 2}
_MERGE = {"world_size": 1, "gpus_per_node": 1, "merge_expert_lora_on_save": True}


class _Table(ExternalTable):
    part_key_format = "shard_{index}.weight"


class _Tiny(nn.Module):
    def __init__(self, *, table: bool = False):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        if table:
            self.table = _Table(rows=4, dim=2, part_count=1)


def _ctx(topology: dict | None = None, *, peft: bool = False, table: bool = False, **fields):
    """``peft`` hands it a PeftModel, ``table`` a model holding an external table."""
    model = _Tiny(table=table)
    model = get_peft_model(model, LoraConfig(r=1, target_modules=["proj"])) if peft else model
    config = make_parallelism_config(**topology) if topology else None
    return save_context(model, config, is_save_rank=True, **fields)


_EXPERT_LORA = {"has_ep_layers": True, "has_expert_lora": True}

# None = nobody owns the save.
CASES = [
    ("PEFT adapters", None, {"peft": True}, save_peft_adapters),
    (
        "PEFT beats PP, EP and FSDP2 (adapter-only)",
        _PP,
        {"peft": True, "has_ep_layers": True, "toolkit_fsdp2": True},
        save_peft_adapters,
    ),
    ("PEFT under accelerate-managed FSDP falls through", None, {"peft": True, "accelerate_manages_fsdp": True}, None),
    (
        "PEFT with merged expert LoRA takes the EP save",
        _MERGE,
        {"peft": True, **_EXPERT_LORA},
        save_ep_gathered_checkpoint,
    ),
    ("PP", _PP, {}, save_pp_checkpoint),
    (
        "PP beats expert LoRA, EP layers and FSDP2 (a stage is a partial model)",
        _PP,
        {**_EXPERT_LORA, "toolkit_fsdp2": True},
        save_pp_checkpoint,
    ),
    ("expert LoRA adapters", None, _EXPERT_LORA, save_expert_lora_adapters),
    ("expert LoRA beats EP+TP", _EP_TP, _EXPERT_LORA, save_expert_lora_adapters),
    ("merged expert LoRA takes the EP save", _MERGE, _EXPERT_LORA, save_ep_gathered_checkpoint),
    ("EP-only", None, {"has_ep_layers": True}, save_ep_gathered_checkpoint),
    ("EP+TP", _EP_TP, {"has_ep_layers": True}, save_ep_gathered_checkpoint),
    ("EP+CP", _CP, {"has_ep_layers": True}, save_ep_gathered_checkpoint),
    ("EP+TP before its layers are found (EP group + TP mode)", _EP_TP, {}, save_ep_gathered_checkpoint),
    ("CP-only", _CP, {"cp_wrapper": object()}, save_cp_checkpoint),
    ("CP with no live wrapper falls through", _CP, {}, None),
    ("TP-only", _TP, {}, save_tp_checkpoint),
    ("TP+DP (toolkit FSDP2)", _TP_DP, {"toolkit_fsdp2": True}, save_tp_checkpoint),
    ("FSDP2 dense", None, {"toolkit_fsdp2": True}, save_fsdp2_checkpoint),
    ("accelerate-managed FSDP", None, {"accelerate_manages_fsdp": True}, None),
    ("plain / DDP", None, {}, None),
    ("plain / DDP holding an external table", None, {"table": True}, save_unsharded_checkpoint),
    ("FSDP2 holding an external table", None, {"table": True, "toolkit_fsdp2": True}, save_fsdp2_checkpoint),
    ("PEFT adapters of a model holding an external table", None, {"table": True, "peft": True}, save_peft_adapters),
    (
        "accelerate-managed FSDP holding an external table falls through (refused at trainer construction)",
        None,
        {"table": True, "accelerate_manages_fsdp": True},
        None,
    ),
]


@pytest.mark.parametrize("desc,topology,fields,expected", CASES)
def test_ladder_routes_each_mode(desc, topology, fields, expected):
    saver = select_checkpoint_saver(_ctx(topology, **fields))
    assert saver is expected, f"{desc}: routed to {getattr(saver, '__name__', saver)}"


@pytest.mark.parametrize("desc,topology,fields,expected", CASES)
def test_save_checkpoint_reports_whether_a_mode_owned_the_save(desc, topology, fields, expected, monkeypatch):
    """False sends ``save_model`` to the base Trainer; under TP that gathers from one rank and barriers the world,
    hanging the job."""
    ran = []
    for name in (
        "save_peft_adapters",
        "save_pp_checkpoint",
        "save_expert_lora_adapters",
        "save_ep_sharded_checkpoint",
        "save_ep_gathered_checkpoint",
        "save_cp_checkpoint",
        "save_tp_checkpoint",
        "save_fsdp2_checkpoint",
        "save_unsharded_checkpoint",
    ):
        monkeypatch.setattr(save_mod, name, lambda ctx, out, _n=name: ran.append(_n))

    handled = save_checkpoint(_ctx(topology, **fields), "/never/written")

    assert handled is (expected is not None), desc
    assert ran == ([expected.__name__] if expected is not None else []), desc


@pytest.mark.parametrize("world_size, expected", [(2, save_ep_sharded_checkpoint), (1, save_ep_gathered_checkpoint)])
def test_a_sharded_ep_save_needs_more_than_one_rank(world_size, expected, monkeypatch):
    monkeypatch.setattr(save_mod, "get_global_world_size", lambda: world_size)
    assert select_checkpoint_saver(_ctx(has_ep_layers=True, save_sharded_ep=True)) is expected


def test_find_peft_model_none_for_plain_module():
    assert find_peft_model(nn.Linear(2, 2)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
