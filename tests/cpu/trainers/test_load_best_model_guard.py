#!/usr/bin/env python
"""Tests for the config-time ``load_best_model_at_end`` refusal.

The failure this guards is a full run that trains to completion and then dies at the export step,
because the checkpoint loader refuses to reload base weights into an EP/CP/TP-transformed model. The
guard must fire for every shape whose end-of-run reload is refused, and for no shape that reloads
fine — a false positive blocks a working run just as hard.

Run: pytest tests/cpu/trainers/test_load_best_model_guard.py
"""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.validation import ParallelismValidationMixin

# A real PretrainedConfig, not a namespace: get_peft_model reads the config through ``to_dict()``
# for its tied-weights probe, and the guard reads the expert count off it.
MOE_CONFIG = PretrainedConfig(num_local_experts=8)
DENSE_CONFIG = PretrainedConfig()


class _Model(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.config = config


def _validate(
    *,
    ep=1,
    cp=1,
    tp=1,
    grouped_gemm=False,
    moe=False,
    load_best=True,
    peft=False,
    expert_lora=None,
    merge_expert_lora=False,
):
    model = _Model(MOE_CONFIG if moe else DENSE_CONFIG)
    if peft:
        model = get_peft_model(model, LoraConfig(target_modules=["fc"]))
        model.config = MOE_CONFIG if moe else DENSE_CONFIG
    stub = SimpleNamespace(
        parallelism_config=ParallelismConfig(
            world_size=8,
            gpus_per_node=8,
            ep_size=ep,
            cp_size=cp,
            tp_size=tp,
            use_grouped_gemm=grouped_gemm,
            expert_lora=expert_lora,
            merge_expert_lora_on_save=merge_expert_lora,
        ),
        args=SimpleNamespace(load_best_model_at_end=load_best),
        model=model,
    )
    ParallelismValidationMixin._validate_load_best_model_reloadable(stub)


def test_flag_off_never_raises():
    _validate(ep=8, moe=True, load_best=False)


def test_plain_data_parallel_is_allowed():
    """Dense FSDP2 reloads the best checkpoint in place — refusing it would block the common case."""
    _validate()


@pytest.mark.parametrize(
    ("kwargs", "shape"),
    [
        ({"cp": 8}, "CP"),
        ({"ep": 8, "moe": True}, "EP"),
        ({"moe": True, "grouped_gemm": True}, "grouped-GEMM MoE (needs_ep_wrappers without EP)"),
        ({"tp": 8, "moe": True, "grouped_gemm": True}, "MoE pure TP with EP wrappers — experts load at construction"),
        ({"tp": 2}, "dense TP + DP — FSDP2 over TP, a 2-D placement the reload does not invert"),
    ],
)
def test_shapes_whose_reload_is_refused_raise_at_config_time(kwargs, shape):
    """Every shape here reaches a loader refusal that fires only AFTER the full run."""
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        _validate(**kwargs)


def test_dense_model_under_grouped_gemm_is_allowed():
    """``needs_ep_wrappers`` is True whenever grouped GEMM is on, but the wrappers only attach to a
    MoE model — a dense run there reloads normally and must not be refused."""
    _validate(grouped_gemm=True)


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe-without-ep-wrappers"])
def test_pure_tp_is_allowed(moe):
    """``tp_size == world_size`` shards into DTensors on a 1-D TP mesh, which the loader reloads by
    ``distribute_tensor`` into the live placements (``_load_tp``). A blanket ``tp_size > 1`` arm
    would refuse a configuration that works."""
    _validate(tp=8, moe=moe)


@pytest.mark.parametrize("shape", [{"ep": 8, "moe": True}, {"cp": 8}, {"tp": 8, "moe": True}])
def test_peft_runs_are_exempt_on_every_refused_shape(shape):
    """Adapters reload in place, so the shapes refused for a full fine-tune must stay allowed here —
    without this the guard would block every LoRA + EP/CP/TP run at construction."""
    _validate(peft=True, **shape)


def test_native_expert_lora_is_exempt():
    """``expert_lora`` is the EP-native adapter path: no PEFT wrapper, same in-place reload."""
    _validate(ep=8, moe=True, expert_lora=ExpertLoraSpec(r=8, alpha=16.0))


def test_merged_expert_lora_is_not_exempt():
    """``merge_expert_lora_on_save`` folds the adapter into a full base checkpoint, so the run ships
    base weights and the loader refuses it — exempting it as "an adapter run" defers that refusal to
    the end of training, which is exactly what this guard exists to prevent."""
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        _validate(ep=8, moe=True, expert_lora=ExpertLoraSpec(r=8, alpha=16.0), merge_expert_lora=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
