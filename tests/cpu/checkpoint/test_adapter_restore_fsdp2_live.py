#!/usr/bin/env python
"""An adapter resume onto an FSDP2-sharded PEFT model writes the saved values into the live sharded adapters, on two
gloo ranks. PEFT gathers every DTensor it selects into a CPU copy, so a restore copying into what PEFT's selection
returns would land in temporaries and resume the adapters from initialization; the selection runs over meta stand-ins,
and a key PEFT answers with a tensor it derived from one is refused."""

import re

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.adapters import write_adapter_file
from halo.checkpoint.parallel import adapters as adapters_mod
from halo.checkpoint.parallel.adapters import live_adapter_tensors, restore_adapters
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG

WORLD = 2


def _saved_value(index: int, shape) -> torch.Tensor:
    return torch.full(tuple(shape), 0.5 + 0.01 * index)


def _sharded_lora_model(modules_to_save: list[str] | None) -> torch.nn.Module:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    peft_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], modules_to_save=modules_to_save)
    model = get_peft_model(model, peft_config)
    mesh = init_device_mesh("cpu", (WORLD,))
    for layer in model.base_model.model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _restore_worker(rank: int, adapter_dir: str, modules_to_save: list[str] | None) -> None:
    model = _sharded_lora_model(modules_to_save)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    assert params and all(isinstance(param, DTensor) for param in params.values()), "premise: sharded adapters"

    live = live_adapter_tensors(model)
    assert {id(tensor) for tensor in live.values()} == {id(param) for param in params.values()}, (
        "the restore's copy targets are not the live trained params"
    )

    keys = sorted(live)
    if rank == 0:
        model.peft_config["default"].save_pretrained(adapter_dir)
        write_adapter_file({key: _saved_value(i, live[key].shape) for i, key in enumerate(keys)}, adapter_dir)
    dist.barrier()
    assert restore_adapters(adapter_dir, model, is_cp_mode=False) is not None

    key_of = {id(tensor): key for key, tensor in live.items()}
    stale = [
        name
        for name, param in params.items()
        if not torch.equal(param.full_tensor(), _saved_value(keys.index(key_of[id(param)]), param.shape))
    ]
    assert not stale, f"{len(stale)} sharded adapters kept their pre-resume values, e.g. {stale[:2]}"


@pytest.mark.parametrize("modules_to_save", [None, ["lm_head"]], ids=["lora", "lora_and_modules_to_save"])
def test_resume_writes_the_live_sharded_adapters(tmp_path, modules_to_save):
    run_gloo_ranks(_restore_worker, WORLD, str(tmp_path), modules_to_save)


def test_a_key_peft_answers_with_a_tensor_derived_from_a_stand_in_is_refused(monkeypatch):
    """A restore's copy into a meta tensor writes nothing and raises nothing."""
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    select = adapters_mod.get_peft_model_state_dict
    derived = sorted(select(model))[1]

    def selection_deriving_one_key(*args, **kwargs):
        selected = select(*args, **kwargs)
        return {**selected, derived: selected[derived].clone()}

    monkeypatch.setattr(adapters_mod, "get_peft_model_state_dict", selection_deriving_one_key)
    with pytest.raises(RuntimeError, match=rf"built from the meta stand-ins(?s:.*){re.escape(derived)}"):
        live_adapter_tensors(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
