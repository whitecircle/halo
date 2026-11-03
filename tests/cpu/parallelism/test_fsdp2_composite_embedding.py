#!/usr/bin/env python
"""The FSDP2 group holding a composite's input embedding is the module whose forward reads it, under ``.model`` or at a
root holding ``language_model`` itself; a group below the reader leaves it a DTensor when the reader embeds."""

import datetime
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh

from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.models.module_tree import find_module_path, input_embedding_backbone
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_LLAMA4_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    tiny_family_model,
)

_POLICY = mixed_precision_policy(torch.bfloat16, fp32_reduce=True, cast_forward_inputs=True)
# Each is a composite with no ``.model``: the root's forward embeds and hands ``inputs_embeds`` down.
_ROOT_EMBEDDING = {
    "llama4": TINY_LLAMA4_FAMILY,
    "nemotron_h_omni": TINY_UNWRAPPED_COMPOSITE_FAMILIES["nemotron_h_omni"],
}
_ROSTER = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# ``CAUSAL_LM``, whose tuner calls the composite's ``.forward()``, as a training script's adapter does.
_LORA = LoraConfig(r=2, target_modules=r".*language_model.*(q_proj|v_proj)", task_type="CAUSAL_LM")


def _forward_backward(model) -> torch.Tensor:
    # Across the vocabulary, so every rank's shard of the embedding takes a gradient.
    ids = torch.arange(0, 128, 8).unsqueeze(0)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    return loss


@pytest.fixture
def mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("name", sorted(_ROSTER))
def test_the_roster_keeps_its_embedding_group(name):
    """Every shipped family reaches it under ``.model``."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[name])
    assert find_module_path(model, input_embedding_backbone(model)) == "model"


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_a_composite_with_no_model_groups_the_embedding_at_its_root(name):
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name])
    assert getattr(model, "model", None) is None and getattr(model, "language_model", None) is not None
    assert input_embedding_backbone(model) is model


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_a_wrapper_above_that_composite_takes_the_group(name):
    """The tuner reaches the composite through ``.forward()``, so a group there would never unshard."""
    torch.manual_seed(0)
    model = get_peft_model(tiny_family_model(_ROOT_EMBEDDING[name]), _LORA)
    assert input_embedding_backbone(model) is model


def test_a_text_class_under_a_root_model_is_not_mistaken_for_a_composite():
    """No ``language_model`` on the root: its own backbone embeds."""

    class _Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _Backbone()

    model = _Root()
    assert input_embedding_backbone(model) is model.transformer


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
@pytest.mark.parametrize("reshard_after_forward", [False, True], ids=["zero2", "zero3"])
def test_the_wrapped_root_embeds_and_runs_backward(name, reshard_after_forward, mesh):
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name]).to(torch.bfloat16)
    model.train()
    apply_fsdp2_per_layer(model, mesh, _POLICY, reshard_after_forward, IdentityParamSet())
    _forward_backward(model)
    embedding = model.get_input_embeddings().weight
    assert embedding.grad is not None and bool((embedding.grad.to_local() != 0).any())


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_an_adapter_on_the_wrapped_composite_trains(name, mesh):
    """The composite reached through PEFT's tuner, which runs no hook of its own."""
    torch.manual_seed(0)
    model = get_peft_model(tiny_family_model(_ROOT_EMBEDDING[name]), _LORA)
    model.to(torch.bfloat16).train()
    apply_fsdp2_per_layer(model, mesh, _POLICY, False, IdentityParamSet())
    _forward_backward(model)
    adapters = [param for key, param in model.named_parameters() if "lora_B" in key]
    assert adapters and any(param.grad is not None and bool((param.grad.to_local() != 0).any()) for param in adapters)


def _two_rank_worker(rank: int, name: str, out: str) -> None:
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name]).to(torch.bfloat16).train()
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (2,)), _POLICY, False, IdentityParamSet())
    loss = _forward_backward(model)
    embedding = model.get_input_embeddings().weight
    with open(f"{out}.{rank}", "w") as handle:
        json.dump(
            {
                "loss": float(loss),
                "sharded": list(embedding.to_local().shape) != list(embedding.shape),
                "grad": float(embedding.grad.to_local().abs().sum()),
            },
            handle,
        )


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_two_ranks_share_the_embedding_and_agree(name, tmp_path):
    """A group actually sharded across ranks, what a one-rank mesh cannot show."""
    out = str(tmp_path / "rank")
    run_gloo_ranks(_two_rank_worker, 2, name, out, pg_timeout=datetime.timedelta(seconds=120))
    ranks = [json.loads(Path(f"{out}.{rank}").read_text()) for rank in range(2)]
    assert all(rank["sharded"] for rank in ranks), ranks
    assert ranks[0]["loss"] == ranks[1]["loss"], ranks
    assert all(rank["grad"] > 0 for rank in ranks), ranks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
