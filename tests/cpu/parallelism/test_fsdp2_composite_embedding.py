#!/usr/bin/env python
"""The FSDP2 group holding a composite's input embedding is the module whose forward reads it, under ``.model`` or at a
root holding ``language_model`` itself; a group below the reader leaves it a DTensor when the reader embeds."""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from trl.trainer.utils import selective_log_softmax

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.reading import load_full_state_dict
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import (
    IdentityParamSet,
    apply_fsdp2_per_layer,
    mixed_precision_policy,
    reshard_fsdp2_modules,
)
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.module_tree import find_module_path, input_embedding_backbone
from halo.models.nemotron_h import apply_nemotron_h_patches
from halo.trainers.grpo.logprob_scoring import LogprobScoringMixin
from tests.common.checkpoint_io import save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_LLAMA4_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    fill_release_scale_bias,
    tiny_family_model,
)

_POLICY = mixed_precision_policy(torch.bfloat16, fp32_reduce=True, cast_forward_inputs=True)
# Each is a composite with no ``.model``: the root's forward embeds and hands ``inputs_embeds`` down.
_ROOT_EMBEDDING = {
    "llama4": TINY_LLAMA4_FAMILY,
    "nemotron_h_omni": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"],
}
_ROSTER = {
    name: family
    for name, family in {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}.items()
    if name not in _ROOT_EMBEDDING
}
_NEMOTRON_ROUTES = {"composite": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"], "text": TINY_MOE_FAMILIES["nemotron_h"]}
# ``CAUSAL_LM``, whose tuner calls the composite's ``.forward()``, as a training script's adapter does.
_LORA = LoraConfig(r=2, target_modules=r".*language_model.*(q_proj|v_proj)", task_type="CAUSAL_LM")


# bf16 logits against the sweep's fp32 ones, at a small tiny head's scale.
_LOGP_ATOL = 5e-2
_COMPLETION = 4


class _Scorer(LogprobScoringMixin):
    """What the chunked path reads off a trainer."""

    temperature = 1.0
    accelerator = SimpleNamespace(unwrap_model=lambda module: module)


def _rows(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.stack([torch.arange(start, start + 80, 5) for start in range(batch)])
    return ids, torch.ones_like(ids)


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


def _grads_reached(module) -> bool:
    gradient = module.weight.grad
    gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
    return gradient is not None and bool((gradient != 0).any())


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
@pytest.mark.parametrize("batch", [1, 2], ids=["dense", "padded"])
def test_the_chunked_path_scores_the_wrapped_composite_as_its_forward(name, batch, mesh):
    """Through the root's forward, whose hook unshards the embedding and the head the path reads below it."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name]).to(torch.bfloat16)
    model.train()
    apply_fsdp2_per_layer(model, mesh, _POLICY, False, IdentityParamSet())
    ids, mask = _rows(batch)
    # First, on the sharded groups: a forward before it leaves them unsharded until the next backward.
    logps, _ = _Scorer()._chunked_completion_logps(model, ids, mask, _COMPLETION, batch)
    logps.sum().backward()
    assert _grads_reached(model.get_input_embeddings()), "the embedding the path reads took no gradient"
    assert _grads_reached(model.get_output_embeddings()), "the head the path reads took no gradient"
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    forward = selective_log_softmax(logits[:, :-1][:, -_COMPLETION:].float(), ids[:, -_COMPLETION:])
    torch.testing.assert_close(logps.detach(), forward, atol=_LOGP_ATOL, rtol=0.0)


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_the_chunked_path_needs_the_roots_forward(name, mesh):
    """Without it the embedding stays sharded, which is what the redirection is for."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name]).to(torch.bfloat16)
    apply_fsdp2_per_layer(model, mesh, _POLICY, False, IdentityParamSet())
    ids, mask = _rows(1)
    with pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        _Scorer()._chunked_logps_impl(model, ids, mask, _COMPLETION, 1, False)


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_the_chunked_path_reaches_a_composite_through_its_adapter(name, mesh):
    """The tuner holds the group, and its forward is the one redirected."""
    torch.manual_seed(0)
    model = get_peft_model(tiny_family_model(_ROOT_EMBEDDING[name]), _LORA)
    model.to(torch.bfloat16).train()
    apply_fsdp2_per_layer(model, mesh, _POLICY, False, IdentityParamSet())
    ids, mask = _rows(1)
    logps, _ = _Scorer()._chunked_completion_logps(model, ids, mask, _COMPLETION, 1)
    logps.sum().backward()
    adapters = [param for key, param in model.named_parameters() if "lora_B" in key]
    assert adapters and any(param.grad is not None and bool((param.grad.to_local() != 0).any()) for param in adapters)


def _chunked_two_rank_worker(rank: int, name: str, out: str) -> None:
    torch.manual_seed(0)
    model = tiny_family_model(_ROOT_EMBEDDING[name]).to(torch.bfloat16).train()
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (2,)), _POLICY, False, IdentityParamSet())
    ids, mask = _rows(1)
    logps, _ = _Scorer()._chunked_completion_logps(model, ids, mask, _COMPLETION, 1)
    logps.sum().backward()
    embedding, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    with open(f"{out}.{rank}", "w") as handle:
        json.dump(
            {
                "logps": logps.tolist(),
                "sharded": [list(p.to_local().shape) != list(p.shape) for p in (embedding, head)],
                "grads": [float(p.grad.to_local().abs().sum()) for p in (embedding, head)],
            },
            handle,
        )


@pytest.mark.parametrize("name", sorted(_ROOT_EMBEDDING))
def test_two_ranks_score_the_chunked_path_alike(name, tmp_path):
    """Both groups sharded across ranks, each rank unsharding through the same root hook."""
    out = str(tmp_path / "rank")
    run_gloo_ranks(_chunked_two_rank_worker, 2, name, out, pg_timeout=datetime.timedelta(seconds=120))
    ranks = [json.loads(Path(f"{out}.{rank}").read_text()) for rank in range(2)]
    assert all(all(rank["sharded"]) for rank in ranks), ranks
    assert ranks[0]["logps"] == ranks[1]["logps"], ranks
    assert all(all(grad > 0 for grad in rank["grads"]) for rank in ranks), ranks


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


def _nemotron_model(name: str):
    """With the bias refilled after the cast, as a release stores it in fp32."""
    torch.manual_seed(0)
    model = tiny_family_model(_NEMOTRON_ROUTES[name]).to(torch.bfloat16)
    fill_release_scale_bias(model)
    return model


def _saving_worker(rank: int, name: str, out: str) -> None:
    """One step's reduce through the wrap a run holds at ``ep_group_size`` 1: experts sharded as DTensors, the embedding
    at the composite's root, the router and latent projections left out."""
    apply_nemotron_h_patches()
    parallelism_config = ParallelismConfig(ep_size=1)
    model = apply_ep_layers(_nemotron_model(name), parallelism_config.ep_config()).train()
    assert parallelism_config.experts_fsdp_managed, "premise: the experts are what FSDP2 shards here"
    left_out = []
    for _, layer in named_ep_layers(model):
        experts = {id(param) for _, param in layer.expert_named_params()}
        left_out += [param for param in layer.parameters() if id(param) not in experts]
    apply_fsdp2_per_layer(model, init_device_mesh("cpu", (2,)), _POLICY, False, IdentityParamSet(left_out))
    _forward_backward(model)
    reshard_fsdp2_modules(model)
    assert all(
        isinstance(param, DTensor) for _, layer in named_ep_layers(model) for _, param in layer.expert_named_params()
    )
    save_ep_gathered_checkpoint(save_context(model, parallelism_config, has_ep_layers=True, toolkit_fsdp2=True), out)
    dist.barrier()


@pytest.mark.parametrize("name", sorted(_NEMOTRON_ROUTES))
def test_two_ranks_write_the_release_names_from_the_sharded_wrap(name, tmp_path):
    """No optimizer step ran, so the gather owes back what a plain save of the same model stores, name for name and
    dtype for dtype, the fp32 bias included."""
    out, plain = tmp_path / "gathered", tmp_path / "plain"
    run_gloo_ranks(_saving_worker, 2, name, str(out), pg_timeout=datetime.timedelta(seconds=120))
    _nemotron_model(name).save_pretrained(plain)
    written, reference = load_full_state_dict(str(out)), load_full_state_dict(str(plain))
    assert written.keys() == reference.keys()
    differing = [
        key
        for key, tensor in reference.items()
        if written[key].dtype != tensor.dtype or not torch.equal(written[key], tensor)
    ]
    assert not differing, differing[:5]
    assert any(key.endswith("experts.0.up_proj.weight") for key in written), "the expert banks went unwritten"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
