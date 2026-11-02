#!/usr/bin/env python
"""Every copy the FSDP2 data-parallel wrap leaves on more than one rank starts from the first copy, on four gloo
ranks, for every family of the tiny rosters.

FSDP2, unlike DDP, broadcasts no module state, and those copies only average their gradients, so a tensor the ranks
drew from their own RNG streams would keep each rank's draw for the whole run while every loss stays finite. The base
model is drawn from one shared seed, as a checkpoint load gives every rank the same weights; what follows is drawn
from each rank's own stream (``ADAPTER_SEED + rank``), as it is whenever the streams have parted before the draw.

HSDP (``dp_replicate=2`` x ``dp_shard=2``) through ``apply_fsdp2_for_dp``, dense and MoE: PEFT LoRA on the
attention projections and one row of the frozen input embedding (the row a vocabulary resize adds) drawn per rank.
After the wrap every shard is the first replica's, and one optimizer step later every shard and the logits of a
shared batch are still identical across each ``dp_replicate`` group.

Every MoE family on the grouped-GEMM EP layers at ``ep_size=1`` with ``fsdp_shard_ep1_experts=False``, wrapped by the
trainer's own mode ladder: the experts, router and native expert LoRA adapters are FSDP-ignored whole copies, the
adapters drawn per rank. Under HSDP and under 1-D full-shard each copy is DP rank 0's after the wrap. At
``ep_size=2`` the experts are distributed, and the wrap leaves every rank's expert weights as that rank holds them.

Run: python tests/cpu/parallelism/test_fsdp2_replicas_start_identical.py
"""

import datetime
import functools
import json
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import halo.distributed.fsdp as fsdp
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.mesh import MeshDim, build_dp_mesh
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.runtime.dtensor import to_local
from tests.common.distributed import group_max_abs_diff, max_abs_diff_across
from tests.common.gloo import run_gloo_ranks
from tests.common.numerics import max_or_nan
from tests.common.parallelism import make_parallelism_config
from tests.common.replica_start import ADAPTER_SEED, BASE_SEED, ModeLadder, per_rank_draws, per_rank_lora_model
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model

WORLD = 4
REPLICAS = 2
FAMILIES = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
BATCH_SEED = 200
TOKENS = 12
EXPERT_LORA = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)
# The EP-layer shapes the trainer's mode ladder wraps, by name: HSDP over two simulated NVLink domains of two
# ranks, 1-D full-shard over one domain of four, and two-way expert distribution in that domain.
EP_SHAPES = {
    "ep1_hsdp": {"gpus_per_node": 2, "use_hsdp": True, "fsdp_shard_ep1_experts": False},
    "ep1": {"gpus_per_node": 4, "fsdp_shard_ep1_experts": False},
    "ep2": {"gpus_per_node": 4, "ep_size": 2},
}
REPLICATED_EP_SHAPES = ("ep1_hsdp", "ep1")
PG_TIMEOUT = datetime.timedelta(seconds=120)


def _max_abs(tensors) -> float:
    """The largest ``|x|`` over ``tensors``, empty ones skipped; ``0.0`` for none."""
    return max_or_nan((tensor.abs().max().item() for tensor in tensors if tensor.numel()), default=0.0)


def _replica_spread(model) -> tuple[int, float]:
    """The parameters on the 2-D HSDP mesh and the largest ``|Δ|`` of any non-empty local shard across its
    ``dp_replicate`` group. COLLECTIVE."""
    shards = [
        param.data
        for param in model.parameters()
        if isinstance(param.data, DTensor) and param.data.device_mesh.ndim == 2
    ]
    spread = max_or_nan(
        (
            group_max_abs_diff(to_local(shard), shard.device_mesh[MeshDim.DP_REPLICATE].get_group())
            for shard in shards
            if to_local(shard).numel()
        ),
        default=float("nan"),
    )
    return len(shards), spread


def _grad_spread(model) -> float:
    """The largest ``|Δ|`` of any trainable non-empty shard's gradient across its ``dp_replicate`` group. COLLECTIVE."""
    grads = [param.grad for param in model.parameters() if param.requires_grad and isinstance(param.grad, DTensor)]
    return max_or_nan(
        (
            group_max_abs_diff(to_local(grad), grad.device_mesh[MeshDim.DP_REPLICATE].get_group())
            for grad in grads
            if to_local(grad).numel()
        ),
        default=float("nan"),
    )


def _first_replica_tensor(drawn: list[torch.Tensor], mesh) -> torch.Tensor:
    """The tensor the first replica's ranks hold together: each keeps the dim-0 chunk of its own draw at its shard
    index (FSDP2's ``torch.chunk`` sharding, empty past the last chunk). ``drawn`` is indexed by global rank."""
    first = mesh.mesh[0].tolist()
    shards = len(first)
    chunks = [torch.chunk(drawn[rank], shards, dim=0) for rank in first]
    return torch.cat([chunk[index] for index, chunk in enumerate(chunks) if index < len(chunk)])


def _gather_world(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Every rank's copy of ``tensor``, indexed by global rank. COLLECTIVE."""
    local = tensor.detach().contiguous()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return gathered


def _first_replica_gap(model, drawn: dict[str, list[torch.Tensor]]) -> float:
    """The largest ``|Δ|`` between any parameter on the HSDP mesh and the first replica's copy of it, read from what
    every rank held before the wrap. COLLECTIVE (one gather per parameter over its mesh)."""
    gaps = []
    for name, param in model.named_parameters():
        if isinstance(param.data, DTensor):
            gaps.append(param.data.full_tensor() - _first_replica_tensor(drawn[name], param.data.device_mesh))
    return _max_abs(gaps)


def _forward_spread(model, vocab: int) -> float:
    """The largest ``|Δ|`` of the logits one shared batch, the drawn embedding row's token included, gives across
    the ``dp_replicate`` group: what the replicas compute from the parameters FSDP2 gathers, not only the shards they
    store. Read after a step, since a zero-init ``lora_B`` hides ``lora_A`` from the forward before one. COLLECTIVE."""
    mesh = next(param.data.device_mesh for param in model.parameters() if isinstance(param.data, DTensor))
    ids = torch.randint(0, vocab, (1, TOKENS), generator=torch.Generator().manual_seed(BATCH_SEED))
    ids[0, -1] = vocab - 1
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    # A forward under SHARD_GRAD_OP leaves the gathered parameters registered.
    fsdp.reshard_fsdp2_modules(model)
    return group_max_abs_diff(logits, mesh[MeshDim.DP_REPLICATE].get_group())


def _hsdp_lora_step(rank: int, family: str) -> dict:
    """Build, adapt, wrap over the HSDP mesh, take one step on this rank's own batch, and record."""
    model = per_rank_lora_model(FAMILIES[family], rank)
    drawn = max_abs_diff_across(per_rank_draws(model))
    before_wrap = {name: _gather_world(param.data) for name, param in model.named_parameters()}

    fsdp.apply_fsdp2_for_dp(model, WORLD, compute_dtype=torch.float32, fp32_reduce=True, dp_replicate_size=REPLICAS)
    sharded, at_wrap = _replica_spread(model)
    first_replica = _first_replica_gap(model, before_wrap)

    trainable = [param for param in model.parameters() if param.requires_grad]
    before_step = [to_local(param.data).clone() for param in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    vocab = model.get_input_embeddings().weight.shape[0]
    ids = torch.randint(0, vocab, (2, TOKENS), generator=torch.Generator().manual_seed(BATCH_SEED + rank))
    model(input_ids=ids, labels=ids).loss.backward()
    grads = _grad_spread(model)
    optimizer.step()
    _, after_step = _replica_spread(model)
    forward = _forward_spread(model, vocab)
    moved = sum(not torch.equal(old, to_local(param.data)) for old, param in zip(before_step, trainable, strict=True))
    return {
        "drawn": drawn,
        "sharded": sharded,
        "at_wrap": at_wrap,
        "first_replica": first_replica,
        "forward": forward,
        "grads": grads,
        "after_step": after_step,
        "moved": moved,
    }


def _ep_shape_wrap(rank: int, family: str, fields: dict) -> dict:
    """Build ``family`` on the EP layers of ``fields`` with per-rank native expert adapters, wrap it through the
    trainer's mode ladder, and record what the wrap left in every EP-layer parameter."""
    torch.manual_seed(BASE_SEED)
    model = tiny_family_model(FAMILIES[family])
    config = make_parallelism_config(
        world_size=WORLD, rank=rank, use_grouped_gemm=True, expert_lora=EXPERT_LORA, **fields
    )
    ep_config = config.ep_config()
    torch.manual_seed(ADAPTER_SEED + rank)
    apply_ep_layers(model, ep_config)
    layers = [layer for _, layer in named_ep_layers(model)]
    # ``parameters()`` order, the same on every rank: the adapter accessors iterate a frozenset of names.
    params = [param for layer in layers for param in layer.parameters()]
    adapter_ids = {id(param) for layer in layers for _, param in layer.expert_lora_named_params()}
    expert_ids = {id(param) for layer in layers for _, param in layer.expert_named_params()}
    adapters = [param for param in params if id(param) in adapter_ids]
    base_experts = [param for param in params if id(param) in expert_ids - adapter_ids]
    drawn = max_abs_diff_across([param.data for param in adapters])
    distributed = max_abs_diff_across([param.data for param in base_experts])
    own_experts = [param.data.clone() for param in base_experts]
    rank0 = [param.data.clone() for param in params]
    for tensor in rank0:
        dist.broadcast(tensor, src=0)

    ModeLadder(model, config, ep_config)._wrap_for_parallel_mode()
    after = [param.data for param in params]
    return {
        "adapters": len(adapters),
        "drawn": drawn,
        "distributed": distributed,
        "plain": all(not isinstance(tensor, DTensor) for tensor in after),
        "spread": max_abs_diff_across(after),
        "rank0_gap": _max_abs([new - old for new, old in zip(after, rank0, strict=True)]),
        "own_expert_gap": _max_abs([param.data - old for param, old in zip(base_experts, own_experts, strict=True)]),
    }


def _hsdp_worker(rank: int, family: str, result_dir: str) -> None:
    with patch.object(fsdp, "build_dp_mesh", functools.partial(build_dp_mesh, device_type="cpu")):
        record = _hsdp_lora_step(rank, family)
    with open(f"{result_dir}/{rank}.json", "w") as f:
        json.dump(record, f)


def _ep_worker(rank: int, family: str, result_dir: str) -> None:
    with (
        patch.object(fsdp, "build_dp_mesh", functools.partial(build_dp_mesh, device_type="cpu")),
        patch("halo.distributed.parallelism_config.validate_nvlink_domain_against_fabric"),
    ):
        record = {shape: _ep_shape_wrap(rank, family, fields) for shape, fields in EP_SHAPES.items()}
    with open(f"{result_dir}/{rank}.json", "w") as f:
        json.dump(record, f)


def _run(worker, family: str, tmp_path_factory) -> list[dict]:
    result_dir = tmp_path_factory.mktemp(f"{worker.__name__}_{family}")
    run_gloo_ranks(worker, WORLD, family, str(result_dir), pg_timeout=PG_TIMEOUT)
    return [json.loads((result_dir / f"{rank}.json").read_text()) for rank in range(WORLD)]


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def hsdp(request, tmp_path_factory) -> list[dict]:
    return _run(_hsdp_worker, request.param, tmp_path_factory)


@pytest.fixture(scope="module", params=sorted(TINY_MOE_FAMILIES))
def ep(request, tmp_path_factory) -> list[dict]:
    return _run(_ep_worker, request.param, tmp_path_factory)


def test_premise_ranks_drew_their_own_adapters(hsdp):
    assert all(record["drawn"] > 0 for record in hsdp), hsdp


def test_premise_the_wrap_runs_hsdp(hsdp):
    assert all(record["sharded"] > 0 for record in hsdp), hsdp


def test_hsdp_replicas_start_from_the_first_replicas_shards(hsdp):
    """Adapters and the frozen embedding row alike: every shard on the HSDP mesh, not only the trainable ones, and
    the first replica's copy, not another replica's."""
    assert all(record["at_wrap"] == 0.0 for record in hsdp), hsdp
    assert all(record["first_replica"] == 0.0 for record in hsdp), hsdp


def test_hsdp_replicas_compute_the_same_forward_after_a_step(hsdp):
    assert all(record["forward"] == 0.0 for record in hsdp), hsdp


def test_hsdp_replicas_stay_identical_through_a_step(hsdp):
    assert all(record["grads"] == 0.0 for record in hsdp), hsdp
    assert all(record["moved"] > 0 for record in hsdp), hsdp
    assert all(record["after_step"] == 0.0 for record in hsdp), hsdp


@pytest.mark.parametrize("shape", REPLICATED_EP_SHAPES)
def test_premise_ep1_expert_adapters_drawn_per_rank_outside_fsdp2(ep, shape):
    ranks = [record[shape] for record in ep]
    assert all(record["adapters"] > 0 and record["drawn"] > 0 for record in ranks), ranks
    assert all(record["plain"] for record in ranks), ranks


@pytest.mark.parametrize("shape", REPLICATED_EP_SHAPES)
def test_ep1_whole_copies_start_from_dp_rank_0(ep, shape):
    """The experts, router and native expert adapters the wrap leaves out: identical on every rank, rank 0's."""
    ranks = [record[shape] for record in ep]
    assert all(record["spread"] == 0.0 and record["rank0_gap"] == 0.0 for record in ranks), ranks


def test_distributed_experts_keep_each_ranks_own_weights(ep):
    ranks = [record["ep2"] for record in ep]
    assert all(record["distributed"] > 0 and record["plain"] for record in ranks), ranks
    assert all(record["own_expert_gap"] == 0.0 for record in ranks), ranks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
