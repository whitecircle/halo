#!/usr/bin/env python
"""Under QLoRA every rank's floating-point parameters start from rank 0's, on two gloo ranks, for every family of
the tiny rosters.

QLoRA skips FSDP2 (``fully_shard`` cannot wrap bnb's packed 4-bit storage), so every parameter is a whole copy on
every rank and ``_sync_qlora_grads`` only averages the gradients: an adapter the ranks drew from their own RNG streams
would keep each rank's draw for the whole run while every loss stays finite. The base model is drawn from one shared
seed, as a checkpoint load gives every rank the same weights; the LoRA adapters on the attention projections and one
row of the frozen input embedding (the row a vocabulary resize adds) are drawn from each rank's own stream. A frozen
``uint8`` parameter stands in for the packed 4-bit weights, which turn the trainer's mode ladder onto the QLoRA sync.
After the ladder every floating-point parameter is rank 0's, and the packed storage is each rank's own.

Run: python tests/cpu/parallelism/test_qlora_replicas_start_identical.py
"""

import datetime
import json
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from halo.model_loading.dtype import has_packed_quantized_params
from tests.common.distributed import max_abs_diff_across
from tests.common.gloo import run_gloo_ranks
from tests.common.numerics import max_or_nan
from tests.common.parallelism import make_parallelism_config
from tests.common.replica_start import ModeLadder, per_rank_draws, per_rank_lora_model
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES

WORLD = 2
FAMILIES = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
# The stand-in for bnb's packed 4-bit storage: rank-specific bytes, so a broadcast of it would show.
PACKED_SHAPE = (8, 4)
PG_TIMEOUT = datetime.timedelta(seconds=120)


def _qlora_ladder(rank: int, family: str) -> dict:
    """Build, adapt, add the packed stand-in, run the trainer's mode ladder, and record."""
    model = per_rank_lora_model(FAMILIES[family], rank)
    packed = torch.nn.Parameter(torch.full(PACKED_SHAPE, rank, dtype=torch.uint8), requires_grad=False)
    model.base_model.model.register_parameter("packed_4bit_stand_in", packed)
    floating = [param for param in model.parameters() if param.is_floating_point()]
    drawn = max_abs_diff_across(per_rank_draws(model))
    rank0 = [param.data.clone() for param in floating]
    for tensor in rank0:
        dist.broadcast(tensor, src=0)

    config = make_parallelism_config(world_size=WORLD, gpus_per_node=WORLD, rank=rank, use_grouped_gemm=False)
    wrapper = ModeLadder(model, config)
    wrapper._wrap_for_parallel_mode()
    return {
        "qlora_path": has_packed_quantized_params(model) and getattr(wrapper, "_qlora_grad_sync", False),
        "drawn": drawn,
        "spread": max_abs_diff_across([param.data for param in floating]),
        "rank0_gap": max_or_nan(
            ((param.data - old).abs().max().item() for param, old in zip(floating, rank0, strict=True)), default=0.0
        ),
        "packed_kept": torch.equal(packed.data, torch.full(PACKED_SHAPE, rank, dtype=torch.uint8)),
    }


def _worker(rank: int, family: str, result_dir: str) -> None:
    with patch("halo.distributed.parallelism_config.validate_nvlink_domain_against_fabric"):
        record = _qlora_ladder(rank, family)
    with open(f"{result_dir}/{rank}.json", "w") as f:
        json.dump(record, f)


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def records(request, tmp_path_factory) -> list[dict]:
    result_dir = tmp_path_factory.mktemp(f"qlora_{request.param}")
    run_gloo_ranks(_worker, WORLD, request.param, str(result_dir), pg_timeout=PG_TIMEOUT)
    return [json.loads((result_dir / f"{rank}.json").read_text()) for rank in range(WORLD)]


def test_premise_the_ladder_takes_the_qlora_sync(records):
    assert all(record["qlora_path"] for record in records), records


def test_premise_ranks_drew_their_own_adapters(records):
    assert all(record["drawn"] > 0 for record in records), records


def test_floating_point_parameters_start_from_rank_0(records):
    """Adapters and the frozen embedding row alike."""
    assert all(record["spread"] == 0.0 and record["rank0_gap"] == 0.0 for record in records), records


def test_packed_storage_is_left_to_each_rank(records):
    assert all(record["packed_kept"] for record in records), records


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
