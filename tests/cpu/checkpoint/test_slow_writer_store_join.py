#!/usr/bin/env python
"""A checkpoint writer slower than the collective watchdog must not abort its peers.

Every rank writes its optimizer shard and its sharded-EP shard at once, a gathered save's writer flushes
its last part and index alone, and each pipeline stage's writer closes its own parts; on slow shared
storage the slowest of those writes can outlast ``DIST_NCCL_TIMEOUT_MINUTES``. The verdict that follows
each write is joined over the c10d store (``DIST_STORE_TIMEOUT_HOURS``), so the peers wait the writer out
instead of aborting in a collective. Two ranks (four for the pipeline), real gloo, the collective timeout
cut to 3 s right before the save against a 6 s write.

    python tests/cpu/checkpoint/test_slow_writer_store_join.py
"""

from __future__ import annotations

import datetime
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d as c10d
from transformers import Qwen3Config, Qwen3ForCausalLM

import halo.checkpoint.parallel.ep as ep_saver
import halo.checkpoint.parallel.writing as gathered_writer
from halo.checkpoint.parallel.ep import save_ep_sharded_checkpoint
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.parallel.writing import stream_gathered_checkpoint
from halo.checkpoint.writing import StreamingShardWriter
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from tests.common.checkpoint_io import (
    save_context,
    shard_round_trip_model,
    shard_round_trip_optimizer,
    shard_store_context,
    step_on_seeded_data,
)
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
PP_WORLD_SIZE = 4
SLOW_RANK = 1
# The collective bound the save runs under, and a write twice as long: a peer waiting in a collective fails.
SAVE_PG_TIMEOUT = datetime.timedelta(seconds=3)
SLOW_WRITE_S = 6.0
SEED = 20261025


def _arm_save_timeout() -> None:
    """Line the ranks up under the launch's generous bound, then cut it for the save: the setup's skew must not
    read as a collective timing out."""
    dist.barrier()
    c10d._set_pg_timeout(SAVE_PG_TIMEOUT)


def _slowed(write):
    def slow(*args, **kwargs):
        time.sleep(SLOW_WRITE_S)
        return write(*args, **kwargs)

    return slow


class _StubEPLayer(StubEPLayerBase):
    """One rank's experts of a two-way EP group spanning the world, the layout a sharded EP save takes."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 8, 12))
        self.down_proj = nn.Parameter(torch.randn(2, 6, 8))
        self.ep_config = SimpleNamespace(
            ep_size=WORLD_SIZE, ep_group_size=WORLD_SIZE, expert_tp_size=1, num_ep_groups=1
        )

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]


def _ep_model() -> nn.Module:
    layer = nn.Module()
    layer.mlp = _StubEPLayer()
    backbone = nn.Module()
    backbone.layers = nn.ModuleList([layer])
    model = nn.Module()
    model.model = backbone
    model.config = SimpleNamespace(
        model_type="gpt_oss", auto_map=None, tie_word_embeddings=False, save_pretrained=lambda output_dir: None
    )
    return model


def _save_optimizer_shards(rank: int, checkpoint: str) -> None:
    model = shard_round_trip_model(SEED)
    optimizer = shard_round_trip_optimizer(model)
    step_on_seeded_data(model, optimizer, SEED + rank, 1)
    config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, rank=rank)
    store = OptimizerShardStore(shard_store_context(model, optimizer, config))
    if rank == SLOW_RANK:
        torch.save = _slowed(torch.save)
    _arm_save_timeout()
    store.save(checkpoint)


def _save_ep_shards(rank: int, checkpoint: str) -> None:
    model = _ep_model()
    context = save_context(model, has_ep_layers=True, save_sharded_ep=True)
    if rank == SLOW_RANK:
        ep_saver.save_safetensors_file = _slowed(ep_saver.save_safetensors_file)
    _arm_save_timeout()
    save_ep_sharded_checkpoint(context, checkpoint)


def _finalize_gathered_save(rank: int, checkpoint: str) -> None:
    writer_rank = rank == 0
    if writer_rank:
        gathered_writer.StreamingShardWriter.close_as_hf_checkpoint = _slowed(
            StreamingShardWriter.close_as_hf_checkpoint
        )
    _arm_save_timeout()
    stream_gathered_checkpoint(
        nn.Linear(2, 2),
        [{"weight": torch.zeros(2, 2)}],
        checkpoint,
        is_save_rank=writer_rank,
        max_shard_size="5GB",
        normalize=dict,
    )


SAVERS = {
    "optimizer shards": _save_optimizer_shards,
    "sharded EP": _save_ep_shards,
    "gathered finalize": _finalize_gathered_save,
}


def _saver_worker(rank: int, root: str, saver: str) -> None:
    checkpoint = os.path.join(root, "checkpoint")
    os.makedirs(checkpoint, exist_ok=True)
    try:
        SAVERS[saver](rank, checkpoint)
        outcome = "saved"
    except Exception as exc:  # the verdict file must be written whatever the rank raised
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"outcome_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)


def _pp_stage_worker(rank: int, root: str) -> None:
    """pp2 x dp2: stage 1's writer closes its last part slowly while every other rank waits on it."""
    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            pad_token_id=0,
            eos_token_id=1,
        )
    )
    config = make_parallelism_config(world_size=PP_WORLD_SIZE, gpus_per_node=2, rank=rank, pp_size=2)
    stage = build_pipeline_stage(model, config.pp_rank, config.pp_size)
    if config.pp_rank == 1:
        StreamingShardWriter.close = _slowed(StreamingShardWriter.close)
    try:
        _arm_save_timeout()
        save_pp_checkpoint(save_context(stage, config), root)
        outcome = "saved"
    except Exception as exc:  # the verdict file must be written whatever the rank raised
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"outcome_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)


def _outcomes(root, world_size: int) -> list[str]:
    return [json.loads((root / f"outcome_{rank}.json").read_text()) for rank in range(world_size)]


@pytest.mark.parametrize("saver", list(SAVERS))
def test_the_peers_wait_out_a_writer_slower_than_the_watchdog(tmp_path, saver):
    run_gloo_ranks(_saver_worker, WORLD_SIZE, str(tmp_path), saver)
    assert _outcomes(tmp_path, WORLD_SIZE) == ["saved"] * WORLD_SIZE


def test_the_pipeline_stages_wait_out_a_slow_stage_writer(tmp_path):
    run_gloo_ranks(_pp_stage_worker, PP_WORLD_SIZE, str(tmp_path))
    assert _outcomes(tmp_path, PP_WORLD_SIZE) == ["saved"] * PP_WORLD_SIZE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
