#!/usr/bin/env python
"""MLA ``kv_a_proj_with_mqa`` keeps its FULL gradient under selective attention TP.

That projection emits ``[kv_lora_rank | qk_rope_head_dim]`` and the two halves need DIFFERENT
reductions over the TP group. The ``kv_lora_rank`` rows reach ``kv_b_proj`` (colwise), whose
``Replicate`` input redistribute all-reduces them in backward — already complete, so AVG is
idempotent. The ``qk_rope_head_dim`` rows bypass ``kv_b_proj``: ``k_rot`` is expanded to this rank's
LOCAL head count and concatenated into ``key_states``, so it never crosses a DTensor boundary and
each rank's gradient covers only its own heads — the true gradient is the SUM.

The weight stays a plain replica, so the trainer's replicated bucket (``_sync_tp_replicated_grads``,
``ReduceOp.AVG``) divides those rope rows by ``tp_size``. Nothing raises: the lora rows stay right,
the loss curve stays plausible, and the rope projection trains on 1/tp_size of its gradient.

This drives the REAL ``apply_tp_to_attention_only`` on a real Glm4MoeLite across a real 2-rank gloo
group and compares the AVG-reduced gradient against a single-process, non-TP reference. Without
``register_mla_rope_grad_reduction`` the rope rows come out ~tp_size x too small and this fails.

    python tests/cpu/parallelism/test_tp_mla_rope_grad.py
"""

import copy
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

from src.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from tests.common.ports import free_port

WORLD_SIZE = 2
KV_LORA_RANK = 16
QK_ROPE_HEAD_DIM = 8
# Loose enough for fp32 accumulation-order differences between the TP and reference graphs, tight
# enough that the 1/tp_size defect (relative error ~= 0.5 at tp_size=2) cannot pass.
TOL = 1e-3


def _tiny_mla_model():
    """Smallest Glm4MoeLite that still exercises MLA + MoE. Seeded, so every rank builds it alike."""
    config = AutoConfig.for_model(
        "glm4_moe_lite",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        q_lora_rank=16,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=8,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config, dtype=torch.float32)


def _backward(model, input_ids):
    model.zero_grad(set_to_none=True)
    model(input_ids=input_ids, labels=input_ids).loss.backward()


def _kv_a_grads(model) -> list[torch.Tensor]:
    """Per-layer ``kv_a_proj_with_mqa.weight`` grads (local shard if it ever became a DTensor)."""
    grads = []
    for layer in model.model.layers:
        grad = layer.self_attn.kv_a_proj_with_mqa.weight.grad
        grads.append(grad.to_local() if isinstance(grad, DTensor) else grad)
    return grads


def _worker(rank: int, out_path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        torch.manual_seed(0)
        input_ids = torch.randint(0, 64, (2, 8))

        model = _tiny_mla_model()
        unsplit = copy.deepcopy(model)
        _backward(unsplit, input_ids)
        reference = [g.clone() for g in _kv_a_grads(unsplit)]

        tp_mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        apply_tp_to_attention_only(model, tp_mesh)
        _backward(model, input_ids)

        failures = []
        for layer_idx, (grad, ref_grad) in enumerate(zip(_kv_a_grads(model), reference, strict=True)):
            if isinstance(model.model.layers[layer_idx].self_attn.kv_a_proj_with_mqa.weight, DTensor):
                failures.append(f"layer {layer_idx}: weight became a DTensor — the AVG bucket would skip it")
            # Exactly what _sync_tp_replicated_grads does to a plain replicated parameter.
            averaged = grad.clone()
            dist.all_reduce(averaged, op=dist.ReduceOp.AVG)

            for label, rows in (
                ("lora", slice(0, KV_LORA_RANK)),
                ("rope", slice(KV_LORA_RANK, None)),
            ):
                err = ((averaged[rows] - ref_grad[rows]).norm() / ref_grad[rows].norm().clamp_min(1e-30)).item()
                if not (err < TOL):
                    ratio = (ref_grad[rows].norm() / averaged[rows].norm().clamp_min(1e-30)).item()
                    failures.append(
                        f"layer {layer_idx} {label} rows: AVG-reduced grad differs from the non-TP "
                        f"reference by rel_err={err:.3e} (||ref||/||avg||={ratio:.4f}; "
                        f"{WORLD_SIZE:.4f} means the rank-partial rows were averaged, not summed)"
                    )

        # EVERY rank writes its own verdict: each holds a different column shard, so a defect can
        # show on one rank alone, and a rank-0-only artifact would report that run as green.
        with open(f"{out_path}.{rank}", "w") as fh:
            fh.write("PASS" if not failures else "FAIL: " + "; ".join(failures))
    finally:
        dist.destroy_process_group()


def test_mla_kv_a_proj_rope_gradient_survives_the_replicated_avg(tmp_path):
    out = str(tmp_path / "result.txt")
    mp.start_processes(_worker, args=(out, free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    results = {}
    for rank in range(WORLD_SIZE):
        with open(f"{out}.{rank}") as fh:
            results[rank] = fh.read()
    assert set(results.values()) == {"PASS"}, results


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
