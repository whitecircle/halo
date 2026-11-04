#!/usr/bin/env python
"""MLA ``kv_a_proj_with_mqa`` keeps its full gradient under selective attention TP. Its rope rows reach only local
heads, so the true gradient is the SUM; AVG leaves them 1/tp_size too small."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

from halo.distributed.tensor_parallel.parallelize_attention import apply_tp_to_attention_only
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
KV_LORA_RANK = 16
QK_ROPE_HEAD_DIM = 8
# Loose enough for fp32 accumulation-order differences, tight enough that the 1/tp_size defect (~0.5 at tp_size=2)
# cannot pass.
TOL = 1e-3


def _tiny_mla_model():
    """Seeded, so every rank builds it alike."""
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
    """Local shard if it ever became a DTensor."""
    grads = []
    for layer in model.model.layers:
        grad = layer.self_attn.kv_a_proj_with_mqa.weight.grad
        grads.append(grad.to_local() if isinstance(grad, DTensor) else grad)
    return grads


def _worker(rank: int, out_path: str) -> None:
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
        # What _sync_tp_replicated_grads does to a plain replicated parameter.
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

    # Each rank holds a different column shard, so a defect can show on one rank alone.
    with open(f"{out_path}.{rank}", "w") as fh:
        fh.write("PASS" if not failures else "FAIL: " + "; ".join(failures))


def test_mla_kv_a_proj_rope_gradient_survives_the_replicated_avg(tmp_path):
    out = str(tmp_path / "result.txt")
    run_gloo_ranks(_worker, WORLD_SIZE, out)
    results = {}
    for rank in range(WORLD_SIZE):
        with open(f"{out}.{rank}") as fh:
            results[rank] = fh.read()
    assert set(results.values()) == {"PASS"}, results


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
