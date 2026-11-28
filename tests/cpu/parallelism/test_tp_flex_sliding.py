#!/usr/bin/env python
"""Under TP (and FSDP2 + TP), ``attention_call`` matches the head count each attention call receives and ``_flex_serves``
admits every masked layer: local heads on ``colwise``, full heads on ``colwise_gather_output``."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import DistributedConfig
from transformers.distributed.parallel_utils import _get_parameter_plan
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from halo.distributed.fsdp import apply_fsdp2_for_tp
from halo.distributed.mesh import build_dp_tp_mesh, tp_submesh
from halo.distributed.tensor_parallel import hf_plan, parallelize_attention
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.parallel import tp as tp_loading
from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.layout import TP_OUTPUT_SHARD_SIZE_ATTR, AttentionCall, attention_call
from halo.models.attention.sdpa_flex_sliding import flex_attention_calls
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_SLIDING_DENSE_FAMILIES,
    build_tiny_family_checkpoint,
    tiny_family_model,
)

TP = 2
DP = 2
PACKED_LENGTHS = (150, 30, 90)
HF_CASES = ("mistral", "cohere2", "olmo3")
CASES = (*HF_CASES, "cohere2_moe")
EXPECTED_HEADS = {
    "mistral": (2, 1),
    "cohere2": (2, 1),
    "olmo3": (4, 2),
    "cohere2_moe": (2, 1),
}


def _apply_break(model: nn.Module, broken: str | None) -> None:
    if broken is None:
        return
    if broken == "no_stamp":
        for module in model.modules():
            if hasattr(module, TP_OUTPUT_SHARD_SIZE_ATTR):
                delattr(module, TP_OUTPUT_SHARD_SIZE_ATTR)
        return
    if broken == "stamp_gather_output":
        plan = model._tp_plan
        stamped = 0
        for name, module in model.named_modules():
            if _get_parameter_plan(name, plan, is_weight=False) == "colwise_gather_output" and isinstance(
                module, nn.Linear
            ):
                setattr(module, TP_OUTPUT_SHARD_SIZE_ATTR, TP)
                stamped += 1
        assert stamped > 0, "premise: the plan carries colwise_gather_output projections"
        return
    if broken == "stamp_one_on_colwise":
        overwritten = 0
        for module in model.modules():
            if hasattr(module, TP_OUTPUT_SHARD_SIZE_ATTR):
                setattr(module, TP_OUTPUT_SHARD_SIZE_ATTR, 1)
                overwritten += 1
        assert overwritten > 0, "premise: at least one colwise projection was stamped"
        return
    raise ValueError(f"unknown negative control {broken!r}")


def _load_case(case: str, checkpoint: str | None, mesh) -> nn.Module:
    if case in TINY_SLIDING_DENSE_FAMILIES:
        family = TINY_SLIDING_DENSE_FAMILIES[case]
        with tp_loading._toolkit_tp_mesh(mesh, tp_submesh(mesh)):
            model = family.load_class.from_pretrained(
                checkpoint, distributed_config=DistributedConfig(tp_plan="auto", tp_size=TP), dtype=torch.float32
            )
        finalize_loaded_model(model, weights_dir=checkpoint)
        hf_plan.finish_hf_tp_load(model, mesh, TP)
        return model
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[case])
    parallelize_attention.apply_tp_to_attention_only(model, mesh)
    return model


def _worker(rank: int, case: str, checkpoint: str | None, dp: int, broken: str | None, out_path: str) -> None:
    mesh = build_dp_tp_mesh(TP, dp, device_type="cpu")
    model = _load_case(case, checkpoint, mesh)
    _apply_break(model, broken)
    if dp > 1:
        apply_fsdp2_for_tp(model, mesh, compute_dtype=torch.float32, fp32_reduce=False, dp_size=dp)

    resharded_calls = [list(call) for call in sorted(flex_attention_calls(model), key=str)]
    dp_rank = mesh["dp"].get_local_rank() if dp > 1 else 0
    text_config = model.config.get_text_config()
    ids = torch.randint(
        3, text_config.vocab_size, (1, sum(PACKED_LENGTHS)), generator=torch.Generator().manual_seed(dp_rank)
    )
    position_ids = torch.cat([torch.arange(n) for n in PACKED_LENGTHS]).unsqueeze(0)

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    calls = []

    def spy(module, query, key, value, attention_mask, **kwargs):
        received = AttentionCall(
            heads=query.shape[1],
            kv_heads=key.shape[1],
            head_dim=query.shape[-1],
            v_head_dim=value.shape[-1],
            window=kwargs.get("sliding_window") or None,
            scaling=kwargs.get("scaling"),
            softcap=kwargs.get("softcap") or None,
        )
        declared = attention_call(module)
        admitted = sdpa_flex_sliding._flex_serves(
            module,
            query,
            key,
            value,
            attention_mask,
            kwargs.get("sliding_window"),
            kwargs.get("scaling"),
        )
        calls.append({"declared": list(declared), "received": list(received), "admitted": admitted})
        return sdpa(module, query, key, value, attention_mask, **kwargs)

    with torch.no_grad(), patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {"sdpa": spy}):
        model(input_ids=ids, position_ids=position_ids, use_cache=False)

    with open(f"{out_path}.{rank}", "w") as fh:
        json.dump(
            {
                "layers": text_config.num_hidden_layers,
                "resharded_calls": resharded_calls,
                "calls": calls,
            },
            fh,
        )


def _results(out_path: str, world: int) -> list[dict]:
    results = []
    for rank in range(world):
        with open(f"{out_path}.{rank}") as fh:
            results.append(json.load(fh))
    return results


@pytest.fixture(scope="module")
def hf_checkpoints(tmp_path_factory):
    paths = {}

    def get_checkpoint(case: str) -> str | None:
        if case not in TINY_SLIDING_DENSE_FAMILIES:
            return None
        if case not in paths:
            path = tmp_path_factory.mktemp(case)
            build_tiny_family_checkpoint(TINY_SLIDING_DENSE_FAMILIES[case], str(path))
            paths[case] = str(path)
        return paths[case]

    return get_checkpoint


def _assert_admitted(results: list[dict], case: str) -> None:
    want_heads = list(EXPECTED_HEADS[case])
    for rank, result in enumerate(results):
        assert len(result["calls"]) == result["layers"], f"rank {rank}: {result}"
        assert {tuple(c[:2]) for c in result["resharded_calls"]} == {tuple(want_heads)}, f"rank {rank}: {result}"
        for record in result["calls"]:
            assert record["received"][:2] == want_heads, f"rank {rank}: {record}"
            assert record["declared"] == record["received"], f"rank {rank}: {record}"
            assert record["admitted"] is True, f"rank {rank}: {record}"


@pytest.mark.parametrize("case", CASES)
def test_tp_attention_calls_match_runtime_and_flex_admits_them(case, hf_checkpoints, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_worker, TP, case, hf_checkpoints(case), 1, None, out)
    _assert_admitted(_results(out, TP), case)


@pytest.mark.parametrize("case", CASES)
def test_fsdp2_tp_attention_calls_match_runtime_and_flex_admits_them(case, hf_checkpoints, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_worker, DP * TP, case, hf_checkpoints(case), DP, None, out)
    _assert_admitted(_results(out, DP * TP), case)


@pytest.mark.parametrize(
    ("case", "broken"),
    [
        ("mistral", "no_stamp"),
        ("olmo3", "stamp_gather_output"),
        ("mistral", "stamp_one_on_colwise"),
    ],
)
def test_negative_control_wrong_tp_stamp_is_rejected(case, broken, hf_checkpoints, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_worker, TP, case, hf_checkpoints(case), 1, broken, out)
    for rank, result in enumerate(_results(out, TP)):
        assert len(result["calls"]) == result["layers"], f"rank {rank}: {result}"
        for record in result["calls"]:
            assert record["declared"] != record["received"], f"rank {rank} ({broken}): {record}"
            assert record["admitted"] is False, f"rank {rank} ({broken}): {record}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
