#!/usr/bin/env python
"""The head as its own FSDP2 unit on real gloo ranks, through the trainer's exclusions and the DP wrap: each rank trains
a different row, and every gradient must come out as the mean of the two rows' single-process gradients. A hard head
sharing a group with the trunk under ZeRO-3 segfaults here."""

import datetime
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.distributed import fsdp as fsdp_module
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import (
    apply_fsdp2_for_dp,
    complete_fsdp2_post_backward,
    fsdp2_param_groups,
    reject_unfinished_fsdp2_backward,
    reshard_fsdp2_modules,
)
from halo.distributed.mesh import build_dp_mesh
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.liger import cross_entropy
from halo.models.mtp.head import freeze_trunk_for_head
from halo.trainers.parallelism.ep_layers import EPLayersMixin
from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.mtp_heads import install_term, tiny_model_with_head
from tests.common.parallelism import single_process_ep_config

_WORLD = 2
_WEIGHT = 0.7
_COLLECTIVE_TIMEOUT = datetime.timedelta(seconds=120)
# Observed: 0 on the head and at most 5e-5 on the trunk, the float summation order of a mean over shards.
_HEAD_BAR = 1e-4
_TRUNK_BAR = 1e-3
# Over each bucket's tensors together: a tensor of small norm alone sits past it in a mean of two bf16 gradients.
_BF16_BAR = 0.06
# With no head at all, this wrap's trunk stands 5% from a plain bf16 Qwen3.5 and 6e-6 from a plain GLM.
_BF16_TRUNK_BAR = 0.1
# One row's own gradient against the mean: above this the rows differ enough for the bars to catch a missing reduce.
_ROWS_DIFFER = 1e-2
_LOSSES = ("hard", "soft")
_GRADED_SCOPES = ("joint", "detached")
_ZERO_MODES = {"zero2": False, "zero3": True}


def _matrix(*, loss_types=_LOSSES, zero_modes=tuple(_ZERO_MODES)) -> list[dict]:
    rows = [
        {"loss_type": loss_type, "scope": scope, "zero": zero, "gc": gc}
        for loss_type in loss_types
        for scope in _GRADED_SCOPES
        for zero in zero_modes
        for gc in (False, True)
    ]
    rows += [
        {"loss_type": loss_type, "scope": "head_only", "zero": zero, "gc": False}
        for loss_type in loss_types
        for zero in zero_modes
    ]
    return rows


def _tag(row: dict) -> str:
    return "-".join(str(value) for value in row.values())


class _ModeSetup(ModeSetupMixin, EPLayersMixin):
    """What the trainer's exclusions read."""

    def __init__(self, model, *, experts_fsdp_managed: bool):
        self.model = model
        self.parallelism_config = SimpleNamespace(experts_fsdp_managed=experts_fsdp_managed)
        self._param_id_caches = {}


def _patch_experts(model, head, **ep_fields) -> None:
    """The ep1 shape: every expert local, the trunk's and the head's EP layers both managed by FSDP2."""
    ep_config = single_process_ep_config(**ep_fields)
    apply_ep_layers(model, ep_config)
    shell = nn.Module()
    shell.add_module("head", head)
    apply_ep_layers(shell, ep_config)


def _built(name: str, row: dict, *, ep1: bool, fp32_router: bool):
    """An fp32 router means a bf16 run with fp32 routers, as the shipped recipes train; every other row computes at
    fp32, where two shard layouts differ by the summation order alone."""
    model, head = tiny_model_with_head(name, dtype=torch.bfloat16 if fp32_router else torch.float32)
    if ep1:
        _patch_experts(model, head, fp32_router=fp32_router)
    install_term(model, head, weight=_WEIGHT, loss_type=row["loss_type"], scope=row["scope"])
    if row["scope"] == "head_only":
        freeze_trunk_for_head(model, head)
    return model, head


def _router_of(head: nn.Module) -> nn.Module:
    return next(module for path, module in head.named_modules() if path.endswith(("mlp.gate", "mixer.gate")))


def _checkpoint(model) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.config.use_cache = False


def _row_of_rank(rank: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(4, 90, (_WORLD, 24), generator=generator)
    labels = input_ids.clone()
    labels[:, : 3 + 2 * rank] = LABEL_IGNORE_INDEX
    return {"input_ids": input_ids[rank : rank + 1], "labels": labels[rank : rank + 1], "use_cache": False}


def _single_process_gradients(name: str, row: dict, *, ep1: bool, fp32_router: bool) -> list[dict[str, torch.Tensor]]:
    per_row = []
    for rank in range(_WORLD):
        model, _ = _built(name, row, ep1=ep1, fp32_router=fp32_router)
        if row["gc"]:
            _checkpoint(model)
        model(**_row_of_rank(rank)).loss.backward()
        per_row.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    return per_row


def _whole(grad: torch.Tensor) -> torch.Tensor:
    return grad.full_tensor().detach() if hasattr(grad, "full_tensor") else grad.detach()


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-20))


def _bucket(name: str, head_names: set[str]) -> str:
    if name in head_names:
        return "head"
    if "lm_head" in name or "embed_tokens" in name:
        return "embeddings"
    return "trunk"


def _run_row(rank: int, name: str, row: dict, *, ep1: bool, fp32_router: bool) -> dict:
    per_row = _single_process_gradients(name, row, ep1=ep1, fp32_router=fp32_router)
    mean = {key: (per_row[0][key] + per_row[1][key]) / _WORLD for key in per_row[0]}

    model, head = _built(name, row, ep1=ep1, fp32_router=fp32_router)
    head_prefix = next(prefix for prefix, module in model.named_modules() if module is head)
    head_names = {f"{head_prefix}.{key}" for key, _param in head.named_parameters()}
    excluded = _ModeSetup(model, experts_fsdp_managed=ep1)._fsdp_excluded_params()
    apply_fsdp2_for_dp(
        model,
        _WORLD,
        compute_dtype=torch.bfloat16 if fp32_router else torch.float32,
        fp32_reduce=True,
        ignored_params=excluded or None,
        reshard_after_forward=_ZERO_MODES[row["zero"]],
        ignored_params_replicated=True,
    )
    if row["gc"]:
        _checkpoint(model)
    groups = fsdp2_param_groups(model)
    head_group = head._get_fsdp_state()._fsdp_param_group
    router_groups = int(fp32_router and _router_of(head)._get_fsdp_state()._fsdp_param_group is not head_group)
    trunk_modules = [m for m in model.modules() if hasattr(m, "_get_fsdp_state") and m is not head]
    shared = [m for m in trunk_modules if m._get_fsdp_state()._fsdp_param_group is head_group]
    model(**_row_of_rank(rank)).loss.backward()
    late = complete_fsdp2_post_backward(groups)
    reject_unfinished_fsdp2_backward(groups)
    reshard_fsdp2_modules(model)

    # Gathers are collectives, so every rank takes every gradient in one order before any rank-local reading.
    whole = {key: _whole(param.grad) for key, param in model.named_parameters() if param.grad is not None}
    errors = {"head": 0.0, "embeddings": 0.0, "trunk": 0.0}
    gap_squared, norm_squared = dict.fromkeys(errors, 0.0), dict.fromkeys(errors, 0.0)
    for key in sorted(set(mean) & set(whole)):
        bucket = _bucket(key, head_names)
        got, expected = whole[key].float(), mean[key].float()
        errors[bucket] = max(errors[bucket], _relative(got, expected))
        gap_squared[bucket] += float((got - expected).norm() ** 2)
        norm_squared[bucket] += float(expected.norm() ** 2)
    pooled = {bucket: (gap_squared[bucket] / max(norm_squared[bucket], 1e-40)) ** 0.5 for bucket in errors}
    trunk_dtensors = sum(
        hasattr(param.data, "full_tensor") for key, param in model.named_parameters() if key not in head_names
    )
    return {
        "errors": errors,
        "pooled": pooled,
        "own_row": max(_relative(whole[key].float(), per_row[rank][key].float()) for key in head_names & set(whole)),
        "router_groups": router_groups,
        "late": late,
        "missing": sorted(set(mean) - set(whole)),
        "extra": sorted(set(whole) - set(mean)),
        "head_group": head_group is not None,
        "head_group_shared": len(shared),
        "groups": len(groups),
        "trunk_dtensors": trunk_dtensors,
        "head_grad_zero": sorted(key for key in head_names & set(whole) if whole[key].abs().sum() == 0),
    }


def _ranks_worker(
    rank: int, directory: str, name: str, rows: list[dict], ep1: bool, fp32_router: bool, chunk_rows: int
) -> None:
    fsdp_module.build_dp_mesh = functools.partial(build_dp_mesh, device_type="cpu")
    if chunk_rows:
        cross_entropy.logit_chunk_rows = lambda _vocab: chunk_rows
    results = {_tag(row): _run_row(rank, name, row, ep1=ep1, fp32_router=fp32_router) for row in rows}
    (Path(directory) / f"rank_{rank}.json").write_text(json.dumps(results))


def _results(
    tmp_path, name: str, rows: list[dict], *, ep1: bool = False, fp32_router: bool = False, chunk_rows: int = 0
) -> dict[str, dict]:
    run_gloo_ranks(
        _ranks_worker, _WORLD, str(tmp_path), name, rows, ep1, fp32_router, chunk_rows, pg_timeout=_COLLECTIVE_TIMEOUT
    )
    first, second = (json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(_WORLD))
    assert first.keys() == {_tag(row) for row in rows}
    assert {tag: value["errors"] for tag, value in first.items()} == {
        tag: value["errors"] for tag, value in second.items()
    }
    return first


def _assert_row(tag: str, result: dict, *, errors: str = "errors", head_bar=_HEAD_BAR, trunk_bar=_TRUNK_BAR) -> None:
    scope = next(scope for scope in (*_GRADED_SCOPES, "head_only") if scope in tag)
    assert result["head_group"], f"{tag}: the head holds no FSDP2 group of its own"
    assert result["head_group_shared"] == 0, f"{tag}: the head shares its group with {result['head_group_shared']}"
    assert result["late"] == 0, f"{tag}: {result['late']} group(s) reduced only in the completion sweep"
    assert not result["missing"] and not result["extra"], f"{tag}: {result['missing']}, {result['extra']}"
    assert not result["head_grad_zero"], f"{tag}: {result['head_grad_zero']}"
    assert result[errors]["head"] < head_bar, f"{tag}: {result[errors]}"
    assert result[errors]["embeddings"] < head_bar, f"{tag}: {result[errors]}"
    assert result[errors]["trunk"] < trunk_bar, f"{tag}: {result[errors]}"
    assert result["own_row"] > _ROWS_DIFFER, f"{tag}: the rows barely differ, so the bars would miss a missing mean"
    if scope == "head_only":
        assert result["groups"] == 1 + result["router_groups"], (
            f"{tag}: a frozen trunk holds {result['groups']} groups"
        )
        assert result["trunk_dtensors"] == 0, f"{tag}: {result['trunk_dtensors']} frozen params stayed sharded"
    else:
        assert result["groups"] > 2 and result["trunk_dtensors"] > 0, f"{tag}: the trunk went unsharded"


@pytest.mark.parametrize("name", ["glm4_moe_lite", "qwen3_5_moe_text", "nemotron_h"])
def test_every_gradient_is_the_mean_of_the_ranks_rows_across_scopes_and_zero_modes(name, tmp_path):
    for tag, result in _results(tmp_path, name, _matrix()).items():
        _assert_row(tag, result)


@pytest.mark.parametrize("name", ["qwen3_5", "qwen3_next", "qwen3_5_moe", "nemotron_h_omni", "nemotron_h_no_latent"])
def test_the_other_families_hold_the_mean_on_the_hardest_rows(name, tmp_path):
    rows = [
        {"loss_type": "soft", "scope": "joint", "zero": "zero3", "gc": True},
        {"loss_type": "hard", "scope": "joint", "zero": "zero3", "gc": False},
        {"loss_type": "soft", "scope": "head_only", "zero": "zero2", "gc": False},
    ]
    for tag, result in _results(tmp_path, name, rows).items():
        _assert_row(tag, result)


@pytest.mark.parametrize("name", ["glm4_moe_lite", "qwen3_5_moe_text", "nemotron_h"])
def test_ep1_experts_managed_by_fsdp2_reduce_in_the_heads_group(name, tmp_path):
    rows = [
        {"loss_type": "soft", "scope": "joint", "zero": "zero2", "gc": True},
        {"loss_type": "hard", "scope": "joint", "zero": "zero3", "gc": True},
        {"loss_type": "soft", "scope": "detached", "zero": "zero3", "gc": True},
        {"loss_type": "hard", "scope": "head_only", "zero": "zero3", "gc": False},
    ]
    for tag, result in _results(tmp_path, name, rows, ep1=True).items():
        _assert_row(tag, result)


@pytest.mark.parametrize("name", ["glm4_moe_lite", "qwen3_5_moe_text", "nemotron_h"])
def test_an_fp32_router_splits_off_the_heads_group_and_still_reduces_at_bf16_compute(name, tmp_path):
    """FSDP2 asserts one dtype per group, and the head's forward reads the gate's weight off the module."""
    rows = [
        {"loss_type": "soft", "scope": "joint", "zero": "zero2", "gc": True},
        {"loss_type": "hard", "scope": "joint", "zero": "zero3", "gc": False},
        {"loss_type": "hard", "scope": "head_only", "zero": "zero3", "gc": False},
    ]
    for tag, result in _results(tmp_path, name, rows, ep1=True, fp32_router=True).items():
        assert result["router_groups"] == 1, f"{tag}: the router took no group of its own"
        _assert_row(tag, result, errors="pooled", head_bar=_BF16_BAR, trunk_bar=_BF16_TRUNK_BAR)


@pytest.mark.parametrize("name", ["glm4_moe_lite", "qwen3_5_moe_text"])
def test_a_long_row_scored_in_checkpointed_chunks_holds_the_mean(name, tmp_path):
    """The chunks recompute during backward and read the output embedding then, which a shard group must hold gathered."""
    rows = [
        {"loss_type": "hard", "scope": "joint", "zero": "zero3", "gc": False},
        {"loss_type": "soft", "scope": "detached", "zero": "zero3", "gc": True},
        {"loss_type": "soft", "scope": "joint", "zero": "zero2", "gc": True},
        {"loss_type": "hard", "scope": "head_only", "zero": "zero2", "gc": False},
    ]
    for tag, result in _results(tmp_path, name, rows, chunk_rows=5).items():
        _assert_row(tag, result)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
