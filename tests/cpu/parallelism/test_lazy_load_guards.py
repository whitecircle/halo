#!/usr/bin/env python
"""CPU tests for the lazy loaders' load-time guards.

The lazy path bypasses ``from_pretrained`` entirely: it reads safetensors slices and REPLACES the
meta-device parameter with whatever came off disk. Four things ``from_pretrained`` does for free
therefore have to be done here, and each is silent corruption when it is not:

* the disk tensor's shape is checked against the live one (else the checkpoint's shape wins over the
  config's, silently);
* a per-expert fusion covers this rank's WHOLE expert range (else a missing global index shifts
  every later expert down a local slot);
* checkpoint keys that align to no model tensor are reported (``from_pretrained``'s unexpected-key
  warning), and two keys claiming one tensor is refused rather than decided by index order;
* the per-rank shard I/O is fenced, so a torn shard on one rank does not strand its peers in the
  next collective.

    python tests/cpu/parallelism/test_lazy_load_guards.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, GptOssConfig, GptOssForCausalLM

import src.distributed.expert_parallel.lazy_loader as ep_lazy_loader
import src.distributed.expert_parallel.loading as ep_loading
import src.distributed.runtime as runtime
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.lazy_loader import (
    EPWeightPlanner,
    ExpertFuser,
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
    load_ep_model_lazy,
)
from tests.common.models import TINY_GPTOSS_CONFIG

E_GLOBAL, M, H = 8, 6, 4
LOADER_LOGGER = "src.distributed.expert_parallel.lazy_loader"

# Loaders that open the lazy safetensors path and must reach it through the shared rank-0 gate.
_LAZY_GATE_CALL_SITES = (
    Path(__file__).resolve().parents[3] / "src/distributed/expert_parallel/loading.py",
    Path(__file__).resolve().parents[3] / "src/distributed/loading/model_loading.py",
)


class _Experts(nn.Module):
    def __init__(self, num_experts: int = E_GLOBAL, intermediate: int = M, hidden: int = H):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, intermediate))


class _Mlp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.experts = _Experts(**kwargs)


class _Layer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mlp = _Mlp(**kwargs)


class _MoEModel(nn.Module):
    """``layers.0.mlp.experts.{gate_up_proj,down_proj}`` — the fused layout every family's shell holds
    before EP patching, with the expert axis at dim 0."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(**kwargs)])


def _load(tmp_path, model: nn.Module, tensors: dict[str, torch.Tensor], plans: list[WeightPlan]) -> None:
    save_file(tensors, str(tmp_path / "model.safetensors"))
    SafetensorsWeightLoader(str(tmp_path), ["model.safetensors"], device="cpu").load_into_model(model, plans)


# --------------------------------------------------------------------------------------------
# Shape validation (both paths install the disk tensor, so nothing else can catch a mismatch)
# --------------------------------------------------------------------------------------------


def test_a_replicated_tensor_of_the_wrong_shape_is_refused(tmp_path):
    """A checkpoint↔config mismatch (a patch_vocab-shrunk checkpoint on the base config) must raise
    the way ``from_pretrained`` does, not silently reshape the model to the checkpoint."""
    model = nn.Linear(H, 3, bias=False)
    plan = WeightPlan(WeightAction.REPLICATE, "model.safetensors", "weight", "weight")

    with pytest.raises(RuntimeError) as excinfo:
        _load(tmp_path, model, {"weight": torch.randn(5, H)}, [plan])

    message = str(excinfo.value)
    assert "'weight'" in message
    assert f"(5, {H})" in message and f"(3, {H})" in message, message
    # And the model is unchanged — the guard runs BEFORE the assignment.
    assert tuple(model.weight.shape) == (3, H)


def test_an_expert_slice_shorter_than_the_rank_range_is_refused(tmp_path):
    """A checkpoint holding fewer experts than the config declares: the ranged safetensors read
    returns the rows that exist instead of raising, so only the shape check sees it."""
    model = _MoEModel()
    on_disk = {"layers.0.mlp.experts.gate_up_proj": torch.randn(E_GLOBAL - 2, 2 * M, H)}
    plan = WeightPlan(
        WeightAction.EXPERT_SHARD,
        "model.safetensors",
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.gate_up_proj",
        shard_dim=0,
        shard_start=4,
        shard_end=8,
    )

    with pytest.raises(RuntimeError, match="expert slice"):
        _load(tmp_path, model, on_disk, [plan])


def test_a_fused_checkpoint_with_MORE_experts_than_the_config_is_refused(tmp_path):
    """The mirror of the short-slice case, and the silent one: a ranged read is satisfied by a
    LONGER axis, so a checkpoint carrying more experts than the config declares loads experts
    0..N-1 at exactly the expected shape. No rank ever owns the rest — the model trains and exports
    as if they were never in the file."""
    model = _MoEModel()
    on_disk = {"layers.0.mlp.experts.gate_up_proj": torch.randn(E_GLOBAL * 2, 2 * M, H)}
    plan = WeightPlan(
        WeightAction.EXPERT_SHARD,
        "model.safetensors",
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.gate_up_proj",
        shard_dim=0,
        shard_start=4,
        shard_end=8,
        shard_total=E_GLOBAL,
    )

    with pytest.raises(RuntimeError, match="config declares"):
        _load(tmp_path, model, on_disk, [plan])


def test_a_correctly_shaped_expert_slice_still_loads(tmp_path):
    """Anti-vacuity: the guard rejects the mismatch, not every expert-sharded read."""
    model = _MoEModel()
    full = torch.randn(E_GLOBAL, 2 * M, H)
    plan = WeightPlan(
        WeightAction.EXPERT_SHARD,
        "model.safetensors",
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.gate_up_proj",
        shard_dim=0,
        shard_start=4,
        shard_end=8,
        shard_total=E_GLOBAL,
    )

    _load(tmp_path, model, {"layers.0.mlp.experts.gate_up_proj": full}, [plan])

    assert torch.equal(model.layers[0].mlp.experts.gate_up_proj.detach(), full[4:8])


def test_the_planner_records_the_configs_expert_count_on_every_shard_plan():
    """The count has to travel with the plan, or the check above is unreachable in production."""
    ep_config = SimpleNamespace(expert_start_idx=4, expert_end_idx=8, num_experts=E_GLOBAL)
    planner = EPWeightPlanner(ep_config)
    key = "layers.0.mlp.experts.gate_up_proj"
    plans = planner.build({key: "model.safetensors"}, {key: key}, {key})
    shard_plans = [p for p in plans if p.action == WeightAction.EXPERT_SHARD]
    assert shard_plans and all(p.shard_total == E_GLOBAL for p in shard_plans)


# --------------------------------------------------------------------------------------------
# ExpertFuser: full local range + the fused result's shape
# --------------------------------------------------------------------------------------------


def _per_expert_checkpoint(tmp_path, experts: list[int], intermediate: int = M) -> dict[str, str]:
    """A per-expert (Qwen3/Bailing-style) checkpoint for one layer. Returns its weight map."""
    tensors: dict[str, torch.Tensor] = {}
    for idx in experts:
        prefix = f"layers.0.mlp.experts.{idx}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.randn(intermediate, H)
        tensors[f"{prefix}.up_proj.weight"] = torch.randn(intermediate, H)
        tensors[f"{prefix}.down_proj.weight"] = torch.randn(H, intermediate)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return dict.fromkeys(tensors, "model.safetensors")


def _fuse(tmp_path, model: nn.Module, weight_map: dict[str, str], ep_start: int, ep_end: int) -> set[str]:
    fuser = ExpertFuser(ep_start, ep_end)
    model_keys = set(model.state_dict())
    tasks = fuser.detect_tasks(weight_map, {key: key for key in weight_map}, model_keys)
    assert tasks, "fixture built no fusion task — the harness, not the guard, is broken"
    return fuser.execute(tasks, model, str(tmp_path), dtype=None, device="cpu")


def test_a_fusion_missing_a_global_expert_index_is_refused(tmp_path):
    """The fuse stacks ``sorted(expert_dict)`` into local slots, so a global index the checkpoint does
    not carry silently promotes every later expert into the wrong slot."""
    weight_map = _per_expert_checkpoint(tmp_path, experts=[0, 1, 3])

    with pytest.raises(RuntimeError, match="global expert index"):
        _fuse(tmp_path, _MoEModel(), weight_map, ep_start=0, ep_end=4)


def test_a_fused_tensor_of_the_wrong_shape_is_refused(tmp_path):
    """The full range can be present and still disagree with the config — an intermediate_size the
    model was not built for."""
    weight_map = _per_expert_checkpoint(tmp_path, experts=[0, 1, 2, 3], intermediate=M + 1)

    with pytest.raises(RuntimeError, match="gate_up_proj"):
        _fuse(tmp_path, _MoEModel(), weight_map, ep_start=0, ep_end=4)


def test_a_complete_in_range_fusion_still_loads(tmp_path):
    """Anti-vacuity: both fuser guards pass for the checkpoint they are meant to admit."""
    weight_map = _per_expert_checkpoint(tmp_path, experts=list(range(E_GLOBAL)))
    model = _MoEModel()

    fused = _fuse(tmp_path, model, weight_map, ep_start=4, ep_end=8)

    assert fused == {"layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj"}
    assert tuple(model.layers[0].mlp.experts.gate_up_proj.shape) == (4, 2 * M, H)


# --------------------------------------------------------------------------------------------
# EPWeightPlanner: unexpected checkpoint keys, and one model tensor claimed twice
# --------------------------------------------------------------------------------------------


def _identity_plan(weight_map: dict[str, str], model_keys: set[str], disk_to_model=None):
    return EPWeightPlanner(None).build(weight_map, disk_to_model or {k: k for k in weight_map}, model_keys)


def test_checkpoint_keys_matching_no_model_tensor_are_reported_once(caplog):
    """``from_pretrained`` names its unexpected keys. Skipping them without a word lets an unapplied
    hub rename take a whole module out of the load with nothing anywhere to say so."""
    weight_map = dict.fromkeys(["model.kept.weight", "model.ghost.weight", "model.other.weight"], "s.safetensors")

    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        plans = _identity_plan(weight_map, {"model.kept.weight"})

    assert [plan.model_key for plan in plans] == ["model.kept.weight"]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert "2 checkpoint key(s)" in message
    assert "model.ghost.weight" in message and "model.other.weight" in message


def test_per_expert_keys_are_not_reported_as_unexpected(caplog):
    """They align to no fused model key BY CONSTRUCTION — the ExpertFuser consumes them, and a fusion
    that claims none of them raises there. Reporting them would bury the real hits under a wall."""
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.{proj}.weight" for i in range(4) for proj in ("gate_proj", "down_proj")],
        "s.safetensors",
    )

    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        _identity_plan(weight_map, {"layers.0.mlp.experts.gate_up_proj"})

    assert [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING] == []


def test_undeclared_per_expert_suffix_is_reported(caplog):
    """The fuser consumes only DECLARED projection suffixes — a per-expert key with any other
    suffix is skipped by detect_tasks and read by nothing, so silence would hide a whole module."""
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.mystery_proj.weight" for i in range(2)],
        "s.safetensors",
    )
    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        _identity_plan(weight_map, {"layers.0.mlp.experts.gate_up_proj"})
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2 checkpoint key(s)" in warnings[0].getMessage()


def test_expert_index_beyond_the_config_is_reported(caplog):
    """A checkpoint declaring MORE experts than the config: no rank owns the tail under any
    ep_size, so those keys silently vanish without this report."""
    ep_config = SimpleNamespace(expert_start_idx=0, expert_end_idx=4, num_experts=4)
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(6)],
        "s.safetensors",
    )
    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        EPWeightPlanner(ep_config).build(weight_map, {k: k for k in weight_map}, {"layers.0.mlp.experts.gate_up_proj"})
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert "2 checkpoint key(s)" in message and "experts.4" in message and "experts.5" in message


def test_two_checkpoint_keys_claiming_one_model_tensor_are_refused():
    """Dropping whichever key the index yields second makes the loaded weight a function of shard
    ordering — a coin flip between two tensors, taken silently."""
    weight_map = dict.fromkeys(["decoder.attn.weight", "model.attn.weight"], "s.safetensors")
    disk_to_model = dict.fromkeys(weight_map, "model.attn.weight")

    with pytest.raises(RuntimeError, match="both align to"):
        _identity_plan(weight_map, {"model.attn.weight"}, disk_to_model)


# --------------------------------------------------------------------------------------------
# Finding 4: the per-rank load is fenced by the consensus seam, not by a bare barrier
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(0)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG)).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("ep_fence_ckpt")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture
def _stub_ep_patching(monkeypatch):
    monkeypatch.setattr(ep_lazy_loader, "patch_moe_model_for_ep", lambda model, *args, **kwargs: model)
    monkeypatch.setattr(ep_lazy_loader, "create_ep_buffers", lambda *args, **kwargs: None)


@pytest.fixture
def rejections(monkeypatch) -> list[tuple[str | None, str]]:
    """Record every trip through the consensus seam. ``DeferredRankFailure.reject`` resolves
    ``reject_across_ranks`` from its module globals, so patching it there sees the real call."""
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=RuntimeError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(f"{what}: {local_reason}")

    monkeypatch.setattr(runtime, "reject_across_ranks", _record)
    return calls


def _load_ep(path: str):
    return load_ep_model_lazy(
        path,
        EPConfig(ep_size=1, world_size=1, gpus_per_node=1),
        AutoConfig.from_pretrained(path, trust_remote_code=False),
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )


def test_the_load_always_closes_on_the_consensus_collective(checkpoint, rejections, _stub_ep_patching):
    """The fence is unconditional: every rank reaches it on the success path too. A reject reached
    only by the failing rank would itself be the desync it exists to prevent."""
    _load_ep(checkpoint)

    assert len(rejections) == 1, rejections
    reason, what = rejections[0]
    assert reason is None
    assert "EP lazy load" in what and checkpoint in what


def test_a_rank_local_shard_failure_surfaces_through_the_consensus_seam(
    checkpoint, rejections, _stub_ep_patching, monkeypatch
):
    """Each rank opens only the shards holding its own experts, so a torn one is a rank-LOCAL raise
    sitting before a collective: unfenced it strands every peer until the NCCL watchdog fires and
    blames the collective for a disk error."""
    torn = "stale NFS file handle: model-00002-of-00003.safetensors"

    def _fail(self, *args, **kwargs):
        raise OSError(torn)

    monkeypatch.setattr(SafetensorsWeightLoader, "load_into_model", _fail)

    with pytest.raises(RuntimeError) as excinfo:
        _load_ep(checkpoint)

    assert len(rejections) == 1, rejections
    reason, what = rejections[0]
    assert reason == f"OSError: {torn}", reason
    assert "EP lazy load" in what
    # The disk error reaches the peers as a REASON — not a lone OSError on one rank.
    assert not isinstance(excinfo.value, OSError)
    assert torn in str(excinfo.value)


# --- The lazy-vs-fallback gate ------------------------------------------------------------------


def test_the_lazy_gate_is_decided_by_rank_zero_alone(monkeypatch):
    """Every rank must take the branch RANK 0 chose, never the one its own filesystem suggests.

    The lazy and fallback branches enter different store phases and different world collectives, and
    every input to the choice is a best-effort per-rank probe — a partially populated cache on a
    non-shared filesystem answers differently on different ranks. A split verdict is a hang with no
    diagnostic on either side, so the verdict is broadcast. The rule lives once rather than being
    spelled out at each of its three call sites (EP, EP+TP, PP stage).
    """
    monkeypatch.setattr(ep_loading, "has_safetensors_checkpoint", lambda _dir: True)
    monkeypatch.setattr(ep_loading, "broadcast_from_rank0", lambda _value: "rank0-verdict")

    assert ep_loading.decide_lazy_loadable("/any/dir", lambda _dir: False) == "rank0-verdict"
    assert ep_loading.decide_lazy_loadable("/any/dir", lambda _dir: True) == "rank0-verdict"


@pytest.mark.parametrize(
    ("local_dir", "has_safetensors", "layout_supported", "expected"),
    [
        ("/dir", True, True, True),
        ("/dir", True, False, False),  # a layout the fuser would silently misload (pre-5.14 Zaya)
        ("/dir", False, True, False),  # no safetensors to stream
        (None, True, True, False),  # this rank never resolved a local snapshot
    ],
)
def test_the_lazy_gate_needs_a_resolved_dir_safetensors_and_a_supported_layout(
    monkeypatch, local_dir, has_safetensors, layout_supported, expected
):
    """All three conditions, and the layout probe is only asked about a directory that exists."""
    asked: list[str] = []
    monkeypatch.setattr(ep_loading, "has_safetensors_checkpoint", lambda _dir: has_safetensors)
    monkeypatch.setattr(ep_loading, "broadcast_from_rank0", lambda value: value)

    def _layout(path):
        asked.append(path)
        return layout_supported

    assert ep_loading.decide_lazy_loadable(local_dir, _layout) is expected
    if local_dir is None or not has_safetensors:
        assert asked == [], "the layout probe was handed a directory that does not hold a checkpoint"


@pytest.mark.parametrize("call_site", _LAZY_GATE_CALL_SITES, ids=lambda p: p.name)
def test_a_lazy_loader_call_site_goes_through_the_one_gate(call_site):
    """One gate, read from the source — a re-inlined copy is how the rule drifts.

    Each loader that opens the lazy path would otherwise carry its own
    ``broadcast_from_rank0(local_dir is not None and has_safetensors... and lazy_loader_supports...)``
    line, restating the "rank 0 decides" reasoning at each and able to drop it at one.
    """
    source = call_site.read_text(encoding="utf-8")
    assert "decide_lazy_loadable" in source, f"{call_site.name} no longer routes its lazy gate through the helper"
    # The inlined shape, in every wrapping: the layout probe ANDed into the caller's own expression.
    assert "and lazy_loader_supports_checkpoint(" not in " ".join(source.split()), (
        f"{call_site.name} re-inlined the layout probe into its own gate expression instead of "
        f"handing it to decide_lazy_loadable"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
