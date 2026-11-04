#!/usr/bin/env python
"""The lazy path bypasses ``from_pretrained`` and replaces meta parameters with what came off disk, so each guard
here stops silent corruption."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, GptOssConfig, GptOssForCausalLM

import halo.model_loading.parallel.ep as ep_lazy_loader
import halo.model_loading.parallel.lazy_plan as lazy_plan
import halo.model_loading.parallel.model_source as model_source
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.config import EPConfig
from halo.model_loading.lazy_safetensors.materialization import SafetensorsWeightLoader, WeightAction, WeightPlan
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.expert_fuser import ExpertFuser
from halo.model_loading.parallel.lazy_plan import WeightPlanner
from halo.runtime import consensus
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.repo import REPO_ROOT

E_GLOBAL, M, H = 8, 6, 4
LOADER_LOGGER = "halo.model_loading.parallel.lazy_plan"

_LAZY_GATE_CALL_SITES = (
    REPO_ROOT / "halo/model_loading/parallel/ep.py",
    REPO_ROOT / "halo/model_loading/parallel/pp.py",
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
    """The fused layout every family's shell holds before EP patching."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(**kwargs)])


def _load(tmp_path, model: nn.Module, tensors: dict[str, torch.Tensor], plans: list[WeightPlan]) -> None:
    save_file(tensors, str(tmp_path / "model.safetensors"))
    SafetensorsWeightLoader(
        str(tmp_path), resolve_checkpoint_weights(str(tmp_path)).weight_map, device="cpu"
    ).load_into_model(model, plans)


def test_a_replicated_tensor_of_the_wrong_shape_is_refused(tmp_path):
    """A patch_vocab-grown checkpoint on the base config must raise rather than reshape the model."""
    model = nn.Linear(H, 3, bias=False)
    plan = WeightPlan(WeightAction.REPLICATE, "model.safetensors", "weight", "weight")

    with pytest.raises(RuntimeError) as excinfo:
        _load(tmp_path, model, {"weight": torch.randn(5, H)}, [plan])

    message = str(excinfo.value)
    assert "'weight'" in message
    assert f"(5, {H})" in message and f"(3, {H})" in message, message
    # The guard runs before the assignment.
    assert tuple(model.weight.shape) == (3, H)


def test_an_expert_slice_shorter_than_the_rank_range_is_refused(tmp_path):
    """The ranged read returns the rows that exist instead of raising."""
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
    """A ranged read is satisfied by a longer axis, so experts past the config are never owned and silently dropped."""
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


def test_an_fp8_disk_tensor_is_refused(tmp_path):
    """fp8 must not be cast to bf16 without the block scales beside it."""
    model = nn.Linear(H, 3, bias=False)
    plan = WeightPlan(WeightAction.REPLICATE, "model.safetensors", "weight", "weight")
    save_file({"weight": torch.randn(3, H).to(torch.float8_e4m3fn)}, str(tmp_path / "model.safetensors"))
    loader = SafetensorsWeightLoader(str(tmp_path), resolve_checkpoint_weights(str(tmp_path)).weight_map, device="cpu")

    with pytest.raises(ValueError, match=r"'weight'.*Dequantize it to bf16"):
        loader.load_into_model(model, [plan], dtype=torch.bfloat16)
    assert model.weight.dtype == torch.float32


def test_the_planner_records_the_configs_expert_count_on_every_shard_plan():
    """Without the count on the plan, the check above is unreachable in production."""
    ep_config = SimpleNamespace(expert_start_idx=4, expert_end_idx=8, num_experts=E_GLOBAL)
    planner = WeightPlanner(ep_config)
    key = "layers.0.mlp.experts.gate_up_proj"
    plans = planner.build({key: "model.safetensors"}, {key: key}, {key})
    shard_plans = [p for p in plans if p.action == WeightAction.EXPERT_SHARD]
    assert shard_plans and all(p.shard_total == E_GLOBAL for p in shard_plans)


def _per_expert_checkpoint(
    tmp_path, experts: list[int], intermediate: int = M, dtype: torch.dtype = torch.float32
) -> dict[str, str]:
    tensors: dict[str, torch.Tensor] = {}
    for idx in experts:
        prefix = f"layers.0.mlp.experts.{idx}"
        tensors[f"{prefix}.gate_proj.weight"] = torch.randn(intermediate, H).to(dtype)
        tensors[f"{prefix}.up_proj.weight"] = torch.randn(intermediate, H).to(dtype)
        tensors[f"{prefix}.down_proj.weight"] = torch.randn(H, intermediate).to(dtype)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return dict.fromkeys(tensors, "model.safetensors")


def _fuse(tmp_path, model: nn.Module, weight_map: dict[str, str], ep_start: int, ep_end: int, dtype=None) -> set[str]:
    fuser = ExpertFuser(ep_start, ep_end)
    model_keys = set(model.state_dict())
    tasks = fuser.detect_tasks(weight_map, {key: key for key in weight_map}, model_keys)
    assert tasks, "fixture built no fusion task — the harness, not the guard, is broken"
    return fuser.execute(tasks, model, str(tmp_path), dtype=dtype, device="cpu")


def test_a_fusion_missing_a_global_expert_index_is_refused(tmp_path):
    """The fuse stacks ``sorted(expert_dict)`` into local slots, so a missing index promotes every later expert into
    the wrong slot."""
    weight_map = _per_expert_checkpoint(tmp_path, experts=[0, 1, 3])

    with pytest.raises(RuntimeError, match="global expert index"):
        _fuse(tmp_path, _MoEModel(), weight_map, ep_start=0, ep_end=4)


def test_a_fused_tensor_of_the_wrong_shape_is_refused(tmp_path):
    weight_map = _per_expert_checkpoint(tmp_path, experts=[0, 1, 2, 3], intermediate=M + 1)

    with pytest.raises(RuntimeError, match="gate_up_proj"):
        _fuse(tmp_path, _MoEModel(), weight_map, ep_start=0, ep_end=4)


def test_an_fp8_per_expert_fusion_is_refused(tmp_path):
    """Per-expert fp8 weights reach the fuser rather than the loader."""
    weight_map = _per_expert_checkpoint(tmp_path, experts=list(range(E_GLOBAL)), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=r"gate_up_proj.*Dequantize it to bf16"):
        _fuse(tmp_path, _MoEModel(), weight_map, ep_start=4, ep_end=8, dtype=torch.bfloat16)


def test_a_complete_in_range_fusion_still_loads(tmp_path):
    weight_map = _per_expert_checkpoint(tmp_path, experts=list(range(E_GLOBAL)))
    model = _MoEModel()

    fused = _fuse(tmp_path, model, weight_map, ep_start=4, ep_end=8)

    assert fused == {"layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj"}
    assert tuple(model.layers[0].mlp.experts.gate_up_proj.shape) == (4, 2 * M, H)


def _identity_plan(weight_map: dict[str, str], model_keys: set[str], disk_to_model=None):
    return WeightPlanner(None).build(weight_map, disk_to_model or {k: k for k in weight_map}, model_keys)


def test_checkpoint_keys_matching_no_model_tensor_are_reported_once(caplog):
    """An unapplied hub rename could take a whole module out of the load with nothing to say so."""
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
    """The ExpertFuser consumes them, and reporting them would bury the real hits."""
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.{proj}.weight" for i in range(4) for proj in ("gate_proj", "down_proj")],
        "s.safetensors",
    )

    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        _identity_plan(weight_map, {"layers.0.mlp.experts.gate_up_proj"})

    assert [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING] == []


def test_undeclared_per_expert_suffix_is_reported(caplog):
    """detect_tasks skips any other suffix and nothing reads it, so silence would hide a whole module."""
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.mystery_proj.weight" for i in range(2)],
        "s.safetensors",
    )
    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        _identity_plan(weight_map, {"layers.0.mlp.experts.gate_up_proj"})
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "2 checkpoint key(s)" in warnings[0].getMessage()


def test_expert_index_beyond_the_config_is_reported(caplog):
    """No rank owns the tail under any ep_size, so those keys would silently vanish."""
    ep_config = SimpleNamespace(expert_start_idx=0, expert_end_idx=4, num_experts=4)
    weight_map = dict.fromkeys(
        [f"layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(6)],
        "s.safetensors",
    )
    with caplog.at_level(logging.WARNING, logger=LOADER_LOGGER):
        WeightPlanner(ep_config).build(weight_map, {k: k for k in weight_map}, {"layers.0.mlp.experts.gate_up_proj"})
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    message = warnings[0].getMessage()
    assert "2 checkpoint key(s)" in message and "experts.4" in message and "experts.5" in message


def test_two_checkpoint_keys_claiming_one_model_tensor_are_refused():
    """Dropping the second key makes the loaded weight a function of shard ordering."""
    weight_map = dict.fromkeys(["decoder.attn.weight", "model.attn.weight"], "s.safetensors")
    disk_to_model = dict.fromkeys(weight_map, "model.attn.weight")

    with pytest.raises(RuntimeError, match="both align to"):
        _identity_plan(weight_map, {"model.attn.weight"}, disk_to_model)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(0)
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG)).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("ep_fence_ckpt")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture
def _stub_ep_patching(monkeypatch):
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *args, **kwargs: model)


@pytest.fixture
def rejections(monkeypatch) -> list[tuple[str | None, str]]:
    """``DeferredRankFailure.reject`` resolves ``reject_across_ranks`` from its module globals, so patching it there
    sees the real call."""
    calls: list[tuple[str | None, str]] = []

    def _record(local_reason, what, exc_type=RuntimeError):
        calls.append((local_reason, what))
        if local_reason:
            raise exc_type(f"{what}: {local_reason}")

    monkeypatch.setattr(consensus, "reject_across_ranks", _record)
    return calls


def _load_ep(path: str):
    return load_ep_model_lazy(
        path,
        EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=True),
        AutoConfig.from_pretrained(path, trust_remote_code=False),
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )


def test_the_load_always_closes_on_the_consensus_collective(checkpoint, rejections, _stub_ep_patching):
    """A reject reached only by the failing rank would be the desync it exists to prevent."""
    _load_ep(checkpoint)

    assert len(rejections) == 1, rejections
    reason, what = rejections[0]
    assert reason is None
    assert "EP lazy load" in what and checkpoint in what


def test_a_rank_local_shard_failure_surfaces_through_the_consensus_seam(
    checkpoint, rejections, _stub_ep_patching, monkeypatch
):
    """Unfenced, a torn shard strands every peer until the NCCL watchdog blames the collective for a disk error."""
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
    # The disk error reaches the peers as a reason, not a lone OSError on one rank.
    assert not isinstance(excinfo.value, OSError)
    assert torn in str(excinfo.value)


def _ungated(monkeypatch):
    """The EP-sharded refusal has its own tests."""
    monkeypatch.setattr(lazy_plan, "reject_ep_sharded_checkpoint", lambda source: None)


_SOURCE = model_source.ModelSource("org/model", None, "/dir")


def test_the_lazy_gate_is_decided_by_rank_zero_alone(monkeypatch):
    """The branches enter different collectives and every input is a per-rank probe; a split verdict is a hang with
    no diagnostic."""
    _ungated(monkeypatch)
    monkeypatch.setattr(lazy_plan, "has_whole_model_weight_file", lambda _dir, **_kw: True)
    monkeypatch.setattr(lazy_plan, "broadcast_from_rank0", lambda _value: "rank0-verdict")

    for layout_supported in (False, True):
        monkeypatch.setattr(lazy_plan, "is_lazy_loadable", lambda _dir, supported=layout_supported: supported)
        assert lazy_plan.agree_lazy_load(_SOURCE) == "rank0-verdict"


@pytest.mark.parametrize(
    ("has_safetensors", "layout_supported", "expected"),
    [
        (True, True, True),
        (True, False, False),  # A layout the fuser would silently misload (pre-5.14 Zaya).
        (False, True, False),
    ],
)
def test_the_lazy_gate_needs_safetensors_and_a_supported_layout(
    monkeypatch, has_safetensors, layout_supported, expected
):
    _ungated(monkeypatch)
    asked: list[str] = []
    monkeypatch.setattr(lazy_plan, "has_whole_model_weight_file", lambda _dir, **_kw: has_safetensors)
    monkeypatch.setattr(lazy_plan, "broadcast_from_rank0", lambda value: value)

    def _layout(path):
        asked.append(path)
        return layout_supported

    monkeypatch.setattr(lazy_plan, "is_lazy_loadable", _layout)
    assert lazy_plan.agree_lazy_load(_SOURCE) is expected
    if not has_safetensors:
        assert asked == [], "the layout probe was handed a directory that does not hold a checkpoint"


@pytest.mark.parametrize("call_site", _LAZY_GATE_CALL_SITES, ids=lambda p: p.name)
def test_a_lazy_loader_call_site_goes_through_the_one_gate(call_site):
    """Read from the source, since a re-inlined copy is how the rule drifts."""
    source = call_site.read_text(encoding="utf-8")
    assert "agree_lazy_load" in source, f"{call_site.name} no longer routes its lazy gate through the helper"
    # The inlined shape, in every wrapping: the layout probe ANDed into the caller's own expression.
    assert "and is_lazy_loadable(" not in " ".join(source.split()), (
        f"{call_site.name} re-inlined the layout probe into its own gate expression instead of "
        f"handing it to agree_lazy_load"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
