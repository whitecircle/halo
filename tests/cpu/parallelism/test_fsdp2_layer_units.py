#!/usr/bin/env python
"""FSDP2 shard groups of several decoder layers train bit for bit like one-layer groups, on gloo ranks, and save the
same checkpoint.

``apply_fsdp2_per_layer`` hands consecutive decoder layers to one ``fully_shard`` call and replaces torch's
multi-module forward hooks with hooks at the unit's two ends (``_install_unit_hooks``). Each case trains a tiny model
two steps at every unit size and compares the loss, every sharded gradient, every sharded weight and the state dict's
layout with one-layer groups: odd layer counts; gradient checkpointing off, non-reentrant and reentrant, on every layer
and on every other one; ZeRO-2 and ZeRO-3; several rows per backward; gradient-accumulation windows; nested dtype
groups and a layer of another trainable dtype; ep1 EP layers with FSDP-sharded and FSDP-ignored experts; LoRA; a
hybrid gated-DeltaNet stack; a pipeline stage's FSDP drive; HSDP; a TP x DP mesh. The FSDP2 writer and the optimizer
shard store save the same tensors at every unit size, and a run resumes across unit sizes and from ZeRO-2 into ZeRO-3.
Left to its own count the wrap groups two layers under ZeRO-2 and keeps one per group under ZeRO-3.

Negative controls, each run here: torch's own multi-module hooks crash a reentrant recompute; without the first
layer's recompute registration a partly checkpointed unit reduces only in the final callback, after the backward; a
unit that mixes trainable dtypes fails FSDP2's dtype check.

    python tests/cpu/parallelism/test_fsdp2_layer_units.py
"""

import contextlib
import dataclasses
import datetime
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard import _fsdp_param_group, _fsdp_state
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from halo.checkpoint.format import optimizer_shard_filename
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.parallel.save import save_fsdp2_checkpoint
from halo.distributed import fsdp
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, fsdp2_modules, mixed_precision_policy
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from tests.common.checkpoint_io import load_context, safetensors_state_dict, save_context, shard_store_context
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model

WORLD = 2
# HSDP's (replicate, shard) mesh, and the TP run's (dp, tp) mesh.
MESH_4 = (2, 2)
PG_TIMEOUT = datetime.timedelta(seconds=120)
VOCAB = 128
STEPS = 2
LR = 0.05
UNIT_SIZES = (2, 3)
MICROBATCHES = 2
FAMILIES = {
    "dense": (TINY_DENSE_FAMILIES["qwen3"], {}),
    # Gated DeltaNet layers beside attention layers: units span both layer types.
    "hybrid": (TINY_DENSE_FAMILIES["qwen3_5"], {}),
    "moe": (TINY_MOE_FAMILIES["qwen3_moe"], {"num_experts": 4, "moe_intermediate_size": 32}),
}
LORA_TARGETS = ["q_proj", "v_proj"]


@dataclasses.dataclass(frozen=True)
class Case:
    """One training shape; ``variant`` names what is built on the family's tiny model before the FSDP2 wrap."""

    family: str = "dense"
    # "" | "nested" (each MLP in float64, a nested group) | "odd_layer" (layer 1 in float64) | "lora"
    # | "ep_sharded" / "ep_ignored" (ep1 EP layers, the experts FSDP-sharded or left out of every group)
    # | "fp32_rope" (fp32 position embeddings into every layer, which each group's pre-forward casts)
    variant: str = ""
    layers: int = 4
    gc: str = "reentrant"  # "off" | "nonreentrant" | "reentrant"
    every_n_layers: int = 1
    raf: bool = False
    rows: int = 1  # forwards per backward
    window: str = ""  # "" | "defer" (no reduce before the window's last backward) | "keep" (no reshard)
    pipeline: bool = False  # the FSDP calls torch's pipeline stage makes around its microbatches

    @property
    def name(self) -> str:
        changed = {key: value for key, value in dataclasses.asdict(self).items() if value != getattr(Case, key)}
        return "-".join(f"{key}={value}" for key, value in changed.items()) or "base"


DENSE_CASES = [
    Case(layers=layers, gc=gc, raf=raf)
    for layers in (3, 4)
    for gc in ("off", "nonreentrant", "reentrant")
    for raf in (False, True)
] + [Case(layers=layers, raf=raf, rows=3) for layers in (3, 4) for raf in (False, True)]
PARTIAL_GC_CASES = [
    Case(layers=layers, gc=gc, every_n_layers=2) for layers in (3, 4) for gc in ("nonreentrant", "reentrant")
]
SHAPE_CASES = [
    Case(window="defer"),
    Case(window="keep"),
    Case(layers=3, gc="nonreentrant", pipeline=True),
    Case(variant="nested", layers=3),
    Case(variant="odd_layer"),
    Case(variant="fp32_rope"),
    Case(variant="fp32_rope", gc="off"),
    Case(variant="lora", layers=3, gc="nonreentrant"),
    Case(variant="lora", layers=3),
    Case(family="hybrid", layers=5, gc="nonreentrant"),
    Case(family="moe", variant="ep_sharded", layers=3),
    Case(family="moe", variant="ep_ignored", layers=3),
]
CASES = DENSE_CASES + PARTIAL_GC_CASES + SHAPE_CASES
MESH_4_CASES = [Case(layers=3, gc=gc) for gc in ("nonreentrant", "reentrant")]


def _rows(case: Case, step: int, micro: int, rank: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(10_000 * step + 100 * micro + rank)
    return [torch.randint(0, VOCAB, (1, 6 + 3 * i), generator=generator) for i in range(case.rows)]


def _loss(model: nn.Module, rows: list[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for ids in rows:
        logits = model(input_ids=ids).logits.float()
        loss = loss + F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    return loss


def _digest(tensor: torch.Tensor) -> str:
    """This rank's shard of ``tensor``, bit for bit."""
    local = (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().contiguous().reshape(-1)
    return hashlib.sha256(local.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _ep1_config(experts_fsdp_managed: bool) -> EPConfig:
    """COLLECTIVE: ep1 over the world."""
    return EPConfig(
        ep_size=1,
        stage_world_size=dist.get_world_size(),
        nvlink_domain_size=dist.get_world_size(),
        use_grouped_gemm=False,
        experts_fsdp_managed=experts_fsdp_managed,
    )


def _tensor_parallel(model: nn.Module, tp_mesh) -> None:
    """Megatron-style TP on each decoder layer: column-parallel in, row-parallel out."""
    plan = {f"self_attn.{name}": ColwiseParallel() for name in ("q_proj", "k_proj", "v_proj")}
    plan |= {f"mlp.{name}": ColwiseParallel() for name in ("gate_proj", "up_proj")}
    plan |= {"self_attn.o_proj": RowwiseParallel(), "mlp.down_proj": RowwiseParallel()}
    for layer in model.model.layers:
        parallelize_module(layer, tp_mesh, plan)


def _build(case: Case, tp_mesh) -> tuple[nn.Module, list[nn.Parameter]]:
    """``case``'s model under its gradient checkpointing, and the parameters the wrap leaves out."""
    family, overrides = FAMILIES[case.family]
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides={**overrides, "num_hidden_layers": case.layers, "vocab_size": VOCAB})
    ignored: list[nn.Parameter] = []
    if case.variant.startswith("ep_"):
        apply_ep_layers(model, _ep1_config(experts_fsdp_managed=case.variant == "ep_sharded"))
        if case.variant == "ep_ignored":
            # A bf16 run's load dtype: no FSDP2 cast reaches the ignored experts.
            model.to(torch.bfloat16)
            ignored = [p for _, layer in named_ep_layers(model) for p in layer.parameters()]
    elif case.variant == "nested":
        for layer in model.model.layers:
            layer.mlp.to(torch.float64)
    elif case.variant == "odd_layer":
        model.model.layers[1].to(torch.float64)
    elif case.variant == "fp32_rope":
        model.model.rotary_emb.register_forward_hook(lambda _module, _args, output: tuple(t.float() for t in output))
    if tp_mesh is not None:
        _tensor_parallel(model, tp_mesh)
    model.train()
    if case.gc != "off":
        kwargs = {"use_reentrant": case.gc == "reentrant", "every_n_layers": case.every_n_layers}
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs=kwargs)
    if case.variant == "lora":
        torch.manual_seed(1)
        config = LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS, init_lora_weights=False)
        model = get_peft_model(model, config)
    return model, ignored


def _wrap(case: Case, unit: int | None, mesh, tp_mesh=None) -> tuple[nn.Module, int]:
    """``case``'s model wrapped at ``unit`` layers per group (None: the wrap's own count); returns it and its group
    count."""
    model, ignored = _build(case, tp_mesh)
    policy = mixed_precision_policy(torch.bfloat16, fp32_reduce=True, cast_forward_inputs=True)
    groups = apply_fsdp2_per_layer(model, mesh, policy, case.raf, IdentityParamSet(ignored), layers_per_unit=unit)
    return model, groups


def _layout(model: nn.Module) -> list[list]:
    """Every state-dict entry's key, global shape, placements and this rank's local shape."""
    return [
        [key, list(value.shape), str(value.placements), list(value.to_local().shape)]
        if isinstance(value, DTensor)
        else [key, list(value.shape), None, list(value.shape)]
        for key, value in model.state_dict().items()
    ]


class _ReductionLog:
    """Each FSDP2 reduce-scatter as ``(issued after the backward, reduces a decoder-layer group)``: by the root's
    final callback or the trainer's completion rather than a post-backward hook, so nothing overlaps it. The
    groups the trainer's completion reduces are counted apart."""

    def __init__(self):
        self.reduces: list[tuple[bool, bool]] = []
        self.completed = 0
        self._after_backward = False

    @contextlib.contextmanager
    def after_backward(self):
        self._after_backward = True
        try:
            yield
        finally:
            self._after_backward = False

    @contextlib.contextmanager
    def watch(self):
        real_reduce = _fsdp_param_group.foreach_reduce
        real_callback = _fsdp_state.FSDPState._root_post_backward_final_callback

        def foreach_reduce(fsdp_params, *args, **kwargs):
            self.reduces.append((self._after_backward, ".layers." in fsdp_params[0]._param_fqn))
            return real_reduce(fsdp_params, *args, **kwargs)

        def callback(state):
            with self.after_backward():
                return real_callback(state)

        with (
            patch.object(_fsdp_param_group, "foreach_reduce", foreach_reduce),
            patch.object(_fsdp_state.FSDPState, "_root_post_backward_final_callback", callback),
        ):
            yield

    def take(self) -> tuple[int, int, int]:
        """``(reduce-scatters, decoder-layer ones issued after the backward, groups the completion reduced)`` since
        the last take."""
        counts = (len(self.reduces), sum(after and layer for after, layer in self.reduces), self.completed)
        self.reduces.clear()
        self.completed = 0
        return counts


def _set_window_end(model: nn.Module, case: Case, is_last: bool) -> None:
    """The trainer's per-microstep toggles (``_set_window_end``)."""
    for module in fsdp2_modules(model):
        if case.window == "keep":
            module.set_reshard_after_backward(is_last, recurse=False)
        elif case.window == "defer":
            module.set_requires_gradient_sync(is_last, recurse=False)


def _pipeline_backward(model: nn.Module, losses: list[torch.Tensor]) -> None:
    """torch 2.11's pipeline stage around an FSDP2 stage: each microbatch's backward with the stage's reduce,
    reshard and finalization off, then ``perform_reduce_grad``'s post-backward over every state."""
    for loss in losses:
        model.set_is_last_backward(False)
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        loss.backward()
    model.set_is_last_backward(True)
    model.set_reshard_after_backward(True)
    model.set_requires_gradient_sync(True)
    state = fully_shard.state(model)
    for each in state._state_ctx.all_states:
        if each._fsdp_param_group:
            each._fsdp_param_group.post_backward()
    state._root_post_backward_final_callback()


def _step(case: Case, model: nn.Module, step: int, log: _ReductionLog) -> float:
    """One optimizer step's backwards, each followed by the trainer's completion; returns the loss."""
    rank = dist.get_rank()
    groups = fsdp.fsdp2_param_groups(model)
    with log.watch():
        if case.pipeline:
            losses = [_loss(model, _rows(case, step, micro, rank)) for micro in range(MICROBATCHES)]
            _pipeline_backward(model, losses)
            return sum(float(loss) for loss in losses)
        microsteps = MICROBATCHES if case.window else 1
        total = 0.0
        for micro in range(microsteps):
            _set_window_end(model, case, micro == microsteps - 1)
            loss = _loss(model, _rows(case, step, micro, rank))
            loss.backward()
            with log.after_backward():
                log.completed += fsdp.complete_fsdp2_post_backward(groups)
            fsdp.reject_unfinished_fsdp2_backward(groups)
            total += float(loss)
    return total


def _settled(model: nn.Module) -> bool:
    """Whether every FSDP2 state is back to idle with no forward root or unfinished multi-module forward left over:
    a stale forward root skips the next forward's wait for the optimizer step."""
    states = {id(state): state for state in (module._get_fsdp_state() for module in fsdp2_modules(model))}.values()
    root = fully_shard.state(model)
    return (
        root._state_ctx.iter_forward_root is None
        and all(state._training_state == TrainingState.IDLE for state in states)
        and not any(state._modules_to_run_forward for state in states)
    )


def _train(case: Case, unit: int, mesh, tp_mesh=None) -> dict:
    """``STEPS`` SGD steps of ``case`` at ``unit`` layers per group: what a comparison across unit sizes reads."""
    model, groups = _wrap(case, unit, mesh, tp_mesh)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR)
    log = _ReductionLog()
    record = {"groups": groups, "param_groups": len(fsdp.fsdp2_param_groups(model)), "loss": [], "grads": []}
    record |= {"weights": [], "reduces": [], "layer_reduces_after_backward": [], "completed": [], "settled": []}
    for step in range(STEPS):
        record["loss"].append(_step(case, model, step, log))
        record["settled"].append(_settled(model))
        record["grads"].append([_digest(p.grad) if p.grad is not None else None for p in params])
        reduces, after, completed = log.take()
        record["reduces"].append(reduces)
        record["layer_reduces_after_backward"].append(after)
        record["completed"].append(completed)
        optimizer.step()
        model.zero_grad()
        record["weights"].append([_digest(p) for p in params])
    record["layout"] = _layout(model)
    return record


def _attempt(fn, *args, **kwargs) -> dict:
    """``fn``'s record, or the error it raised: every rank builds the same structure, so a raise is on all."""
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _torch_group_hooks_case(mesh) -> dict:
    """Negative control: units on torch's own multi-module hooks."""
    with patch.object(fsdp, "_install_unit_hooks", lambda unit: None):
        return _attempt(_train, Case(), 2, mesh)


def _no_recompute_registration_case(mesh) -> dict:
    """Negative control: the partly checkpointed unit without the first layer's recompute registration, the one
    call the unit hooks make while the group is in ``PRE_BACKWARD`` (FSDP2's own pre-forward makes it in
    ``FORWARD``)."""
    real = _fsdp_param_group.FSDPParamGroup._register_post_backward_hook

    def register(group, args, kwargs):
        if group._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        return real(group, args, kwargs)

    with patch.object(_fsdp_param_group.FSDPParamGroup, "_register_post_backward_hook", register):
        return _attempt(_train, Case(every_n_layers=2), 2, mesh)


def _dtype_blind_units_case(mesh) -> dict:
    """Negative control: the float64 layer grouped with its neighbor whatever its dtype."""

    def blind(layers, splits, left_out, layers_per_unit):
        return [list(range(len(layers)))[i : i + layers_per_unit] for i in range(0, len(layers), layers_per_unit)]

    with patch.object(fsdp, "_layer_units", blind):
        return _attempt(_train, Case(variant="odd_layer"), 2, mesh)


def _resume_case(mesh, tmp: Path) -> dict:
    """AdamW steps saved by the FSDP2 writer and the optimizer shard store at each unit size, every saved tensor
    digested; then one-layer groups resumed from each unit size's checkpoint take one more step, and so does ZeRO-3
    on the wrap's own groups resumed from the two-layer one."""
    case = Case(layers=3)
    rank = dist.get_rank()
    config = make_parallelism_config(world_size=WORLD, gpus_per_node=WORLD, rank=rank)
    out = {}

    def adamw(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)

    def adamw_step(model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
        _step(case, model, step, _ReductionLog())
        optimizer.step()
        model.zero_grad()

    for unit in (1, *UNIT_SIZES):
        model, _ = _wrap(case, unit, mesh)
        optimizer = adamw(model)
        for step in range(STEPS):
            adamw_step(model, optimizer, step)
        directory = tmp / f"unit{unit}"
        save_fsdp2_checkpoint(save_context(model, config, training_checkpoint=True), str(directory))
        OptimizerShardStore(shard_store_context(model, optimizer, config)).save(str(directory))
        shard = torch.load(directory / optimizer_shard_filename(rank), weights_only=False)
        out[f"unit{unit}"] = {
            "weights": {key: _digest(value) for key, value in safetensors_state_dict(directory).items()},
            "moments": {
                f"{name}.{key}": _digest(value)
                for name, state in shard["state"].items()
                for key, value in state.items()
                if torch.is_tensor(value)
            },
        }
        if unit == 1:
            adamw_step(model, optimizer, STEPS)
            out["continued"] = [_digest(p) for p in model.parameters()]

    def resumed(wrap_case: Case, unit: int | None, directory: str) -> list[str]:
        model, _ = _wrap(wrap_case, unit, mesh)
        optimizer = adamw(model)
        load_resume_weights(load_context(model, config, fsdp_wrapped=True), directory, model)
        OptimizerShardStore(shard_store_context(model, optimizer, config)).load(directory)
        adamw_step(model, optimizer, STEPS)
        return [_digest(p) for p in model.parameters()]

    for unit in (1, *UNIT_SIZES):
        out[f"resumed_from_unit{unit}"] = resumed(case, 1, str(tmp / f"unit{unit}"))
    out["zero3_resumed_from_unit2"] = resumed(dataclasses.replace(case, raf=True), None, str(tmp / "unit2"))
    return out


def _worker(rank: int, out: str) -> None:
    mesh = init_device_mesh("cpu", (WORLD,))
    results = {"cases": {}}
    for case in CASES:
        for unit in (1, *UNIT_SIZES):
            results["cases"][f"{case.name}|{unit}"] = _attempt(_train, case, unit, mesh)
    results["torch_group_hooks"] = _torch_group_hooks_case(mesh)
    results["no_recompute_registration"] = _no_recompute_registration_case(mesh)
    results["dtype_blind_units"] = _dtype_blind_units_case(mesh)
    results["resume"] = _attempt(_resume_case, mesh, Path(f"{out}.checkpoints"))
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def _mesh_4_worker(rank: int, out: str) -> None:
    """HSDP (2 replicas of 2 shards), and FSDP2 over the DP dimension of a TP x DP mesh, on 4 ranks."""
    hsdp_mesh = init_device_mesh("cpu", MESH_4, mesh_dim_names=("dp_replicate", "dp_shard"))
    dp_tp_mesh = init_device_mesh("cpu", MESH_4, mesh_dim_names=("dp", "tp"))
    results = {}
    for case in MESH_4_CASES:
        for unit in (1, *UNIT_SIZES):
            results[f"hsdp|{case.name}|{unit}"] = _attempt(_train, case, unit, hsdp_mesh)
            results[f"tp|{case.name}|{unit}"] = _attempt(_train, case, unit, dp_tp_mesh["dp"], dp_tp_mesh["tp"])
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def _spawn(worker, tmp_path_factory, name: str, world: int) -> list[dict]:
    out = str(tmp_path_factory.mktemp(name) / "runs")
    run_gloo_ranks(worker, world, out, pg_timeout=PG_TIMEOUT)
    per_rank = []
    for rank in range(world):
        with open(f"{out}.{rank}") as fh:
            per_rank.append(json.load(fh))
    return per_rank


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _spawn(_worker, tmp_path_factory, "units", WORLD)


@pytest.fixture(scope="module")
def mesh_4_runs(tmp_path_factory):
    return _spawn(_mesh_4_worker, tmp_path_factory, "units_mesh4", math.prod(MESH_4))


@pytest.fixture
def single_rank_mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _ok(record: dict, name: str) -> dict:
    if "error" in record:
        pytest.fail(f"{name} raised: {record['error']}")
    return record


def _assert_same_training(results: dict, prefix: str) -> None:
    one = _ok(results[f"{prefix}|1"], f"{prefix}|1")
    for unit in UNIT_SIZES:
        run = _ok(results[f"{prefix}|{unit}"], f"{prefix}|{unit}")
        for key in ("loss", "grads", "weights", "layout"):
            assert run[key] == one[key], f"{prefix} at {unit} layers per group: {key} differs from one-layer groups"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_units_train_bit_for_bit_like_one_layer_groups(runs, case):
    """Loss, every sharded gradient and weight after each step, and the state dict's keys, global shapes,
    placements and local shapes equal one-layer groups' on every rank."""
    for rank_runs in runs:
        _assert_same_training(rank_runs["cases"], case.name)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_fsdp2_state_settles_after_the_backward(runs, case):
    """After each step's backward every FSDP2 state is idle, with no forward root and no multi-module forward left
    open, at every unit size."""
    for rank_runs in runs:
        for unit in (1, *UNIT_SIZES):
            run = _ok(rank_runs["cases"][f"{case.name}|{unit}"], case.name)
            assert run["settled"] == [True] * STEPS, (unit, run["settled"])


@pytest.mark.parametrize("case", DENSE_CASES, ids=lambda case: case.name)
def test_each_unit_is_one_group_reduced_once_per_pass(runs, case):
    """ceil(layers / unit) decoder-layer groups beside the embedding and root groups, the modules of a unit naming
    one group. Each group reduce-scatters once per backward, a decoder-layer group from its post-backward hook
    inside the backward; at several rows per backward each decoder-layer group reduces once more after it, for
    the recomputes FSDP2's hooks did not cover, as one-layer groups do. torch's final callback visits a unit's
    state once per module and reduces those leftovers itself, so the trainer's completion reduces only the
    one-layer groups'."""
    for rank_runs in runs:
        for unit in (1, *UNIT_SIZES):
            run = _ok(rank_runs["cases"][f"{case.name}|{unit}"], case.name)
            layer_groups = math.ceil(case.layers / unit)
            assert run["groups"] == run["param_groups"] == layer_groups + 2, (unit, run["groups"], run["param_groups"])
            after = 0 if case.rows == 1 else layer_groups
            assert run["layer_reduces_after_backward"] == [after] * STEPS, (unit, run["layer_reduces_after_backward"])
            assert run["reduces"] == [layer_groups + 2 + after] * STEPS, (unit, run["reduces"])
            one_layer_groups = case.layers if unit == 1 else int(case.layers % unit == 1)
            completed = 0 if case.rows == 1 else one_layer_groups
            assert run["completed"] == [completed] * STEPS, (unit, run["completed"])


def test_nested_groups_leave_their_layers_in_one_unit(runs):
    """Three layers, each MLP a nested float64 group: the layers' own float32 parameters decide their units, so
    two units at two layers per group (and one at three) beside the three nested, embedding and root groups."""
    case = Case(variant="nested", layers=3)
    for rank_runs in runs:
        for unit in (1, *UNIT_SIZES):
            run = _ok(rank_runs["cases"][f"{case.name}|{unit}"], case.name)
            expected = math.ceil(case.layers / unit) + case.layers + 2
            assert run["groups"] == run["param_groups"] == expected, (unit, run["groups"], run["param_groups"])


def test_a_layer_of_another_dtype_opens_its_own_unit(runs):
    """The float64 layer 1 of four splits the first pair: units [0], [1], [2, 3] at two layers per group."""
    for rank_runs in runs:
        run = _ok(rank_runs["cases"][f"{Case(variant='odd_layer').name}|2"], "odd_layer")
        assert run["groups"] == 3 + 2, run["groups"]


@pytest.mark.parametrize("case", PARTIAL_GC_CASES, ids=lambda case: case.name)
def test_a_partly_checkpointed_unit_reduces_inside_the_backward(runs, case):
    """Every other layer checkpointed: a unit whose first layer alone is recomputed still reduce-scatters from its
    post-backward hook, beside the rest of the backward, not from the final callback after it."""
    for rank_runs in runs:
        for unit in (1, *UNIT_SIZES):
            run = _ok(rank_runs["cases"][f"{case.name}|{unit}"], case.name)
            assert run["layer_reduces_after_backward"] == [0] * STEPS, (unit, run["layer_reduces_after_backward"])


def test_negative_control_torch_group_hooks_crash_a_reentrant_recompute(runs):
    """torch starts the unit at its last layer's recompute and reshards it after that layer's backward, so the
    first layer recomputes on sharded parameters."""
    for rank_runs in runs:
        error = rank_runs["torch_group_hooks"].get("error", "")
        assert "mixed torch.Tensor and DTensor" in error, rank_runs["torch_group_hooks"]


def test_negative_control_without_the_recompute_registration_the_reduce_waits_for_the_callback(runs):
    """The partly checkpointed reentrant run minus the first layer's registration: the units' reduce-scatters move
    to the final callback, which the inside-the-backward assertion above refuses."""
    for rank_runs in runs:
        run = _ok(rank_runs["no_recompute_registration"], "no_recompute_registration")
        assert all(count > 0 for count in run["layer_reduces_after_backward"]), run["layer_reduces_after_backward"]


def test_negative_control_a_unit_mixing_dtypes_fails_fsdp2s_dtype_check(runs):
    for rank_runs in runs:
        error = rank_runs["dtype_blind_units"].get("error", "")
        assert "uniform original parameter dtype" in error, rank_runs["dtype_blind_units"]


def test_the_saved_checkpoint_is_the_same_at_every_unit_size(runs):
    """The FSDP2 writer's tensors and this rank's optimizer-shard moments, tensor for tensor."""
    for rank_runs in runs:
        resume = _ok(rank_runs["resume"], "resume")
        assert resume["unit1"]["weights"] and resume["unit1"]["moments"]
        for unit in UNIT_SIZES:
            assert resume[f"unit{unit}"] == resume["unit1"], unit


def test_a_run_resumes_across_unit_sizes(runs):
    """One-layer groups resumed from each unit size's checkpoint, ZeRO-3 resumed from the two-layer ZeRO-2 one
    included, take the step the uninterrupted run takes."""
    for rank_runs in runs:
        resume = _ok(rank_runs["resume"], "resume")
        for unit in (1, *UNIT_SIZES):
            assert resume[f"resumed_from_unit{unit}"] == resume["continued"], unit
        assert resume["zero3_resumed_from_unit2"] == resume["continued"]


@pytest.mark.parametrize("mesh", ["hsdp", "tp"])
@pytest.mark.parametrize("case", MESH_4_CASES, ids=lambda case: case.name)
def test_units_on_hsdp_and_tp_meshes_train_bit_for_bit(mesh_4_runs, mesh, case):
    for rank_runs in mesh_4_runs:
        _assert_same_training(rank_runs, f"{mesh}|{case.name}")


@pytest.mark.parametrize(("raf", "expected"), [(False, [[0, 1], [2, 3]]), (True, [[0], [1], [2], [3]])])
def test_the_wrap_groups_layers_under_zero2_and_not_under_zero3(single_rank_mesh, raf, expected):
    """The wrap's own unit count: two layers per group under ZeRO-2, one under ZeRO-3, whose backward would hold
    the running and the prefetched unit gathered."""
    model, _ = _build(Case(gc="off"), None)
    policy = mixed_precision_policy(torch.bfloat16, fp32_reduce=True, cast_forward_inputs=True)
    groups = apply_fsdp2_per_layer(model, single_rank_mesh, policy, raf, IdentityParamSet())
    states = [layer._get_fsdp_state() for layer in model.model.layers]
    units = [[index for index, state in enumerate(states) if state is each] for each in dict.fromkeys(states)]
    assert units == expected and groups == len(expected) + 2, (units, groups)


def test_the_unit_hooks_internals_exist(single_rank_mesh):
    """The FSDP2 members the unit hooks drive, on the installed torch: the multi-module hook handle they remove, the
    state's forward hooks, policy and training state they read, and the group's post-backward registration,
    unshard and training state. The input cast they share with FSDP2's pre-forward is pinned by the fp32_rope
    cases."""
    layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(2))
    fully_shard(list(layers), mesh=single_rank_mesh)
    state = layers[0]._get_fsdp_state()
    assert state is layers[1]._get_fsdp_state() and isinstance(layers[1], FSDPModule)
    assert state._post_forward_hook_handle is state._pre_forward_hook_handle
    assert callable(state._pre_forward) and callable(state._post_forward)
    assert isinstance(state._mp_policy, MixedPrecisionPolicy) and isinstance(state._training_state, TrainingState)
    group = state._fsdp_param_group
    assert (
        callable(group._register_post_backward_hook) and callable(group.unshard) and callable(group.wait_for_unshard)
    )
    assert isinstance(group.is_unsharded, bool) and isinstance(group._training_state, TrainingState)
    state._pre_forward_hook_handle.remove()
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in layers)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
