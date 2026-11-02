#!/usr/bin/env python
"""A ``fully_shard`` layer recomputed K>1 times in one backward reaches the optimizer with every segment's
gradient, through the trainer's own ``training_step``, on 2 gloo ranks.

Halo runs MoE, EP and CP gradient checkpointing reentrant, around each decoder layer's ``__call__``, and
env-GRPO scores every turn row in its own forward, so one backward recomputes each layer K = rows times.
torch 2.11's FSDP2 registers its post-backward hook in only one of those recomputes: the other K-1 segments
stay in the unsharded gradient, the layer groups stay gathered past the optimizer step, and their sharded
gradients survive ``model.zero_grad()``. ``DistributedTrainerMixin.training_step`` completes that
post-backward after the backward and refuses one still undone.

Two families run through ``apply_fsdp2_per_layer`` and scoped reentrant checkpointing: a tiny dense Qwen3
(also with its MLPs held in another dtype, so each layer carries a nested shard group) and a tiny Qwen3-MoE
under EP layers at ep1 with FSDP-sharded experts; the dense one also under HSDP on 4 ranks. Each is trained 3 SGD
steps beside an unsharded model
without checkpointing that averages its gradients over the ranks, with HF's step order (backward per
microbatch, clip, step, ``model.zero_grad()``). At K=1 the step must leave every gradient and weight bit for
bit what the same step without the completion leaves. The non-finite step guard reads its verdict off the
completed gradient: a K>1 step whose backward overflows on one rank is skipped everywhere, and nothing of it
reaches a later step.

    python tests/cpu/parallelism/test_fsdp2_reentrant_recompute.py
"""

import contextlib
import datetime
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import _fsdp_param_group
from torch.distributed.tensor import DTensor
from transformers import Trainer

from halo.distributed import fsdp
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, fsdp2_modules, mixed_precision_policy
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism import grad_sync
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model

WORLD = 2
PG_TIMEOUT = datetime.timedelta(seconds=120)
SEED = 0
VOCAB = 128
STEPS = 3
LR = 0.05
# The clip threshold is this fraction of the reference's own norm, so every step's clip scales.
CLIP_FRACTION = 0.5
# Forwards of each decoder layer per backward: env-GRPO's turn rows per microbatch.
K_VALUES = (1, 2, 4)
WINDOW_K_VALUES = (1, 4)
GRAD_ACCUM = 3
# fp32 throughout: FSDP2's reduce-scatter and the reference's all-reduce sum in different orders (observed
# below 1e-7); the dropped segments this guards against are 30-80% of the layer gradient.
PARITY_RTOL = 1e-5
FAMILY_OVERRIDES = {
    "dense": {"num_hidden_layers": 2, "vocab_size": VOCAB},
    "moe_ep1": {
        "num_hidden_layers": 2,
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "head_dim": 16,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_experts": 4,
    },
}
# Each window arms FSDP2's post-backward work only on the window's last backward (``_set_window_end``).
WINDOWS = {
    "defer": {"fsdp_reshard_after_backward": True, "fsdp_defer_grad_sync": True},
    "keep_gathered": {"fsdp_reshard_after_backward": False, "fsdp_defer_grad_sync": False},
    "both": {"fsdp_reshard_after_backward": False, "fsdp_defer_grad_sync": True},
}
NO_WINDOW = {"fsdp_reshard_after_backward": True, "fsdp_defer_grad_sync": False}
# HSDP over 4 ranks: 2 replicas of 2 shards, each group's reduce a reduce-scatter then an all-reduce.
HSDP_MESH = (2, 2)
HSDP_K_VALUES = (1, 4)
# The step whose backward overflows on the last rank, and the rows each of its backwards recomputes.
POISONED_STEP = 1
POISONED_K = 4


def _rows(k: int, step: int, micro: int, rank: int) -> list[torch.Tensor]:
    """``k`` token rows of different lengths, distinct per step, microbatch and rank."""
    generator = torch.Generator().manual_seed(10_000 * step + 100 * micro + rank)
    lengths = [5 + 3 * i + int(torch.randint(0, 4, (1,), generator=generator)) for i in range(k)]
    return [torch.randint(0, VOCAB, (1, length), generator=generator) for length in lengths]


def _rows_loss(model: nn.Module, rows: list[torch.Tensor]) -> torch.Tensor:
    """Next-token cross entropy summed over the rows, each row its own forward."""
    loss = 0.0
    for ids in rows:
        logits = model(input_ids=ids).logits
        loss = loss + F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1), reduction="sum")
    return loss


class _InfGrad(torch.autograd.Function):
    """Identity forward, infinite backward: a finite loss whose every gradient overflows."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * math.inf


class _PoisonedRows(list):
    """Rows whose backward overflows."""


class _RowsBackward:
    """The step the trainer seam wraps: one backward through every row's forward. The step guard extends
    transformers' own norm read."""

    _get_grad_norm = Trainer._get_grad_norm

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = _rows_loss(model, inputs)
        if isinstance(inputs, _PoisonedRows):
            loss = _InfGrad.apply(loss)
        loss.backward()
        return loss.detach()


class _Trainer(DistributedTrainerMixin, _RowsBackward):
    """``DistributedTrainerMixin.training_step`` and its clips over the minimum state they read."""

    def __init__(self, model: nn.Module, ep_config: EPConfig | None, window: dict):
        self.model = model
        self.parallelism_config = SimpleNamespace(is_tp_mode=False, **window)
        self.accelerator = SimpleNamespace(sync_gradients=True, distributed_type=DistributedType.NO)
        self.state = SimpleNamespace(global_step=0)
        self.optimizer = None
        self._ep_config = ep_config
        self._fsdp_wrapped = True
        self._pp_runner = None
        self._pp_stage_group = None
        self._pp_chain_group = None
        self._device_mesh = None
        self._memory_margin_checked = True
        self._attention_work_bound = True
        self._attention_layout = None
        self._param_id_caches = {}
        self._install_grad_accum_window()
        if ep_config is not None:
            self._install_ep_clip()

    def clip(self, max_norm: float) -> torch.Tensor:
        """The clip HF runs on the sync step: the EP clip under EP layers, torch's own on plain FSDP2."""
        if self._ep_config is not None:
            return self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)


def _param_groups(model: nn.Module) -> list:
    """Each FSDP2 module's parameter group, read off torch's own state rather than the code under test."""
    groups = (module._get_fsdp_state()._fsdp_param_group for module in fsdp2_modules(model))
    return [group for group in groups if group is not None]


def _is_layer_group(group) -> bool:
    return ".layers." in repr(group)


def _build(family: str, ep_config: EPConfig) -> nn.Module:
    torch.manual_seed(SEED)
    if family == "dense":
        model = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"], overrides=FAMILY_OVERRIDES[family])
    else:
        model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"], overrides=FAMILY_OVERRIDES[family])
        apply_ep_layers(model, ep_config)
    return model.train()


def _sharded(family: str, ep_config: EPConfig, mesh, *, raf: bool, nested: bool) -> nn.Module:
    """``family``'s model under scoped reentrant checkpointing and the toolkit's per-layer FSDP2 wrap; ``nested``
    holds each MLP in float64, which the wrap splits into a shard group inside its layer's."""
    model = _build(family, ep_config)
    if nested:
        for layer in model.model.layers:
            layer.mlp.to(torch.float64)
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": True})
    policy = mixed_precision_policy(torch.float32, fp32_reduce=True, cast_forward_inputs=True)
    apply_fsdp2_per_layer(model, mesh, policy, reshard_after_forward=raf, ignored_params=IdentityParamSet())
    return model


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """COLLECTIVE for a DTensor."""
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().clone()


def _grads(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros(p.shape) if p.grad is None else _full(p.grad) for p in params]


def _rel(actual: list[torch.Tensor], reference: list[torch.Tensor]) -> float:
    num = sum(((a - r) ** 2).sum() for a, r in zip(actual, reference, strict=True))
    den = sum((r**2).sum() for r in reference)
    return math.sqrt(float(num) / float(den))


def _average(params: list[nn.Parameter]) -> None:
    """The reference trains data-parallel too: its grads averaged over the ranks in place."""
    for param in params:
        dist.all_reduce(param.grad)
        param.grad /= dist.get_world_size()


class _ReductionLog:
    """FSDP2's reduce-scatters per group, and every reduction event the reduce-scatter stream records, against
    the waits the compute stream makes. On CPU an event is ``None``; a token stands in for each, so a reduction
    whose completion the compute stream never waited on shows up as an unwaited token."""

    def __init__(self, groups: list):
        self._owner = {id(fsdp_param): id(group) for group in groups for fsdp_param in group.fsdp_params}
        # Read at each event: the root's lazy init gives every group the shared communication context.
        self._group = groups[0]
        self.reduces: dict[int, int] = {}
        self.recorded: list[object] = []
        self.waited: set[int] = set()

    @contextlib.contextmanager
    def watch(self):
        real_reduce = _fsdp_param_group.foreach_reduce
        compute_stream = torch.cpu.current_stream()

        def foreach_reduce(fsdp_params, *args, **kwargs):
            owner = self._owner[id(fsdp_params[0])]
            self.reduces[owner] = self.reduces.get(owner, 0) + 1
            return real_reduce(fsdp_params, *args, **kwargs)

        def record_event(stream):
            if stream is not getattr(self._group.comm_ctx, "reduce_scatter_stream", None):
                return None
            token = object()
            self.recorded.append(token)
            return token

        def wait_event(stream, event):
            if stream is compute_stream and event is not None:
                self.waited.add(id(event))

        with (
            patch.object(_fsdp_param_group, "foreach_reduce", foreach_reduce),
            patch.object(torch.cpu.Stream, "record_event", record_event),
            patch.object(torch.cpu.Stream, "wait_event", wait_event),
        ):
            yield

    def take(self) -> tuple[dict[int, int], int, int]:
        """This window's reduce-scatter counts by group ``id``, the reductions recorded and those left unwaited;
        then reset."""
        recorded, unwaited = len(self.recorded), sum(1 for token in self.recorded if id(token) not in self.waited)
        reduces, self.reduces, self.recorded, self.waited = self.reduces, {}, [], set()
        return reduces, recorded, unwaited


def _train(
    family: str,
    ep_config: EPConfig,
    mesh,
    *,
    k: int,
    raf: bool,
    micro: int,
    nested: bool = False,
    window: dict = NO_WINDOW,
    complete: bool = True,
) -> dict:
    """``STEPS`` optimizer steps of ``micro`` microbatches of ``k`` rows through the trainer's step; ``complete``
    off runs the same step without the post-backward completion (the K=1 twin). Every collective runs on
    every rank in the same order, before anything is compared."""
    reference = _build(family, ep_config)
    model = _sharded(family, ep_config, mesh, raf=raf, nested=nested)
    trainer = _Trainer(model, None if family == "dense" else ep_config, window)
    if not complete:
        trainer._finish_fsdp2_backward = lambda model: None
    names = [name for name, _ in reference.named_parameters()]
    ref_params = list(reference.parameters())
    params = [param for _, param in model.named_parameters()]
    is_layer = [".layers." in name for name in names]
    trainer.optimizer = torch.optim.SGD(params, lr=LR)
    ref_optimizer = torch.optim.SGD(ref_params, lr=LR)
    groups = _param_groups(model)
    log = _ReductionLog(groups)
    record = {"grad_rel": [], "layer_grad_rel": [], "norm_rel": [], "clipped_rel": [], "sharded_after_step": []}
    record |= {"micro": [], "recorded": 0, "unwaited": 0, "grads": [], "weights": []}
    for step in range(STEPS):
        trainer.state.global_step = step
        for index in range(micro):
            rows = _rows(k, step, index, dist.get_rank())
            trainer.accelerator.sync_gradients = index == micro - 1
            with log.watch():
                trainer.training_step(model, rows)
            reduces, recorded, unwaited = log.take()
            record["recorded"] += recorded
            record["unwaited"] += unwaited
            record["micro"].append(
                {
                    "sync": trainer.accelerator.sync_gradients,
                    "layer_reduces": [reduces.get(id(group), 0) for group in groups if _is_layer_group(group)],
                    "any_reduce": sum(reduces.values()),
                    "layers_gathered": [group.is_unsharded for group in groups if _is_layer_group(group)],
                }
            )
            _rows_loss(reference, rows).backward()
        _average(ref_params)
        grads, ref_grads = _grads(params), [p.grad.detach().clone() for p in ref_params]
        record["grad_rel"].append(_rel(grads, ref_grads))
        record["layer_grad_rel"].append(
            _rel(
                [g for g, layer in zip(grads, is_layer, strict=True) if layer],
                [r for r, layer in zip(ref_grads, is_layer, strict=True) if layer],
            )
        )
        ref_norm = torch.nn.utils.get_total_norm([p.grad for p in ref_params])
        max_norm = CLIP_FRACTION * float(ref_norm)
        torch.nn.utils.clip_grad_norm_(ref_params, max_norm)
        norm = _full(trainer.clip(max_norm))
        record["norm_rel"].append(abs(float(norm) - float(ref_norm)) / float(ref_norm))
        record["clipped_rel"].append(_rel(_grads(params), [p.grad for p in ref_params]))
        if not complete or k == 1:
            record["grads"].append([g.tolist() for g in _grads(params)])
        trainer.optimizer.step()
        ref_optimizer.step()
        model.zero_grad()
        reference.zero_grad()
        record["sharded_after_step"].append(
            all(group.is_sharded for group in groups) and all(isinstance(p, DTensor) for p in model.parameters())
        )
        if not complete or k == 1:
            record["weights"].append([_full(p).tolist() for p in params])
    model.eval()
    reference.eval()
    with torch.no_grad():
        rows = _rows(k, STEPS, 0, dist.get_rank())
        logits = [model(input_ids=ids).logits.flatten() for ids in rows]
        ref_logits = [reference(input_ids=ids).logits.flatten() for ids in rows]
    record["logit_rel"] = _rel(logits, ref_logits)
    return record


def _skipped_step_run(mesh) -> dict:
    """``STEPS`` steps of the dense model at ``POISONED_K`` rows per backward, ``POISONED_STEP``'s backward
    overflowing on the last rank, through the trainer's step and the step guard on its optimizer; the reference
    takes no step there."""
    rank = dist.get_rank()
    reference = _build("dense", None)
    model = _sharded("dense", None, mesh, raf=False, nested=False)
    trainer = _Trainer(model, None, NO_WINDOW)
    ref_params = list(reference.parameters())
    params = list(model.parameters())
    trainer.optimizer = torch.optim.SGD(params, lr=LR)
    trainer._tr_loss, trainer._tr_loss_before_step = torch.zeros(()), torch.zeros(())
    trainer._install_nonfinite_step_skip()
    ref_optimizer = torch.optim.SGD(ref_params, lr=LR)
    record = {"grad_rel": [], "weights": []}
    for step in range(STEPS):
        trainer.state.global_step = step
        rows = _rows(POISONED_K, step, 0, rank)
        poisoned = step == POISONED_STEP
        trainer.training_step(model, _PoisonedRows(rows) if poisoned and rank == WORLD - 1 else rows)
        if not poisoned:
            _rows_loss(reference, rows).backward()
            _average(ref_params)
            record["grad_rel"].append(_rel(_grads(params), [p.grad for p in ref_params]))
        trainer._get_grad_norm(model, grad_norm=trainer.clip(1.0))
        trainer.optimizer.step()
        if not poisoned:
            torch.nn.utils.clip_grad_norm_(ref_params, 1.0)
            ref_optimizer.step()
        model.zero_grad()
        reference.zero_grad()
        record["weights"].append([_full(p).tolist() for p in params])
    model.eval()
    reference.eval()
    with torch.no_grad():
        rows = _rows(POISONED_K, STEPS, 0, rank)
        record["logit_rel"] = _rel(
            [model(input_ids=ids).logits.flatten() for ids in rows],
            [reference(input_ids=ids).logits.flatten() for ids in rows],
        )
    record["skipped"] = trainer._nonfinite_steps_skipped
    return record


def _guard_cases(ep_config: EPConfig, mesh) -> dict:
    """The guard on a backward the completion cannot finish, and on each leftover injected by hand."""
    out = {}
    model = _sharded("dense", ep_config, mesh, raf=False, nested=False)
    trainer = _Trainer(model, ep_config=None, window=NO_WINDOW)
    try:
        with patch.object(grad_sync, "complete_fsdp2_post_backward", lambda groups: 0):
            trainer.training_step(model, _rows(4, 0, 0, dist.get_rank()))
        out["uncompleted_backward"] = None
    except RuntimeError as error:
        out["uncompleted_backward"] = str(error)
    for injected in ("unreduced", "gathered"):
        model = _sharded("dense", ep_config, mesh, raf=False, nested=False)
        trainer = _Trainer(model, ep_config=None, window=NO_WINDOW)
        trainer.training_step(model, _rows(1, 0, 0, dist.get_rank()))
        layers = [
            module for module in fsdp2_modules(model) if ".layers." in repr(module._get_fsdp_state()._fsdp_param_group)
        ]
        group = layers[0]._get_fsdp_state()._fsdp_param_group
        if injected == "unreduced":
            fsdp_param = group.fsdp_params[0]
            fsdp_param.unsharded_param.grad = torch.ones_like(fsdp_param.unsharded_param)
        else:
            layers[0].unshard()
        try:
            fsdp.reject_unfinished_fsdp2_backward(_param_groups(model))
            out[injected] = None
        except RuntimeError as error:
            out[injected] = str(error)
    return out


def _scenarios() -> dict[str, dict]:
    cases = {}
    for family in FAMILY_OVERRIDES:
        for raf in (False, True):
            for micro in (1, GRAD_ACCUM):
                for nested in (False, True) if family == "dense" else (False,):
                    shape = {"family": family, "raf": raf, "micro": micro, "nested": nested}
                    prefix = f"{family}-raf{raf}-micro{micro}-nested{nested}"
                    for k in K_VALUES:
                        cases[f"{prefix}-k{k}"] = {**shape, "k": k}
                    cases[f"{prefix}-k1-twin"] = {**shape, "k": 1, "complete": False}
        for name, window in WINDOWS.items():
            for k in WINDOW_K_VALUES:
                cases[f"{family}-window_{name}-k{k}"] = {
                    "family": family,
                    "k": k,
                    "raf": False,
                    "micro": GRAD_ACCUM,
                    "window": window,
                }
    return cases


def _ep1_config() -> EPConfig:
    """COLLECTIVE: ep1 over the 2 ranks, experts sharded by FSDP2 (``fsdp_shard_ep1_experts``)."""
    return EPConfig(
        ep_size=1, stage_world_size=WORLD, nvlink_domain_size=WORLD, use_grouped_gemm=False, experts_fsdp_managed=True
    )


def _worker(rank: int, out: str) -> None:
    mesh = init_device_mesh("cpu", (WORLD,))
    ep_config = _ep1_config()
    results = {}
    for name, case in _scenarios().items():
        # Both ranks hold the same structure, so a run that raises raises on both before diverging.
        try:
            results[name] = _train(ep_config=ep_config, mesh=mesh, **case)
        except Exception as error:
            results[name] = {"error": f"{type(error).__name__}: {error}"}
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def _guard_worker(rank: int, out: str) -> None:
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(_guard_cases(_ep1_config(), init_device_mesh("cpu", (WORLD,))), fh)


def _skip_worker(rank: int, out: str) -> None:
    try:
        result = _skipped_step_run(init_device_mesh("cpu", (WORLD,)))
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    with open(f"{out}.{rank}", "w") as fh:
        json.dump({"completed": result}, fh)


def _hsdp_worker(rank: int, out: str) -> None:
    """The dense model under HSDP, 2 replicas of 2 shards: a completion's reduce-scatter and all-reduce."""
    mesh = init_device_mesh("cpu", HSDP_MESH, mesh_dim_names=("dp_replicate", "dp_shard"))
    results = {}
    for k in HSDP_K_VALUES:
        for complete in (True, False) if k == 1 else (True,):
            name = f"k{k}" if complete else f"k{k}-twin"
            try:
                results[name] = _train("dense", None, mesh, k=k, raf=False, micro=1, complete=complete)
            except Exception as error:
                results[name] = {"error": f"{type(error).__name__}: {error}"}
    with open(f"{out}.{rank}", "w") as fh:
        json.dump(results, fh)


def _spawn(worker, tmp_path_factory, name: str, world: int = WORLD) -> list[dict]:
    out = str(tmp_path_factory.mktemp(name) / "runs")
    run_gloo_ranks(worker, world, out, pg_timeout=PG_TIMEOUT)
    per_rank = []
    for rank in range(world):
        with open(f"{out}.{rank}") as fh:
            per_rank.append(json.load(fh))
    return per_rank


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _spawn(_worker, tmp_path_factory, "reentrant")


@pytest.fixture(scope="module")
def guard_runs(tmp_path_factory):
    return _spawn(_guard_worker, tmp_path_factory, "guard")


@pytest.fixture(scope="module")
def skip_runs(tmp_path_factory):
    return _spawn(_skip_worker, tmp_path_factory, "skip")


@pytest.fixture(scope="module")
def hsdp_runs(tmp_path_factory):
    return _spawn(_hsdp_worker, tmp_path_factory, "hsdp", world=HSDP_MESH[0] * HSDP_MESH[1])


def _run(rank_runs: dict, name: str) -> dict:
    run = rank_runs[name]
    if "error" in run:
        pytest.fail(f"{name} raised: {run['error']}")
    return run


MAIN = [
    f"{family}-raf{raf}-micro{micro}-nested{nested}"
    for family in FAMILY_OVERRIDES
    for raf in (False, True)
    for micro in (1, GRAD_ACCUM)
    for nested in ((False, True) if family == "dense" else (False,))
]


@pytest.mark.parametrize("case", MAIN)
def test_every_recomputed_segment_reaches_the_sharded_gradient(runs, case):
    """Each step's gradient, read off the sharded parameters the optimizer holds, is the reference's: the K-1
    segments FSDP2's hooks missed are reduce-scattered too, nested groups included."""
    for rank_runs in runs:
        for k in K_VALUES:
            run = _run(rank_runs, f"{case}-k{k}")
            assert max(run["layer_grad_rel"]) < PARITY_RTOL, (k, run["layer_grad_rel"])
            assert max(run["grad_rel"]) < PARITY_RTOL, (k, run["grad_rel"])


@pytest.mark.parametrize("case", MAIN)
def test_one_forward_per_backward_leaves_the_step_bit_identical(runs, case):
    """At K=1 the completion is a no-op: every gradient and weight equals the step without it, and the step
    issues one reduce-scatter per layer group; a K>1 step costs exactly one more."""
    for rank_runs in runs:
        run, twin = _run(rank_runs, f"{case}-k1"), _run(rank_runs, f"{case}-k1-twin")
        assert run["grads"] == twin["grads"]
        assert run["weights"] == twin["weights"]
        for k in K_VALUES:
            sync_steps = [m for m in _run(rank_runs, f"{case}-k{k}")["micro"] if m["sync"]]
            assert all(m["layer_reduces"] == [1 if k == 1 else 2] * len(m["layer_reduces"]) for m in sync_steps), k


@pytest.mark.parametrize("case", MAIN)
def test_clip_norms_the_gradient_the_optimizer_steps(runs, case):
    """The sync step's clip returns the reference's total norm and scales the sharded gradients to its
    clipped ones (torch's clip on plain FSDP2, the EP clip under EP layers)."""
    for rank_runs in runs:
        for k in K_VALUES:
            run = _run(rank_runs, f"{case}-k{k}")
            assert max(run["norm_rel"]) < PARITY_RTOL, (k, run["norm_rel"])
            assert max(run["clipped_rel"]) < PARITY_RTOL, (k, run["clipped_rel"])


@pytest.mark.parametrize("case", MAIN)
def test_no_gradient_carries_across_model_zero_grad(runs, case):
    """Steps 2 and 3 match the reference too: HF's ``model.zero_grad()`` clears the registered parameters only,
    so a group left gathered keeps its sharded gradient into the next step."""
    for rank_runs in runs:
        for k in K_VALUES:
            run = _run(rank_runs, f"{case}-k{k}")
            assert max(run["grad_rel"][1:]) < PARITY_RTOL, (k, run["grad_rel"])


@pytest.mark.parametrize("case", MAIN)
def test_groups_reshard_and_the_next_forward_sees_the_step(runs, case):
    """After every step each group is sharded and registers its DTensor shards, and a forward after the last
    step computes with the stepped weights, not a gathered pre-step copy."""
    for rank_runs in runs:
        for k in K_VALUES:
            run = _run(rank_runs, f"{case}-k{k}")
            assert all(run["sharded_after_step"]), (k, run["sharded_after_step"])
            assert run["logit_rel"] < PARITY_RTOL, (k, run["logit_rel"])


@pytest.mark.parametrize("case", MAIN)
def test_the_compute_stream_waits_on_every_reduction(runs, case):
    """Every reduction the step issues, the completion's extra reduce-scatter per layer group included, is
    waited on by the compute stream before the step returns, where the clip and the optimizer read it."""
    for rank_runs in runs:
        for k in K_VALUES[1:]:
            run = _run(rank_runs, f"{case}-k{k}")
            assert run["recorded"] > 0 and run["unwaited"] == 0, (k, run["recorded"], run["unwaited"])
            assert all(m["layer_reduces"] == [2] * len(m["layer_reduces"]) for m in run["micro"]), k


@pytest.mark.parametrize("family", list(FAMILY_OVERRIDES))
@pytest.mark.parametrize("window", list(WINDOWS))
def test_accumulation_windows_keep_their_semantics(runs, family, window):
    """Inside a grad-accum window the completion follows each group's armed flags: a deferred reduce issues no
    reduce-scatter before the window's last backward, nor does the completion; a kept-gathered window keeps
    its layers gathered between microsteps, and every other microstep ends them sharded; the last backward
    reduces every segment."""
    flags = WINDOWS[window]
    for rank_runs in runs:
        for k in WINDOW_K_VALUES:
            run = _run(rank_runs, f"{family}-window_{window}-k{k}")
            assert max(run["grad_rel"]) < PARITY_RTOL, (k, run["grad_rel"])
            assert all(run["sharded_after_step"]), k
            assert run["unwaited"] == 0, k
            for microstep in run["micro"]:
                kept = not flags["fsdp_reshard_after_backward"] and not microstep["sync"]
                assert microstep["layers_gathered"] == [kept] * len(microstep["layers_gathered"]), (k, microstep)
                if flags["fsdp_defer_grad_sync"] and not microstep["sync"]:
                    assert microstep["any_reduce"] == 0, (k, microstep)
                if microstep["sync"]:
                    assert microstep["layer_reduces"] == [1 if k == 1 else 2] * len(microstep["layer_reduces"]), k


def test_hsdp_completes_the_reduce_scatter_and_the_replica_all_reduce(hsdp_runs):
    """Under HSDP each completed group reduce-scatters over its shard group and all-reduces over its replicas:
    every segment reaches the gradient, waited on, and K=1 stays bit-identical."""
    for rank_runs in hsdp_runs:
        run, twin = _run(rank_runs, "k1"), _run(rank_runs, "k1-twin")
        assert run["grads"] == twin["grads"] and run["weights"] == twin["weights"]
        for k in HSDP_K_VALUES:
            run = _run(rank_runs, f"k{k}")
            assert max(run["grad_rel"]) < PARITY_RTOL, (k, run["grad_rel"])
            assert run["norm_rel"] and max(run["norm_rel"]) < PARITY_RTOL, (k, run["norm_rel"])
            assert all(run["sharded_after_step"]) and run["logit_rel"] < PARITY_RTOL, k
            assert run["recorded"] > 0 and run["unwaited"] == 0, k
            expected = 1 if k == 1 else 2
            assert all(m["layer_reduces"] == [expected] * len(m["layer_reduces"]) for m in run["micro"]), k


def test_a_recomputed_step_that_overflows_on_one_rank_is_skipped_on_every_rank(skip_runs):
    """The step guard's norm is the completed gradient's: the poisoned step leaves every rank's weights bit for
    bit, and the steps around it, the first one after it included, match the reference that never took it."""
    for rank_runs in skip_runs:
        run = _run(rank_runs, "completed")
        assert run["skipped"] == 1, run["skipped"]
        assert run["weights"][POISONED_STEP] == run["weights"][POISONED_STEP - 1]
        assert max(run["grad_rel"]) < PARITY_RTOL, run["grad_rel"]
        assert run["logit_rel"] < PARITY_RTOL, run["logit_rel"]


def test_the_guard_refuses_a_backward_left_unfinished(guard_runs):
    """With the completion disabled a K>1 step raises instead of stepping partial gradients."""
    for rank_runs in guard_runs:
        message = rank_runs["uncompleted_backward"]
        assert message is not None and "post-backward left undone" in message


@pytest.mark.parametrize(
    ("injected", "unreduced", "gathered"),
    [("unreduced", 1, 0), ("gathered", 0, 1)],
)
def test_the_guard_names_each_leftover(guard_runs, injected, unreduced, gathered):
    """An unsharded gradient on a group whose reduce is armed, or a group still gathered with its reshard
    armed, each raises, counted under its own leg."""
    for rank_runs in guard_runs:
        message = rank_runs[injected]
        assert message is not None and "post-backward left undone" in message, message
        assert f"{unreduced} group(s) still hold an unreduced gradient" in message, message
        assert f"and {gathered} armed to reshard are still gathered" in message, message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
