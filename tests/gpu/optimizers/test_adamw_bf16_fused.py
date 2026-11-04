#!/usr/bin/env python
"""The fused AdamWBF16 step matches per-parameter launches bit for bit on CUDA, at Qwen3-8B's FSDP2 scale shapes.

At 64 ranks the 2-element q/k norm shards move every later grad of a layer off the 8-element boundary. A step past the
CUDA launch queue can't be queued whole, so it is timed as it runs under a ``_host_bound`` key.
"""

import copy
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from halo.optimizers.adamw_bf16 import AdamWBF16, _tensors_per_launch
from halo.runtime.dtensor import to_local
from tests.common.adamw_bf16_reference import PerParameterAdamWBF16, fused_kernel_mismatches
from tests.common.device_step_skip import adamw_bf16_builder, device_skip_mismatches
from tests.common.harness import gpu_test_main, record_check

# (shape, dtype, grad dtype, group): group 0 decays, group 1 does not and runs at its own lr.
_SHAPES = (
    ((4099,), torch.bfloat16, torch.bfloat16, 0),
    ((33, 65), torch.bfloat16, torch.bfloat16, 0),
    ((1,), torch.bfloat16, torch.bfloat16, 1),
    ((0,), torch.bfloat16, torch.bfloat16, 1),
    ((17,), torch.float32, torch.float32, 0),
    ((1024,), torch.bfloat16, torch.bfloat16, 1),
    ((3, 1031), torch.bfloat16, torch.bfloat16, 0),
    ((2048,), torch.bfloat16, torch.float32, 0),
)
# Its grad is absent on odd steps.
_INTERMITTENT = 5
_SCALED_STEPS = {2, 5}
_STEPS = 6
# A dense model's parameter count, at 64k elements each, to show the per-parameter launch cost.
_TIMING_PARAMS = 400
# One Qwen3-8B decoder layer's full shapes: q, k, v, o, q_norm, k_norm, gate, up, down, two RMSNorms.
_QWEN3_8B_LAYER = (
    (4096, 4096),
    (1024, 4096),
    (1024, 4096),
    (4096, 4096),
    (128,),
    (128,),
    (12288, 4096),
    (12288, 4096),
    (4096, 12288),
    (4096,),
    (4096,),
)
_QWEN3_8B_VOCAB = 151936
# World size, decoder layers, and whether the embedding, head and final norm unit is included.
_SCALE_SHAPES = {
    "qwen3_8b_fsdp8": (8, 36, True),
    "qwen3_8b_fsdp64": (64, 36, True),
    "qwen3_8b_180_layers_fsdp32": (32, 180, False),
}
_GPU_TIMING_REPS = 5
# How many times a sleep must outlast a step's unslept enqueue before a host still finishing after it is taken to be
# held by a full launch queue.
_SLEEP_PAST_ENQUEUE = 4


def _named_params(device: str) -> list[tuple[str, nn.Parameter]]:
    generator = torch.Generator().manual_seed(0)
    named = []
    for i, (shape, dtype, grad_dtype, _) in enumerate(_SHAPES):
        p = nn.Parameter((torch.randn(shape, generator=generator) * 0.02).to(device=device, dtype=dtype))
        p.grad_dtype = grad_dtype
        named.append((f"p{i}", p))
    return named


def _groups(named: list[tuple[str, nn.Parameter]]) -> list[dict]:
    members = [[(n, p) for (n, p), (*_, g) in zip(named, _SHAPES, strict=True) if g == group] for group in (0, 1)]
    return [
        {"params": [p for _, p in members[0]], "param_names": [n for n, _ in members[0]], "weight_decay": 0.1},
        {
            "params": [p for _, p in members[1]],
            "param_names": [n for n, _ in members[1]],
            "weight_decay": 0.0,
            "lr": 3e-3,
        },
    ]


def _state_mismatches(a: torch.optim.Optimizer, b: torch.optim.Optimizer, params_a, params_b, label: str) -> list[str]:
    found = []
    for i, (pa, pb) in enumerate(zip(params_a, params_b, strict=True)):
        if not torch.equal(to_local(pa.detach()), to_local(pb.detach())):
            found.append(f"{label} param[{i}]")
        sa, sb = a.state[pa], b.state[pb]
        if sa.get("step") != sb.get("step"):
            found.append(f"{label} step[{i}] {sa.get('step')} != {sb.get('step')}")
        for key in ("exp_avg", "exp_avg_sq"):
            if key in sa and not torch.equal(to_local(sa[key]), to_local(sb[key])):
                found.append(f"{label} {key}[{i}]")
    return found


def _step_both(fused, oracle, params_f, params_o, grads, step: int) -> None:
    for i, (pf, po, grad) in enumerate(zip(params_f, params_o, grads, strict=True)):
        absent = i == _INTERMITTENT and step % 2
        pf.grad = None if absent else grad.clone()
        po.grad = None if absent else grad.clone()
    if step in _SCALED_STEPS:
        fused.defer_grad_scale(torch.tensor(0.3, device="cuda"))
        oracle.defer_grad_scale(torch.tensor(0.3, device="cuda"))
    fused.step()
    oracle.step()


def _grads_for(params, step: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(100 + step)
    return [torch.randn(p.shape, generator=generator, device="cuda").to(p.grad_dtype) for p in params]


def test_fused_kernel_matches_per_parameter_kernel():
    mismatches = fused_kernel_mismatches("cuda")
    assert not mismatches, mismatches
    assert fused_kernel_mismatches("cuda", seed_shift=1), "a shifted seed changed no output: the check is blind"


def test_device_skip_is_exact():
    build = adamw_bf16_builder("cuda")
    mismatches = device_skip_mismatches(build, build)
    assert not mismatches, mismatches


def test_step_matches_per_parameter_optimizer():
    named_f, named_o = _named_params("cuda"), _named_params("cuda")
    initial = [p.detach().clone() for _, p in named_f]
    fused = AdamWBF16(_groups(named_f), lr=1e-2)
    oracle = PerParameterAdamWBF16(_groups(named_o), lr=1e-2)
    params_f, params_o = [p for _, p in named_f], [p for _, p in named_o]
    for step in range(1, _STEPS + 1):
        _step_both(fused, oracle, params_f, params_o, _grads_for(params_f, step), step)
        found = _state_mismatches(fused, oracle, params_f, params_o, f"step {step}")
        assert not found, found
    moved = [not torch.equal(p.detach(), p0) for p, p0 in zip(params_f, initial, strict=True) if p.numel()]
    assert all(moved), f"a param never moved, so its equality proves nothing: {moved}"


class _Shards(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.wide = nn.Parameter(torch.randn(64, 33, generator=generator).bfloat16())
        self.uneven = nn.Parameter(torch.randn(3, 5, generator=generator).bfloat16())
        self.row = nn.Parameter(torch.randn(1, 8, generator=generator).bfloat16())
        self.bias = nn.Parameter(torch.randn(7, generator=generator).bfloat16())

    def forward(self, coefficients: list[torch.Tensor]) -> torch.Tensor:
        """A loss whose gradient for each param is its coefficient."""
        return sum((p * c).sum() for p, c in zip(self.parameters(), coefficients, strict=True))


def test_fsdp2_shards_match():
    models = [_Shards().cuda(), _Shards().cuda()]
    for model in models:
        fully_shard(model)
    fused = AdamWBF16(models[0].named_parameters(), lr=1e-2, weight_decay=0.1)
    oracle = PerParameterAdamWBF16(models[1].named_parameters(), lr=1e-2, weight_decay=0.1)
    params_f, params_o = list(models[0].parameters()), list(models[1].parameters())
    found = []
    for step in range(1, 5):
        generator = torch.Generator(device="cuda").manual_seed(step)
        coefficients = [torch.randn(p.shape, generator=generator, device="cuda").bfloat16() for p in params_f]
        for model, optimizer in zip(models, (fused, oracle), strict=True):
            optimizer.zero_grad()
            model(coefficients).backward()
        if len({to_local(p.grad).untyped_storage().data_ptr() for p in params_f}) != 1:
            found.append(f"rank {dist.get_rank()} step {step}: grads are not views into one reduce-scatter output")
        fused.step()
        oracle.step()
        found += _state_mismatches(fused, oracle, params_f, params_o, f"rank {dist.get_rank()} step {step}")
    empty_shards = torch.tensor([sum(to_local(p).numel() == 0 for p in params_f)], device="cuda")
    verdict = torch.tensor([0 if found else 1], device="cuda")
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    dist.all_reduce(empty_shards)
    assert verdict.item() == 1, found or "another rank's shards diverged"
    assert empty_shards.item() >= 1, "no rank held an empty shard: the empty-local-shard case went unexercised"


def test_resumed_state_matches():
    named_o, named_f = _named_params("cuda"), _named_params("cuda")
    oracle = PerParameterAdamWBF16(_groups(named_o), lr=1e-2)
    params_o, params_f = [p for _, p in named_o], [p for _, p in named_f]
    for step in range(1, 4):
        for p, grad in zip(params_o, _grads_for(params_o, step), strict=True):
            p.grad = grad
        oracle.step()
    with torch.no_grad():
        for pf, po in zip(params_f, params_o, strict=True):
            pf.copy_(po)
    oracle.state[params_o[1]]["step"] += 4
    fused = AdamWBF16(_groups(named_f), lr=1e-2)
    fused.load_state_dict(copy.deepcopy(oracle.state_dict()))
    for step in range(4, 7):
        _step_both(fused, oracle, params_f, params_o, _grads_for(params_f, step), step)
        found = _state_mismatches(fused, oracle, params_f, params_o, f"resumed step {step}")
        assert not found, found


def _fsdp_shards(world: int, layers: int, *, embeddings: bool) -> list[list[tuple[int, ...]]]:
    units = [list(_QWEN3_8B_LAYER)] * layers
    if embeddings:
        units.append([(_QWEN3_8B_VOCAB, 4096), (_QWEN3_8B_VOCAB, 4096), (4096,)])
    return [[(shape[0] // world, *shape[1:]) for shape in unit] for unit in units]


def _scale_case(units: list[list[tuple[int, ...]]]) -> tuple[AdamWBF16, PerParameterAdamWBF16]:
    """Fused and oracle optimizers sharing gradients viewed into one buffer per unit, as FSDP's reduce-scatter output
    is."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    shapes, grads = [], []
    for unit in units:
        sizes = [torch.Size(shape).numel() for shape in unit]
        flat_grads = torch.randn(sum(sizes), generator=generator, device="cuda").bfloat16()
        grads += [chunk.view(shape) for chunk, shape in zip(flat_grads.split(sizes), unit, strict=True)]
        shapes += unit
    optimizers = []
    for optimizer_cls in (AdamWBF16, PerParameterAdamWBF16):
        generator = torch.Generator(device="cuda").manual_seed(4)
        params = [
            nn.Parameter((torch.randn(shape, generator=generator, device="cuda") * 0.02).bfloat16())
            for shape in shapes
        ]
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        optimizers.append(optimizer_cls([(f"s{i}", p) for i, p in enumerate(params)], lr=1e-3, weight_decay=0.1))
    return optimizers[0], optimizers[1]


def _timed_step(optimizer: torch.optim.Optimizer, sleep_cycles: int) -> tuple[float, bool, float]:
    """GPU ms behind a ``sleep_cycles`` sleep on the stream, whether the host finished enqueueing first, and the
    sleep's ms."""
    torch.cuda.synchronize()
    slept, start, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    slept.record()
    if sleep_cycles:
        torch.cuda._sleep(sleep_cycles)
    start.record()
    optimizer.step()
    end.record()
    ahead = not start.query()
    end.synchronize()
    return start.elapsed_time(end), ahead, slept.elapsed_time(start)


def _gpu_ms_per_step(optimizer: torch.optim.Optimizer) -> tuple[float, bool]:
    """Median GPU time of one step, and whether its launches ran back to back. A sleep holds the stream while the host
    enqueues; a step past the queue's capacity blocks the host however long the sleep, so it is timed without one."""
    torch.cuda.synchronize()
    begin = time.perf_counter()
    optimizer.step()
    enqueue_ms = (time.perf_counter() - begin) * 1e3
    times, cycles, held = [], 1 << 27, 0
    while len(times) < _GPU_TIMING_REPS:
        ms, ahead, slept_ms = _timed_step(optimizer, cycles)
        if ahead:
            times.append(ms)
            continue
        held = held + 1 if not times and slept_ms > _SLEEP_PAST_ENQUEUE * enqueue_ms else 0
        if held == 2:
            unslept = [_timed_step(optimizer, 0)[0] for _ in range(_GPU_TIMING_REPS)]
            return statistics.median(unslept), False
        cycles *= 2
        assert cycles < 1 << 36, f"the host never got ahead of a {slept_ms:.0f} ms sleep"
    return statistics.median(times), True


def _scale_mismatches(label: str, units: list[list[tuple[int, ...]]], metrics: dict) -> list[str]:
    """Two steps compared bit for bit, then GPU time under ``label``: ``gpu_ms_per_step_*`` for launches that ran back
    to back, ``*_host_bound`` for a step the launch queue held the host on."""
    fused, oracle = _scale_case(units)
    params_f, params_o = fused.param_groups[0]["params"], oracle.param_groups[0]["params"]
    found = []
    for step in (1, 2):
        fused.step()
        oracle.step()
        found += _state_mismatches(fused, oracle, params_f, params_o, f"{label} step {step}")
    if not found:
        back_to_back = {}
        for name, optimizer in (("fused", fused), ("per_parameter", oracle)):
            ms, back_to_back[name] = _gpu_ms_per_step(optimizer)
            metrics[f"gpu_ms_per_step_{label}_{name}{'' if back_to_back[name] else '_host_bound'}"] = ms
        if all(back_to_back.values()):
            metrics[f"gpu_time_ratio_{label}"] = (
                metrics[f"gpu_ms_per_step_{label}_fused"] / metrics[f"gpu_ms_per_step_{label}_per_parameter"]
            )
    del fused, oracle, params_f, params_o
    torch.cuda.empty_cache()
    return found


def _check_scale_shapes(metrics: dict) -> None:
    found = []
    for label, (world, layers, embeddings) in _SCALE_SHAPES.items():
        found += _scale_mismatches(label, _fsdp_shards(world, layers, embeddings=embeddings), metrics)
    assert not found, found[:20]


def _check_fused_scale_steps_back_to_back(metrics: dict) -> None:
    """The fused step is a few launches and pinned copies, so the host gets ahead of the stream at every scale
    shape."""
    if _tensors_per_launch(torch.device("cuda", torch.cuda.current_device())) == 1:
        return
    unqueued = [label for label in _SCALE_SHAPES if f"gpu_ms_per_step_{label}_fused" not in metrics]
    assert not unqueued, f"the fused step did not run back to back at {unqueued}"


def _host_ms_per_step(optimizer_cls) -> float:
    params = [nn.Parameter(torch.randn(256, 256, device="cuda").bfloat16()) for _ in range(_TIMING_PARAMS)]
    optimizer = optimizer_cls([(f"t{i}", p) for i, p in enumerate(params)], lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    for _ in range(3):
        optimizer.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        optimizer.step()
    host = (time.perf_counter() - start) / 10
    torch.cuda.synchronize()
    return host * 1e3


@gpu_test_main(exact_world_size=2, prefix="adamw_bf16_fused", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "fused_kernel_matches_per_parameter_kernel", test_fused_kernel_matches_per_parameter_kernel)
    record_check(checks, "step_matches_per_parameter_optimizer", test_step_matches_per_parameter_optimizer)
    record_check(checks, "fsdp2_shards_match", test_fsdp2_shards_match)
    record_check(checks, "resumed_state_matches", test_resumed_state_matches)
    record_check(checks, "device_skip_is_exact", test_device_skip_is_exact)
    metrics = {
        f"host_ms_per_step_{_TIMING_PARAMS}_params_fused": _host_ms_per_step(AdamWBF16),
        f"host_ms_per_step_{_TIMING_PARAMS}_params_per_parameter": _host_ms_per_step(PerParameterAdamWBF16),
    }
    record_check(checks, "scale_shapes_match", lambda: _check_scale_shapes(metrics))
    record_check(checks, "fused_scale_steps_run_back_to_back", lambda: _check_fused_scale_steps_back_to_back(metrics))
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
