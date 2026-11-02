#!/usr/bin/env python
"""The fused AdamWBF16 launch: its host tables, and its kernel against the per-parameter one it replaced.

One launch steps every bf16 parameter of a dtype and specialization group, so each program must find its tensor and
that tensor's block index from the tables alone, and key its rounding noise exactly as a launch over that tensor would.
The kernel comparison runs under Triton's interpreter in a fresh process (``TRITON_INTERPRET`` is read when a
kernel is defined); the CUDA leg, where FMA contraction applies, is ``tests/gpu/optimizers/test_adamw_bf16_fused.py``.
The compiled-code check builds both kernels for sm_90, sm_100 and sm_103 without a GPU, each under one tensor's launch
specialization and at the table size that target launches with: the full table on sm_100/103, one tensor elsewhere.

Run: python tests/cpu/optimizers/test_adamw_bf16_fused_launch.py
"""

import collections
import math
import re

import pytest
import torch
import torch.nn as nn
import triton
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import native_specialize_impl

from halo.optimizers import adamw_bf16
from halo.optimizers.adamw_bf16 import (
    _ADAM_MAX_REGISTERS,
    _ADAM_NUM_WARPS,
    _BLOCKS_PER_PROGRAM,
    _MAX_TENSORS_PER_LAUNCH,
    _MULTI_TENSOR_LAUNCH_CAPABILITIES,
    _ROLE_ROWS,
    _ROW_FIRST_PROGRAM,
    _ROW_NUMEL,
    _ROW_SEED,
    _SR_KEY,
    AdamWBF16,
    _adam_bf16_sr_kernel,
    _divisibility,
    _fused_update,
    _FusedUpdate,
    _launch_tables,
    _specialization,
    _tensors_per_launch,
)
from halo.optimizers.stochastic_rounding import SR_BLOCK_SIZE, sr_param_index, sr_seed_pair
from tests.common.adamw_bf16_reference import _per_parameter_kernel
from tests.common.repo import probe_findings

_HPARAMS = (0.9, 0.999, 1.0 / 3.0, 0.1, 1e-8, 1.0 - 1e-3)
_MARKER = "FUSED_ADAMW_MISMATCHES:"
_PROBE = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
from tests.common.adamw_bf16_reference import fused_kernel_mismatches
findings = fused_kernel_mismatches("cpu")
if not fused_kernel_mismatches("cpu", seed_shift=1):
    findings.append("a shifted seed changed no output: the comparison is blind to the rounding noise")
print({_MARKER!r} + "|".join(findings))
"""
_LOOKUP_MARKER = "ADAMW_LOOKUP_MISMATCHES:"
_LOOKUP_PROBE = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
from tests.common.adamw_bf16_reference import lookup_mismatches
print({_LOOKUP_MARKER!r} + "|".join(lookup_mismatches()))
"""
# The architectures the training images target: Hopper and data-center Blackwell.
_TARGETS = tuple(GPUTarget("cuda", arch, 32) for arch in (90, 100, 103))
# Hopper, the two Blackwell capabilities the multi-tensor launch is gated to, and two Blackwell ones outside it.
_CAPABILITIES = ((9, 0), (10, 0), (10, 3), (11, 0), (12, 0))
_LAYOUT_DEF = re.compile(r"^(#\w+) = (#ttg\.\w+<.*>)$", re.M)
_TENSOR_OP = re.compile(r"^\s*(?:(%[\w.]+) = )?(\w+\.\w+)\b(.*)$")
_SSA_NAME = re.compile(r"%[\w.]+")
_ATTRIBUTES = re.compile(r"\{[^{}]*\}")
_LAYOUT_NAME = re.compile(r"#\w+")
# An op with several results (``%a:2 = ``, ``%a, %b = ``), which _TENSOR_OP does not take apart.
_MULTI_RESULT = re.compile(r"^\s*%[\w.]+(?::\d+|, %)")
_GLOBAL_ACCESS = re.compile(r"^\s+(?:@%p\d+\s+)?((?:ld|st)\.global[\w.]*)", re.M)
_FLOAT_OP = re.compile(r"^\s*(?:@%p\d+\s+)?((?:fma|mul|add|sub|div|sqrt|rcp|cvt|max|min|neg|abs|mad)\.[\w.]+)\s", re.M)
# The fused kernel's own reads: the int64 index rows and the fp32 scalar rows.
_TABLE_LOADS = ("ld.global.b64", "ld.global.v2.b64", "ld.global.v4.b64", "ld.global.b32")
# (grad dtype, grad scale, _FusedUpdate.divisible): what step() meets (aligned shards, fresh tensors of an odd size, an
# FSDP2 grad view off the boundary, fused-parameter slices and their grad views, an fp32 grad off the boundary), two
# other partial alignments, and none.
_SPECIALIZATIONS = (
    ("bf16", False, (True, True, True, True, True)),
    ("bf16", True, (False, True, True, True, True)),
    ("bf16", False, (True, True, False, True, True)),
    ("bf16", True, (False, False, False, True, True)),
    ("bf16", True, (True, False, False, True, True)),
    ("fp32", True, (False, True, False, True, True)),
    ("bf16", True, (True, False, False, True, False)),
    ("bf16", False, (True, False, True, False, False)),
    ("bf16", True, (False, False, False, False, False)),
)


def _update(
    p: torch.Tensor, seed: int, *, grad: torch.Tensor | None = None, hparams: tuple = _HPARAMS
) -> _FusedUpdate:
    grad = torch.zeros_like(p) if grad is None else grad
    tensors = (p, grad, torch.zeros_like(p), torch.zeros_like(p))
    return _FusedUpdate(p, tensors, hparams, seed, _divisibility(tensors))


class _CudaParameter(nn.Parameter):
    """A CPU parameter ``step()`` routes to the fused launch, as it would a CUDA one."""

    @property
    def is_cuda(self) -> bool:
        return True


@pytest.fixture
def capability(monkeypatch):
    """Set the compute capability every device reports, read afresh by the launch-size gate."""
    _tensors_per_launch.cache_clear()
    yield lambda major_minor: monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: major_minor)
    _tensors_per_launch.cache_clear()


def _compiled_asm(
    kernel, target: GPUTarget, signature: dict, constexprs: dict, divisible: tuple[str, ...], **options
) -> dict:
    """``kernel`` compiled for ``target`` at the step's launch width (and ``options``) with ``divisible`` arguments
    marked 16-divisible, as Triton's launch specialization marks a 16-byte-aligned pointer or a multiple-of-16
    integer."""
    names = kernel.arg_names
    attrs = {(names.index(name),): [["tt.divisibility", 16]] for name in divisible}
    source = ASTSource(fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs)
    asm = triton.compile(source, target=target, options={"num_warps": _ADAM_NUM_WARPS, **options}).asm
    # With a GPU visible, a compile that drops the target builds for that GPU instead of raising.
    assert f'ttg.target = "cuda:{target.arch}"' in asm["ttgir"], f"not compiled for sm_{target.arch}"
    return asm


def _store_dataflow(ttgir: str, interned: dict) -> list[int]:
    """Each tensor store's value, address and mask as a graph of the tensor ops that compute it, each op with its
    layouts resolved: the per-element code. The layout decides which register of its thread an element sits in, and
    the compiled FMA association can differ between registers. Graphs are hash-consed into ``interned``, so two
    kernels whose stores get the same ids compute every element alike, wherever the compiler placed each op (inside
    the block loop or hoisted above it). Scalars, which the two kernels source differently, are opaque leaves."""
    layouts = dict(_LAYOUT_DEF.findall(ttgir))
    defs, stores = {}, []
    for line in ttgir.splitlines():
        match = _TENSOR_OP.match(line)
        if not match:
            assert not (_MULTI_RESULT.match(line) and "tensor<" in line), f"unparsed tensor op: {line.strip()}"
            continue
        result, op, rest = match.groups()
        rest = rest.split(" loc(")[0]
        operands, _, types = _ATTRIBUTES.sub("", rest).partition(" : ")
        if "tensor<" not in types:
            continue
        node = (
            op,
            _SSA_NAME.sub("%", operands).strip() + "".join(_ATTRIBUTES.findall(rest)),
            _LAYOUT_NAME.sub(lambda name: layouts.get(name.group(0), name.group(0)), types),
            tuple(_SSA_NAME.findall(operands)),
        )
        if result:
            defs[result] = node
        if op == "tt.store":
            stores.append(node)

    ids: dict[str, int] = {}

    def intern(node: tuple) -> int:
        op, attributes, types, operands = node
        key = (op, attributes, types, tuple(value_id(name) for name in operands))
        return interned.setdefault(key, len(interned))

    def value_id(name: str) -> int:
        if name not in defs:
            return interned.setdefault(("scalar",), len(interned))
        if name not in ids:
            ids[name] = intern(defs[name])
        return ids[name]

    return [intern(store) for store in stores]


def _launcher_marks_divisible(arg) -> bool:
    """Whether Triton's launcher specializes ``arg``, a tensor or an int, as divisible."""
    return native_specialize_impl(BaseBackend, arg, False, True, True)[1] == "D"


def test_tables_address_every_tensor_and_count_its_programs():
    """Each role's element offset lands on the tensor's own storage (behind the base too), first programs
    accumulate each tensor's blocks in programs of ``_BLOCKS_PER_PROGRAM``, and the scalars are the fp32 a float
    argument becomes."""
    per_program = SR_BLOCK_SIZE * _BLOCKS_PER_PROGRAM
    shared = torch.zeros(2 * per_program + 5000, dtype=torch.bfloat16)
    params = [
        shared[2000:3025],
        torch.zeros(1, dtype=torch.bfloat16),
        shared[7:2106],
        shared[4500 : 4500 + per_program + 1],
        shared[4500 + per_program + 3 : 4500 + 2 * per_program + 3],
    ]
    updates = [_update(p, seed) for p, seed in zip(params, (11, 22, 33, 44, 55), strict=True)]

    index, hparams, num_programs = _launch_tables(updates)
    rows = index.tolist()

    programs = [1, 1, 1, 2, 1]
    assert [triton.cdiv(triton.cdiv(p.numel(), SR_BLOCK_SIZE), _BLOCKS_PER_PROGRAM) for p in params] == programs
    assert num_programs == sum(programs)
    assert rows[_ROW_FIRST_PROGRAM] == [sum(programs[:i]) for i in range(len(programs))]
    assert rows[_ROW_NUMEL] == [p.numel() for p in params]
    assert rows[_ROW_SEED] == [11, 22, 33, 44, 55]
    for row, role in zip(_ROLE_ROWS, range(4), strict=True):
        base = updates[0].tensors[role]
        for update, offset in zip(updates, rows[row], strict=True):
            tensor = update.tensors[role]
            assert base.data_ptr() + offset * tensor.element_size() == tensor.data_ptr()
    assert rows[_ROLE_ROWS[0]][2] < 0, "the third param sits before the base: a negative offset"
    expected = torch.tensor([[torch.tensor(h, dtype=torch.float32).item()] * len(updates) for h in _HPARAMS])
    assert hparams.dtype == torch.float32 and torch.equal(hparams, expected)


def test_tables_keep_per_tensor_scalars_and_seeds():
    """Params at different steps (a resumed run) share a launch: each column carries its own step's scalars."""
    first = _update(torch.zeros(4, dtype=torch.bfloat16), 5, hparams=(0.9, 0.999, 0.5, 0.25, 1e-8, 1.0))
    second = _update(torch.zeros(4, dtype=torch.bfloat16), 6, hparams=(0.8, 0.99, 0.125, 0.75, 1e-6, 0.9))

    index, hparams, _ = _launch_tables([first, second])

    assert index.tolist()[_ROW_SEED] == [5, 6]
    assert torch.equal(hparams[:, 0], torch.tensor(first.hparams, dtype=torch.float32))
    assert torch.equal(hparams[:, 1], torch.tensor(second.hparams, dtype=torch.float32))


@pytest.mark.parametrize(
    "grad",
    [
        torch.zeros(8, 4, dtype=torch.bfloat16).t(),  # strided: the kernel indexes flat storage
        torch.zeros(64, dtype=torch.bfloat16)[::2],  # viewable as 1-D, yet not contiguous
        torch.zeros(31, dtype=torch.bfloat16),  # a different element count
    ],
)
def test_tables_refuse_tensors_the_kernel_would_misread(grad):
    p = torch.zeros(4, 8, dtype=torch.bfloat16)
    good = _update(torch.zeros(3, dtype=torch.bfloat16), 1)
    with pytest.raises(ValueError, match="contiguous, element-aligned local shards"):
        _launch_tables([good, _update(p, 2, grad=grad)])


def test_group_key_separates_what_one_launch_cannot_mix():
    p = torch.zeros(4, dtype=torch.bfloat16)
    assert _update(p, 1).group_key == _update(torch.zeros(9, dtype=torch.bfloat16), 2).group_key
    assert _update(p, 1).group_key != _update(p, 1, grad=torch.zeros(4)).group_key
    aligned = _update(torch.zeros(32, dtype=torch.bfloat16), 1)
    assert all(aligned.divisible) and aligned.group_key != _update(p, 1).group_key
    buffer = torch.zeros(64, dtype=torch.bfloat16)
    shifted_grad = _update(torch.zeros(32, dtype=torch.bfloat16), 1, grad=buffer.narrow(0, 4, 32))
    assert shifted_grad.group_key != aligned.group_key, "one role's alignment alone keys a launch apart"


def test_divisibility_is_each_tensors_launch_specialization():
    """Each flag is what Triton's launcher marks in a launch over that tensor alone: its element count, then each of
    its four tensors' starts, a 16-byte-aligned fp32 grad included."""
    bf16 = torch.zeros(4096, dtype=torch.bfloat16)
    fp32 = torch.zeros(4096)
    updates = [
        _update(bf16.narrow(0, 64, 1024), 1),
        _update(bf16.narrow(0, 64, 1032), 1),
        _update(bf16.narrow(0, 68, 1024), 1),
        _update(bf16.narrow(0, 64, 1024), 1, grad=bf16.narrow(0, 2049, 1024)),
        _update(bf16.narrow(0, 64, 1024), 1, grad=fp32.narrow(0, 4, 1024)),
        _update(bf16.narrow(0, 64, 1024), 1, grad=fp32.narrow(0, 2, 1024)),
        _update(bf16.narrow(0, 65, 1), 1),
    ]
    for update in updates:
        launcher = (update.tensors[0].numel(), *update.tensors)
        assert update.divisible == tuple(map(_launcher_marks_divisible, launcher)), update.divisible
    assert [update.divisible for update in updates] == [
        (True, True, True, True, True),
        (False, True, True, True, True),
        (True, False, True, True, True),
        (True, True, False, True, True),
        (True, True, True, True, True),
        (True, True, False, True, True),
        (False, False, True, True, True),
    ]
    p = torch.zeros(16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one specialization"):
        _specialization((_update(p, 1).divisible, _update(p, 1, grad=bf16.narrow(0, 1, 16)).divisible))


def test_fused_update_carries_its_param_step_scalars_and_seed():
    """What ``step()`` hands the kernel per param: its own local tensors, the Adam scalars of its step and group,
    and the first seed of its name's pair, for params at mixed steps across two groups."""
    params = {f"w{i}": nn.Parameter(torch.zeros(32, dtype=torch.bfloat16)) for i in range(4)}
    groups = [
        {"params": [params["w0"], params["w1"]], "param_names": ["w0", "w1"], "weight_decay": 0.1},
        {
            "params": [params["w2"], params["w3"]],
            "param_names": ["w2", "w3"],
            "weight_decay": 0.0,
            "lr": 3e-3,
            "betas": (0.8, 0.95),
            "eps": 1e-6,
        },
    ]
    optimizer = AdamWBF16(groups, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    steps = {"w0": 1, "w1": 7, "w2": 2, "w3": 40}
    checked = 0
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        for p, name in zip(group["params"], group["param_names"], strict=True):
            p.grad = torch.zeros_like(p)
            state = {"step": steps[name], "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
            update = _fused_update(p, name, state, group)
            step = steps[name]
            assert update.param is p
            assert [t.data_ptr() for t in update.tensors] == [
                t.data_ptr() for t in (p, p.grad, state["exp_avg"], state["exp_avg_sq"])
            ]
            assert update.hparams == (
                beta1,
                beta2,
                lr / (1.0 - beta1**step),
                math.sqrt(1.0 - beta2**step),
                eps,
                1.0 - lr * wd,
            ), name
            assert update.seed == sr_seed_pair(_SR_KEY, step, sr_param_index(name))[0], name
            checked += 1
    assert checked == 4


@pytest.mark.parametrize("major_minor", _CAPABILITIES, ids=lambda c: f"sm{c[0]}{c[1]}")
def test_only_sm100_and_sm103_launch_many_tensors_at_once(capability, major_minor):
    capability(major_minor)
    expected = _MAX_TENSORS_PER_LAUNCH if major_minor in {(10, 0), (10, 3)} else 1
    assert _tensors_per_launch(torch.device("cuda", 0)) == expected


@pytest.mark.parametrize("major_minor", [(10, 0), (9, 0)], ids=["sm100", "sm90"])
def test_step_launches_full_chunks_from_each_steps_tensors(monkeypatch, capability, major_minor):
    """``step()`` sends a launch out once its group key holds as many non-empty updates as one launch steps there
    (``_MAX_TENSORS_PER_LAUNCH`` on sm_100, one on sm_90), in param order, skips empty shards and a key left empty,
    hands every launch the deferred clip scale, bumps every fused param's version, and builds each launch's table
    from that step's local tensors and seeds: grads reallocated between steps (an FSDP2 reshard) are addressed where
    they now live."""
    capability(major_minor)
    launches = []
    monkeypatch.setattr(
        adamw_bf16, "_launch_fused", lambda updates, index, _, __, scale: launches.append((updates[:], index, scale))
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    # Two full launches of the aligned key go out mid-step and leave it empty; the odd-sized key ends partial.
    sizes = [32] * 250 + [33] * 5 + [0] + [32] * 262 + [33] * 5
    params = [_CudaParameter(torch.zeros(n, dtype=torch.bfloat16)) for n in sizes]
    optimizer = AdamWBF16([(f"w{i}", p) for i, p in enumerate(params)], lr=1e-2)

    for step in (1, 2):
        # Allocated while the previous grads still live, so every grad moves.
        grads = [torch.randn(n).bfloat16() for n in sizes]
        moved = [
            p.grad is None or p.grad.data_ptr() != g.data_ptr()
            for p, g in zip(params, grads, strict=True)
            if g.numel()
        ]
        assert all(moved), "a grad kept its storage, so a stale table would go unseen"
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        versions = [p._version for p in params]
        scale = torch.tensor(0.5) if step == 2 else None
        if scale is not None:
            optimizer.defer_grad_scale(scale)
        launches.clear()
        optimizer.step()

        aligned = [id(p) for p, n in zip(params, sizes, strict=True) if n == 32]
        odd = [id(p) for p, n in zip(params, sizes, strict=True) if n == 33]
        if major_minor == (10, 0):
            cap = _MAX_TENSORS_PER_LAUNCH
            expected = [aligned[:cap], aligned[cap : 2 * cap], odd]
        else:
            expected = [[id(p)] for p, n in zip(params, sizes, strict=True) if n]
        assert [[id(update.param) for update in updates] for updates, _, _ in launches] == expected, step
        assert all(
            launched is None if scale is None else torch.equal(launched, scale) for _, _, launched in launches
        ), "every launch, the ones sent mid-step included, takes the step's deferred scale"
        assert [p._version for p in params] == [v + 1 for v in versions], "every fused param's version moves"
        for updates, index, _ in launches:
            rows = index.tolist()
            assert rows[_ROW_SEED] == [update.seed for update in updates]
            for row, role in zip(_ROLE_ROWS, range(4), strict=True):
                base = updates[0].tensors[role]
                for update, offset in zip(updates, rows[row], strict=True):
                    tensor = update.tensors[role]
                    assert base.data_ptr() + offset * tensor.element_size() == tensor.data_ptr()
            assert [update.tensors[1].data_ptr() for update in updates] == [
                update.param.grad.data_ptr() for update in updates
            ]


@pytest.mark.parametrize("target", _TARGETS, ids=lambda target: f"sm{target.arch}")
@pytest.mark.parametrize(
    ("grad_dtype", "has_scale", "divisible"),
    _SPECIALIZATIONS,
    ids=[f"{g}-{'scale' if s else 'no_scale'}-{''.join(str(int(f)) for f in d)}" for g, s, d in _SPECIALIZATIONS],
)
def test_compiled_kernel_matches_a_lone_tensor_launch(grad_dtype, has_scale, divisible, target):
    """Compiled for each target under a tensor's launch specialization and the table that target launches with, the
    fused kernel runs the per-parameter kernel's per-element code: the same graph of tensor ops behind every store,
    each op in the same layout, and the same vector widths and float ops (the PTX schedule may place them around the
    table loads and the block loop)."""
    scale_ptr = "*fp32" if has_scale else f"*{grad_dtype}"
    oracle_signature = {
        "p_ptr": "*bf16",
        "grad_ptr": f"*{grad_dtype}",
        **dict.fromkeys(("ea_ptr", "easq_ptr"), "*bf16"),
        **dict.fromkeys(("beta1", "beta2", "step_size", "bc2_sqrt", "eps", "wd_factor"), "fp32"),
        "n_elements": "i32",
        "seed": "i32",
        "grad_scale_ptr": scale_ptr,
        "HAS_GRAD_SCALE": "constexpr",
        "BLOCK_SIZE": "constexpr",
    }
    roles = [name for name, flag in zip(("p", "grad", "ea", "easq"), divisible[1:], strict=True) if flag]
    oracle = _compiled_asm(
        _per_parameter_kernel,
        target,
        oracle_signature,
        {"HAS_GRAD_SCALE": has_scale, "BLOCK_SIZE": SR_BLOCK_SIZE},
        (*(f"{role}_ptr" for role in roles), "grad_scale_ptr", *(("n_elements",) if divisible[0] else ())),
    )
    specialization = _specialization([divisible])
    fused_signature = {
        "p_base": "*bf16",
        "grad_base": f"*{grad_dtype}",
        **dict.fromkeys(("ea_base", "easq_base"), "*bf16"),
        "index_table": "*i64",
        "hparam_table": "*fp32",
        "num_tensors": "i32",
        "grad_scale_ptr": scale_ptr,
        **dict.fromkeys(
            ("HAS_GRAD_SCALE", *specialization, "TABLE_SIZE", "BLOCKS_PER_PROGRAM", "BLOCK_SIZE"), "constexpr"
        ),
    }
    # The launcher marks each base by the launch's first tensor of that role, which shares the role's flag.
    fused = _compiled_asm(
        _adam_bf16_sr_kernel,
        target,
        fused_signature,
        {
            "HAS_GRAD_SCALE": has_scale,
            "TABLE_SIZE": _MAX_TENSORS_PER_LAUNCH
            if divmod(target.arch, 10) in _MULTI_TENSOR_LAUNCH_CAPABILITIES
            else 1,
            "BLOCKS_PER_PROGRAM": _BLOCKS_PER_PROGRAM,
            "BLOCK_SIZE": SR_BLOCK_SIZE,
            **specialization,
        },
        (*(f"{role}_base" for role in roles), "index_table", "hparam_table", "grad_scale_ptr"),
        maxnreg=_ADAM_MAX_REGISTERS,
    )

    def data_accesses(ptx: str) -> collections.Counter:
        return collections.Counter(op for op in _GLOBAL_ACCESS.findall(ptx) if op not in _TABLE_LOADS)

    def float_ops(ptx: str) -> collections.Counter:
        return collections.Counter(op for op in _FLOAT_OP.findall(ptx) if ".f32" in op or ".bf16" in op)

    expected = data_accesses(oracle["ptx"])
    vectorized = divisible[0] and any(divisible[1:])
    assert any(".v" in op for op in expected) == vectorized, f"the oracle's specialization did not hold: {expected}"
    interned: dict = {}
    assert _store_dataflow(fused["ttgir"], interned) == _store_dataflow(oracle["ttgir"], interned)
    assert data_accesses(fused["ptx"]) == expected
    assert float_ops(fused["ptx"]) == float_ops(oracle["ptx"])


def test_lookup_matches_the_block_scan_interpreted():
    """Every program resolves each block it steps to the tensor and block index the per-block scan of the
    block-start row gave, block for block: one tensor and two (TABLE_SIZE 1 and 2), three in a table padded to the
    cap, a full table at the cap with empty tensors at both ends and a table just past a power of two, both ragged
    over empty, one-block and multi-program tensors."""
    assert probe_findings(_LOOKUP_PROBE, _LOOKUP_MARKER) == []


def test_fused_kernel_matches_per_parameter_kernel_interpreted():
    """Bit for bit on every output against the per-parameter launch: empty, one-element, block-boundary and
    multi-block tensors at mixed steps, a tensor of several programs, odd-offset views, weight decay on and off, both
    grad-scale modes, three chained steps."""
    assert probe_findings(_PROBE, _MARKER) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
