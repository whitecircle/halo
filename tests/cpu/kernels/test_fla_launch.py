"""FLA's launch options, set by ``finalize_loaded_model`` through one seam (``apply_fla_launch_options``).

The delta rule's state kernel autotunes over ``num_warps=2`` configs only on Blackwell: its ``num_warps=4`` configs
give run-to-run different, partly wrong outputs on sm_10x, and the stock autotuner picks them at Qwen3.5/3.6 shapes.
These fail if the seam stops restricting the kernel on Blackwell, starts restricting it on another architecture, or
leaves an already-picked ``num_warps=4`` config in the autotuner's cache. The GPU behaviour itself is pinned by
``tests/gpu/kernels/test_fla_blackwell_autotune.py``.

Under ``HALO_FLA_FIXED_NB`` FLA's norm kernels launch at a fixed ``NB``, tune on the real one, and compute the same.
The compiled code for ``NB=1`` and ``NB=3`` is compared for Hopper and Blackwell targets (compiled on the CPU, never
launched), with a used constexpr as the control that must change it. The launch path runs FLA's real launchers on CPU
tensors with Triton's launch and benchmark stubbed, so the ``NB`` each launch carries and each tuning key are read
directly. These fail if the wrapper stops fixing ``NB`` (keyword or positional, launch or warm-up), hides the real
``NB`` from a callable grid or from tuning, wraps a kernel whose body reads ``NB``, if a kernel leaves the table, or if
``finalize_loaded_model`` stops applying it.
"""

from types import ModuleType

import pytest
import torch
import triton
import triton.language as tl
from fla.modules import fused_norm_gate, l2norm
from fla.ops.common import chunk_delta_h
from torch import nn
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

from halo.kernels import fla_launch
from halo.kernels.fla_launch import FIXED_NB_FLAG, apply_fla_launch_options, triton_autotuner
from halo.model_loading.buffer_repair import finalize_loaded_model

_STATE_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
# Rows past two 65,536 boundaries: FLA launches these with NB=3.
_ROWS = 2 * 65536 + 1
_D = 16
# One value for every constexpr the table's kernels declare; BT is the autotuned one the control varies.
_CONSTEXPRS = {
    "D": 128,
    "BD": 128,
    "BT": 32,
    "ACTIVATION": "swish",
    "IS_RMS_NORM": True,
    "HAS_WEIGHT": True,
    "HAS_BIAS": False,
    "HAS_RESIDUAL": False,
    "STORE_RESIDUAL_OUT": False,
    "HAS_DRESIDUAL": False,
    "STORE_DRESIDUAL": False,
    "RECOMPUTE_OUTPUT": False,
}
_SCALARS = {"eps": "fp32", "T": "i32", "BS": "i32"}
# Spelled out rather than read from the table, so a kernel dropped from it fails the launch checks.
_KERNELS = {
    l2norm: ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
    fused_norm_gate: ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"),
}


def _state_autotuner() -> Autotuner:
    return triton_autotuner(getattr(chunk_delta_h, _STATE_KERNEL))


def _warps() -> set[int]:
    return {config.num_warps for config in _state_autotuner().configs}


def _registered():
    for module, kernel_names in _KERNELS.items():
        for kernel_name in kernel_names:
            yield f"{module.__name__}.{kernel_name}", getattr(module, kernel_name)


def _jit(kernel) -> JITFunction:
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


def _fixed(kernel) -> bool:
    return isinstance(triton_autotuner(kernel).fn, fla_launch._FixedNB)


def _table_of(kernel) -> dict[ModuleType, tuple[str, ...]]:
    """A one-kernel table: ``kernel`` as the only entry of a module of its own."""
    module = ModuleType("halo_test_fla_module")
    module.kernel = kernel
    return {module: ("kernel",)}


@pytest.fixture
def gpu(monkeypatch):
    """Pretend to run on a GPU of the given major capability, and restore the state kernel's configs and cache."""
    autotuner = _state_autotuner()
    configs, cache = list(autotuner.configs), dict(autotuner.cache)
    monkeypatch.delenv(FIXED_NB_FLAG, raising=False)

    def as_major(major: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, 0))

    yield as_major
    autotuner.configs, autotuner.cache = configs, cache


@pytest.fixture
def restored(monkeypatch):
    """Undo the norm kernels' wrapping and any tuning picks after each test: they are FLA's process-wide objects."""
    for _, kernel in _registered():
        tuner = triton_autotuner(kernel)
        monkeypatch.setattr(tuner, "fn", tuner.fn)
        monkeypatch.setattr(tuner, "cache", dict(tuner.cache))
    monkeypatch.delenv(FIXED_NB_FLAG, raising=False)
    return monkeypatch


@pytest.fixture
def launches(restored):
    """Stub Triton's launch and benchmark: record each launch's kernel and ``NB``, compile and run nothing."""
    recorded = []

    def run(self, *args, grid, warmup, **kwargs):
        recorded.append((self.__name__, kwargs["NB"]))

    restored.setattr(JITFunction, "run", run)
    restored.setattr(Autotuner, "_bench", lambda self, *args, config, **meta: [float(config.num_warps)] * 3)
    for _, kernel in _registered():
        restored.setattr(triton_autotuner(kernel), "cache_results", False)
    restored.setattr(fused_norm_gate, "get_multiprocessor_count", lambda index: 4)
    return recorded


def test_the_locked_fla_offers_num_warps_4_here():
    """Non-vacuity guard: without ``num_warps=4`` configs to drop, the restriction tests below would prove nothing."""
    assert 4 in _warps()


def test_blackwell_keeps_only_num_warps_2(gpu):
    gpu(10)
    apply_fla_launch_options()
    assert _warps() == {2}


@pytest.mark.parametrize("major", [8, 9, 12])
def test_other_architectures_keep_every_config(gpu, major):
    before = _warps()
    gpu(major)
    apply_fla_launch_options()
    assert _warps() == before


def test_cpu_only_process_keeps_every_config(gpu, monkeypatch):
    before = _warps()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    apply_fla_launch_options()
    assert _warps() == before


def test_finalizing_a_loaded_model_applies_the_restriction(gpu):
    gpu(10)
    finalize_loaded_model(torch.nn.Linear(2, 2))
    assert _warps() == {2}


def test_a_cached_num_warps_4_pick_is_dropped(gpu):
    autotuner = _state_autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 4)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    apply_fla_launch_options()
    assert autotuner.cache == {}


def test_a_cached_num_warps_2_pick_survives(gpu):
    autotuner = _state_autotuner()
    picked = next(config for config in autotuner.configs if config.num_warps == 2)
    autotuner.cache = {("shape",): picked}
    gpu(10)
    apply_fla_launch_options()
    assert autotuner.cache == {("shape",): picked}


def test_unautotuned_state_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.setattr(chunk_delta_h, _STATE_KERNEL, lambda *args: None)
    with pytest.raises(RuntimeError, match="not an autotuned FLA kernel here; re-check the Blackwell restriction"):
        apply_fla_launch_options()


def test_moved_state_kernel_fails_loud(gpu, monkeypatch):
    gpu(10)
    monkeypatch.delattr(chunk_delta_h, _STATE_KERNEL)
    with pytest.raises(RuntimeError, match="re-check the Blackwell restriction"):
        apply_fla_launch_options()


def test_one_call_sets_both_options_on_blackwell_with_the_flag(gpu, restored):
    gpu(10)
    restored.setenv(FIXED_NB_FLAG, "1")
    apply_fla_launch_options()
    assert _warps() == {2}
    assert all(_fixed(kernel) for _, kernel in _registered())


def _launch_all():
    """Each table kernel once, through FLA's own launchers, on ``_ROWS`` rows."""
    x, g, dy = (torch.randn(_ROWS, _D) for _ in range(3))
    weight = torch.ones(_D)
    y, rstd = l2norm.l2norm_fwd(x)
    l2norm.l2norm_bwd(y, rstd, dy)
    fused_norm_gate.layer_norm_gated_fwd(x, g, weight, None, is_rms_norm=True)
    fused_norm_gate.layer_norm_gated_bwd(dy, x, g, weight, None, rstd=rstd, is_rms_norm=True)


def _compile(fn: JITFunction, constexprs: dict, capability: int) -> bytes:
    names = fn.arg_names
    constant = {names[p.num] for p in fn.params if p.is_constexpr}
    signature = {name: "constexpr" if name in constant else _SCALARS.get(name, "*bf16") for name in names}
    source = ASTSource(
        fn=fn, signature=signature, constexprs={(names.index(name),): constexprs[name] for name in constant}
    )
    return triton.compile(source, target=GPUTarget("cuda", capability, 32), options={"num_warps": 4}).asm["cubin"]


def test_the_table_holds_the_spelled_out_kernels():
    assert fla_launch.FIXED_NB_KERNELS == _KERNELS


@pytest.mark.parametrize("capability", [90, 100])
def test_an_unread_nb_compiles_to_the_same_binary(capability):
    for qualname, kernel in _registered():
        fn = _jit(kernel)
        one = _compile(fn, {**_CONSTEXPRS, "NB": 1}, capability)
        assert one == _compile(fn, {**_CONSTEXPRS, "NB": 3}, capability), qualname
        assert one != _compile(fn, {**_CONSTEXPRS, "NB": 1, "BT": 16}, capability), f"{qualname}: control"


def test_flag_off_launches_carry_the_real_nb(launches):
    apply_fla_launch_options()
    _launch_all()
    assert len(launches) == 4 and {nb for _, nb in launches} == {3}, launches


def test_flag_on_launches_fixed_nb_and_tunes_on_the_real_one(restored, launches):
    restored.setenv(FIXED_NB_FLAG, "1")
    apply_fla_launch_options()
    _launch_all()
    assert sorted(name for name, _ in launches) == sorted(_jit(k).__name__ for _, k in _registered())
    assert {nb for _, nb in launches} == {1}, launches
    for qualname, kernel in _registered():
        tuner = triton_autotuner(kernel)
        nb_slot = tuner.keys.index("NB")
        assert [key[nb_slot] for key in tuner.cache] == [3], (qualname, list(tuner.cache))


def test_the_disk_cache_key_still_reaches_the_jit_function(restored):
    originals = {qualname: _jit(kernel) for qualname, kernel in _registered()}
    restored.setenv(FIXED_NB_FLAG, "1")
    apply_fla_launch_options()
    apply_fla_launch_options()
    for qualname, kernel in _registered():
        wrapper = triton_autotuner(kernel).fn
        assert isinstance(wrapper, fla_launch._FixedNB) and wrapper.fn is originals[qualname], qualname
        assert wrapper.cache_key == originals[qualname].cache_key


def test_finalize_loaded_model_applies_it_only_with_the_flag(restored):
    finalize_loaded_model(nn.Linear(2, 2))
    assert not any(_fixed(k) for _, k in _registered())
    restored.setenv(FIXED_NB_FLAG, "1")
    finalize_loaded_model(nn.Linear(2, 2))
    assert all(_fixed(k) for _, k in _registered())


@triton.autotune(configs=[triton.Config({}, num_warps=1)], key=["NB"])
@triton.jit
def _reads_nb_kernel(out, NB: tl.constexpr):
    tl.store(out, NB)


@triton.autotune(configs=[triton.Config({}, num_warps=1)], key=["D"])
@triton.jit
def _takes_no_nb_kernel(out, D: tl.constexpr):
    tl.store(out, D)


@triton.autotune(configs=[triton.Config({}, num_warps=1)], key=["NB"])
@triton.jit
def _positional_nb_kernel(out, NB: tl.constexpr, D: tl.constexpr):
    tl.store(out, D)


def test_a_positional_nb_is_fixed_in_place(restored):
    recorded = []
    restored.setattr(JITFunction, "run", lambda self, *args, grid, warmup, **kwargs: recorded.append((args, kwargs)))
    restored.setattr(_positional_nb_kernel, "fn", _positional_nb_kernel.fn)
    restored.setenv(FIXED_NB_FLAG, "1")
    restored.setattr(fla_launch, "FIXED_NB_KERNELS", _table_of(_positional_nb_kernel))
    apply_fla_launch_options()
    out = torch.empty(1)
    _positional_nb_kernel[(1,)](out, 3, 7)
    _positional_nb_kernel[(1,)](out, 3, D=7)
    assert [(args[1:], kwargs.get("D")) for args, kwargs in recorded] == [((1, 7), None), ((1,), 7)], recorded


def test_warmup_fixes_nb_and_a_callable_grid_sees_the_real_one(restored):
    recorded = []

    def run(self, *args, grid, warmup, **kwargs):
        bound = {**dict(zip(self.arg_names, args, strict=False)), **kwargs}
        recorded.append((warmup, bound["NB"], grid(bound) if callable(grid) else grid))

    restored.setattr(JITFunction, "run", run)
    restored.setattr(_positional_nb_kernel, "fn", _positional_nb_kernel.fn)
    restored.setenv(FIXED_NB_FLAG, "1")
    restored.setattr(fla_launch, "FIXED_NB_KERNELS", _table_of(_positional_nb_kernel))
    apply_fla_launch_options()
    out = torch.empty(1)

    def grid(meta):
        return (meta["NB"],)

    _positional_nb_kernel[grid](out, 3, 7)
    _positional_nb_kernel[grid](out, NB=3, D=7)
    _positional_nb_kernel.warmup(out, 3, 7, grid=grid)
    _positional_nb_kernel.warmup(out, NB=3, D=7, grid=grid)
    _positional_nb_kernel.warmup(out, 3, 7, grid=(2,))
    assert recorded == [(False, 1, (3,))] * 2 + [(True, 1, (3,))] * 2 + [(True, 1, (2,))], recorded


@pytest.mark.parametrize(
    "kernel",
    [_reads_nb_kernel, _takes_no_nb_kernel, triton.jit(_reads_nb_kernel.fn.fn), None],
    ids=["reads_nb", "takes_no_nb", "not_autotuned", "missing"],
)
def test_a_kernel_that_no_longer_fits_raises(restored, kernel):
    restored.setenv(FIXED_NB_FLAG, "1")
    restored.setattr(fla_launch, "FIXED_NB_KERNELS", _table_of(kernel))
    with pytest.raises(RuntimeError, match=FIXED_NB_FLAG):
        apply_fla_launch_options()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
