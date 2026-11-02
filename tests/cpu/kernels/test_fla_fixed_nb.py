"""``HALO_FLA_FIXED_NB``: FLA's norm kernels launch at a fixed ``NB``, tune on the real one, and compute the same.

The compiled code for ``NB=1`` and ``NB=3`` is compared for Hopper and Blackwell targets (compiled on the CPU, never
launched), with a used constexpr as the control that must change it. The launch path runs FLA's real launchers on CPU
tensors with Triton's launch and benchmark stubbed, so the ``NB`` each launch carries and each tuning key are read
directly. These fail if the wrapper stops fixing ``NB`` (keyword or positional, launch or warm-up), hides the real
``NB`` from a callable grid or from tuning, wraps a kernel whose body reads ``NB``, imports FLA itself, if a kernel
leaves the table, or if ``finalize_loaded_model`` stops applying it.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from fla.modules import fused_norm_gate, l2norm
from torch import nn
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

from halo.kernels import fla_fixed_nb
from halo.kernels.fla_autotune import triton_autotuner
from halo.kernels.fla_fixed_nb import FIXED_NB_FLAG, fix_fla_norm_nb
from halo.model_loading.buffer_repair import finalize_loaded_model

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
    "fla.modules.l2norm": ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
    "fla.modules.fused_norm_gate": ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"),
}


def _registered():
    for module_name, kernel_names in _KERNELS.items():
        for kernel_name in kernel_names:
            yield f"{module_name}.{kernel_name}", getattr(sys.modules[module_name], kernel_name)


def _jit(kernel) -> JITFunction:
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


@pytest.fixture
def restored(monkeypatch):
    """Undo the wrapping and any tuning picks after each test: the kernels are FLA's process-wide objects."""
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


@pytest.mark.parametrize("capability", [90, 100])
def test_an_unread_nb_compiles_to_the_same_binary(capability):
    for qualname, kernel in _registered():
        fn = _jit(kernel)
        one = _compile(fn, {**_CONSTEXPRS, "NB": 1}, capability)
        assert one == _compile(fn, {**_CONSTEXPRS, "NB": 3}, capability), qualname
        assert one != _compile(fn, {**_CONSTEXPRS, "NB": 1, "BT": 16}, capability), f"{qualname}: control"


def test_flag_off_launches_carry_the_real_nb(launches):
    fix_fla_norm_nb()
    _launch_all()
    assert len(launches) == 4 and {nb for _, nb in launches} == {3}, launches


def test_flag_on_launches_fixed_nb_and_tunes_on_the_real_one(restored, launches):
    restored.setenv(FIXED_NB_FLAG, "1")
    fix_fla_norm_nb()
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
    fix_fla_norm_nb()
    fix_fla_norm_nb()
    for qualname, kernel in _registered():
        wrapper = triton_autotuner(kernel).fn
        assert isinstance(wrapper, fla_fixed_nb._FixedNB) and wrapper.fn is originals[qualname], qualname
        assert wrapper.cache_key == originals[qualname].cache_key


def test_finalize_loaded_model_applies_it_only_with_the_flag(restored):
    finalize_loaded_model(nn.Linear(2, 2))
    assert not any(isinstance(triton_autotuner(k).fn, fla_fixed_nb._FixedNB) for _, k in _registered())
    restored.setenv(FIXED_NB_FLAG, "1")
    finalize_loaded_model(nn.Linear(2, 2))
    assert all(isinstance(triton_autotuner(k).fn, fla_fixed_nb._FixedNB) for _, k in _registered())


def test_an_importable_module_not_yet_imported_is_left_alone(restored, tmp_path):
    (tmp_path / "halo_test_unimported_fla.py").write_text("")
    restored.syspath_prepend(str(tmp_path))
    restored.setenv(FIXED_NB_FLAG, "1")
    restored.setattr(fla_fixed_nb, "FIXED_NB_KERNELS", {"halo_test_unimported_fla": ("kernel",)})
    fix_fla_norm_nb()
    assert "halo_test_unimported_fla" not in sys.modules


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
    restored.setitem(sys.modules, "halo_test_fla_module", SimpleNamespace(kernel=_positional_nb_kernel))
    restored.setattr(fla_fixed_nb, "FIXED_NB_KERNELS", {"halo_test_fla_module": ("kernel",)})
    fix_fla_norm_nb()
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
    restored.setitem(sys.modules, "halo_test_fla_module", SimpleNamespace(kernel=_positional_nb_kernel))
    restored.setattr(fla_fixed_nb, "FIXED_NB_KERNELS", {"halo_test_fla_module": ("kernel",)})
    fix_fla_norm_nb()
    out = torch.empty(1)

    def grid(meta):
        return (meta["NB"],)

    _positional_nb_kernel[grid](out, 3, 7)
    _positional_nb_kernel[grid](out, NB=3, D=7)
    _positional_nb_kernel.warmup(out, 3, 7, grid=grid)
    _positional_nb_kernel.warmup(out, NB=3, D=7, grid=grid)
    _positional_nb_kernel.warmup(out, 3, 7, grid=(2,))
    assert recorded == [(False, 1, (3,))] * 2 + [(True, 1, (3,))] * 2 + [(True, 1, (2,))], recorded


@pytest.mark.parametrize("kernel", [_reads_nb_kernel, _takes_no_nb_kernel, triton.jit(_reads_nb_kernel.fn.fn)])
def test_a_kernel_that_no_longer_fits_raises(restored, kernel):
    restored.setenv(FIXED_NB_FLAG, "1")
    restored.setitem(sys.modules, "halo_test_fla_module", SimpleNamespace(kernel=kernel))
    restored.setattr(fla_fixed_nb, "FIXED_NB_KERNELS", {"halo_test_fla_module": ("kernel",)})
    with pytest.raises(RuntimeError, match=FIXED_NB_FLAG):
        fix_fla_norm_nb()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
