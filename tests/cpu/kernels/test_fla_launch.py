"""FLA's launch options through ``apply_fla_launch_options``. ``num_warps=4`` configs give run-to-run different,
partly wrong outputs on sm_10x, and the stock autotuner picks them at Qwen3.5/3.6 shapes; the q/k norm's constant-NB
launch keeps one compiled kernel and one tuned config per width."""

import ast
import inspect
import types

import pytest
import torch
import triton
import triton.language as tl
from fla.modules import l2norm as fla_l2norm
from fla.ops.common import chunk_delta_h
from triton.runtime.autotuner import Autotuner

from halo.kernels import fla_launch
from halo.kernels.fla_launch import (
    apply_fla_launch_options,
    patch_fla_l2norm_nb,
    triton_autotuner,
)
from halo.model_loading import finalization
from halo.model_loading.finalization import finalize_loaded_model

_STATE_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def _state_autotuner() -> Autotuner:
    return triton_autotuner(getattr(chunk_delta_h, _STATE_KERNEL))


def _warps() -> set[int]:
    return {config.num_warps for config in _state_autotuner().configs}


@pytest.fixture
def gpu(monkeypatch):
    """Restores the state kernel's configs and cache."""
    autotuner = _state_autotuner()
    configs, cache = list(autotuner.configs), dict(autotuner.cache)

    def as_major(major: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (major, 0))

    yield as_major
    autotuner.configs, autotuner.cache = configs, cache


def test_the_locked_fla_offers_num_warps_4_here():
    """Non-vacuity: without ``num_warps=4`` configs to drop, the restriction tests prove nothing."""
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


L2NORM_KNOB = "HALO_FLA_L2NORM_FIXED_NB"
_CONFIGS = [triton.Config({"BT": 8}, num_warps=1), triton.Config({"BT": 16}, num_warps=2)]


@triton.jit
def _ignores_nb(x, D: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    tl.store(x, tl.load(x) + D + BT)


@triton.jit
def _reads_nb(x, D: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    tl.store(x, tl.load(x) + NB)


def _tuner(jit_function, keys=("D", "NB")) -> Autotuner:
    return triton.autotune(configs=_CONFIGS, key=list(keys))(jit_function)


@pytest.fixture
def fake_l2norm(monkeypatch):
    """A module of two autotuned kernels keyed on ``D`` and ``NB``, as fla 0.4.2's."""
    module = types.ModuleType("fla.modules.l2norm")
    module.l2norm_fwd_kernel, module.l2norm_bwd_kernel = _tuner(_ignores_nb), _tuner(_ignores_nb)
    monkeypatch.setattr(fla_launch, "fla_l2norm", module)
    return module


def test_install_pins_nb_out_of_the_tuning_key_and_is_idempotent(fake_l2norm):
    patch_fla_l2norm_nb()
    wrapped = [fake_l2norm.l2norm_fwd_kernel.fn, fake_l2norm.l2norm_bwd_kernel.fn]
    assert all(isinstance(fn, fla_launch._ConstantNB) for fn in wrapped)
    assert fake_l2norm.l2norm_fwd_kernel.keys == ["D"] == fake_l2norm.l2norm_bwd_kernel.keys
    patch_fla_l2norm_nb()
    assert [fake_l2norm.l2norm_fwd_kernel.fn, fake_l2norm.l2norm_bwd_kernel.fn] == wrapped


def test_a_launch_runs_with_a_constant_nb_and_reads_through(fake_l2norm):
    class Recorder:
        arg_names = ["x", "D", "NB", "BT"]

        def run(self, *args, **kwargs):
            return kwargs

    launcher = fla_launch._ConstantNB(Recorder())
    first, second = launcher.run(NB=1, BT=8, D=128), launcher.run(NB=57, BT=8, D=128)
    assert first == second == {"NB": 1, "BT": 8, "D": 128}
    assert launcher.arg_names == Recorder.arg_names


def test_a_kernel_reading_nb_refuses_and_stays_as_it_was(fake_l2norm):
    fake_l2norm.l2norm_bwd_kernel = _tuner(_reads_nb)
    with pytest.raises(RuntimeError, match="reads NB in its body"):
        patch_fla_l2norm_nb()
    assert fake_l2norm.l2norm_bwd_kernel.keys == ["D", "NB"]
    assert not isinstance(fake_l2norm.l2norm_bwd_kernel.fn, fla_launch._ConstantNB)


@pytest.mark.parametrize(
    "kernel", [lambda: None, lambda: _tuner(_ignores_nb, keys=("D",))], ids=["not-autotuned", "not-keyed-on-nb"]
)
def test_a_kernel_this_fla_does_not_key_on_nb_fails_loud(fake_l2norm, kernel):
    fake_l2norm.l2norm_fwd_kernel = kernel()
    with pytest.raises(RuntimeError, match="not an autotuned FLA kernel keyed on NB"):
        patch_fla_l2norm_nb()


def test_the_locked_fla_kernels_pass_the_install_checks():
    """Non-vacuity: the checks above refuse a fla whose kernels do not look like these, so these must pass them."""
    for name in fla_launch._L2NORM_KERNELS:
        tuner = triton_autotuner(getattr(fla_l2norm, name))
        assert "NB" in tuner.keys or isinstance(tuner.fn, fla_launch._ConstantNB)
        assert not fla_launch._reads_nb(tuner.fn.fn if isinstance(tuner.fn, fla_launch._ConstantNB) else tuner.fn)


def test_the_knob_installs_the_launch_and_off_leaves_fla_alone(fake_l2norm, monkeypatch):
    monkeypatch.delenv(L2NORM_KNOB, raising=False)
    apply_fla_launch_options()
    assert fake_l2norm.l2norm_fwd_kernel.keys == ["D", "NB"]
    monkeypatch.setenv(L2NORM_KNOB, "1")
    apply_fla_launch_options()
    assert fake_l2norm.l2norm_fwd_kernel.keys == ["D"]


@pytest.mark.parametrize("restrict_first", [True, False], ids=["restriction-first", "norm-first"])
def test_the_blackwell_restriction_and_the_norm_launch_compose(gpu, fake_l2norm, monkeypatch, restrict_first):
    monkeypatch.setenv(L2NORM_KNOB, "1")
    gpu(10)
    if not restrict_first:
        patch_fla_l2norm_nb()
    apply_fla_launch_options()
    assert _warps() == {2}
    assert isinstance(fake_l2norm.l2norm_fwd_kernel.fn, fla_launch._ConstantNB)


def test_a_launch_source_without_a_function_fails_loud(fake_l2norm):
    fake_l2norm.l2norm_bwd_kernel.fn = types.SimpleNamespace(src="NB = 1", __name__="l2norm_bwd_kernel")
    with pytest.raises(RuntimeError, match="l2norm_bwd_kernel has no function definition.*patch_fla_l2norm_nb"):
        patch_fla_l2norm_nb()


def test_the_warm_rows_are_the_nb_bucket_the_locked_fla_computes():
    """FLA buckets rows as ``NB = cdiv(T, 2048 * 32)``: the warm size must be one bucket's worth, so a change of that
    formula fails here instead of silently warming another bucket."""
    source = ast.parse(inspect.getsource(fla_l2norm.l2norm_fwd))
    (assignment,) = (
        node for node in ast.walk(source) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "NB"
    )
    divisor = eval(compile(ast.Expression(assignment.value.args[1]), "<fla>", "eval"))  # noqa: S307
    assert assignment.value.func.attr == "cdiv" and divisor == fla_launch._L2NORM_WARM_ROWS


class _GatedDeltaRule(torch.nn.Module):
    def __init__(self, head_k_dim: int, device_type: str = "cuda"):
        super().__init__()
        self.head_k_dim = head_k_dim
        self._device = torch.device(device_type)

    def parameters(self, recurse=True):
        return iter([types.SimpleNamespace(device=self._device)])


@pytest.fixture
def l2norm_launches(monkeypatch):
    """Every forward and backward the warm-up launches, with the launches and their CUDA tensors replaced."""
    launches = []

    def forward(x):
        launches.append(("fwd", x.shape[-1], x.dtype))
        return x, x[:, 0]

    def backward(y, rstd, x):
        launches.append(("bwd", x.shape[-1], x.dtype))

    monkeypatch.setattr(fla_launch, "fla_l2norm", types.SimpleNamespace(l2norm_fwd=forward, l2norm_bwd=backward))
    monkeypatch.setattr(torch, "randn", lambda *shape, device, dtype: torch.zeros(shape, dtype=dtype))
    return launches


def _mixers(*widths, device_type="cuda"):
    return torch.nn.ModuleList(_GatedDeltaRule(width, device_type) for width in widths)


def test_warming_without_the_knob_launches_nothing(l2norm_launches, monkeypatch):
    monkeypatch.delenv(L2NORM_KNOB, raising=False)
    fla_launch.warm_fla_l2norm(_mixers(128))
    assert l2norm_launches == []


def test_warming_off_cuda_or_without_a_gated_delta_rule_launches_nothing(l2norm_launches, monkeypatch):
    monkeypatch.setenv(L2NORM_KNOB, "1")
    fla_launch.warm_fla_l2norm(_mixers(128, device_type="cpu"))
    fla_launch.warm_fla_l2norm(torch.nn.Linear(2, 2))
    assert l2norm_launches == []


def test_warming_runs_each_width_once_in_every_dtype_both_ways(l2norm_launches, monkeypatch):
    monkeypatch.setenv(L2NORM_KNOB, "1")
    fla_launch.warm_fla_l2norm(_mixers(128, 128, 64))
    expected = {
        (direction, width, dtype)
        for direction in ("fwd", "bwd")
        for width in (64, 128)
        for dtype in fla_launch._L2NORM_WARM_DTYPES
    }
    assert len(l2norm_launches) == len(expected) and set(l2norm_launches) == expected


def test_finalizing_warms_the_norm_of_the_models_own_layers(monkeypatch):
    warmed = []
    monkeypatch.setattr(finalization, "warm_fla_l2norm", warmed.append)
    model = torch.nn.Linear(2, 2)
    finalize_loaded_model(model)
    assert warmed == [model]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
