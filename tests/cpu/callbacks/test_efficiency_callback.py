#!/usr/bin/env python
"""Tests for EfficiencyCallback. Run: python tests/cpu/callbacks/test_efficiency_callback.py"""

import sys

import pytest
import torch

from tests.common.parallelism import make_parallelism_config

# Single-process topology: a test that exercises no parallel axis still owes the callback a real
# config, because every divisor it applies is read off one.
_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)

# Mock helpers


class MockParam:
    """A lightweight mock for torch.nn.Parameter."""

    def __init__(self, numel_val, dtype=torch.bfloat16, requires_grad=True):
        self._numel = numel_val
        self.dtype = dtype
        self.requires_grad = requires_grad
        self._data = torch.randn(numel_val, dtype=dtype) if numel_val <= 1000 else None

    def numel(self):
        return self._numel


class MockModel:
    """A mock model with configurable named_parameters and config."""

    def __init__(self, named_params=None, config=None, dtype=None):
        self._named_params = named_params or []
        self.config = config
        if dtype is not None:
            self.dtype = dtype

    def parameters(self):
        return iter([p for _, p in self._named_params])

    def named_parameters(self):
        return iter(self._named_params)


class MockConfig:
    """Mock model config with arbitrary attributes.

    ``get_text_config`` mirrors ``PretrainedConfig``: every real config exposes it, returning the
    language tower's config (itself, for a text-only model).
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_text_config(self):
        return self


class MockTrainingArgs:
    """Mock TrainingArguments with common fields."""

    def __init__(self, **kwargs):
        self.bf16 = kwargs.get("bf16", False)
        self.fp16 = kwargs.get("fp16", False)
        self.tf32 = kwargs.get("tf32", False)
        self.include_num_input_tokens_seen = kwargs.get("include_num_input_tokens_seen", True)
        self.logging_steps = kwargs.get("logging_steps", 1)
        self.max_length = kwargs.get("max_length", 2048)
        self.per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)


class MockTrainerState:
    """Mock TrainerState."""

    def __init__(self, global_step=0, max_steps=100, num_input_tokens_seen=None):
        self.global_step = global_step
        self.max_steps = max_steps
        self.num_input_tokens_seen = num_input_tokens_seen


# Tests


def test_classify_gpu_name():
    from src.hardware import _classify_gpu_name

    cases = [
        ("NVIDIA H100 80GB HBM3", "H100_SXM"),
        ("NVIDIA H100 PCIE", "H100_PCIE"),
        ("NVIDIA H100 NVL", "H100_NVL"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA B200", "B200"),
        ("NVIDIA GB200", "GB200"),
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA A6000", "A6000"),
        ("Unknown GPU", None),
    ]
    for gpu_name, expected in cases:
        result = _classify_gpu_name(gpu_name.upper())
        assert result == expected, f"_classify_gpu_name({gpu_name!r}) = {result!r}, expected {expected!r}"


def test_detect_precision_bf16_arg():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs(bf16=True)
    model = MockModel()
    assert _detect_precision(model, args) == "bf16"


def test_detect_precision_fp16_arg():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs(fp16=True)
    model = MockModel()
    assert _detect_precision(model, args) == "fp16"


def test_detect_precision_tf32_arg():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs(tf32=True)
    model = MockModel()
    assert _detect_precision(model, args) == "tf32"


def test_detect_precision_model_dtype():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs()
    model = MockModel(dtype=torch.bfloat16)
    assert _detect_precision(model, args) == "bf16"


def test_detect_precision_param_fallback():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs()
    p = MockParam(100, dtype=torch.float16)
    model = MockModel(named_params=[("w", p)])
    # Model has no dtype attribute, so the function falls through to first param
    result = _detect_precision(model, args)
    assert result == "fp16", f"Expected 'fp16' from param fallback, got {result!r}"


def test_detect_precision_empty_model():
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs()
    model = MockModel()  # no dtype, no params
    result = _detect_precision(model, args)
    assert result == "bf16", f"Expected 'bf16' default, got {result!r}"


def test_detect_precision_fp32_is_tf32_aware():
    """An fp32-param model maps to the precision its matmuls actually run at.

    On Ampere+ fp32 matmuls execute on TF32 Tensor Cores when
    float32_matmul_precision='high' (the toolkit/NGC default) — keying MFU off the
    true-fp32 peak then yields MFU > 100% (the achieved TF32 FLOPs exceed the fp32
    peak). 'highest' is true fp32; 'medium' uses bf16.
    """
    from src.callbacks.efficiency import _detect_precision

    args = MockTrainingArgs()
    model = MockModel(dtype=torch.float32)
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert _detect_precision(model, args) == "tf32"
        torch.set_float32_matmul_precision("medium")
        assert _detect_precision(model, args) == "bf16"
        torch.set_float32_matmul_precision("highest")
        # 'highest' disables TF32 only if allow_tf32 is also off
        if not torch.backends.cuda.matmul.allow_tf32:
            assert _detect_precision(model, args) == "fp32"
    finally:
        torch.set_float32_matmul_precision(prev)


def test_every_low_precision_mode_has_a_peak_key():
    """A ``lowp_precision`` the peak table does not spell falls back to the parameter dtype — i.e.
    silently to the bf16 peak, the very mis-scaling the map exists to prevent, one mode later."""
    from src.callbacks.efficiency import _PRECISION_KEY_BY_LOWP
    from src.distributed.parallelism_config import LOWP_PRECISIONS

    assert set(_PRECISION_KEY_BY_LOWP) == set(LOWP_PRECISIONS) - {"bf16"}


@pytest.mark.parametrize(
    ("lowp_precision", "expected"),
    [("fp8", "fp8"), ("fp4", "fp4"), ("mxfp4", "fp4"), ("bf16", "bf16")],
)
def test_low_precision_compute_scores_against_its_own_peak(monkeypatch, lowp_precision, expected):
    """MFU divides by the peak the GEMMs are actually issued at.

    ``lowp_precision`` keeps the masters in bf16 while the matmuls run in fp8/fp4, so a key derived
    from the parameter dtype charges an fp8 run against the bf16 peak and every reported utilization
    reads ~2x high (4x for fp4). nvfp4 and mxfp4 share the 4-bit MMA, hence one peak.
    """
    from src.callbacks import efficiency
    from src.hardware import GPU_PEAK_FLOPS

    monkeypatch.setattr(efficiency, "detect_gpu_model", lambda: "B300")
    callback = efficiency.EfficiencyCallback(
        make_parallelism_config(world_size=1, gpus_per_node=1, lowp_precision=lowp_precision)
    )
    callback._initialize_metrics(MockTrainingArgs(bf16=True))

    assert callback.state.precision == expected
    assert callback.mfu.precision == expected
    assert callback.state.gpu_peak_flops == GPU_PEAK_FLOPS["B300"].flops[expected]


def test_get_gpu_peak_flops():
    from src.hardware import get_gpu_peak_flops

    assert get_gpu_peak_flops("A100", "fp8") is None  # No FP8 on Ampere
    assert get_gpu_peak_flops("Unknown", "bf16") is None


def test_is_expert_param():
    from src.callbacks.model_flops import _is_expert_param

    # EP fused expert params -- True
    assert _is_expert_param("model.layers.0.mlp.gate_up_proj") is True
    assert _is_expert_param("model.layers.0.mlp.down_proj") is True

    # GptOss grouped-GEMM de-interleaved experts (use_grouped_gemm, on by default) -- True.
    # The fused gate_up_proj is split into gate_proj_gmm/up_proj_gmm; missing these
    # undercounts GptOss experts by the gate+up share (~2/3), inflating S-MFU ~3x.
    assert _is_expert_param("model.layers.0.mlp.gate_proj_gmm") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj_gmm") is True
    assert _is_expert_param("model.layers.0.mlp.gate_proj_gmm_bias") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj_gmm_bias") is True

    # Separate-projection experts + biases (Qwen3, Bailing, GptOss expert-TP shards) -- True
    assert _is_expert_param("model.layers.0.mlp.gate_proj") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj") is True
    assert _is_expert_param("model.layers.0.mlp.gate_up_proj_bias") is True
    assert _is_expert_param("model.layers.0.mlp.down_proj_bias") is True

    # Shared expert -- False
    assert _is_expert_param("model.layers.0.mlp.shared_expert.down_proj") is False

    # Router / gate -- False
    assert _is_expert_param("model.layers.0.mlp.gate.weight") is False
    assert _is_expert_param("model.layers.0.mlp.router.weight") is False

    # HF MoE unpatched -- True
    assert _is_expert_param("model.layers.0.mlp.experts.0.w1.weight") is True

    # EP individual experts -- True
    # Pre-5.14 Zaya nesting ("local_experts") is unreadable by the current stack — not expert params.
    assert _is_expert_param("model.layers.0.mlp.local_experts.w1") is False

    # Attention -- False
    assert _is_expert_param("model.layers.0.self_attn.q_proj.weight") is False


def test_detect_moe_config_num_local_experts():
    from src.models.moe_balancing import detect_moe_experts_topk

    config = MockConfig(num_local_experts=32, num_experts_per_tok=4)
    model = MockModel(config=config)
    ne, tk = detect_moe_experts_topk(model)
    assert ne == 32, f"Expected num_experts=32, got {ne}"
    assert tk == 4, f"Expected top_k=4, got {tk}"


def test_detect_moe_config_num_experts():
    from src.models.moe_balancing import detect_moe_experts_topk

    config = MockConfig(num_experts=16, top_k=2)
    model = MockModel(config=config)
    ne, tk = detect_moe_experts_topk(model)
    assert ne == 16, f"Expected num_experts=16, got {ne}"
    assert tk == 2, f"Expected top_k=2, got {tk}"


def test_detect_moe_config_no_moe():
    from src.models.moe_balancing import detect_moe_experts_topk

    config = MockConfig(hidden_size=4096)  # no MoE attrs
    model = MockModel(config=config)
    ne, tk = detect_moe_experts_topk(model)
    assert ne == 0 and tk == 0, f"Expected (0, 0), got ({ne}, {tk})"


def test_detect_moe_config_no_config():
    from src.models.moe_balancing import detect_moe_experts_topk

    model = MockModel()  # no config attribute at all
    model.config = None
    ne, tk = detect_moe_experts_topk(model)
    assert ne == 0 and tk == 0, f"Expected (0, 0), got ({ne}, {tk})"


def test_estimate_model_flops():
    from src.callbacks.model_flops import estimate_model_flops_per_token

    # 1B trainable params -> 6e9 flops per token
    num_params = 1_000_000_000
    # Create a single large parameter to represent ~1B params
    # We use a small tensor but override numel via a wrapper
    p = MockParam(num_params, dtype=torch.bfloat16, requires_grad=True)
    model = MockModel(named_params=[("weight", p)])
    flops = estimate_model_flops_per_token(model)
    assert flops == 6 * num_params, f"Expected {6 * num_params}, got {flops}"


def test_compute_expert_params():
    from src.callbacks.model_flops import compute_expert_params

    expert_p1 = MockParam(1000, requires_grad=True)
    expert_p2 = MockParam(2000, requires_grad=True)
    shared_p = MockParam(500, requires_grad=True)
    attn_p = MockParam(3000, requires_grad=True)

    named_params = [
        ("model.layers.0.mlp.gate_up_proj", expert_p1),
        ("model.layers.0.mlp.down_proj", expert_p2),
        ("model.layers.0.mlp.shared_expert.down_proj", shared_p),
        ("model.layers.0.self_attn.q_proj.weight", attn_p),
    ]
    model = MockModel(named_params=named_params)
    total_expert = compute_expert_params(model)
    assert total_expert == 3000.0, f"Expected 3000.0, got {total_expert}"


def test_efficiency_callback_init():
    from src.callbacks.efficiency import EfficiencyCallback

    parallelism = make_parallelism_config(ep_size=2, tp_size=2)
    cb = EfficiencyCallback(
        parallelism,
        n_warmup_steps=5,
        num_full_model_params=10e9,
    )
    assert cb.state.n_warmup_steps == 5
    assert cb._parallelism_config is parallelism
    assert cb._full_model_params == 10e9


def test_warmup_skipping():
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=5)
    cb.state.global_start_step = 0

    args = MockTrainingArgs()
    state = MockTrainerState(global_step=2, num_input_tokens_seen=1000)
    control = None

    # Before warmup, elapsed_step should stay 0
    cb.state.step_start_time = 1.0  # fake start time
    cb.on_step_end(args, state, control)
    assert cb.state.elapsed_step == 0, f"Expected elapsed_step=0 during warmup, got {cb.state.elapsed_step}"


# Core accounting: estimate_model_flops_per_token, MFU, S-MFU, throughput.
# These pin the *values* of the numbers the callback reports to wandb.


def test_estimate_model_flops_includes_attention():
    """6N param term PLUS the 12*L*S*H attention term when a config is present."""
    from src.callbacks.model_flops import estimate_model_flops_per_token

    n_params = 1_000_000
    p = MockParam(n_params, requires_grad=True)
    config = MockConfig(num_hidden_layers=4, hidden_size=128)
    model = MockModel(named_params=[("w", p)], config=config)

    seq_len = 512
    flops = estimate_model_flops_per_token(model, seq_length=seq_len)
    expected = 6 * n_params + 12 * 4 * seq_len * 128
    assert flops == expected, f"expected {expected}, got {flops}"


def test_estimate_model_flops_config_fallback():
    """When param counting yields zero AND no requires_grad params, fall to 6N over all params."""
    from src.callbacks.model_flops import estimate_model_flops_per_token

    # All params frozen → requires_grad sum is 0, falls to all-params sum.
    p = MockParam(2000, requires_grad=False)
    model = MockModel(named_params=[("w", p)], config=MockConfig())
    flops = estimate_model_flops_per_token(model, seq_length=128)
    # No num_hidden_layers/hidden_size on the bare config → 0 attention flops.
    assert flops == 6 * 2000


def test_compute_mfu_value():
    """MFU% = (tokens * flops_per_token / step_time) / peak * 100, exactly."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.state.model_flops_per_token = 6.0e9  # 6 GFLOP/token
    cb.state.gpu_peak_flops = 1.0e15  # 1 PFLOP/s
    cb.state.elapsed_time = 1.0  # so avg == step on the first measured step

    tokens = 1000
    step_time = 0.5
    cb._compute_mfu(tokens, step_time)

    # step_flops = 1000 * 6e9 = 6e12; /0.5s = 1.2e13 flops/s; /1e15 *100 = 1.2%
    assert cb.mfu.step_mfu_percent == pytest.approx(1.2, abs=1e-6)
    assert cb.mfu.step_tflops_per_sec == pytest.approx(12.0, abs=1e-6)  # 1.2e13 / 1e12 TFLOPS
    # total_flops accumulates the step flops.
    assert cb.state.total_flops == pytest.approx(6e12, rel=1e-9)


def test_compute_mfu_distributed_efficiency():
    """params_ratio > 1.05 turns MFU into a distributed-efficiency speedup figure."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.state.model_flops_per_token = 6.0e9
    cb.state.gpu_peak_flops = 1.0e15
    cb.state.elapsed_time = 1.0
    cb.mfu.params_ratio = 4.0  # local model is 1/4 of the full model

    cb._compute_mfu(step_tokens_seen=1000, step_time=0.5)
    # step_mfu = 1.2% → distributed_efficiency = 4.0 * 1.2 / 100 = 0.048
    assert cb.mfu.step_distributed_efficiency == pytest.approx(0.05, abs=1e-2)


def test_compute_mfu_zeroed_when_unconfigured():
    """No peak flops / no model flops → all MFU fields zeroed (not stale)."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.state.model_flops_per_token = None
    cb.state.gpu_peak_flops = None
    cb._compute_mfu(1000, 0.5)
    assert cb.mfu.step_mfu_percent == 0.0
    assert cb.mfu.step_tflops_per_sec == 0.0


def test_compute_smfu_sparse_value():
    """S-MFU uses the active (sparse) flops/token, which is lower than dense MFU."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.state.active_model_flops_per_token = 1.5e9  # quarter of the dense below
    cb.state.gpu_peak_flops = 1.0e15
    cb.state.elapsed_time = 1.0

    cb._compute_smfu(step_tokens_seen=1000, step_time=0.5)
    # step_active_flops = 1000 * 1.5e9 = 1.5e12; /0.5 = 3e12; /1e15*100 = 0.3%
    assert cb.smfu.step_smfu_percent == pytest.approx(0.3, abs=1e-6)
    assert cb.smfu.step_smfu_tflops_per_sec == pytest.approx(3.0, abs=1e-6)


def test_compute_token_metrics_per_gpu_and_cluster(monkeypatch):
    """Per-GPU tokens = cluster/world; cluster tokens = per_gpu * data_parallel_size * cp."""
    from src.callbacks import efficiency
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(make_parallelism_config(tp_size=2, world_size=4, gpus_per_node=4))
    cb.training_args = MockTrainingArgs()
    # Pretend world_size is 4 (no real dist init → patch the helper).
    monkeypatch.setattr(efficiency, "get_global_world_size", lambda: 4)

    state = MockTrainerState(num_input_tokens_seen=8000)
    cb.state.step_start_tokens_seen = 0  # so the step delta is the full 8000
    cb.state.elapsed_time = 2.0  # nonzero so the avg-TPS divisor is valid

    step_tokens = cb._compute_token_metrics(state, step_time=2.0)
    # cluster delta = 8000; per-gpu = 8000 / world(4) = 2000.
    assert step_tokens == 2000
    # per-GPU TPS = 2000 / 2.0s = 1000.
    assert cb.tps.step_tokens_per_second == pytest.approx(1000.0, abs=1e-6)
    # data_parallel_size = 4 / max(tp=2) = 2; cluster tokens = 2000 * 2 * cp(1) = 4000.
    # cluster TPS = 4000 / 2.0 = 2000.
    assert cb.tps.step_cluster_tokens_per_second == pytest.approx(2000.0, abs=1e-6)


def test_compute_token_metrics_cp_divides_per_gpu(monkeypatch):
    """Under CP the Trainer over-counts (full sequence per rank) → divide by cp_size."""
    from src.callbacks import efficiency
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(make_parallelism_config(cp_size=2, world_size=4, gpus_per_node=4))
    cb.training_args = MockTrainingArgs()
    monkeypatch.setattr(efficiency, "get_global_world_size", lambda: 4)  # dp = 4 / cp(2) = 2

    state = MockTrainerState(num_input_tokens_seen=8000)
    cb.state.step_start_tokens_seen = 0
    cb.state.elapsed_time = 1.0  # nonzero so the avg-TPS divisor is valid

    step_tokens = cb._compute_token_metrics(state, step_time=1.0)
    # per-gpu raw = 8000/4 = 2000; CP halves it → 1000.
    assert step_tokens == 1000
    # cluster = per_gpu(1000) * data_parallel_size(2) * cp(2) = 4000 (reconstructs full seq).
    assert cb.tps.step_cluster_tokens_per_second == pytest.approx(4000.0, abs=1e-6)


def test_on_log_gates_diagnostics():
    """MFU diagnostics stay OUT of the headline log unless report_mfu_diagnostics=True."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=0)
    cb.state.global_start_step = 0
    cb.tps.step_tokens_per_second = 1234.0
    cb.mfu.step_mfu_percent = 42.0

    logs: dict = {}
    cb.on_log(MockTrainingArgs(), MockTrainerState(global_step=5), None, logs=logs)
    assert "step_tokens_per_second" in logs  # headline always present
    assert "step_mfu_percent" not in logs  # diagnostic gated off by default

    cb._report_mfu_diagnostics = True
    logs2: dict = {}
    cb.on_log(MockTrainingArgs(), MockTrainerState(global_step=5), None, logs=logs2)
    assert logs2["step_mfu_percent"] == 42.0


def test_initialize_metrics_survives_unmeasurable_model():
    """A model the estimator cannot measure disables MFU instead of killing the run.

    ``estimate_model_flops_per_token`` fails loud (no parameters AND no hidden_size /
    num_hidden_layers in the config); the callback is observability, so it must degrade.
    """
    from src.callbacks.efficiency import EfficiencyCallback
    from src.callbacks.model_flops import estimate_model_flops_per_token

    shell = MockModel(named_params=[], config=MockConfig(model_type="mystery"))
    with pytest.raises(ValueError):  # the estimator itself stays fail-loud
        estimate_model_flops_per_token(shell)

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.model_ref = shell
    cb._initialize_metrics(MockTrainingArgs(bf16=True))  # must not raise

    assert cb.state.model_flops_per_token is None, "MFU must be disabled, not guessed"
    assert cb.state.active_model_flops_per_token is None, "S-MFU must be disabled too"


def test_initialize_metrics_measures_normal_model():
    """The degrade path must not swallow the normal case: a countable model still gets FLOPS/token."""
    from src.callbacks.efficiency import EfficiencyCallback

    model = MockModel(
        named_params=[("model.layers.0.self_attn.q_proj.weight", MockParam(1_000_000))],
        config=MockConfig(hidden_size=128, num_hidden_layers=2),
    )
    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.model_ref = model
    cb._initialize_metrics(MockTrainingArgs(bf16=True, max_length=512))

    assert cb.state.model_flops_per_token == 6 * 1_000_000 + 12 * 2 * 512 * 128
    assert cb.mfu.local_params == 1_000_000.0


def test_on_log_skips_during_warmup():
    """Within the warmup window on_log emits nothing."""
    from src.callbacks.efficiency import EfficiencyCallback

    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=3)
    cb.state.global_start_step = 0
    cb.tps.step_tokens_per_second = 999.0

    logs: dict = {}
    cb.on_log(MockTrainingArgs(), MockTrainerState(global_step=2), None, logs=logs)
    assert logs == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
