#!/usr/bin/env python

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from halo.callbacks import efficiency, factory
from halo.callbacks.efficiency import EfficiencyCallback, MFUMetrics, StepTimeMetrics, ThroughputMetrics
from halo.callbacks.model_flops import RankFlopsModel, _is_expert_param, estimate_rank_flops
from halo.distributed.parallelism_rules import SUPPORTED_AXIS_SETS
from halo.hardware import GPU_PEAK_FLOPS, _classify_gpu_name, get_gpu_peak_flops, peak_precision_key
from halo.kernels.lowp.spec import LowpSpec
from halo.models.moe.config_fields import resolve_expert_count, resolve_router_topk
from tests.common.parallelism import make_parallelism_config

# Every divisor the callback applies is read off a real config, so axis-free tests still pass one.
_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)


class MockParam:
    def __init__(self, numel_val, dtype=torch.bfloat16, requires_grad=True):
        self._numel = numel_val
        self.dtype = dtype
        self.requires_grad = requires_grad
        self._data = torch.randn(numel_val, dtype=dtype) if numel_val <= 1000 else None

    def numel(self):
        return self._numel


class MockModel:
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
    """``get_text_config`` mirrors ``PretrainedConfig``: the language tower's config, itself for a text-only model."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_text_config(self):
        return self


class MockTrainingArgs:
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
    def __init__(self, global_step=0, max_steps=100, num_input_tokens_seen=None):
        self.global_step = global_step
        self.max_steps = max_steps
        self.num_input_tokens_seen = num_input_tokens_seen


_DENSE_FLOPS = RankFlopsModel(
    model_flops_per_token=6.0e9,
    active_model_flops_per_token=6.0e9,
    attention_flops_per_token=0.0,
    attention_layout=None,
    attention_share=1.0,
    local_params=1.0e9,
    full_params=1.0e9,
    params_ratio=1.0,
    num_experts=0,
    top_k=0,
    sparsity_factor=1.0,
    local_expert_params=0.0,
    local_active_params=1.0e9,
    full_active_params=1.0e9,
)


def _rank_flops(model, parallelism=_NO_PARALLELISM, seq_len=128) -> RankFlopsModel:
    return estimate_rank_flops(model, seq_len=seq_len, parallelism_config=parallelism, full_model_params=None)


def _precision(model, args) -> str:
    cb = EfficiencyCallback(make_parallelism_config(world_size=1, gpus_per_node=1))
    cb.on_init_end(args, MockTrainerState(), None, model=model)
    return cb.precision


def _peaked(
    flops: RankFlopsModel | None = _DENSE_FLOPS, peak: float | None = 1.0e15, parallelism=_NO_PARALLELISM
) -> EfficiencyCallback:
    cb = EfficiencyCallback(parallelism, n_warmup_steps=0)
    cb.flops_model, cb.gpu_peak_flops = flops, peak
    return cb


class _HostClock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _HostClock:
    host = _HostClock()
    monkeypatch.setattr(efficiency, "time", host)
    return host


def _measured_step(clock: _HostClock, cb, *, tokens=1000, step_time=0.5, step=1) -> None:
    """One optimizer step through the hooks, then the log where the callback records it. ``tokens=None``: uncounted."""
    cb.training_args = cb.training_args or MockTrainingArgs()
    cb.on_step_begin(cb.training_args, MockTrainerState(global_step=step - 1), None)
    clock.now += step_time
    cb.on_step_end(cb.training_args, MockTrainerState(global_step=step, num_input_tokens_seen=tokens), None)
    cb.on_log(cb.training_args, MockTrainerState(global_step=step), None, logs={})


def test_classify_gpu_name():
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


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (MockTrainingArgs(bf16=True), "bf16"),
        (MockTrainingArgs(fp16=True), "fp16"),
        (MockTrainingArgs(tf32=True), "tf32"),
        (MockTrainingArgs(), "bf16"),  # no flag, a model with no dtype and no params: the default
    ],
)
def test_the_training_arguments_pick_the_precision(args, expected):
    assert _precision(MockModel(), args) == expected


def test_the_model_dtype_picks_the_precision():
    assert _precision(MockModel(dtype=torch.bfloat16), MockTrainingArgs()) == "bf16"


def test_the_first_parameter_picks_the_precision_without_a_model_dtype():
    model = MockModel(named_params=[("w", MockParam(100, dtype=torch.float16))])
    assert _precision(model, MockTrainingArgs()) == "fp16"


def test_the_fp32_peak_key_is_tf32_aware():
    """On Ampere+ fp32 matmuls run on TF32 Tensor Cores at ``float32_matmul_precision='high'``, so keying MFU off the
    true-fp32 peak yields MFU > 100%. 'highest' is true fp32; 'medium' uses bf16."""
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        assert peak_precision_key(torch.float32) == "tf32"
        torch.set_float32_matmul_precision("medium")
        assert peak_precision_key(torch.float32) == "bf16"
        torch.set_float32_matmul_precision("highest")
        # 'highest' disables TF32 only if allow_tf32 is also off
        if not torch.backends.cuda.matmul.allow_tf32:
            assert peak_precision_key(torch.float32) == "fp32"
    finally:
        torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize(
    ("lowp_precision", "expected"),
    [("fp8", "fp8"), ("fp4", "fp4"), ("mxfp4", "fp4"), ("bf16", "bf16")],
)
def test_low_precision_compute_scores_against_its_own_peak(monkeypatch, lowp_precision, expected):
    """A key from the parameter dtype charges an fp8 run against the bf16 peak, so utilization reads ~2x high (4x for
    fp4). nvfp4 and mxfp4 share the 4-bit MMA, hence one peak."""
    monkeypatch.setattr(efficiency, "detect_gpu_model", lambda: "B300")
    callback = efficiency.EfficiencyCallback(
        make_parallelism_config(world_size=1, gpus_per_node=1, lowp=LowpSpec(precision=lowp_precision))
    )
    callback.on_init_end(MockTrainingArgs(bf16=True), MockTrainerState(), None)

    assert callback.precision == expected
    assert callback.gpu_peak_flops == GPU_PEAK_FLOPS["B300"].flops[expected]


def test_a_detected_gpu_without_a_peak_for_the_precision_warns(monkeypatch, caplog):
    """A missing table entry zeroes MFU, S-MFU and TFLOP/s for the whole run, so INFO can't be the only trace."""
    monkeypatch.setattr(efficiency, "detect_gpu_model", lambda: "A100")
    callback = efficiency.EfficiencyCallback(
        make_parallelism_config(world_size=1, gpus_per_node=1, lowp=LowpSpec(precision="fp8"))
    )
    with caplog.at_level(logging.WARNING, logger=efficiency.__name__):
        callback.on_init_end(MockTrainingArgs(bf16=True), MockTrainerState(), None)

    assert callback.gpu_peak_flops is None, "test premise: A100 has no fp8 peak"
    assert [r for r in caplog.records if "has no fp8 peak for A100" in r.getMessage()]


def test_get_gpu_peak_flops():
    assert get_gpu_peak_flops("A100", "fp8") is None  # No FP8 on Ampere
    assert get_gpu_peak_flops("Unknown", "bf16") is None


def test_is_expert_param():
    # EP fused experts
    assert _is_expert_param("model.layers.0.mlp.gate_up_proj") is True
    assert _is_expert_param("model.layers.0.mlp.down_proj") is True

    # GptOss grouped-GEMM splits gate_up_proj; missing these undercounts its experts by ~2/3, inflating S-MFU ~3x.
    assert _is_expert_param("model.layers.0.mlp.gate_proj_gmm") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj_gmm") is True
    assert _is_expert_param("model.layers.0.mlp.gate_proj_gmm_bias") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj_gmm_bias") is True

    # Separate-projection experts and biases (Qwen3, Bailing, GptOss expert-TP shards)
    assert _is_expert_param("model.layers.0.mlp.gate_proj") is True
    assert _is_expert_param("model.layers.0.mlp.up_proj") is True
    assert _is_expert_param("model.layers.0.mlp.gate_up_proj_bias") is True
    assert _is_expert_param("model.layers.0.mlp.down_proj_bias") is True

    assert _is_expert_param("model.layers.0.mlp.shared_expert.down_proj") is False

    assert _is_expert_param("model.layers.0.mlp.gate.weight") is False
    assert _is_expert_param("model.layers.0.mlp.router.weight") is False

    # HF MoE unpatched
    assert _is_expert_param("model.layers.0.mlp.experts.0.w1.weight") is True

    # Pre-5.14 Zaya nesting is unreadable by the current stack.
    assert _is_expert_param("model.layers.0.mlp.local_experts.w1") is False

    assert _is_expert_param("model.layers.0.self_attn.q_proj.weight") is False


def test_detect_moe_config_num_local_experts():
    config = MockConfig(num_local_experts=32, num_experts_per_tok=4)
    model = MockModel(config=config)
    ne, tk = (resolve_expert_count(model.config), resolve_router_topk(model.config))
    assert ne == 32, f"Expected num_experts=32, got {ne}"
    assert tk == 4, f"Expected top_k=4, got {tk}"


def test_detect_moe_config_num_experts():
    config = MockConfig(num_experts=16, top_k=2)
    model = MockModel(config=config)
    ne, tk = (resolve_expert_count(model.config), resolve_router_topk(model.config))
    assert ne == 16, f"Expected num_experts=16, got {ne}"
    assert tk == 2, f"Expected top_k=2, got {tk}"


def test_detect_moe_config_no_moe():
    config = MockConfig(hidden_size=4096)
    model = MockModel(config=config)
    ne, tk = (resolve_expert_count(model.config), resolve_router_topk(model.config))
    assert ne == 0 and tk == 0, f"Expected (0, 0), got ({ne}, {tk})"


def test_detect_moe_config_no_config():
    model = MockModel()
    model.config = None
    ne, tk = (resolve_expert_count(model.config), resolve_router_topk(model.config))
    assert ne == 0 and tk == 0, f"Expected (0, 0), got ({ne}, {tk})"


def test_estimate_linear_flops():
    num_params = 1_000_000_000
    # One parameter with an overridden numel stands for the 1B; no config, no attention term.
    model = MockModel(named_params=[("weight", MockParam(num_params, dtype=torch.bfloat16, requires_grad=True))])
    flops = _rank_flops(model).model_flops_per_token
    assert flops == 6 * num_params, f"Expected {6 * num_params}, got {flops}"


def test_a_fully_frozen_rank_is_charged_forward_and_input_gradient():
    """A frozen stage or base runs the forward and the input gradient its upstream needs, but no weight gradient."""
    model = torch.nn.Linear(10, 10, bias=False).requires_grad_(False)
    model.config = None
    flops = _rank_flops(model)
    assert flops.model_flops_per_token == 4 * model.weight.numel()
    assert flops.active_model_flops_per_token == flops.model_flops_per_token


def test_expert_params_count_routed_experts_only():
    named_params = [
        ("model.layers.0.mlp.gate_up_proj", MockParam(1000, requires_grad=True)),
        ("model.layers.0.mlp.down_proj", MockParam(2000, requires_grad=True)),
        ("model.layers.0.mlp.shared_expert.down_proj", MockParam(500, requires_grad=True)),
        ("model.layers.0.self_attn.q_proj.weight", MockParam(3000, requires_grad=True)),
    ]
    model = MockModel(named_params=named_params, config=MockConfig(num_experts=4, num_experts_per_tok=1))
    total_expert = _rank_flops(model).local_expert_params
    assert total_expert == 3000.0, f"Expected 3000.0, got {total_expert}"


def test_warmup_skipping():
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=5)
    cb.on_step_end(MockTrainingArgs(), MockTrainerState(global_step=2, num_input_tokens_seen=1000), None)
    assert cb.time == StepTimeMetrics(), "a warmup step was measured"


def test_the_measured_step_times_are_every_step_after_warmup_unrounded(clock):
    """The benchmarks read the step-time median off them, finer than the logged two decimals."""
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=1)
    for step, step_time in enumerate((9.0, 0.123456, 0.5), start=1):
        _measured_step(clock, cb, tokens=1000 * step, step_time=step_time, step=step)

    assert cb.measured_step_times == pytest.approx((0.123456, 0.5), rel=1e-12)
    assert cb.time.avg_step_time_seconds == 0.31


def test_estimate_model_flops_includes_attention():
    n_params = 1_000_000
    p = MockParam(n_params, requires_grad=True)
    config = MockConfig(num_hidden_layers=4, hidden_size=128)
    model = MockModel(named_params=[("w", p)], config=config)

    seq_len = 512
    flops = _rank_flops(model, seq_len=seq_len).model_flops_per_token
    expected = 6 * n_params + 12 * 4 * seq_len * 128
    assert flops == expected, f"expected {expected}, got {flops}"


def test_mfu_value(clock):
    cb = _peaked()
    _measured_step(clock, cb, tokens=1000, step_time=0.5)
    # step_flops = 1000 * 6e9 = 6e12; /0.5s = 1.2e13 flops/s; /1e15 *100 = 1.2%
    assert cb.mfu.step_mfu_percent == pytest.approx(1.2, abs=1e-6)
    assert cb.mfu.step_tflops_per_sec == pytest.approx(12.0, abs=1e-6)  # 1.2e13 / 1e12 TFLOPS
    assert cb.mfu.avg_tflops_per_sec == cb.mfu.step_tflops_per_sec, "one step: the average is the step"


def test_mfu_distributed_efficiency(clock):
    """params_ratio > 1.05 turns MFU into a distributed-efficiency speedup figure."""
    cb = _peaked(replace(_DENSE_FLOPS, full_params=4.0e9, params_ratio=4.0))
    _measured_step(clock, cb, tokens=1000, step_time=0.5)
    # step_mfu = 1.2% → distributed_efficiency = 4.0 * 1.2 / 100 = 0.048
    assert cb.mfu.step_distributed_efficiency == pytest.approx(0.05, abs=1e-2)


@pytest.mark.parametrize(("flops", "peak"), [(None, 1.0e15), (_DENSE_FLOPS, None)])
def test_mfu_zeroed_when_unconfigured(clock, flops, peak):
    cb = _peaked(flops, peak)
    _measured_step(clock, cb)
    assert cb.mfu == MFUMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_smfu_sparse_value(clock):
    cb = _peaked(replace(_DENSE_FLOPS, active_model_flops_per_token=1.5e9, sparsity_factor=0.25))
    _measured_step(clock, cb, tokens=1000, step_time=0.5)
    # step_active_flops = 1000 * 1.5e9 = 1.5e12; /0.5 = 3e12; /1e15*100 = 0.3%
    assert cb.smfu.step_smfu_percent == pytest.approx(0.3, abs=1e-6)
    assert cb.smfu.step_smfu_tflops_per_sec == pytest.approx(3.0, abs=1e-6)


# Each axis as (config fields, world size, GPUs per node, data-parallel replicas, ranks splitting one sequence).
# A batch is shared by the ranks of one TP, expert-TP or CP group and one pipeline chain; each EP rank draws its own.
_AXES = {
    "dp": ({}, 4, 4, 4, 1),
    "ep": ({"ep_size": 4}, 4, 4, 4, 1),
    "tp": ({"tp_size": 2}, 4, 4, 2, 1),
    "cp": ({"cp_size": 2}, 4, 4, 2, 2),
    "etp": ({"expert_tp_size": 2}, 4, 4, 2, 1),
    "ep_tp": ({"ep_size": 2, "tp_size": 2}, 4, 4, 2, 1),
    "ep_cp": ({"ep_size": 4, "cp_size": 4}, 4, 4, 1, 4),
    "ep_etp": ({"ep_size": 2, "expert_tp_size": 2}, 4, 4, 2, 1),
    "pp": ({"pp_size": 2}, 4, 2, 2, 1),
    "pp_etp": ({"pp_size": 2, "expert_tp_size": 2}, 8, 4, 2, 1),
    "pp_ep": ({"pp_size": 2, "ep_size": 2}, 4, 2, 2, 1),
}


def _axis_config(axis: str):
    fields, world, gpus_per_node, _, _ = _AXES[axis]
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, **fields)


def test_the_throughput_rows_cover_every_supported_axis_set():
    assert {_axis_config(axis).active_axes for axis in _AXES} == SUPPORTED_AXIS_SETS


@pytest.mark.parametrize("axis", sorted(_AXES))
def test_token_throughput_counts_each_shared_batch_once(monkeypatch, clock, axis):
    """The cluster figure is each replica's batch once and tokens/s/GPU is that over the world size. MFU charges a rank
    every token it runs through its parameter slice: the whole batch under TP, expert-TP and PP, its 1/cp under CP."""
    _, world, _, replicas, sequence_split = _AXES[axis]
    batch_tokens, step_time = 8000, 2.0
    cb = _peaked(parallelism=_axis_config(axis))
    monkeypatch.setattr(efficiency, "get_global_world_size", lambda: world)
    _measured_step(clock, cb, tokens=batch_tokens * world, step_time=step_time)

    cluster = batch_tokens * replicas / step_time
    assert cb.tps.step_cluster_tokens_per_second == pytest.approx(cluster)
    assert cb.tps.step_tokens_per_second == pytest.approx(cluster / world)
    assert cb.tps.avg_tokens_per_second == cb.tps.step_tokens_per_second, "one step: the average is the step"
    rank_tokens = batch_tokens / sequence_split
    assert cb.mfu.step_tflops_per_sec == pytest.approx(
        rank_tokens * _DENSE_FLOPS.model_flops_per_token / step_time / 1e12
    )


@pytest.mark.parametrize("report_mfu_diagnostics", [False, True])
def test_on_log_gates_diagnostics(report_mfu_diagnostics):
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=0, report_mfu_diagnostics=report_mfu_diagnostics)
    cb.tps = ThroughputMetrics(step_tokens_per_second=1234.0)
    cb.mfu = MFUMetrics(step_mfu_percent=42.0)

    logs: dict = {}
    cb.on_log(MockTrainingArgs(), MockTrainerState(global_step=5), None, logs=logs)
    assert logs["step_tokens_per_second"] == 1234.0
    assert logs.get("step_mfu_percent") == (42.0 if report_mfu_diagnostics else None)


def test_initialize_metrics_survives_unmeasurable_model():
    """The estimator fails loud, but the callback is observability, so it disables MFU instead of killing the run."""
    shell = MockModel(named_params=[], config=MockConfig(model_type="mystery"))
    with pytest.raises(ValueError):
        _rank_flops(shell)

    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.on_init_end(MockTrainingArgs(bf16=True), MockTrainerState(), None, model=shell)  # must not raise

    assert cb.flops_model is None, "MFU and S-MFU must be disabled, not guessed"


def test_initialize_metrics_measures_normal_model():
    model = MockModel(
        named_params=[("model.layers.0.self_attn.q_proj.weight", MockParam(1_000_000))],
        config=MockConfig(hidden_size=128, num_hidden_layers=2),
    )
    cb = EfficiencyCallback(_NO_PARALLELISM)
    cb.on_init_end(MockTrainingArgs(bf16=True, max_length=512), MockTrainerState(), None, model=model)

    assert cb.flops_model.model_flops_per_token == 6 * 1_000_000 + 12 * 2 * 512 * 128
    assert cb.flops_model.local_params == 1_000_000.0


@pytest.mark.parametrize("max_length", [None, 0, -1])
def test_a_one_sided_budget_keeps_its_declared_length(max_length):
    """A prompt budget with no completion cap still sizes the MFU length; a non-positive ``max_length`` is no bound."""
    model = MockModel(
        named_params=[("model.layers.0.self_attn.q_proj.weight", MockParam(1_000_000))],
        config=MockConfig(hidden_size=128, num_hidden_layers=2),
    )
    cb = EfficiencyCallback(_NO_PARALLELISM, script_args=SimpleNamespace(max_prompt_length=300))
    cb.on_init_end(MockTrainingArgs(bf16=True, max_length=max_length), MockTrainerState(), None, model=model)

    assert cb.flops_model.model_flops_per_token == 6 * 1_000_000 + 12 * 2 * 300 * 128


# The logs dict's key order, which wandb/tb series and the bench lines read.
_LOGGED_KEYS = (
    "step_time_seconds",
    "avg_step_time_seconds",
    "step_tokens_per_second",
    "avg_tokens_per_second",
    "step_cluster_tokens_per_second",
    "avg_cluster_tokens_per_second",
    "allocated_gb",
    "reserved_gb",
    "peak_allocated_gb",
    "training_peak_allocated_gb",
    "step_mfu_percent",
    "avg_mfu_percent",
    "step_tflops_per_sec",
    "avg_tflops_per_sec",
    "step_distributed_efficiency",
    "avg_distributed_efficiency",
    "step_smfu_percent",
    "avg_smfu_percent",
    "step_smfu_tflops_per_sec",
    "avg_smfu_tflops_per_sec",
    "step_smfu_distributed_efficiency",
    "avg_smfu_distributed_efficiency",
)


def test_logged_metric_keys_keep_their_order(monkeypatch):
    """On a MoE model, so S-MFU is emitted too."""
    monkeypatch.setattr(efficiency, "detect_gpu_model", lambda: "B300")
    model = MockModel(
        named_params=[
            ("model.layers.0.mlp.experts.0.w1.weight", MockParam(4000)),
            ("model.layers.0.self_attn.q_proj.weight", MockParam(1000)),
        ],
        config=MockConfig(num_experts=8, num_experts_per_tok=2, hidden_size=16, num_hidden_layers=1),
    )
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=0, report_mfu_diagnostics=True)
    args = MockTrainingArgs(bf16=True)
    cb.on_init_end(args, MockTrainerState(), None, model=model)
    cb.on_train_begin(args, MockTrainerState(), None, model=model)
    cb.on_step_begin(args, MockTrainerState(), None, model=model)
    cb.on_step_end(args, MockTrainerState(global_step=1, num_input_tokens_seen=1000), None, model=model)
    logs: dict = {}
    cb.on_log(args, MockTrainerState(global_step=1), None, logs=logs)
    assert tuple(logs) == _LOGGED_KEYS


def test_on_log_skips_during_warmup():
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=3)
    cb.tps = ThroughputMetrics(step_tokens_per_second=999.0)

    logs: dict = {}
    cb.on_log(MockTrainingArgs(), MockTrainerState(global_step=2), None, logs=logs)
    assert logs == {}


def test_the_wired_length_bound_spans_the_training_config_and_the_script_args(clock):
    """GRPO's config declares neither max_length nor max_prompt_length and the scripts carry the prompt budget on their
    own args, so the wired bound reads both. An uncounted step falls back to ``batch x accumulation x 150 x fill``."""
    script_args = SimpleNamespace(
        moe_balancing="none",
        enable_efficiency_metrics=True,
        enable_moe_metrics=False,
        enable_torch_profiler=False,
        num_full_model_params=None,
        report_mfu_diagnostics=False,
        max_prompt_length=100,
    )
    training_config = SimpleNamespace(
        max_completion_length=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        include_num_input_tokens_seen="no",
        logging_steps=1,
        bf16=True,
        fp16=False,
        tf32=False,
    )
    model = torch.nn.Linear(2, 2)
    model.config = None
    _, callback = factory.build_run_callbacks(
        script_args, training_config, model, _NO_PARALLELISM, balancing_mode="none"
    )
    callback.on_init_end(training_config, MockTrainerState(), None)
    _measured_step(clock, callback, tokens=None, step_time=1.0, step=3)  # past the default warmup
    assert callback.tps.step_tokens_per_second == int(150 * efficiency._ESTIMATE_FILL_FRACTION)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
