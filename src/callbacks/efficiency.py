"""MFU / throughput accounting: step time, token throughput, MFU / S-MFU and GPU memory per step."""

import time
from dataclasses import dataclass

import torch
import transformers
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.utils import logging

from src.callbacks.model_flops import (
    ASSUMED_MAX_SEQ_LEN,
    compute_expert_params,
    estimate_attention_flops,
    estimate_model_flops_per_token,
)
from src.callbacks.parameter_stats import count_model_parameters
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import get_global_world_size, is_global_main_process
from src.hardware import detect_gpu_model, get_gpu_peak_flops
from src.models.loading.dtype import resolve_training_dtype
from src.models.moe_balancing import detect_moe_experts_topk

logger = logging.get_logger(__name__)

_ESTIMATE_FILL_FRACTION = 0.8

# Smallest params_ratio that counts as sharding-derived speed-up when scoring distributed
# efficiency. Plain FSDP/DP sits at ~1.001 (the ratio is 1 plus rounding), which is not one.
_DISTRIBUTED_EFFICIENCY_PARAMS_RATIO_MIN = 1.05

# Paired trainers (DPO/KTO/reward) concatenate chosen and rejected into one ``input_ids``.
_MAX_ROWS_PER_EXAMPLE = 2.0

# Headroom over that worst case; a larger per-step delta is counter corruption, not real work.
_TOKEN_COUNT_SANITY_FACTOR = 1.25

_TERA = 10**12

# Decimals every reported metric is rounded to before it reaches the logs dict.
_DISPLAY_DECIMALS = 2


def resolve_max_seq_len(*sources) -> int | None:
    """Upper bound on tokens per sequence declared across ``sources``, or None when none declares one.

    Several sources: on the RL surface the bound is split across objects (``trl.GRPOConfig`` declares
    neither ``max_length`` nor ``max_prompt_length`` — the scripts carry the latter on their own args).
    """

    def first_positive(field: str) -> int | None:
        return next((int(value) for source in sources if (value := getattr(source, field, None))), None)

    max_length = first_positive("max_length")
    if max_length:
        return max_length
    prompt = first_positive("max_prompt_length")
    completion = first_positive("max_completion_length")
    return (prompt or 0) + (completion or 0) if (prompt or completion) else None


@dataclass
class State:
    """Internal state of the efficiency callback."""

    n_warmup_steps: int = 0
    step_start_time: float = 0.0
    elapsed_time: float = 0.0
    elapsed_step: int = 0
    step_start_tokens_seen: int = 0
    elapsed_tokens_seen: int = 0  # per-GPU
    elapsed_cluster_tokens: int = 0  # cluster-wide unique
    global_start_step: int = 0

    total_flops: float = 0.0
    gpu_model: str | None = None
    precision: str = "bf16"
    gpu_peak_flops: float | None = None
    model_flops_per_token: float | None = None

    total_active_flops: float = 0.0
    active_model_flops_per_token: float | None = None


@dataclass
class Time:
    """Time-related metrics."""

    step_time_seconds: float = 0.0
    avg_step_time_seconds: float = 0.0


@dataclass
class TPS:
    """Tokens per second metrics.

    step/avg_tokens_per_second: per-GPU tokens (used for MFU).
    step/avg_cluster_tokens_per_second: unique tokens processed by the cluster.
    """

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0
    step_cluster_tokens_per_second: float = 0.0
    avg_cluster_tokens_per_second: float = 0.0


@dataclass
class MFU:
    """Model FLOPS Utilization metrics.

    Tracks *local* MFU (params each GPU computes) and a *distributed efficiency* ratio = speed-up
    over a hypothetical single-GPU baseline holding the full model.
    """

    step_mfu_percent: float = 0.0
    avg_mfu_percent: float = 0.0
    step_tflops_per_sec: float = 0.0
    avg_tflops_per_sec: float = 0.0

    step_distributed_efficiency: float = 1.0
    avg_distributed_efficiency: float = 1.0

    gpu_model: str = "Unknown"
    precision: str = "bf16"

    local_params: float = 0.0
    full_params: float = 0.0
    params_ratio: float = 1.0


@dataclass
class SMFU:
    """Sparse Model FLOPS Utilization metrics (MoE-aware).

    Only ``top_k`` of ``num_experts`` fire per token, so S-MFU uses
    ``N_active = shared + (top_k/num_experts)·expert_params`` instead of ``N_local``. Dense models:
    S-MFU == MFU (sparsity_factor = 1.0). Reference: MoE-CAP (arXiv:2412.07067).
    """

    step_smfu_percent: float = 0.0
    avg_smfu_percent: float = 0.0
    step_smfu_tflops_per_sec: float = 0.0
    avg_smfu_tflops_per_sec: float = 0.0

    step_smfu_distributed_efficiency: float = 1.0
    avg_smfu_distributed_efficiency: float = 1.0

    local_active_params: float = 0.0
    local_expert_params: float = 0.0
    full_active_params: float = 0.0
    sparsity_factor: float = 1.0
    num_experts: int = 0
    top_k: int = 0


@dataclass
class Memory:
    """GPU memory metrics in GB."""

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_allocated_gb: float = 0.0  # per-step
    training_peak_allocated_gb: float = 0.0  # since training started


def _per_step_metrics(metrics) -> dict[str, float]:
    """The per-step / running-average fields of a metrics dataclass (``step_*`` / ``avg_*``).

    The remaining fields are run constants reported once in the setup summary; shipping them every
    logging step would add non-numeric values and flat series to wandb/tb.
    """
    return {name: value for name, value in vars(metrics).items() if name.startswith(("step_", "avg_"))}


# What a utilization report degrades to when the FLOPS/token or the GPU peak is unknown. Zeroed rather
# than left untouched: an unset report would republish the previous step's figures.
_UNKNOWN_UTILIZATION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _utilization_report(
    step_flops: float,
    step_time: float,
    total_flops: float,
    elapsed_time: float,
    peak_flops: float,
    ratio: float | None,
) -> tuple[float, float, float, float, float, float]:
    """``(step %, avg %, step TFLOP/s, avg TFLOP/s, step efficiency, avg efficiency)`` for one FLOPS series.

    MFU and S-MFU are the same arithmetic over different counters, so both read it here. ``ratio`` is
    the full-model-over-local parameter ratio scaling distributed efficiency, or ``None`` when there
    is no meaningful full-model baseline (efficiency stays 1.0).
    """
    step_flops_per_sec = step_flops / step_time
    step_percent = (step_flops_per_sec / peak_flops) * 100
    avg_flops_per_sec = total_flops / elapsed_time
    avg_percent = (avg_flops_per_sec / peak_flops) * 100
    efficiency = (
        (1.0, 1.0)
        if ratio is None
        else (
            round(ratio * step_percent / 100.0, _DISPLAY_DECIMALS),
            round(ratio * avg_percent / 100.0, _DISPLAY_DECIMALS),
        )
    )
    return (
        round(step_percent, _DISPLAY_DECIMALS),
        round(avg_percent, _DISPLAY_DECIMALS),
        round(step_flops_per_sec / _TERA, _DISPLAY_DECIMALS),
        round(avg_flops_per_sec / _TERA, _DISPLAY_DECIMALS),
        *efficiency,
    )


_PRECISION_KEY_BY_DTYPE = {torch.bfloat16: "bf16", torch.float16: "fp16"}

# Peak-FLOPS key per ``lowp_precision`` mode: low-precision compute decides the peak the matmuls
# actually issue at, whatever the master weights' dtype. Both fp4 recipes (nvfp4, mxfp4) run the same
# 4-bit MMA. ``"bf16"`` (low precision off) is absent — the dtype decides there.
_PRECISION_KEY_BY_LOWP = {"fp8": "fp8", "fp4": "fp4", "mxfp4": "fp4"}


def _fp32_compute_precision() -> str:
    """Precision key for an fp32-parameter model, accounting for TF32.

    ``float32_matmul_precision`` routes fp32 matmuls to Tensor Cores (``high``→TF32, ``medium``→bf16),
    whose peak is higher — keying MFU off ``"fp32"`` there would overstate TFLOPS past peak.
    """
    precision = torch.get_float32_matmul_precision()
    if precision == "medium":
        return "bf16"
    if precision == "high" or torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "fp32"


def _precision_key(dtype: torch.dtype | None) -> str | None:
    """Peak-FLOPS key for a parameter dtype, or ``None`` when the table has no entry for it."""
    if dtype == torch.float32:
        return _fp32_compute_precision()
    return _PRECISION_KEY_BY_DTYPE.get(dtype)


def _detect_precision(model: torch.nn.Module | None, training_args, lowp_precision: str = "bf16") -> str:
    """Peak-FLOPS precision key for the matmuls this run issues.

    ``lowp_precision`` wins where it is on: the masters stay bf16/fp32 while the GEMMs run in fp8/fp4,
    so the parameter dtype names the wrong peak and every MFU figure would be scaled by 2-4x. Off, the
    training arguments decide, falling back to the model dtype — ``model`` is None until the Trainer
    hands over a model reference, where the args flags are the only signal (bf16 is the default).
    """
    lowp_key = _PRECISION_KEY_BY_LOWP.get(lowp_precision)
    if lowp_key is not None:
        return lowp_key

    declared = resolve_training_dtype(training_args)
    if declared != torch.float32:
        return _PRECISION_KEY_BY_DTYPE[declared]
    if training_args.tf32:
        return "tf32"

    if model is None:
        return "bf16"

    key = _precision_key(getattr(model, "dtype", None))
    if key is None:
        first_param = next(model.parameters(), None)
        key = _precision_key(first_param.dtype) if first_param is not None else None
    return key or "bf16"


class EfficiencyCallback(transformers.TrainerCallback):
    """Tracks MFU (Model FLOPS Utilization) and token throughput during training.

    MFU = (per_gpu_tokens x 6 x local_params / step_time) / peak_gpu_flops x 100 %

    Token accounting: HF gathers ``num_input_tokens_seen`` summed across ranks, so the per-rank count
    divides by ``world_size`` — never by ``ep_size`` (EP ⊥ DP: each EP rank sees a distinct batch) —
    and again by ``cp_size``, since CP splits the sequence *inside* ``forward()``.

    Args:
        parallelism_config: the run's validated axis sizes; every divisor here is read off it.
        num_full_model_params: total full-model params; enables ``distributed_efficiency = params_ratio x local_mfu``.
    """

    def __init__(
        self,
        parallelism_config: ParallelismConfig,
        n_warmup_steps=2,
        num_full_model_params: float | None = None,
        report_mfu_diagnostics: bool = False,
        max_seq_len: int | None = None,
    ):
        self.state = State(n_warmup_steps)
        self.time = Time()
        self.tps = TPS()
        self.mfu = MFU()
        self.smfu = SMFU()
        self.memory = Memory()
        self.model_ref = None
        self._warned_token_estimate = False
        self._parallelism_config = parallelism_config
        self._full_model_params = num_full_model_params
        # Caller-resolved: the GRPO configs declare neither max_length nor max_prompt_length.
        self._max_seq_len_override = max_seq_len
        # MFU / S-MFU / achieved TFLOPS are setup-dependent, so they stay out of the headline logs dict.
        self._report_mfu_diagnostics = report_mfu_diagnostics

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Tri-state in transformers 5 ("no" | "all" | "non_padding"): "no" is truthy, so a plain
        # falsiness check never fires.
        if args.include_num_input_tokens_seen in ("no", False):
            logger.warning("--include_num_input_tokens_seen not enabled. Using fallback token estimation.")
        if args.logging_steps != 1:
            logger.info(f"logging_steps={args.logging_steps}. Metrics logged every {args.logging_steps} steps.")

        self._try_get_model_reference(kwargs)
        self._initialize_metrics(args)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.model_ref is None:
            self._try_get_model_reference(kwargs)
            if self.model_ref is not None:
                self._initialize_metrics(args)

        self.state.global_start_step = state.global_step

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        if self._is_warmup_step(state):
            return
        logs.update(_per_step_metrics(self.time))
        logs.update(self.tps.__dict__)
        logs.update(self.memory.__dict__)
        if self._report_mfu_diagnostics:
            logs.update(_per_step_metrics(self.mfu))
            if self.smfu.sparsity_factor < 1.0:
                logs.update(_per_step_metrics(self.smfu))

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.model_ref is None:
            self._try_get_model_reference(kwargs)
            if self.model_ref is not None:
                self._initialize_metrics(args)

        self.state.step_start_time = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._is_warmup_step(state):
            if state.num_input_tokens_seen is not None:
                self.state.step_start_tokens_seen = state.num_input_tokens_seen
            return

        step_time = self._compute_step_time()
        step_tokens_seen = self._compute_token_metrics(state, step_time)
        self._compute_mfu(step_tokens_seen, step_time)
        self._compute_smfu(step_tokens_seen, step_time)
        self._compute_memory()

        if state.num_input_tokens_seen is not None:
            self.state.step_start_tokens_seen = state.num_input_tokens_seen

    def _is_warmup_step(self, state: TrainerState) -> bool:
        """Whether this step falls in the warmup window excluded from the averages.

        HF increments ``global_step`` before firing ``on_step_end``, so the first completed step
        arrives as ``global_start_step + 1`` — hence the inclusive bound.
        """
        return state.global_step <= self.state.global_start_step + self.state.n_warmup_steps

    def _compute_step_time(self) -> float:
        """Update time tracking and return the step duration."""
        current_time = time.perf_counter()
        step_time = current_time - self.state.step_start_time
        self.state.elapsed_time += step_time
        self.state.elapsed_step += 1
        avg_step_time = self.state.elapsed_time / self.state.elapsed_step

        self.time.step_time_seconds = round(step_time, _DISPLAY_DECIMALS)
        self.time.avg_step_time_seconds = round(avg_step_time, _DISPLAY_DECIMALS)
        return step_time

    def _fallback_step_tokens(self, world_size: int, measured_cluster: int | None) -> int:
        """Substitute the padded-length estimate for an unusable measured token delta, loudly.

        Trainers whose collator emits no ``input_ids`` key (SMPO, embedding) never advance HF's
        ``num_input_tokens_seen``, so every step lands here and throughput becomes a config-derived
        constant that looks measured; the warning names that cause.
        """
        if not self._warned_token_estimate:
            self._warned_token_estimate = True
            cause = (
                f"the measured cluster delta ({measured_cluster}) was non-positive or above the plausible upper bound"
                if measured_cluster is not None
                else "no usable num_input_tokens_seen was available"
            )
            logger.warning(
                f"EfficiencyCallback: {cause}, so token throughput and MFU fall back to a "
                "padded-length ESTIMATE (per_device_train_batch_size x gradient_accumulation_steps x "
                f"max_seq x {_ESTIMATE_FILL_FRACTION}), not a measurement. If this repeats every step "
                "the trainer's collator emits no 'input_ids' key (SMPO and embedding do not), so "
                "tokens/s is a constant rescaling of step time — treat it as an estimate."
            )
        return self._estimate_step_tokens() * world_size

    def _compute_token_metrics(self, state: TrainerState, step_time: float) -> int:
        """Compute per-rank and cluster token throughput. Returns per-rank step tokens."""
        parallelism = self._parallelism_config
        world_size = get_global_world_size()

        if state.num_input_tokens_seen is not None and self.state.step_start_tokens_seen is not None:
            try:
                step_tokens_cluster = state.num_input_tokens_seen - self.state.step_start_tokens_seen
                # Clamp implausible deltas only against a real length bound (GRPO's 2048 guess is none).
                _, has_len_bound = self._max_seq_len()
                nominal_cluster = self._nominal_step_tokens() * world_size
                upper_bound = nominal_cluster * _MAX_ROWS_PER_EXAMPLE * _TOKEN_COUNT_SANITY_FACTOR
                if step_tokens_cluster <= 0 or (
                    has_len_bound and nominal_cluster > 0 and step_tokens_cluster > upper_bound
                ):
                    step_tokens_cluster = self._fallback_step_tokens(world_size, step_tokens_cluster)
            except (TypeError, AttributeError):
                step_tokens_cluster = self._fallback_step_tokens(world_size, None)
        else:
            step_tokens_cluster = self._fallback_step_tokens(world_size, None)

        # Under CP the Trainer counts the full sequence (CP splits inside forward()).
        step_tokens_seen = step_tokens_cluster // world_size if world_size > 1 else step_tokens_cluster
        if parallelism.cp_size > 1:
            step_tokens_seen = step_tokens_seen // parallelism.cp_size
        self.state.elapsed_tokens_seen += step_tokens_seen

        self.tps.step_tokens_per_second = round(
            step_tokens_seen / step_time,
            _DISPLAY_DECIMALS,
        )
        self.tps.avg_tokens_per_second = round(
            self.state.elapsed_tokens_seen / self.state.elapsed_time,
            _DISPLAY_DECIMALS,
        )

        # Only ranks sharing the SAME batch are non-independent (TP/ETP, CP, and the pp_size ranks
        # of one pipeline chain); EP ⊥ DP, so data_parallel_size never divides ep_size out.
        step_cluster_tokens = step_tokens_seen * parallelism.data_parallel_size * parallelism.cp_size
        self.state.elapsed_cluster_tokens += step_cluster_tokens

        self.tps.step_cluster_tokens_per_second = round(
            step_cluster_tokens / step_time,
            _DISPLAY_DECIMALS,
        )
        self.tps.avg_cluster_tokens_per_second = round(
            self.state.elapsed_cluster_tokens / self.state.elapsed_time,
            _DISPLAY_DECIMALS,
        )
        return step_tokens_seen

    def _compute_mfu(self, step_tokens_seen: int, step_time: float):
        """Compute Model FLOPs Utilization metrics."""
        if self.state.model_flops_per_token and self.state.gpu_peak_flops:
            step_flops = step_tokens_seen * self.state.model_flops_per_token
            self.state.total_flops += step_flops
            report = _utilization_report(
                step_flops,
                step_time,
                self.state.total_flops,
                self.state.elapsed_time,
                self.state.gpu_peak_flops,
                self.mfu.params_ratio if self.mfu.params_ratio > _DISTRIBUTED_EFFICIENCY_PARAMS_RATIO_MIN else None,
            )
        else:
            report = _UNKNOWN_UTILIZATION
        (
            self.mfu.step_mfu_percent,
            self.mfu.avg_mfu_percent,
            self.mfu.step_tflops_per_sec,
            self.mfu.avg_tflops_per_sec,
            self.mfu.step_distributed_efficiency,
            self.mfu.avg_distributed_efficiency,
        ) = report

    def _compute_smfu(self, step_tokens_seen: int, step_time: float):
        """Compute Sparse MFU (MoE-aware) metrics."""
        if self.state.active_model_flops_per_token and self.state.gpu_peak_flops:
            step_active_flops = step_tokens_seen * self.state.active_model_flops_per_token
            self.state.total_active_flops += step_active_flops
            has_baseline = self.smfu.full_active_params > self.smfu.local_active_params > 0
            report = _utilization_report(
                step_active_flops,
                step_time,
                self.state.total_active_flops,
                self.state.elapsed_time,
                self.state.gpu_peak_flops,
                self.smfu.full_active_params / self.smfu.local_active_params if has_baseline else None,
            )
        else:
            report = _UNKNOWN_UTILIZATION
        (
            self.smfu.step_smfu_percent,
            self.smfu.avg_smfu_percent,
            self.smfu.step_smfu_tflops_per_sec,
            self.smfu.avg_smfu_tflops_per_sec,
            self.smfu.step_smfu_distributed_efficiency,
            self.smfu.avg_smfu_distributed_efficiency,
        ) = report

    def _compute_memory(self):
        """Update GPU memory tracking metrics."""
        if torch.cuda.is_available():
            self.memory.allocated_gb = round(
                torch.cuda.memory_allocated() / (1024**3),
                _DISPLAY_DECIMALS,
            )
            self.memory.reserved_gb = round(
                torch.cuda.memory_reserved() / (1024**3),
                _DISPLAY_DECIMALS,
            )
            step_peak = torch.cuda.max_memory_allocated() / (1024**3)
            self.memory.peak_allocated_gb = round(step_peak, _DISPLAY_DECIMALS)
            self.memory.training_peak_allocated_gb = round(
                max(self.memory.training_peak_allocated_gb, step_peak),
                _DISPLAY_DECIMALS,
            )

    def _max_seq_len(self) -> tuple[int, bool]:
        """Return ``(max_seq, is_real_bound)``; ``is_real_bound`` is False when nothing declares a bound."""
        resolved = self._max_seq_len_override or resolve_max_seq_len(self.training_args)
        return (int(resolved), True) if resolved else (ASSUMED_MAX_SEQ_LEN, False)

    def _nominal_step_tokens(self) -> int:
        """PER-GPU tokens a step would carry with every sequence padded to the length bound."""
        batch_size = getattr(self.training_args, "per_device_train_batch_size", 1)
        grad_accum = getattr(self.training_args, "gradient_accumulation_steps", 1)
        return int(batch_size * grad_accum * self._max_seq_len()[0])

    def _estimate_step_tokens(self) -> int:
        """Fallback: estimate PER-GPU tokens when num_input_tokens_seen is unavailable."""
        return int(self._nominal_step_tokens() * _ESTIMATE_FILL_FRACTION)

    def _try_get_model_reference(self, kwargs):
        """Capture the model from the HF callback kwargs (``model`` — the only handle HF passes)."""
        if self.model_ref is not None:
            return

        model = kwargs.get("model")
        if model is not None:
            self.model_ref = model

    def _initialize_metrics(self, args):
        """Initialize GPU detection, peak FLOPS, and local param counting."""
        self.training_args = args  # before _max_seq_len(), which reads it
        self.state.gpu_model = detect_gpu_model()
        self.state.precision = _detect_precision(self.model_ref, args, self._parallelism_config.lowp_precision)

        if self.state.gpu_model:
            self.state.gpu_peak_flops = get_gpu_peak_flops(
                self.state.gpu_model,
                self.state.precision,
            )

        self.mfu.gpu_model = self.state.gpu_model or "Unknown"
        self.mfu.precision = self.state.precision

        if self.model_ref:
            try:
                self._initialize_model_flops(self.model_ref, self._max_seq_len()[0])
            except Exception as exc:
                # Observability must never abort a run; unset estimates just disable MFU reporting.
                self.state.model_flops_per_token = None
                self.state.active_model_flops_per_token = None
                logger.warning(f"MFU reporting disabled — could not estimate model FLOPS/token: {exc}")

        if not is_global_main_process():
            return
        if self.state.gpu_model:
            logger.info(f"Detected GPU: {self.state.gpu_model}, Precision: {self.state.precision}")
            if self.state.gpu_peak_flops:
                logger.info(f"GPU Peak FLOPS: {self.state.gpu_peak_flops / _TERA:.1f} TFLOPS")
        else:
            logger.warning("Could not detect GPU model, MFU calculation may be inaccurate")

        if self.state.model_flops_per_token:
            logger.info(f"Model FLOPS/token: {self.state.model_flops_per_token / 1e12:.4f} TFLOPS")
            logger.info(f"Local params: {self.mfu.local_params / 1e9:.2f}B")
            if self.mfu.params_ratio > 1.0:
                logger.info(
                    f"Full model params: {self.mfu.full_params / 1e9:.2f}B (ratio: {self.mfu.params_ratio:.2f}x)"
                )
            if self.smfu.sparsity_factor < 1.0:
                logger.info(
                    f"S-MFU: {self.smfu.num_experts} experts, top_k={self.smfu.top_k}, "
                    f"sparsity={self.smfu.sparsity_factor:.3f}"
                )
                logger.info(
                    f"  Expert params: {self.smfu.local_expert_params / 1e9:.2f}B, "
                    f"Active params: {self.smfu.local_active_params / 1e9:.2f}B"
                )
                logger.info(
                    f"  Active FLOPS/token: "
                    f"{self.state.active_model_flops_per_token / 1e12:.4f} TFLOPS "
                    f"(dense: {self.state.model_flops_per_token / 1e12:.4f} TFLOPS)"
                )
        else:
            logger.warning("Could not get model reference for FLOPS calculation")

    def _initialize_model_flops(self, model: torch.nn.Module, seq_len: int):
        """Derive the dense and active FLOPS/token estimates (and the param counts feeding them).

        Model introspection only — raises on a model it cannot measure; the caller degrades.
        """
        parallelism = self._parallelism_config
        self.state.model_flops_per_token = estimate_model_flops_per_token(
            model, seq_len, parallelism.pp_size, parallelism.tp_size
        )

        local_params, trainable_params = count_model_parameters(model)
        if local_params == 0:
            local_params = trainable_params
        frozen_params = max(local_params - trainable_params, 0.0)

        self.mfu.local_params = float(local_params)

        if self._full_model_params is not None and self._full_model_params > 0:
            self.mfu.full_params = float(self._full_model_params)
            self.mfu.params_ratio = self.mfu.full_params / self.mfu.local_params
        else:
            self.mfu.full_params = self.mfu.local_params
            self.mfu.params_ratio = 1.0

        # The shared router probe, not a caller-declared pair: the balancing wiring and the MoE load
        # metrics read the same one, and a second source classifies a family as MoE for one consumer
        # and dense for another.
        num_experts, top_k = detect_moe_experts_topk(model)

        local_expert_params = 0.0
        sparsity_factor = 1.0
        trainable_expert_params = 0.0
        frozen_expert_params = 0.0

        if num_experts > 0 and top_k > 0:
            sparsity_factor = top_k / num_experts
            local_expert_params = compute_expert_params(model)
            trainable_expert_params = compute_expert_params(model, trainable_only=True)
            frozen_expert_params = max(local_expert_params - trainable_expert_params, 0.0)

        # Only top_k/num_experts fire per token; apply sparsity inside the 6N and 4N buckets separately.
        trainable_active = trainable_params - trainable_expert_params * (1.0 - sparsity_factor)
        frozen_active = frozen_params - frozen_expert_params * (1.0 - sparsity_factor)
        local_active_params = float(trainable_active + frozen_active)

        # FLOPs, not params: this rank holds num_experts/ep_size experts but serves the whole EP
        # group's tokens, so per-rank active FLOPs/token is ep-invariant. expert_tp_size must NOT
        # appear here — it already divides local_expert_params.
        expert_duty = sparsity_factor * parallelism.ep_size
        trainable_active_flops = trainable_params - trainable_expert_params * (1.0 - expert_duty)
        frozen_active_flops = frozen_params - frozen_expert_params * (1.0 - expert_duty)

        attn_flops = estimate_attention_flops(model, seq_len, parallelism.pp_size, parallelism.tp_size)
        self.state.active_model_flops_per_token = 6.0 * trainable_active_flops + 4.0 * frozen_active_flops + attn_flops

        self.smfu.num_experts = num_experts
        self.smfu.top_k = top_k
        self.smfu.sparsity_factor = sparsity_factor
        self.smfu.local_expert_params = local_expert_params
        self.smfu.local_active_params = local_active_params

        if self._full_model_params is not None and self._full_model_params > 0 and sparsity_factor < 1.0:
            # Both axes shard the expert FFN; counting only EP under-reports the full expert bank
            # by expert_tp_size and pushes the difference into the shared-param term.
            full_expert_params = local_expert_params * parallelism.ep_size * parallelism.expert_tp_size
            full_shared_params = self._full_model_params - full_expert_params
            self.smfu.full_active_params = full_shared_params + sparsity_factor * full_expert_params
        else:
            self.smfu.full_active_params = local_active_params
