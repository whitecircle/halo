"""MFU / throughput accounting: step time, token throughput, MFU / S-MFU and GPU memory per step."""

import logging
import weakref
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
import transformers

from halo.callbacks.model_flops import RankFlopsModel, estimate_rank_flops
from halo.diagnostics.phase_timers import BoundaryMark, mark_boundary, seconds_between
from halo.diagnostics.profiling import read_cuda_memory
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import PEAK_KEY_BY_DTYPE, detect_gpu_model, get_gpu_peak_flops, peak_precision_key
from halo.kernels.lowp.deepgemm import use_deepgemm
from halo.kernels.lowp.spec import LOWP_RECIPES
from halo.model_loading.dtype import resolve_training_dtype
from halo.models.attention.layout import AttentionLayout
from halo.models.context_window import bounded_row_length
from halo.runtime.ranks import get_global_world_size, is_global_main_process

logger = logging.getLogger(__name__)

_ESTIMATE_FILL_FRACTION = 0.8
# Fallback when nothing declares a bound. Not a real upper bound, so callers must not clamp to it.
_ASSUMED_MAX_SEQ_LEN = 2048
# Smallest params_ratio that counts as a sharding-derived speed-up; plain FSDP/DP sits at ~1.001.
_DISTRIBUTED_EFFICIENCY_PARAMS_RATIO_MIN = 1.05
# Largest plausible step token delta over the padded nominal: paired trainers have two rows per example, plus headroom.
_TOKEN_DELTA_UPPER_BOUND = 2 * 1.25
_TERA = 10**12
_DISPLAY_DECIMALS = 2


@dataclass
class _RunState:
    n_warmup_steps: int = 0
    global_start_step: int = 0
    # Where the next step's time starts: the last step's end, or its own begin where none precedes it (``None``).
    step_start: BoundaryMark | None = None
    step_start_tokens_seen: int = 0
    elapsed_time: float = 0.0
    step_times: list[float] = field(default_factory=list)
    elapsed_cluster_tokens: int = 0  # cluster-wide unique
    elapsed_data_wait: float = 0.0
    total_flops: float = 0.0
    total_active_flops: float = 0.0


@dataclass
class StepTimeMetrics:
    step_time_seconds: float = 0.0
    avg_step_time_seconds: float = 0.0


@dataclass
class ThroughputMetrics:
    """Unique tokens the cluster processed per second, and their per-GPU share (cluster over world size)."""

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0
    step_cluster_tokens_per_second: float = 0.0
    avg_cluster_tokens_per_second: float = 0.0


@dataclass
class MFUMetrics:
    """MFU of the parameters this GPU computes, and the speed-up over a hypothetical single GPU with the full model."""

    step_mfu_percent: float = 0.0
    avg_mfu_percent: float = 0.0
    step_tflops_per_sec: float = 0.0
    avg_tflops_per_sec: float = 0.0
    step_distributed_efficiency: float = 1.0
    avg_distributed_efficiency: float = 1.0


@dataclass
class _SMFUMetrics:
    """Sparse MFU over the active FLOPs of :class:`~halo.callbacks.model_flops.RankFlopsModel`; MFU when dense."""

    step_smfu_percent: float = 0.0
    avg_smfu_percent: float = 0.0
    step_smfu_tflops_per_sec: float = 0.0
    avg_smfu_tflops_per_sec: float = 0.0
    step_smfu_distributed_efficiency: float = 1.0
    avg_smfu_distributed_efficiency: float = 1.0


@dataclass
class DataWaitMetrics:
    """Seconds an optimizer step spent blocked fetching batches from the train loader (mean over ranks)."""

    step_data_wait_seconds: float = 0.0
    avg_data_wait_seconds: float = 0.0


@dataclass
class _MemoryMetrics:
    """GPU memory in GiB; ``peak_allocated_gb`` is this step's, ``training_peak_allocated_gb`` the run's."""

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    peak_allocated_gb: float = 0.0
    training_peak_allocated_gb: float = 0.0


class _PendingStep(NamedTuple):
    """A measured step whose time is read once its end mark is: its boundary marks and the work it did."""

    start: BoundaryMark
    end: BoundaryMark
    rank_tokens: int
    cluster_tokens: int
    measured_attention_flops: float | None
    data_wait: float


class _StepWork(NamedTuple):
    """What the bound trainer reports for the step that just ended."""

    attention_flops: float | None
    data_wait: float


class _Utilization(NamedTuple):
    step_percent: float
    avg_percent: float
    step_tflops: float
    avg_tflops: float
    step_efficiency: float
    avg_efficiency: float


# Zeroed rather than left as is, so an unknown FLOPS/token or GPU peak never republishes stale figures.
_UNKNOWN_UTILIZATION = _Utilization(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _utilization_report(
    step_flops: float,
    step_time: float,
    total_flops: float,
    elapsed_time: float,
    peak_flops: float,
    ratio: float | None,
) -> _Utilization:
    """One FLOPS series' utilization; ``ratio`` (full-model over local parameters) scales the distributed
    efficiency, which stays 1.0 without a meaningful full-model baseline (``None``)."""
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
    return _Utilization(
        round(step_percent, _DISPLAY_DECIMALS),
        round(avg_percent, _DISPLAY_DECIMALS),
        round(step_flops_per_sec / _TERA, _DISPLAY_DECIMALS),
        round(avg_flops_per_sec / _TERA, _DISPLAY_DECIMALS),
        *efficiency,
    )


def _runs_native_expert_gemms(model: torch.nn.Module | None, lowp_precision: str) -> bool:
    """Whether a grouped EP layer runs the native kernel; a dense linear and a per-expert-loop layer never do."""
    recipe = LOWP_RECIPES.get(lowp_precision)
    if recipe is None or model is None or not use_deepgemm(recipe.quant_format):
        return False
    return any(
        layer._use_grouped_mm and layer._lowp_quant_format == recipe.quant_format
        for _, layer in named_ep_layers(model)
    )


def _detect_precision(model: torch.nn.Module | None, training_args, lowp_precision: str = "bf16") -> str:
    """Peak-FLOPS key for the run's matmuls: the recipe's only where they issue natively, as the simulated path
    multiplies fake-quantized operands in bf16. Then training arguments, then model dtype."""
    if _runs_native_expert_gemms(model, lowp_precision):
        return LOWP_RECIPES[lowp_precision].peak_flops_key

    declared = resolve_training_dtype(training_args)
    if declared != torch.float32:
        return PEAK_KEY_BY_DTYPE[declared]
    if training_args.tf32:
        return "tf32"
    if model is None:
        return "bf16"
    key = peak_precision_key(getattr(model, "dtype", None))
    if key is None:
        first_param = next(model.parameters(), None)
        key = peak_precision_key(first_param.dtype) if first_param is not None else None
    return key or "bf16"


class EfficiencyCallback(transformers.TrainerCallback):
    """Step time, token throughput, MFU / S-MFU and GPU memory, merged into the logs dict.

    Step time runs from the previous step's end to this one's, so host time between steps books to the later one and
    the steps sum to the wall clock. An epoch turnover, evaluation or save is not a step: the next step starts at its
    own begin.

    ``num_input_tokens_seen`` counts attended tokens (padding excluded), summed across ranks. A rank's own tokens
    divide it by ``world_size`` and ``cp_size``, since CP splits the sequence inside ``forward()``; TP, expert-TP and pipeline ranks each run every token through
    their parameter slice. Throughput per GPU is the cluster's unique tokens over ``world_size``. ``max_seq_len`` is
    for runs whose real bound is not the declared length fields (a multi-turn RL trajectory)."""

    def __init__(
        self,
        parallelism_config: ParallelismConfig,
        n_warmup_steps=2,
        num_full_model_params: float | None = None,
        report_mfu_diagnostics: bool = False,
        max_seq_len: int | None = None,
        script_args=None,
    ):
        self.gpu_model: str | None = None
        self.precision = "bf16"
        self.gpu_peak_flops: float | None = None
        self.flops_model: RankFlopsModel | None = None
        self.time = StepTimeMetrics()
        self.tps = ThroughputMetrics()
        self.data_wait = DataWaitMetrics()
        self.mfu = MFUMetrics()
        self.smfu = _SMFUMetrics()
        self.memory = _MemoryMetrics()
        self.model_ref = None
        self.training_args = None
        self._run = _RunState(n_warmup_steps)
        # Read at the next step end or log, when their end mark has passed.
        self._pending_steps: list[_PendingStep] = []
        self._parallelism_config = parallelism_config
        self._full_model_params = num_full_model_params
        self._report_mfu_diagnostics = report_mfu_diagnostics
        self._max_seq_len_override = max_seq_len
        self._script_args = script_args
        self._max_seq_len = _ASSUMED_MAX_SEQ_LEN
        self._has_length_bound = False
        # The trainer accumulating measured attention work per batch.
        self._attention_source = None
        self._warned_token_estimate = False
        self._noted_attention_fallback = False

    def on_init_end(self, args, state, control, **kwargs):
        # Tri-state in transformers 5 ("no" | "all" | "non_padding"): "no" is truthy.
        if args.include_num_input_tokens_seen == "no":
            logger.warning("--include_num_input_tokens_seen not enabled. Using fallback token estimation.")
        if args.logging_steps != 1:
            logger.info(f"logging_steps={args.logging_steps}. Metrics logged every {args.logging_steps} steps.")
        self.model_ref = kwargs.get("model")
        self._initialize_metrics(args)

    def on_train_begin(self, args, state, control, **kwargs):
        self._initialize_once_model_arrives(args, kwargs)
        self._run.global_start_step = state.global_step
        self._run.step_start = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._run.step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._initialize_once_model_arrives(args, kwargs)
        if self._run.step_start is None:
            self._run.step_start = mark_boundary()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        step_end = mark_boundary()
        # Before the warmup gate, so a warmup step's work never leaks into the first measured one.
        work = self._drain_step_work()
        if not self._is_warmup_step(state):
            rank_tokens, cluster_tokens = self._step_tokens(state)
            # One step later the earlier end marks have passed, so reading them waits on nothing.
            self._record_pending_steps()
            self._pending_steps.append(
                _PendingStep(
                    self._run.step_start,
                    step_end,
                    rank_tokens,
                    cluster_tokens,
                    work.attention_flops,
                    work.data_wait,
                )
            )
            self._record_memory()
        self._run.step_start = step_end
        if state.num_input_tokens_seen is not None:
            self._run.step_start_tokens_seen = state.num_input_tokens_seen

    def on_evaluate(self, args, state, control, **kwargs):
        self._run.step_start = None

    def on_save(self, args, state, control, **kwargs):
        self._run.step_start = None

    def on_log(self, args, state, control, logs, **kwargs):
        if self._is_warmup_step(state):
            return
        self._record_pending_steps()
        logs.update(asdict(self.time))
        logs.update(asdict(self.tps))
        logs.update(asdict(self.data_wait))
        logs.update(asdict(self.memory))
        if self._report_mfu_diagnostics:
            logs.update(asdict(self.mfu))
            if self.flops_model is not None and self.flops_model.sparsity_factor < 1.0:
                logs.update(asdict(self.smfu))

    def bind_attention_work_source(self, source) -> AttentionLayout | None:
        """Take the trainer as the per-step supplier of measured attention work and hand it the layout to cost batches
        with; ``None`` keeps its accumulator off and this callback on the config term."""
        self._attention_source = weakref.ref(source)
        return self.flops_model.attention_layout if self.flops_model is not None else None

    @property
    def measured_step_times(self) -> tuple[float, ...]:
        """Every measured step's time in seconds, unrounded and in order."""
        self._record_pending_steps()
        return tuple(self._run.step_times)

    def _initialize_once_model_arrives(self, args, kwargs) -> None:
        """Re-run :meth:`_initialize_metrics` when HF first hands the model over; ``on_init_end`` may precede that."""
        if self.model_ref is None and kwargs.get("model") is not None:
            self.model_ref = kwargs["model"]
            self._initialize_metrics(args)

    def _initialize_metrics(self, args) -> None:
        self.training_args = args
        resolved = self._max_seq_len_override or bounded_row_length(args, self._script_args, one_sided=True)
        self._max_seq_len, self._has_length_bound = (
            (int(resolved), True) if resolved else (_ASSUMED_MAX_SEQ_LEN, False)
        )
        self.gpu_model = detect_gpu_model()
        self.precision = _detect_precision(self.model_ref, args, self._parallelism_config.lowp.precision)
        if self.gpu_model:
            self.gpu_peak_flops = get_gpu_peak_flops(self.gpu_model, self.precision)
        if self.model_ref:
            try:
                self.flops_model = estimate_rank_flops(
                    self.model_ref,
                    seq_len=self._max_seq_len,
                    parallelism_config=self._parallelism_config,
                    full_model_params=self._full_model_params,
                )
            except Exception as exc:
                # An observability failure must not abort a run; unset estimates disable MFU only.
                self.flops_model = None
                logger.warning(f"MFU reporting disabled — could not estimate model FLOPS/token: {exc}")
        if is_global_main_process():
            self._log_setup_summary()

    def _log_setup_summary(self) -> None:
        if self.gpu_model:
            logger.info(f"Detected GPU: {self.gpu_model}, Precision: {self.precision}")
            if self.gpu_peak_flops:
                logger.info(f"GPU Peak FLOPS: {self.gpu_peak_flops / _TERA:.1f} TFLOPS")
            else:
                logger.warning(
                    f"GPU_PEAK_FLOPS has no {self.precision} peak for {self.gpu_model}, so MFU, "
                    f"S-MFU and TFLOP/s are reported as 0. Add the dense peak to its GPU_PEAK_FLOPS entry."
                )
        else:
            logger.warning("Could not detect the GPU model, so MFU, S-MFU and TFLOP/s are reported as 0.")

        flops = self.flops_model
        if flops is None or not flops.model_flops_per_token:
            logger.warning("Could not get model reference for FLOPS calculation")
            return
        logger.info(f"Model FLOPS/token: {flops.model_flops_per_token / _TERA:.4f} TFLOPS")
        if flops.attention_layout is not None:
            logger.info(
                f"  Attention layout (this rank): {flops.attention_layout.describe()} — score term "
                f"measured per document from each batch; config fallback at max_seq_len={self._max_seq_len}: "
                f"{flops.attention_flops_per_token / _TERA:.4f} TFLOPS/token"
            )
        logger.info(f"Local params: {flops.local_params / 1e9:.2f}B")
        if flops.params_ratio > 1.0:
            logger.info(f"Full model params: {flops.full_params / 1e9:.2f}B (ratio: {flops.params_ratio:.2f}x)")
        if flops.sparsity_factor < 1.0:
            logger.info(
                f"S-MFU: {flops.num_experts} experts, top_k={flops.top_k}, sparsity={flops.sparsity_factor:.3f}"
            )
            logger.info(
                f"  Expert params: {flops.local_expert_params / 1e9:.2f}B, "
                f"Active params: {flops.local_active_params / 1e9:.2f}B"
            )
            logger.info(
                f"  Active FLOPS/token: "
                f"{flops.active_model_flops_per_token / _TERA:.4f} TFLOPS "
                f"(dense: {flops.model_flops_per_token / _TERA:.4f} TFLOPS)"
            )

    def _drain_step_work(self) -> _StepWork:
        """COLLECTIVE through a bound source, which also sums the step's input tokens into the throughput's state. This
        rank's measured attention-score FLOPs, or ``None`` with no document measured (the config term stands in), and
        the step's data wait."""
        source = self._attention_source() if self._attention_source is not None else None
        if source is None:
            return _StepWork(None, 0.0)
        flops, documents, data_wait = source.drain_step_work()
        if documents <= 0 or self.flops_model is None:
            return _StepWork(None, data_wait)
        parallelism = self._parallelism_config
        # The rank's share of the layout, 1/tp of each layer's heads, and under CP 1/cp of the sequence's heads.
        share = self.flops_model.attention_share / parallelism.tp_size / parallelism.cp_size
        return _StepWork(flops * share, data_wait)

    def _is_warmup_step(self, state) -> bool:
        """HF increments ``global_step`` before ``on_step_end``: the first completed step is
        ``global_start_step+1``."""
        return state.global_step <= self._run.global_start_step + self._run.n_warmup_steps

    def _record_pending_steps(self) -> None:
        for step in self._pending_steps:
            self._record_step(step)
        self._pending_steps.clear()

    def _record_step(self, step: _PendingStep) -> None:
        step_time = self._record_step_time(seconds_between(step.start, step.end))
        self._record_token_throughput(step.cluster_tokens, step_time)
        self._record_data_wait(step.data_wait)
        flops = self.flops_model
        dense_per_token = active_per_token = mfu_ratio = smfu_ratio = None
        if flops is not None:
            dense_per_token, active_per_token = flops.model_flops_per_token, flops.active_model_flops_per_token
            if flops.params_ratio > _DISTRIBUTED_EFFICIENCY_PARAMS_RATIO_MIN:
                mfu_ratio = flops.params_ratio
            if flops.full_active_params > flops.local_active_params > 0:
                smfu_ratio = flops.full_active_params / flops.local_active_params
        work = (step.rank_tokens, step_time, step.measured_attention_flops)
        report, self._run.total_flops = self._record_utilization(
            dense_per_token, self._run.total_flops, mfu_ratio, *work
        )
        self.mfu = MFUMetrics(*report)
        report, self._run.total_active_flops = self._record_utilization(
            active_per_token, self._run.total_active_flops, smfu_ratio, *work
        )
        self.smfu = _SMFUMetrics(*report)

    def _record_data_wait(self, seconds: float) -> None:
        self._run.elapsed_data_wait += seconds
        self.data_wait = DataWaitMetrics(
            round(seconds, 4),
            round(self._run.elapsed_data_wait / len(self._run.step_times), 4),
        )

    def _record_step_time(self, step_time: float) -> float:
        self._run.elapsed_time += step_time
        self._run.step_times.append(step_time)
        self.time = StepTimeMetrics(
            round(step_time, _DISPLAY_DECIMALS),
            round(self._run.elapsed_time / len(self._run.step_times), _DISPLAY_DECIMALS),
        )
        return step_time

    def _step_tokens(self, state) -> tuple[int, int]:
        """``(tokens this rank computed over, the cluster's unique tokens)`` in the step that just ended."""
        parallelism = self._parallelism_config
        world_size = get_global_world_size()
        if state.num_input_tokens_seen is not None:
            step_tokens_cluster = state.num_input_tokens_seen - self._run.step_start_tokens_seen
            # Clamp implausible deltas only against a real length bound (GRPO's 2048 guess is none).
            nominal_cluster = self._nominal_step_tokens() * world_size
            if step_tokens_cluster <= 0 or (
                self._has_length_bound
                and nominal_cluster > 0
                and step_tokens_cluster > nominal_cluster * _TOKEN_DELTA_UPPER_BOUND
            ):
                step_tokens_cluster = self._fallback_step_tokens(world_size, step_tokens_cluster)
        else:
            step_tokens_cluster = self._fallback_step_tokens(world_size, None)

        # Under CP the Trainer counts the full sequence (CP splits inside forward()).
        rank_tokens = step_tokens_cluster // world_size // parallelism.cp_size
        # Only ranks sharing one batch are non-independent (TP/ETP, CP, a pipeline chain); EP is orthogonal to DP.
        return rank_tokens, rank_tokens * parallelism.data_parallel_size * parallelism.cp_size

    def _record_token_throughput(self, step_cluster_tokens: int, step_time: float) -> None:
        world_size = get_global_world_size()
        self._run.elapsed_cluster_tokens += step_cluster_tokens
        self.tps = ThroughputMetrics(
            round(step_cluster_tokens / world_size / step_time, _DISPLAY_DECIMALS),
            round(self._run.elapsed_cluster_tokens / world_size / self._run.elapsed_time, _DISPLAY_DECIMALS),
            round(step_cluster_tokens / step_time, _DISPLAY_DECIMALS),
            round(self._run.elapsed_cluster_tokens / self._run.elapsed_time, _DISPLAY_DECIMALS),
        )

    def _nominal_step_tokens(self) -> int:
        """Tokens one rank's batch would carry in a step with every sequence padded to the length bound."""
        args = self.training_args
        return int(args.per_device_train_batch_size * args.gradient_accumulation_steps * self._max_seq_len)

    def _fallback_step_tokens(self, world_size: int, measured_cluster: int | None) -> int:
        """The padded-length estimate for an unusable token delta, warned once. A batch holding no ``*input_ids`` tensor
        (online GRPO's loader yields prompts) lands here every step, so its throughput is a config-derived constant."""
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
                "the trainer's batches carry no '*input_ids' tensor to count, so "
                "tokens/s is a constant rescaling of step time — treat it as an estimate."
            )
        return int(self._nominal_step_tokens() * _ESTIMATE_FILL_FRACTION) * world_size

    def _record_utilization(
        self,
        per_token: float | None,
        total_flops: float,
        ratio: float | None,
        rank_tokens: int,
        step_time: float,
        measured_attention_flops: float | None,
    ) -> tuple[_Utilization, float]:
        """One series' utilization and running FLOPs total; MFU and S-MFU differ in per-token figure and ratio."""
        if not (per_token and self.gpu_peak_flops):
            return _UNKNOWN_UTILIZATION, total_flops
        step_flops = rank_tokens * per_token
        attention_per_token = self.flops_model.attention_flops_per_token
        if measured_attention_flops is not None:
            step_flops = step_flops - rank_tokens * attention_per_token + measured_attention_flops
        elif not self._noted_attention_fallback and attention_per_token:
            self._noted_attention_fallback = True
            logger.info(
                "EfficiencyCallback: no measured attention work reached this step, so the "
                "attention-score term is the config model at the declared length bound (every "
                "token in a max_seq_len document). Trainers whose batches carry no 'input_ids' "
                "stay on that model."
            )
        total_flops += step_flops
        report = _utilization_report(
            step_flops, step_time, total_flops, self._run.elapsed_time, self.gpu_peak_flops, ratio
        )
        return report, total_flops

    def _record_memory(self) -> None:
        memory = read_cuda_memory()
        if memory is None:
            return
        self.memory = _MemoryMetrics(
            round(memory.allocated, _DISPLAY_DECIMALS),
            round(memory.reserved, _DISPLAY_DECIMALS),
            round(memory.peak_allocated, _DISPLAY_DECIMALS),
            round(max(self.memory.training_peak_allocated_gb, memory.peak_allocated), _DISPLAY_DECIMALS),
        )
