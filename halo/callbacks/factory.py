"""The run's callback roster, and the reorder that lets metric callbacks reach the report-to integrations."""

from __future__ import annotations

import logging

from transformers import TrainerCallback

from halo.callbacks.balancing import BalancingBiasCallback
from halo.callbacks.efficiency import EfficiencyCallback
from halo.callbacks.moe_metrics import MoEMetricsCallback
from halo.callbacks.parameter_stats import ParameterStatsCallback
from halo.callbacks.torch_profiler import TorchProfilerCallback
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.balancing import BIAS_UPDATE_MODES, BalancingMode
from halo.models.moe.config_fields import resolve_moe_routing

logger = logging.getLogger(__name__)


def _metric_log_period(training_config) -> int:
    """Step period on which MoE load metrics are worth deriving: only ``on_log`` consumes them. A
    ``logging_steps`` below 1 is a ratio of total steps and carries no period, so every step derives."""
    logging_steps = getattr(training_config, "logging_steps", 1) or 1
    return int(logging_steps) if logging_steps >= 1 else 1


def build_run_callbacks(
    args,
    training_config,
    model,
    parallelism_config: ParallelismConfig,
    *,
    balancing_mode: BalancingMode,
    generate_callback: TrainerCallback | None = None,
    max_seq_len: int | None = None,
) -> list[TrainerCallback | type[TrainerCallback]]:
    """The callbacks ``args`` enable, in roster order, for the mode
    :func:`~halo.callbacks.balancing.configure_moe_balancing` returned. Runs no collective.

    Enabling the efficiency callback sets ``training_config.include_num_input_tokens_seen="all"``;
    ``max_seq_len`` is its length bound where the declared length fields do not sum to one (a
    multi-turn RL trajectory).
    """
    if balancing_mode == "auto":
        raise ValueError("build_run_callbacks needs the balancing mode configure_moe_balancing resolved, got 'auto'.")
    routing = resolve_moe_routing(model)
    callbacks: list[TrainerCallback | type[TrainerCallback]] = [ParameterStatsCallback]
    if generate_callback is not None:
        callbacks.append(generate_callback)

    if args.enable_efficiency_metrics:
        # After TrainingArguments.__post_init__, so the bool is never coerced into the tri-state.
        training_config.include_num_input_tokens_seen = "all"
        callbacks.append(
            EfficiencyCallback(
                parallelism_config=parallelism_config,
                num_full_model_params=args.num_full_model_params,
                report_mfu_diagnostics=args.report_mfu_diagnostics,
                max_seq_len=max_seq_len,
                script_args=args,
            )
        )

    # MoEMetricsCallback needs outputs.router_logits, which neither the bias modes' EP layers nor a
    # PP stage's bare-tensor forward populate; BalancingBiasCallback emits moe/* there instead.
    if args.enable_moe_metrics and routing.routes_between_experts and balancing_mode not in BIAS_UPDATE_MODES:
        if parallelism_config.pp_size > 1:
            logger.info(
                "MoEMetricsCallback disabled under pipeline parallelism (pp_size="
                f"{parallelism_config.pp_size}): stage forwards return bare tensors without "
                "router_logits. Use moe_balancing=bias_update for moe/* load metrics under PP."
            )
        else:
            callbacks.append(
                MoEMetricsCallback(
                    topk=routing.top_k,
                    exclude_last_slot=routing.exclude_last_slot,
                    # Only aux_loss requires router logits; under `none` the toolkit adds nothing.
                    enable_router_logits=(balancing_mode == "aux_loss"),
                    log_every_n_steps=_metric_log_period(training_config),
                )
            )

    if balancing_mode in BIAS_UPDATE_MODES:
        callbacks.append(
            BalancingBiasCallback(
                update_rate=args.router_balancing_rate,
                exclude_last_slot=routing.exclude_last_slot,
                log_every_n_steps=_metric_log_period(training_config),
            )
        )

    if args.enable_torch_profiler:
        callbacks.append(
            TorchProfilerCallback(
                output_dir=args.profiler_output_dir,
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                ranks=args.profiler_ranks,
                memory_snapshot=args.profiler_record_memory_snapshot,
            )
        )
    return callbacks


def reorder_integration_callbacks_last(trainer) -> None:
    """Move HF report-to integration callbacks to the tail of the callback handler (idempotent).

    HF wires integrations before user callbacks, so their ``on_log`` would relay the logs dict before
    user callbacks add keys via ``logs.update(...)``.
    """
    handler = trainer.callback_handler
    integrations, others = [], []
    for cb in handler.callbacks:
        if type(cb).__module__.startswith("transformers.integrations"):
            integrations.append(cb)
        else:
            others.append(cb)
    handler.callbacks = others + integrations
