"""Wire the observability and MoE-balancing callbacks a run's ``args`` enable.

Metric callbacks inject keys via ``logs.update(...)`` in ``on_log``, but HF fires its report-to
integrations first, so scripts must call :func:`reorder_integration_callbacks_last` after Trainer
construction or the added metrics never reach wandb/tb.
"""

from __future__ import annotations

import logging

from transformers import TrainerCallback

from src.callbacks.efficiency import EfficiencyCallback, resolve_max_seq_len
from src.callbacks.moe_metrics import MoEMetricsCallback
from src.callbacks.profiler import TorchProfilerCallback
from src.callbacks.router_bias_balancing import RouterBiasBalancingCallback
from src.distributed.expert_parallel.balancing_strategy import agree_balancing_mode, apply_balancing_strategy
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import get_global_world_size
from src.models.moe_balancing import (
    BIAS_UPDATE_MODES,
    detect_moe_experts_topk,
    has_discard_expert_slot,
    resolve_balancing_mode,
)

logger = logging.getLogger(__name__)

# World size past which exact token accounting's per-micro-batch all-gather is worth warning about.
_EXACT_TOKEN_COUNT_WORLD_WARN = 64


def _metric_log_period(training_config) -> int:
    """Step period on which MoE load metrics are worth deriving.

    Only ``on_log`` consumes them, so deriving them more often than the trainer logs adds a host sync
    (plus a collective for :class:`MoEMetricsCallback`) per step with no benefit. A ``logging_steps``
    below 1 is a ratio of total steps and carries no step period, so every step is derived there.
    """
    logging_steps = getattr(training_config, "logging_steps", 1) or 1
    return int(logging_steps) if logging_steps >= 1 else 1


def _detect_moe(model) -> tuple[bool, int, bool]:
    """Return ``(is_moe, top_k, exclude_last_slot)`` from ``model.config`` (``(False, 1, False)`` if not MoE)."""
    num_experts, top_k = detect_moe_experts_topk(model)
    if num_experts <= 1:
        return False, 1, False
    return True, top_k or 1, has_discard_expert_slot(model)


def build_perf_callbacks(
    args,
    training_config,
    model,
    parallelism_config: ParallelismConfig,
    policy_gradient_loss: bool = False,
    syncs_to_external_generator: bool = False,
    max_seq_len: int | None = None,
) -> list[TrainerCallback]:
    """Build the list of performance / balancing callbacks enabled by ``args``.

    Args:
        args: CommonScriptArguments-derived dataclass.
        training_config: TRL/HF training config; mutated to set ``include_num_input_tokens_seen="all"``
            when the efficiency callback is enabled.
        model: loaded model — used for MoE auto-detection and balancing side effects on ``model.config``.
        parallelism_config: the run's ParallelismConfig — the axis sizes plus the derived
            ``data_parallel_size`` every cluster-throughput figure divides by.
        policy_gradient_loss: True for GRPO-family trainers, whose loss never adds the router aux loss;
            makes ``aux_loss`` balancing warn-and-no-op and steers users to ``bias_update``.
        syncs_to_external_generator: True for on-policy RL syncing weights to a live vLLM generator.
            The weight-sync does not forward ``bias_update``'s routing bias, so it is downgraded to
            ``none`` for trainer↔generator routing parity — which, with ``aux_loss`` inert under a
            policy-gradient loss, leaves such runs with no router balancing at all.
        max_seq_len: explicit per-sequence token bound for ``EfficiencyCallback``, for runs whose real
            bound is not the sum of the declared length fields (a multi-turn RL trajectory). Defaults
            to :func:`resolve_max_seq_len` over the training config + script args.

    Returns:
        List of TrainerCallback instances (may be empty).
    """
    callbacks: list[TrainerCallback] = []

    is_moe, top_k, exclude_last_slot = _detect_moe(model)
    mode = agree_balancing_mode(resolve_balancing_mode(args.moe_balancing, model, is_moe))
    if syncs_to_external_generator and mode in BIAS_UPDATE_MODES:
        logger.warning(
            f"moe_balancing={mode} is incompatible with on-policy weight-sync RL: the routing bias "
            "isn't forwarded by the vLLM weight-sync (parameters only — an adopted native slot is a "
            "buffer), so trainer↔generator routing diverges. Downgrading to moe_balancing=none — "
            "with aux_loss also inert under a policy-gradient loss, THIS RUN HAS NO ROUTER BALANCING "
            "AT ALL and router_balancing_rate is unreachable. Families that only balance via "
            "bias_update (Zaya, DeepSeek-V4) train with unbalanced experts here."
        )
        mode = "none"
    apply_balancing_strategy(model, mode, policy_gradient_loss=policy_gradient_loss, is_moe=is_moe)

    if args.enable_efficiency_metrics:
        # After TrainingArguments.__post_init__, so the bool is never coerced into the tri-state.
        training_config.include_num_input_tokens_seen = "all"
        # HF counts inside the micro-batch loop, with a world all-gather plus a host sync each time:
        # cheap at 8 ranks, costly at hundreds. The callback estimates without it.
        if get_global_world_size() >= _EXACT_TOKEN_COUNT_WORLD_WARN:
            logger.warning(
                "enable_efficiency_metrics=true turns on exact token accounting, which adds a "
                "world all-gather and a host sync per micro-batch (%d per optimizer step at this "
                "gradient_accumulation_steps) across %d ranks. Set enable_efficiency_metrics=false "
                "for production runs at this scale; the throughput numbers are a calibration tool.",
                getattr(training_config, "gradient_accumulation_steps", 1),
                get_global_world_size(),
            )
        callbacks.append(
            EfficiencyCallback(
                parallelism_config=parallelism_config,
                num_full_model_params=args.num_full_model_params,
                report_mfu_diagnostics=args.report_mfu_diagnostics,
                max_seq_len=max_seq_len or resolve_max_seq_len(training_config, args),
            )
        )

    # MoEMetricsCallback needs outputs.router_logits, which neither bias_update's EP wrappers nor a PP
    # stage's bare-tensor forward populate; RouterBiasBalancingCallback covers those cases instead.
    if args.enable_moe_metrics and is_moe and mode not in BIAS_UPDATE_MODES:
        if parallelism_config.pp_size > 1:
            logger.info(
                "MoEMetricsCallback disabled under pipeline parallelism (pp_size="
                f"{parallelism_config.pp_size}): stage forwards return bare tensors without "
                "router_logits. Use moe_balancing=bias_update for moe/* load metrics under PP."
            )
        else:
            callbacks.append(
                MoEMetricsCallback(
                    topk=top_k,
                    exclude_last_slot=exclude_last_slot,
                    # Only aux_loss requires router logits; under `none` the toolkit adds nothing.
                    enable_router_logits=(mode == "aux_loss"),
                    log_every_n_steps=_metric_log_period(training_config),
                )
            )

    if mode in BIAS_UPDATE_MODES:
        callbacks.append(
            RouterBiasBalancingCallback(
                update_rate=args.router_balancing_rate,
                exclude_last_slot=exclude_last_slot,
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

    HF wires integrations before user callbacks, so their ``on_log`` relays the logs dict before user
    callbacks add keys via ``logs.update(...)``; reordering makes those keys visible.
    """
    handler = trainer.callback_handler
    integrations, others = [], []
    for cb in handler.callbacks:
        if type(cb).__module__.startswith("transformers.integrations"):
            integrations.append(cb)
        else:
            others.append(cb)
    handler.callbacks = others + integrations
