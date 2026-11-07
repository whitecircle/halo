"""The parity probe after a trainer's first weight push: the trainer and the engine score one fixed token row, and
a gap past the trainer-versus-engine floor means the engine did not take the pushed weights as sent."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch

from halo.models.fp8_rollout import RolloutWeightPrecision
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import refusal_text, run_on_rank0_raising_everywhere
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.rollout.engine_preflight import counting_probe_token_ids
from halo.weight_sync.parity_probe import resolve_parity_probe_mode
from halo.weight_sync.pool import WeightSyncPool

logger = logging.getLogger(__name__)

# Mean absolute log-prob gap per position. The floor on identical weights reaches 0.65 on Laguna; a load that drops or
# scrambles a tensor reads in nats. Block-FP8 weights widen the mean gap by 1.5x (0.064 against 0.042 nats on prompts
# on Qwen3.5-35B-A3B), and the bound grows with it.
_FP8_BLOCK_GAP_GROWTH = 1.5
PARITY_MAX_MEAN_GAP_NATS: dict[RolloutWeightPrecision, float] = {"bf16": 1.0, "fp8_block": 1.0 * _FP8_BLOCK_GAP_GROWTH}


def _scoring_client(client):
    return client.scoring_client if isinstance(client, WeightSyncPool) else client


def _trainer_logps(trainer, token_ids: list[int]) -> list[float]:
    """COLLECTIVE: every rank scores the same row. Log-probs of ``token_ids[1:]`` given the first token, in the model's
    current train/eval mode, as the loss path's own recomputation runs."""
    ids = torch.tensor([token_ids], device=trainer.accelerator.device)
    with torch.no_grad():
        logps, _entropies, _aux = trainer._get_per_token_logps_and_entropies(
            trainer.model, ids, torch.ones_like(ids), len(token_ids) - 1
        )
    return logps[0].float().cpu().tolist()


def _compare_on_rank0(
    client, token_ids: list[int], trainer_logps: list[float], mode: str, precision: RolloutWeightPrecision
) -> None:
    engine_logps = _scoring_client(client).score_completion_logprobs(token_ids[:1], token_ids[1:])
    gaps = [abs(a - b) for a, b in zip(trainer_logps, engine_logps, strict=True)]
    mean_gap = sum(gaps) / len(gaps) if all(map(math.isfinite, gaps)) else math.inf
    bound = PARITY_MAX_MEAN_GAP_NATS[precision]
    if mean_gap > bound:
        message = (
            f"After the first weight push the rollout engine's log-probs on a {len(token_ids)}-token probe row "
            f"differ from the trainer's by {mean_gap:.2f} nats per position (bound {bound}): the "
            f"engine may not have taken the pushed weights as sent. It answers 200 for a tensor its loader does "
            f"not map and keeps the launch value, and a mis-stacked or mis-shaped one serves a different model. "
            f"Compare the pushed names with the engine loader's (the exact-push GPU test's reach mode scales each "
            f"forwarded name in turn), or serve the checkpoint the trainer started from. A "
            f"bf16 trainer-versus-engine floor above the bound (quantized trainers, composites) also reads here: "
            f"HALO_WEIGHT_SYNC_PARITY_PROBE=off silences the probe."
        )
        if mode == "raise":
            raise ValueError(message)
        logger.warning(message)
        return
    logger.info(f"Weight-sync parity probe: the engine's log-probs sit {mean_gap:.3f} nats from the trainer's")


def _run_on_rank0_or_warn(step: Callable[[], None], mode: str) -> bool:
    """COLLECTIVE. ``step`` on rank 0 with its failure raised on every rank; under ``warn`` that failure is logged on
    rank 0 instead and the probe stops, alike on every rank. Returns whether ``step`` ran."""
    try:
        run_on_rank0_raising_everywhere(step, refusal_text, ValueError)
    except ValueError as error:
        if mode == "raise":
            raise
        if is_global_main_process():
            logger.warning(f"Weight-sync parity probe could not run: {error}")
        return False
    return True


def verify_first_push_parity(trainer, client) -> None:
    """COLLECTIVE, after each push; acts on the trainer's first only. ``client`` is the main process's sender, ``None``
    elsewhere. Warns on rank 0 when the engine's log-probs on a fixed row differ from the trainer's by more than the
    bound, or when the engine cannot be probed, and raises on every rank instead under
    ``HALO_WEIGHT_SYNC_PARITY_PROBE=raise``. Skipped, with a line on rank 0, under context parallelism, whose batches
    shard the row."""
    if trainer.first_push_verified:
        return
    trainer.first_push_verified = True
    mode = resolve_parity_probe_mode()
    if mode == "off":
        return
    config = trainer.parallelism_config
    if config.is_cp_mode:
        if is_global_main_process():
            logger.info("Weight-sync parity probe skipped: context parallel batches shard the probe row.")
        return
    token_ids = counting_probe_token_ids(trainer.processing_class)
    windows: list[int | None] = []
    if not _run_on_rank0_or_warn(lambda: windows.append(_scoring_client(client).served_max_model_len()), mode):
        return
    if window := broadcast_from_rank0(windows[0] if windows else None):
        token_ids = token_ids[: window - 1]
    trainer_logps = _trainer_logps(trainer, token_ids)
    _run_on_rank0_or_warn(
        lambda: _compare_on_rank0(client, token_ids, trainer_logps, mode, trainer._rollout_weight_precision), mode
    )
