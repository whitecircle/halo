"""Token-balanced deal of an online GRPO round's rows across ranks: each rank trains the same number of rows, chosen so
the ranks' attended-token totals are close, in place of the contiguous slice the rollout hands each one. Row-mean and
token-global losses read no row's rank, so only the load changes."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist

from halo.runtime.collectives import all_gather_cat
from halo.runtime.consensus import agree_across_ranks, rank_max, reject_across_ranks
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.microbatch_plan import balanced_partition
from halo.trainers.grpo.row_order import permute_rows, reject_microbatch_dependent_objective

logger = logging.getLogger(__name__)

RANK_BALANCE_KNOB = "balance_rank_tokens"
SPREAD_BEFORE_KEY = "rank_balance/tokens_max_over_mean_before"
SPREAD_AFTER_KEY = "rank_balance/tokens_max_over_mean_after"

# The left-padded prompt tensors: each ``(ids key, mask key)`` pair is one prompt a row is scored under, whose forward
# also reads the row's completion. A trainer adding a prompt (SDPG's teacher) declares its pair on ``ROUND_PROMPT_ROWS``.
PROMPT_ROWS: tuple[tuple[str, str], ...] = (("prompt_ids", "prompt_mask"),)
# Every other row tensor of a text round is right-padded to the completion width (or one column wide).
_COMPLETION_KEYS = (
    "completion_ids",
    "completion_mask",
    "advantages",
    "old_per_token_logps",
    "ref_per_token_logps",
    "sampling_per_token_logps",
    "importance_sampling_ratio",
    "tool_mask",
)


@dataclass(frozen=True)
class RankSpread:
    """The heaviest rank's attended tokens over the mean, before and after the deal (1.0 is balanced)."""

    before: float
    after: float


def deal_order(tokens: Sequence[int], ranks: int) -> list[int]:
    """Row indices in the order rank 0's rows, rank 1's, and so on, ``len(tokens) // ranks`` each: heaviest row first
    onto the lightest rank with a free place (ties to the lowest rank and row), each rank's rows then in index order.
    Raises when the rows do not divide over the ranks."""
    if ranks < 1 or len(tokens) % ranks:
        raise ValueError(f"Cannot deal {len(tokens)} rows over {ranks} ranks in equal shares.")
    owned = balanced_partition(tokens, ranks, capacity=len(tokens) // ranks)
    return [row for rows in owned for row in sorted(rows)]


def max_over_mean(tokens: Sequence[int], order: Sequence[int], ranks: int) -> float:
    """The heaviest rank's tokens over the mean for the rows ``order`` hands each rank."""
    share = len(order) // ranks
    loads = [sum(tokens[row] for row in order[r * share : (r + 1) * share]) for r in range(ranks)]
    mean = sum(loads) / ranks
    return max(loads) / mean if mean else 1.0


def reject_unbalanceable_rank_tokens(
    args, parallelism_config, *, ranks: int, misaligned_accumulation: bool, knob: str = RANK_BALANCE_KNOB
) -> None:
    """Refuse a setup where moving rows between ranks would change the objective, split rows that must stay on one
    rank, or balance nothing."""
    reject_microbatch_dependent_objective(args, knob)
    if misaligned_accumulation:
        raise ValueError(
            f"{knob} is refused when gradient_accumulation_steps ({args.gradient_accumulation_steps}) straddles a "
            f"generation round: an optimizer step then takes a subset of its micro-batches and the deal picks which. "
            f"Align them."
        )
    if parallelism_config.is_tp_mode or parallelism_config.is_expert_tp_mode:
        raise ValueError(
            f"{knob} is refused under TP/ETP, where the group's ranks train the same rows and are re-synced from "
            f"their leader: a deal across them would split the shared batch. Unset {knob}."
        )
    if ranks < 2:
        raise ValueError(f"{knob} balances rows across ranks, but this run has {ranks}. Unset {knob}.")


def _pad_width(value: torch.Tensor, width: int, *, left: bool, fill: float) -> torch.Tensor:
    short = width - value.size(1)
    if short <= 0:
        return value
    pad = value.new_full((value.size(0), short, *value.shape[2:]), fill)
    return torch.cat([pad, value] if left else [value, pad], dim=1)


def _trim_to_tokens(dealt: dict[str, torch.Tensor], prompt_rows: Sequence[tuple[str, str]]) -> None:
    """Drop the columns no dealt row attends: each prompt's leading padding and the completion's trailing padding, as
    the rank's own rows would have been padded."""
    for ids_key, mask_key in prompt_rows:
        if mask_key not in dealt:
            continue
        columns = dealt[mask_key].bool().any(dim=0).nonzero()
        first = int(columns[0]) if columns.numel() else 0
        for key in (ids_key, mask_key):
            dealt[key] = dealt[key][:, first:]
    columns = dealt["completion_mask"].bool().any(dim=0).nonzero()
    width = dealt["completion_mask"].size(1)
    keep = int(columns[-1]) + 1 if columns.numel() else 1
    for key in _COMPLETION_KEYS:
        value = dealt.get(key)
        if value is not None and value.ndim > 1 and value.size(1) == width:
            dealt[key] = value[:, :keep]


def deal_rows_across_ranks(
    batch: dict, *, pad_token_id: int, prompt_rows: Sequence[tuple[str, str]] = PROMPT_ROWS
) -> tuple[dict, RankSpread]:
    """COLLECTIVE over the world. ``batch``'s rows dealt by :func:`deal_order` over every rank's attended tokens, each
    rank keeping its row count and receiving the rows the deal gives it, with the round's scalars unchanged. A row's
    tokens are each prompt it is scored under plus the completion that prompt's forward also reads. A round the deal
    would not make lighter at its heaviest rank comes back as it is. Raises on every rank when a rank's round carries a
    per-row value beyond the text-row keys (images, lists), or the ranks' key sets differ."""
    prompt_keys = {key for pair in prompt_rows for key in pair}
    id_keys = {ids_key for ids_key, _ in prompt_rows} | {"completion_ids"}
    known_keys = prompt_keys | set(_COMPLETION_KEYS)
    row_keys = sorted(key for key in known_keys if batch.get(key) is not None)
    rows = batch["completion_mask"].size(0)
    foreign = sorted(
        key
        for key, value in batch.items()
        if key not in known_keys
        and (
            (isinstance(value, torch.Tensor) and value.ndim > 0)
            or (isinstance(value, (list, tuple)) and len(value) == rows)
        )
    )
    reject_across_ranks(
        f"the round carries {foreign}, which rank balancing does not move (text rows only)" if foreign else None,
        "rank-balanced round",
        ValueError,
    )
    if not agree_across_ranks(row_keys):
        raise ValueError("The ranks' rounds carry different row tensors, so their rows cannot be dealt across ranks.")

    ranks = dist.get_world_size()
    local_tokens = sum(
        (batch[mask_key].sum(dim=1) + batch["completion_mask"].sum(dim=1)).long()
        for _, mask_key in prompt_rows
        if batch.get(mask_key) is not None
    )
    tokens = all_gather_cat(local_tokens, 0, None, ranks).tolist()
    order = deal_order(tokens, ranks)
    spread = RankSpread(max_over_mean(tokens, range(len(tokens)), ranks), max_over_mean(tokens, order, ranks))
    if spread.after >= spread.before:
        return batch, RankSpread(spread.before, spread.before)

    widths = rank_max([batch[key].size(1) if batch[key].ndim > 1 else 0 for key in row_keys])
    mine = torch.tensor(order[dist.get_rank() * rows :][:rows], dtype=torch.long)
    dealt: dict[str, torch.Tensor] = {}
    for key, width in zip(row_keys, widths, strict=True):
        value = batch[key]
        if value.ndim > 1:
            fill = pad_token_id if key in id_keys else 1 if key == "importance_sampling_ratio" else 0
            value = _pad_width(value, width, left=key in prompt_keys, fill=fill)
        gathered = all_gather_cat(value, 0, None, ranks)
        dealt[key] = permute_rows({key: gathered}, mine)[key]
    _trim_to_tokens(dealt, prompt_rows)
    return batch | dealt, spread


class RankBalancedRoundMixin:
    """The host provides ``args``, ``parallelism_config``, ``accelerator``, ``_metrics``, ``_misaligned_accumulation``
    and ``pad_token_id``, sets ``_balances_rank_tokens``, calls :meth:`_configure_rank_balance` once the model is placed
    and ends a train round's build in :meth:`_deal_round_across_ranks`, after every stage that reads the round's
    rows and before any stage that arranges them."""

    # The prompts a round's rows are scored under, which the deal moves, weighs and trims together with the completion.
    ROUND_PROMPT_ROWS: ClassVar[tuple[tuple[str, str], ...]] = PROMPT_ROWS

    _balances_rank_tokens = False

    def _configure_rank_balance(self) -> None:
        if not self._balances_rank_tokens:
            return
        reject_unbalanceable_rank_tokens(
            self.args,
            self.parallelism_config,
            ranks=self.accelerator.num_processes,
            misaligned_accumulation=self._misaligned_accumulation,
        )
        if is_global_main_process():
            logger.info("Rank balance: each round's rows are dealt across ranks by attended tokens.")

    def _deal_round_across_ranks(self, batch: dict) -> dict:
        """COLLECTIVE. The train round with its rows dealt across ranks."""
        dealt, spread = deal_rows_across_ranks(
            batch, pad_token_id=self.pad_token_id, prompt_rows=self.ROUND_PROMPT_ROWS
        )
        self._metrics["train"][SPREAD_BEFORE_KEY].append(spread.before)
        self._metrics["train"][SPREAD_AFTER_KEY].append(spread.after)
        return dealt
