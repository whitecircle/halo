"""What a draft head is worth to a server that accepts or rejects its tokens, scored on held-out rows. Each scored
position pairs the head's draft with the trunk's own distribution over the same next token."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from halo.kernels.logprobs import at_least_fp32

# The world fractions a gate reads, as ``_record_loss_metrics`` names them and an eval log prefixes them.
MTP_ACCEPTANCE_TV = "mtp_acceptance_tv"
MTP_ACCEPTANCE_GREEDY = "mtp_acceptance_greedy"
MTP_TOP1 = "mtp_top1"
MTP_SCORED_POSITIONS = "mtp_scored_positions"


class MtpAcceptance(NamedTuple):
    """This rank's sums over its scored positions at temperature 1. ``overlap_sum`` holds ``sum(min(p, q))``, a draft
    sampled from ``q``'s chance of acceptance. ``greedy_sum`` holds ``p(argmax q)``, the same for a greedy draft, which
    is vLLM's default. ``top1_sum`` counts drafts naming the gold token."""

    overlap_sum: torch.Tensor
    greedy_sum: torch.Tensor
    top1_sum: torch.Tensor
    positions: torch.Tensor


@torch.no_grad()
def score_drafts(
    project: Callable[[torch.Tensor], torch.Tensor],
    trunk_rows: torch.Tensor,
    draft_rows: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int,
) -> MtpAcceptance:
    """``trunk_rows`` and ``draft_rows`` are aligned ``[N, H]`` hidden states predicting ``targets``; ``project`` takes
    them to logits. A chunk's two fp32 planes stay under ``chunk_rows`` rows each."""
    overlap = greedy = top1 = trunk_rows.new_zeros((), dtype=torch.float32)
    for trunk, draft, target in zip(
        trunk_rows.split(chunk_rows), draft_rows.split(chunk_rows), targets.split(chunk_rows), strict=True
    ):
        p = torch.softmax(at_least_fp32(project(trunk)), dim=-1)
        q = torch.softmax(at_least_fp32(project(draft)), dim=-1)
        drafted = q.argmax(dim=-1, keepdim=True)
        overlap = overlap + torch.minimum(p, q).sum()
        greedy = greedy + p.gather(-1, drafted).sum()
        top1 = top1 + (drafted.squeeze(-1) == target).sum()
    return MtpAcceptance(overlap, greedy, top1, targets.new_tensor(targets.numel()))
