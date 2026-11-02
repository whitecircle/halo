"""Sequence-axis ops shared by the CP model and the CP trainers."""

from __future__ import annotations

import torch
import torch.distributed.nn.functional as dist_nn

from halo.distributed.context_parallel.config import CPConfig
from halo.kernels.logprobs import next_token_targets


def cp_chunk_bounds(seq_len: int, cp_rank: int, cp_size: int) -> tuple[int, int]:
    """``[start, end)`` of CP rank ``cp_rank``'s contiguous chunk of a ``seq_len``-token sequence."""
    if seq_len % cp_size != 0:
        raise ValueError(f"Sequence length {seq_len} must be divisible by context_parallel_size {cp_size}")
    chunk_size = seq_len // cp_size
    return cp_rank * chunk_size, (cp_rank + 1) * chunk_size


def cp_local_targets(labels: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """The next-token targets of CP rank ``cp_rank``'s chunk positions in the full pre-split ``[batch, seq_len]``
    labels: a non-final rank's last position targets the next chunk's first token, the sequence's last is
    ignored."""
    start, end = cp_chunk_bounds(labels.size(1), cp_rank, cp_size)
    return next_token_targets(labels, start, end - start)


def cp_shift_against_full_labels(
    logits: torch.Tensor, labels: torch.Tensor, cp_rank: int, cp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(shift_logits, shift_labels)`` of CP rank ``cp_rank``'s local logits against the full pre-split
    ``[batch, seq_len]`` labels. A non-final rank's last logit predicts the next chunk's first token, so every
    local logit stays supervised; only the final rank drops its last one."""
    targets = cp_local_targets(labels, cp_rank, cp_size)
    if cp_rank == cp_size - 1:
        return logits[:, :-1], targets[:, :-1]
    return logits, targets


def split_sequence_for_cp(tensor: torch.Tensor, cp_config: CPConfig) -> torch.Tensor:
    """This CP rank's chunk of a ``[batch, seq, ...]`` tensor."""
    if cp_config.cp_size == 1:
        return tensor
    start, end = cp_chunk_bounds(tensor.shape[1], cp_config.cp_rank, cp_config.cp_size)
    return tensor.narrow(1, start, end - start).contiguous()


def cp_sum_rows(local_values: torch.Tensor, cp_config: CPConfig | None) -> torch.Tensor:
    """COLLECTIVE over the CP group: the fp32 group sum of per-row values, whose SUM backward the world-wide mean
    gradient sync cancels."""
    values = local_values.float()
    if cp_config is None or cp_config.cp_size <= 1:
        return values
    return dist_nn.all_reduce(values, group=cp_config.process_group)


def reject_left_padding(attention_mask: torch.Tensor | None) -> None:
    """Raise on a left-padded row: the Ulysses path drops the mask and attends causally, so it tolerates only
    trailing pads, while leading ones would be attended by every real token.

    Checked on every batch (:class:`LeftPaddingGuard`): under a collator that left-pads prompts, a batch whose
    prompts happen to be equal-length certifies nothing about the next one.
    """
    if attention_mask is None:
        return
    # Only rows starting with a pad are scanned: one holding a token is left-padded, an all-zero one holds none.
    if attention_mask[attention_mask[:, 0] == 0].any():
        raise ValueError(
            "Context Parallelism received a LEFT-padded batch (row starts with attention_mask == 0). "
            "The CP attention path ignores the mask and runs dense causal attention, so every real "
            "token would attend the leading pads and the loss would differ from the same batch "
            "without CP. Use right padding (tokenizer.padding_side='right')."
        )


class LeftPaddingGuard:
    """``collator``'s batches, each refused when left-padded (:func:`reject_left_padding`) while still on the host.

    Every loader of a CP trainer collates through it, so the CP forward reads nothing back from the device to check
    its input. It checks the collated ``attention_mask``, the one the SFT and offline GRPO forwards consume; SMPO's
    concat builds its forward rows itself and flushes every pad behind their tokens.
    """

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        batch = self.collator(features)
        reject_left_padding(batch.get("attention_mask"))
        return batch
