"""Sequence-axis ops shared by the CP model and the CP trainers."""

from __future__ import annotations

import torch
import torch.distributed.nn.functional as dist_nn

from halo.distributed.context_parallel.config import CPConfig
from halo.kernels.logprobs import next_token_targets


def cp_chunk_bounds(seq_len: int, cp_rank: int, cp_size: int) -> tuple[int, int]:
    if seq_len % cp_size != 0:
        raise ValueError(f"Sequence length {seq_len} must be divisible by context_parallel_size {cp_size}")
    chunk_size = seq_len // cp_size
    return cp_rank * chunk_size, (cp_rank + 1) * chunk_size


def cp_local_targets(labels: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Next-token targets of this rank's chunk positions in the full ``[batch, seq_len]`` labels: a non-final rank's
    last position targets the next chunk's first token, the sequence's last is ignored."""
    start, end = cp_chunk_bounds(labels.size(1), cp_rank, cp_size)
    return next_token_targets(labels, start, end - start)


def cp_shift_against_full_labels(
    logits: torch.Tensor, labels: torch.Tensor, cp_rank: int, cp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(shift_logits, shift_labels)`` of this rank's logits against the full labels. A non-final rank's last logit
    predicts the next chunk's first token, so only the final rank drops its last one."""
    targets = cp_local_targets(labels, cp_rank, cp_size)
    if cp_rank == cp_size - 1:
        return logits[:, :-1], targets[:, :-1]
    return logits, targets


def split_sequence_for_cp(tensor: torch.Tensor, cp_config: CPConfig) -> torch.Tensor:
    if cp_config.cp_size == 1:
        return tensor
    start, end = cp_chunk_bounds(tensor.shape[1], cp_config.cp_rank, cp_config.cp_size)
    return tensor.narrow(1, start, end - start).contiguous()


def cp_sum_rows(local_values: torch.Tensor, cp_config: CPConfig | None) -> torch.Tensor:
    """COLLECTIVE over the CP group: fp32 sum of per-row values; the mean gradient sync cancels its SUM backward."""
    values = local_values.float()
    if cp_config is None or cp_config.cp_size <= 1:
        return values
    return dist_nn.all_reduce(values, group=cp_config.process_group)


def reject_left_padding(attention_mask: torch.Tensor | None) -> None:
    """Raise on a left-padded row: the Ulysses path drops the mask and attends causally, so leading pads would be
    attended by every real token. Checked on every batch, since an equal-length batch under a left-padding collator
    certifies nothing about the next one."""
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
    """``collator``'s batches, each refused when left-padded while still on the host, so the CP forward reads nothing
    back from the device. SMPO's concat builds its rows itself and flushes every pad behind them."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        batch = self.collator(features)
        reject_left_padding(batch.get("attention_mask"))
        return batch
