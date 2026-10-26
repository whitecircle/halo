"""Fixed-sequence-length padding, and the collator wrapper built on it.

A consumer whose buffer shapes freeze on the first batch (the pipeline schedule's P2P buffers) needs
every later batch at the same width, so an over-length tensor raises rather than truncating loss
tokens.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.data.collators.packing import pad_tail_positions


def pad_to_fixed_length(
    tensors: Mapping[str, tuple[torch.Tensor, int | float]],
    max_length: int,
    hint: str,
    *,
    limit: str = ", over the pipeline's fixed shape",
) -> dict[str, torch.Tensor]:
    """Right-pad each ``{key: (tensor, fill)}`` along its last axis to ``max_length``.

    A tensor with no sequence axis (``dim < 2``, a per-example scalar) passes through, since padding
    it would corrupt its values. ``position_ids`` pads with the :func:`pad_tail_positions` ramp: a
    constant-0 pad would make every pad token its own varlen segment. An over-length tensor raises,
    worded with the caller's ``limit`` phrase and ``hint`` remedy.
    """
    out = {}
    for key, (tensor, fill) in tensors.items():
        if tensor.dim() < 2 or tensor.shape[-1] == max_length:
            out[key] = tensor
        elif tensor.shape[-1] > max_length:
            raise ValueError(
                f"Collated '{key}' is {tensor.shape[-1]} tokens{limit} max_length={max_length}; "
                f"truncating here would silently drop loss tokens. {hint}"
            )
        elif key == "position_ids":
            ramp = pad_tail_positions(max_length - tensor.shape[-1], tensor.dtype)
            out[key] = torch.cat([tensor, ramp.unsqueeze(0).expand(tensor.shape[0], -1)], dim=-1)
        else:
            out[key] = F.pad(tensor, (0, max_length - tensor.shape[-1]), value=fill)
    return out


class FixedShapeCollator:
    """Pad every batch ``inner`` produces to ``max_length`` (:func:`pad_to_fixed_length`).

    ``pad_values`` names the keys to pad and the fill each takes; a key the batch does not carry is
    skipped.
    """

    def __init__(self, inner, max_length: int, pad_values: Mapping[str, int]):
        self.inner = inner
        self.max_length = max_length
        self.pad_values = dict(pad_values)

    @property
    def required_dataset_columns(self) -> tuple[str, ...]:
        """The wrapped collator's own column requirement, forwarded.

        A caller that pins HF's signature columns unions these in. Without the forward the wrapper
        answers the empty tuple, a packing collator's ``seq_lengths`` is pruned from the dataset,
        ``DataCollatorWithPacking`` sees no packing metadata, and every packed row collates as a
        single document that attends across all of them. No error is raised: the loss stays finite.
        """
        return getattr(self.inner, "required_dataset_columns", ())

    def __call__(self, features):
        batch = self.inner(features)
        present = {key: (batch[key], fill) for key, fill in self.pad_values.items() if batch.get(key) is not None}
        batch.update(
            pad_to_fixed_length(
                present,
                self.max_length,
                "Truncate in tokenization, or raise max_length.",
                limit=" but the frozen shape is",
            )
        )
        return batch
