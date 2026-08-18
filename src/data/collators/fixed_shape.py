"""Fixed-sequence-length wrapper around any collator.

A consumer whose buffer shapes freeze on the first batch (the pipeline schedule's P2P buffers)
needs every later batch at the same width. This wrapper pads the keys its caller names and refuses
an over-length one, rather than truncating loss tokens into a shape nothing downstream can detect.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.data.collators.packing import pad_tail_positions


class FixedShapeCollator:
    """Pad every batch ``inner`` produces to ``max_length`` along the sequence axis.

    ``pad_values`` names the keys to pad and the fill each takes; a key the batch does not carry is
    skipped, and a key with no sequence axis (per-example scalars) is left alone — padding one would
    corrupt its values.
    """

    def __init__(self, inner, max_length: int, pad_values: Mapping[str, int]):
        self.inner = inner
        self.max_length = max_length
        self.pad_values = dict(pad_values)

    @property
    def required_dataset_columns(self) -> tuple[str, ...]:
        """The wrapped collator's own column requirement, forwarded.

        A caller that pins HF's signature columns unions these in. Without the forward the wrapper
        answers the empty tuple — a packing collator's ``seq_lengths`` is then pruned from the
        dataset, ``DataCollatorWithPacking`` sees no packing metadata, and every packed row
        collates as ONE document that attends across all of them. Silent: the loss stays finite and
        merely wrong.
        """
        return getattr(self.inner, "required_dataset_columns", ())

    def __call__(self, features):
        batch = self.inner(features)
        for key, pad_value in self.pad_values.items():
            tensor = batch.get(key)
            # dim < 2 = no sequence axis (per-example scalars).
            if tensor is None or tensor.dim() < 2 or tensor.shape[-1] == self.max_length:
                continue
            if tensor.shape[-1] > self.max_length:
                raise ValueError(
                    f"Collated '{key}' is {tensor.shape[-1]} tokens but the frozen shape is "
                    f"max_length={self.max_length}; truncating here would silently drop loss "
                    f"tokens. Truncate in tokenization, or raise max_length."
                )
            elif key == "position_ids":
                # A constant-0 pad would make every pad token its own varlen segment
                # (:data:`~src.data.collators.packing.PAD_TAIL_SEGMENT_CHUNK`); the ramp keeps the tail a
                # handful of no-op segments. Labels there are already the pad value.
                ramp = pad_tail_positions(self.max_length - tensor.shape[-1], tensor.dtype)
                batch[key] = torch.cat([tensor, ramp.unsqueeze(0).expand(tensor.shape[0], -1)], dim=-1)
            else:
                batch[key] = torch.nn.functional.pad(tensor, (0, self.max_length - tensor.shape[-1]), value=pad_value)
        return batch
