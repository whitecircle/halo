"""Token-level loss-mask composition shared across trainer families."""

from collections.abc import Mapping
from typing import Any

import torch


def effective_loss_mask(masks: Mapping[str, torch.Tensor | Any]) -> torch.Tensor | None:
    """The mask of loss-contributing tokens from a batch/result mapping, or ``None`` without one.

    Single home for the composition rule: ``completion_mask ∧ tool_mask`` when a ``tool_mask`` is
    present (TRL-native tool use and the environmental trainer — ``completion_mask`` alone is the
    ATTENTION-valid mask, still counting tool-output tokens the loss never trains on), else
    ``completion_mask``.
    """
    completion_mask = masks.get("completion_mask")
    if completion_mask is None:
        return None
    tool_mask = masks.get("tool_mask")
    return completion_mask if tool_mask is None else completion_mask * tool_mask
