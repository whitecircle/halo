"""Whole-model loads and their placement on this rank's device."""

import gc

import torch

from src.hardware import current_device


def move_model_to_local_device(model: torch.nn.Module) -> torch.nn.Module:
    """Move ``model`` to this rank's CUDA device (or CPU) and free the stale CPU copy."""
    model = model.to(current_device())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
